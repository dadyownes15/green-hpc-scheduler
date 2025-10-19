from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from src.baseline import Baseline
from src.hpc_env import HPCenv


Seconds = int
ActionIndex = int


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    submit_time: int
    run_time: int
    procs: int
    power: float


@dataclass
class PlannerState:
    current_time: int
    available_nodes: int
    running_jobs: List[Tuple[int, int, int]]  # (end_time, job_idx, procs)
    queue: List[int]
    next_arrival_idx: int
    actions: List[ActionIndex]


@dataclass
class _PlanState:
    reward: float
    state: PlannerState


class DynamicProgrammingBaseline(Baseline):
    """
    Dynamic-programming style planner that approximates an optimal schedule for a limited horizon.

    The planner operates on a light-weight copy of the environment dynamics. It performs a
    beam-search over possible action sequences, expanding promising states while pruning others.
    At the end of planning we replay the chosen action list on the real environment to obtain a
    full trace compatible with existing tooling.
    """

    def __init__(
        self,
        config_dict: dict,
        env: HPCenv,
        horizon_seconds: Seconds = 72 * 3600,
        beam_width: int = 8,
        schedule_top_k: int = 4,
        max_iterations: int = 1200,
    ) -> None:
        super().__init__(config_dict, env)
        self.name = "Dynamic Programming Approximation"
        self.horizon_seconds = int(horizon_seconds)
        self.beam_width = max(1, int(beam_width))
        self.schedule_top_k = max(1, int(schedule_top_k))
        self.max_iterations = max(1, int(max_iterations))

        self.max_queue_size = int(self.config_dict["max_queue_size"])
        self.delay_time_list: List[int] = list(self.config_dict["delay_time_list"])
        self.delay_time_list_length = len(self.delay_time_list)
        self.delay_offsets = sorted({0, self.delay_time_list_length // 2, self.delay_time_list_length - 1})
        self.delay_offsets = [offset for offset in self.delay_offsets if 0 <= offset < self.delay_time_list_length]
        self.skip_action_index = self.max_queue_size + self.delay_time_list_length

        self.max_wait_time = max(1, int(self.config_dict.get("max_wait_time", 20000)))
        self.eta = float(self.config_dict.get("eta", 0.0))

        self._restrict_env_to_horizon(self.env)
        self.jobs: List[JobRecord] = self._extract_job_records(self.env)
        self.job_order_keys: List[Tuple[int, int]] = [(job.submit_time, job.job_id) for job in self.jobs]

        if not self.jobs:
            raise ValueError("No jobs found within the requested planning horizon.")

        self.total_nodes = int(getattr(self.env.cluster, "total_node", 256))
        self.carbon = self.env.carbon_intensity

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, seed: int = 42, debug: bool = False):
        """Plan an action list and replay it on the real environment."""
        plan_state = self._beam_search(debug=debug)
        planned_actions = plan_state.state.actions
        return self._replay_actions(planned_actions, seed=seed, debug=debug)

    # ------------------------------------------------------------------ #
    # Planning logic
    # ------------------------------------------------------------------ #
    def _beam_search(self, debug: bool = False) -> _PlanState:
        start_state = self._initial_planner_state()
        frontier: Dict[Tuple, _PlanState] = {self._state_key(start_state): _PlanState(0.0, start_state)}
        completed: List[_PlanState] = []

        iteration = 0
        while frontier and iteration < self.max_iterations:
            iteration += 1
            next_frontier: Dict[Tuple, _PlanState] = {}

            for plan_state in frontier.values():
                state = plan_state.state
                if self._is_terminal(state):
                    completed.append(self._finalize_state(plan_state))
                    continue

                for successor_state, step_reward in self._successors(state):
                    total_reward = plan_state.reward + step_reward
                    key = self._state_key(successor_state)
                    stored = next_frontier.get(key)
                    if stored is None or total_reward > stored.reward:
                        next_frontier[key] = _PlanState(total_reward, successor_state)

            if not next_frontier:
                break

            # Keep only the best beam_width states.
            ordered = sorted(next_frontier.values(), key=lambda item: item.reward, reverse=True)
            frontier = {self._state_key(state.state): state for state in ordered[: self.beam_width]}

            if debug and iteration % 50 == 0:
                best_reward = ordered[0].reward if ordered else float("-inf")
                print(f"[DP] iteration={iteration}, frontier={len(frontier)}, best_reward={best_reward:.3f}")

        if not completed:
            completed = [self._finalize_state(plan_state) for plan_state in frontier.values()]

        if not completed:
            raise RuntimeError("Dynamic programming planner failed to produce any candidate schedule.")

        return max(completed, key=lambda item: item.reward)

    def _initial_planner_state(self) -> PlannerState:
        first_submit = self.jobs[0].submit_time
        state = PlannerState(
            current_time=first_submit,
            available_nodes=self.total_nodes,
            running_jobs=[],
            queue=[],
            next_arrival_idx=0,
            actions=[],
        )
        self._enqueue_arrivals(state, up_to_time=first_submit)
        self._auto_advance_if_idle(state)
        return state

    def _successors(self, state: PlannerState) -> List[Tuple[PlannerState, float]]:
        successors: List[Tuple[PlannerState, float]] = []

        for queue_pos in self._schedule_candidates(state):
            result = self._apply_schedule(state, queue_pos)
            if result is not None:
                successors.append(result)

        if state.current_time < self.horizon_seconds:
            for offset in self.delay_offsets:
                action_idx = self.max_queue_size + offset
                delta = self.delay_time_list[offset]
                result = self._apply_delay(state, delta, action_idx)
                if result is not None:
                    successors.append(result)

            if state.running_jobs:
                result = self._apply_skip(state)
                if result is not None:
                    successors.append(result)

        return successors

    # ------------------------------------------------------------------ #
    # State transition helpers
    # ------------------------------------------------------------------ #
    def _apply_schedule(self, state: PlannerState, queue_pos: int) -> Tuple[PlannerState, float] | None:
        if queue_pos >= len(state.queue):
            return None

        job_idx = state.queue[queue_pos]
        job = self.jobs[job_idx]
        if job.procs > state.available_nodes:
            return None

        new_state = self._clone_state(state)
        job_idx = new_state.queue.pop(queue_pos)
        job = self.jobs[job_idx]

        start_time = new_state.current_time
        assert job.submit_time <= start_time, "Job scheduled before arrival"

        end_time = start_time + job.run_time
        heapq.heappush(new_state.running_jobs, (end_time, job_idx, job.procs))
        new_state.available_nodes -= job.procs

        wait_seconds = start_time - job.submit_time
        wait_penalty = (wait_seconds / self.max_wait_time) * self.eta
        emission = self.carbon.getCarbonEmissions(job.power, start_time, end_time)
        carbon_penalty = float(emission) * (1.0 - self.eta)
        step_reward = -(wait_penalty + carbon_penalty)

        new_state.actions.append(queue_pos)
        self._auto_advance_if_idle(new_state)
        return new_state, step_reward

    def _apply_delay(self, state: PlannerState, delta: int, action_index: int) -> Tuple[PlannerState, float] | None:
        if delta <= 0:
            return None

        new_state = self._clone_state(state)
        target_time = min(state.current_time + delta, self.horizon_seconds)
        if target_time == state.current_time:
            return None

        self._advance_time(new_state, target_time)
        new_state.actions.append(action_index)
        return new_state, 0.0

    def _apply_skip(self, state: PlannerState) -> Tuple[PlannerState, float] | None:
        if not state.running_jobs:
            return None

        earliest_end, _, _ = state.running_jobs[0]
        target_time = min(earliest_end, state.current_time + 3600)
        if target_time <= state.current_time:
            target_time = earliest_end
        if target_time <= state.current_time:
            return None

        new_state = self._clone_state(state)
        self._advance_time(new_state, target_time)
        new_state.actions.append(self.skip_action_index)
        return new_state, 0.0

    def _advance_time(self, state: PlannerState, target_time: int) -> None:
        if target_time <= state.current_time:
            return

        while state.running_jobs and state.running_jobs[0][0] <= target_time:
            end_time, job_idx, procs = heapq.heappop(state.running_jobs)
            state.available_nodes += procs
            # Process arrivals at the exact completion time before continuing
            self._enqueue_arrivals(state, up_to_time=end_time)

        self._enqueue_arrivals(state, up_to_time=target_time)
        state.current_time = target_time
        self._auto_advance_if_idle(state)

    def _auto_advance_if_idle(self, state: PlannerState) -> None:
        if state.running_jobs or state.queue:
            return
        if state.next_arrival_idx >= len(self.jobs):
            state.current_time = min(state.current_time, self.horizon_seconds)
            return

        next_submit = self.jobs[state.next_arrival_idx].submit_time
        target = min(next_submit, self.horizon_seconds)
        if target > state.current_time:
            self._advance_time(state, target)
        else:
            self._enqueue_arrivals(state, up_to_time=state.current_time)

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _schedule_candidates(self, state: PlannerState) -> Sequence[int]:
        candidates: List[Tuple[float, int]] = []
        limit = min(len(state.queue), self.max_queue_size)
        for pos in range(limit):
            job_idx = state.queue[pos]
            job = self.jobs[job_idx]
            if job.procs > state.available_nodes:
                continue
            start = state.current_time
            end = start + job.run_time
            emission = self.carbon.getCarbonEmissions(job.power, start, end)
            candidates.append((float(emission), pos))

        candidates.sort(key=lambda item: item[0])
        return [pos for _, pos in candidates[: self.schedule_top_k]]

    def _enqueue_arrivals(self, state: PlannerState, up_to_time: int) -> None:
        while state.next_arrival_idx < len(self.jobs) and self.jobs[state.next_arrival_idx].submit_time <= up_to_time:
            job_idx = state.next_arrival_idx
            state.next_arrival_idx += 1
            self._insert_job(state.queue, job_idx)

    def _insert_job(self, queue: List[int], job_idx: int) -> None:
        key = self.job_order_keys[job_idx]
        lo, hi = 0, len(queue)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.job_order_keys[queue[mid]] <= key:
                lo = mid + 1
            else:
                hi = mid
        queue.insert(lo, job_idx)

    def _clone_state(self, state: PlannerState) -> PlannerState:
        return PlannerState(
            current_time=state.current_time,
            available_nodes=state.available_nodes,
            running_jobs=list(state.running_jobs),
            queue=list(state.queue),
            next_arrival_idx=state.next_arrival_idx,
            actions=list(state.actions),
        )

    def _state_key(self, state: PlannerState) -> Tuple:
        return (
            state.current_time,
            state.available_nodes,
            state.next_arrival_idx,
            tuple(state.queue),
            tuple(sorted(state.running_jobs)),
        )

    def _is_terminal(self, state: PlannerState) -> bool:
        if state.current_time >= self.horizon_seconds:
            return True
        if state.queue or state.running_jobs:
            return False
        return state.next_arrival_idx >= len(self.jobs)

    def _finalize_state(self, plan_state: _PlanState) -> _PlanState:
        penalty = self._estimate_backlog_penalty(plan_state.state)
        return _PlanState(plan_state.reward - penalty, plan_state.state)

    def _estimate_backlog_penalty(self, state: PlannerState) -> float:
        pending = set(state.queue)
        pending.update(job_idx for _, job_idx, _ in state.running_jobs)
        pending.update(range(state.next_arrival_idx, len(self.jobs)))

        penalty = 0.0
        for job_idx in pending:
            job = self.jobs[job_idx]
            start = max(state.current_time, job.submit_time)
            end = start + job.run_time
            emission = self.carbon.getCarbonEmissions(job.power, start, end)
            penalty += float(emission) * (1.0 - self.eta)
            if job.submit_time <= state.current_time:
                wait = state.current_time - job.submit_time
                penalty += (wait / self.max_wait_time) * self.eta
        return penalty

    # ------------------------------------------------------------------ #
    # Replay on real environment
    # ------------------------------------------------------------------ #
    def _replay_actions(self, actions: Iterable[ActionIndex], seed: int, debug: bool = False):
        eval_env = copy.deepcopy(self.env)
        eval_env.trace_enabled = True
        eval_env.reset(seed=seed, options={})

        total_reward = 0.0
        reward_components = {"wait": 0.0, "carbon": 0.0, "total": 0.0}

        for step_idx, action in enumerate(actions):
            _, rwd, terminated, _, info = eval_env.step(action)
            total_reward += rwd
            reward_components["wait"] += float(info.get("reward_wait", 0.0))
            reward_components["carbon"] += float(info.get("reward_carbon", 0.0))
            reward_components["total"] += float(info.get("reward_total", rwd))

            if debug:
                print(
                    f"[replay] step={step_idx} t={eval_env.current_timestamp} action={action} "
                    f"reward={rwd:.3f} cumulative={total_reward:.3f}"
                )

            if eval_env.current_timestamp >= self.horizon_seconds or terminated:
                break

        reward_components["total"] = float(total_reward)
        return float(total_reward), reward_components, eval_env.get_action_trace()

    # ------------------------------------------------------------------ #
    # Environment preprocessing
    # ------------------------------------------------------------------ #
    def _restrict_env_to_horizon(self, env: HPCenv) -> None:
        if not hasattr(env, "loads"):
            return

        limit = 0
        for job in env.loads.loaded_jobs:
            if job.submit_time <= self.horizon_seconds:
                limit += 1
            else:
                break

        if limit == 0:
            raise ValueError("No jobs available before the planning horizon.")

        env.config_dict["episode_length"] = limit
        env.loads.loaded_jobs = env.loads.loaded_jobs[:limit]
        if getattr(env.loads, "episode_jobs", None):
            env.loads.episode_jobs = env.loads.episode_jobs[:limit]
        env.last_job_in_batch = limit
        env.num_job_in_batch = limit

    def _extract_job_records(self, env: HPCenv) -> List[JobRecord]:
        records: List[JobRecord] = []
        power_per_proc = float(self.config_dict.get("constant_power_per_processor", 500))
        for job in env.loads.loaded_jobs:
            if job.submit_time > self.horizon_seconds:
                break
            power = float(job.power_usage if getattr(job, "power_usage", -1.0) > 0 else power_per_proc * job.request_number_of_processors)
            records.append(
                JobRecord(
                    job_id=int(job.job_id),
                    submit_time=int(job.submit_time),
                    run_time=int(job.run_time),
                    procs=int(job.request_number_of_processors),
                    power=power,
                )
            )
        records.sort(key=lambda rec: (rec.submit_time, rec.job_id))
        return records
