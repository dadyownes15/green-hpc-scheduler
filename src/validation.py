from src.baseline import Baseline, PercentileBaseline, FCFSBaseline, FCFSEasyBackfillBaseline
from src.hpc_env import HPCenv
from src.utils import VideoGenerator, get_config_as_dict, mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from src.carbon_intensity import CarbonIntensity
import configparser
import json
import os
import numpy as np
import collections
import math
import torch

from typing import List, Type, Optional, Sequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
class Validation():
    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """
    def validate_policy(
        self,
        n_eval_episodes: int,
        checkpoints: Optional[Sequence[str]],
        mode: str,
        debug: bool = False,
        checkpoint_dir: str | Path = "logs",
    ):
        if self.model_dir is None:
            raise RuntimeError("Call load_dir(...) before validate_policy().")

        self.mode = mode.lower()
        if self.mode not in {"training", "validation", "test"}:
            raise ValueError("mode must be 'training', 'validation', or 'test'.")

        if debug:
            print("Validating policy on data from:", self.mode)

        if not self.config_dict.get("reward_type"):
            self.config_dict["reward_type"] = "wait_abs_ems"
        # Enable tracing in env so we can collect action traces for analysis
        self.env = ActionMasker(
            HPCenv(config_dict=self.config_dict, mode=self.mode, debug=debug, trace_enabled=True),
            action_mask_fn=mask_fn,
        )

        checkpoint_dir_path = self._resolve_checkpoint_dir(checkpoint_dir)

        if checkpoints is None:
            if not checkpoint_dir_path.is_dir():
                raise FileNotFoundError(f"No logs directory found at '{checkpoint_dir_path}'.")
            checkpoints = sorted(p.name for p in checkpoint_dir_path.iterdir() if p.is_file())
            if not checkpoints:
                raise ValueError(f"No checkpoints found in '{checkpoint_dir_path}'.")
        else:
            checkpoints = list(checkpoints)

        stats_dict: dict[str, dict[str, list]] = {}

        for checkpoint in checkpoints:
            if debug:
                print("Initiating checkpoint:", checkpoint)
            checkpoint_path = self._resolve_checkpoint_path(checkpoint_dir_path, checkpoint)
            model = MaskablePPO.load(str(checkpoint_path), env=self.env)
            stats_dict[checkpoint] = {
                "rewards": [],
                "job_scheduled_history": [],
                "action_traces": [],
                "reward_components": [],
            }

            for i in range(n_eval_episodes):
                if debug and i % 10 == 0:
                    print("Val episode:", i)
                total_reward, job_hist, action_trace, reward_components = self.evaluate_policy(
                    seed=i, model=model, debug=debug
                )
                stats_dict[checkpoint]["rewards"].append(total_reward)
                stats_dict[checkpoint]["job_scheduled_history"].append(job_hist)
                stats_dict[checkpoint]["action_traces"].append(action_trace)
                stats_dict[checkpoint]["reward_components"].append(reward_components)

        
        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        carbon_intensity.set_mode(mode)
        return self.process_metrics(
            stats_dict=stats_dict,
            carbon_intensity_calculator=carbon_intensity,
            config_dict=self.config_dict,
        ), stats_dict

    def _resolve_checkpoint_dir(self, checkpoint_dir: str | Path) -> Path:
        base_dir = Path(self.model_dir)
        provided = Path(checkpoint_dir)
        if provided.is_absolute():
            candidates = [provided]
        else:
            candidates = [base_dir / provided, provided]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_dir: Path, checkpoint_name: str) -> Path:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path
        if checkpoint_path.suffix:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found in '{checkpoint_dir}'.")
        zipped = checkpoint_path.with_suffix(".zip")
        if zipped.exists():
            return zipped
        raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found in '{checkpoint_dir}'.")

    def validate_model(self, n_eval_episodes, model: MaskablePPO, mode: str, debug: bool = False):
        """
        Validate a provided model over a number of episodes.

        Args:
            n_eval_episodes: Number of evaluation episodes.
            model: A trained RL model (e.g., MaskablePPO) to evaluate.
            mode: One of "training", "validation", or "test". Controls which dataset the env uses.
            debug: If True, prints progress information.

        Returns:
            Tuple of (processed_metrics_dict, raw_stats_dict).
        """
        assert getattr(self, 'config_dict', None) is not None, "Call load_dir(...) to set config first."

        self.mode = mode.lower()
        assert self.mode in ["training", "validation", "test"]

        if debug:
            print("Validating provided model on data from:", self.mode)

        print(self.config_dict)

        self.env = ActionMasker(
            HPCenv(config_dict=self.config_dict, mode=self.mode, debug=debug, trace_enabled=True),
            action_mask_fn=mask_fn,
        )

        stats_dict = {
            "model": {
                "rewards": [],
                "job_scheduled_history": [],
                "action_traces": [],
                "reward_components": [],
            }
        }
        total_reward, job_hist, action_trace, reward_components = self.evaluate_policy(
                    seed=1, model=model, debug=debug
                )
        stats_dict["model"]["rewards"].append(total_reward)
        stats_dict["model"]["job_scheduled_history"].append(job_hist)
        stats_dict["model"]["action_traces"].append(action_trace)
        stats_dict["model"]["reward_components"].append(reward_components)

        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        return self.process_metrics(stats_dict=stats_dict, carbon_intensity_calculator=carbon_intensity, config_dict=self.config_dict), stats_dict


    def load_dir(self, model_dir: str = None, config_dict: dict = None):
        """
        Initialize validation context from either a model directory or a config dict.

        - If only `model_dir` is provided, loads `config.json` from that directory.
        - If only `config_dict` is provided, uses it directly (useful for baselines).
        - If both are provided, loads from `model_dir` and applies `config_dict` as overrides.

        Args:
            model_dir: Path to a directory containing a `config.json` and `logs/` with checkpoints.
            config_dict: Configuration dictionary to use directly or to override file-loaded config.
        """
        self.model_dir = model_dir if model_dir is not None else getattr(self, 'model_dir', None)

        # Start from provided config_dict if given
        provided_cfg = None
        if config_dict is not None:
            if not isinstance(config_dict, dict):
                raise TypeError("config_dict must be a dict if provided")
            provided_cfg = config_dict

        file_cfg = None
        if self.model_dir is not None:
            # Prefer joining directly; model_dir may already be absolute
            config_path = os.path.join(self.model_dir, 'config.json')
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.getcwd(), config_path)

            try:
                with open(config_path, 'r') as f:
                    file_cfg = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Configuration file not found at '{config_path}'. Please ensure it exists."
                )
            except json.JSONDecodeError:
                raise ValueError(
                    f"Configuration file at '{config_path}' is not a valid JSON file. Please check its syntax."
                )

        # Decide final config
        if file_cfg is None and provided_cfg is None:
            raise ValueError("Either model_dir or config_dict must be provided to load configuration.")

        if file_cfg is not None and provided_cfg is not None:
            # Override file config with provided values
            self.config_dict = {**file_cfg, **provided_cfg}
        else:
            self.config_dict = provided_cfg or file_cfg

        return self

       
    def run_baselines(self, n_eval_episodes, mode, debug = False):

        baselines = [

            FCFSBaseline(config_dict=self.config_dict, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug, trace_enabled=True)), 
                        PercentileBaseline(config_dict=self.config_dict, percentile=10, mode=mode, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug, trace_enabled=True)),
            PercentileBaseline(config_dict=self.config_dict, percentile=25, mode=mode, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug, trace_enabled=True)),
            PercentileBaseline(config_dict=self.config_dict, percentile = 50, mode=mode, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug, trace_enabled=True)), 
       
                    ]
        
        stats_dict = {}
        for baseline in baselines:
            stats_dict[baseline.name] = {
                "rewards": [],
                "action_traces": [],
                "job_scheduled_history": [],
                "reward_components": [],
            }
            print("Executing baseline: ", baseline.name)
            for i in range(n_eval_episodes): 
                print("Episode ", i)
                reward, reward_components, action_trace = baseline.run(seed=i, debug=debug)
                stats_dict[baseline.name]["rewards"].append(reward)
                stats_dict[baseline.name]["action_traces"].append(action_trace)
                stats_dict[baseline.name]["job_scheduled_history"].append(baseline.env.scheduled_job_history)
                stats_dict[baseline.name]["reward_components"].append(reward_components)
        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        return self.process_metrics(stats_dict=stats_dict, carbon_intensity_calculator=carbon_intensity, config_dict=self.config_dict), stats_dict
    

    def evaluate_policy(self,seed, model : MaskablePPO, debug = False):
        obs, _ = self.env.reset(seed=seed, options={})
        
        terminated = False
        truncated = False
        total_reward = 0.0
        reward_wait_component = 0.0
        reward_carbon_component = 0.0
        reward_total_component = 0.0
        step = 0;
        while not terminated and not truncated:
            action_masks = get_action_masks(self.env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic = True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            # print("step: ", step, "reward: ", reward)
            total_reward += float(reward)
            reward_wait_component += float(info.get('reward_wait', 0.0))
            reward_carbon_component += float(info.get('reward_carbon', 0.0))
            reward_total_component += float(info.get('reward_total', reward))
            step += 1

        job_scheduled_history = self.env.unwrapped.scheduled_job_history
        action_trace = self.env.unwrapped.get_action_trace() if hasattr(self.env.unwrapped, 'get_action_trace') else []
        reward_components = {
            "wait": float(reward_wait_component),
            "carbon": float(reward_carbon_component),
            "total": float(reward_total_component),
        }
        
        return total_reward, job_scheduled_history, action_trace, reward_components

    def process_metrics(self, stats_dict, carbon_intensity_calculator, config_dict):
        """
        Single-episode validation metrics, including:
        - Avg/Max Wait, Avg Response, Avg Slowdown
        - Episode Duration (last finish time)
        - Carbon Emissions (raw and weighted)
        - System Utilization
        - Validation Reward  <-- NEW
        """
        import collections
        import numpy as np

        processed_stats = {}

        for checkpoint, data in stats_dict.items():
            # --- Single-episode guard ---
            rewards_list = data.get('rewards', [])
            num_episodes = len(rewards_list)
            assert num_episodes == 1, f"Expected exactly one episode, got {num_episodes}"

            # Validation reward (assumed scalar per episode)
            env_reward = float(rewards_list[0])

            reward_components_list = data.get("reward_components", [])
            if reward_components_list:
                assert len(reward_components_list) == num_episodes, (
                    "Mismatch between reward components and episode count"
                )
                component_entry = reward_components_list[0] or {}
            else:
                component_entry = {}

            jobs = (data.get('job_scheduled_history', [[]])[0]) if 'job_scheduled_history' in data else []
            assert jobs, "No jobs found in job_scheduled_history[0]"

            # --- Action analysis (trace preferred, else legacy) ---
            total_schedule_actions = 0
            total_delay_fixed_actions = 0
            total_delay_wait_actions = 0
            fixed_delay_counts = collections.defaultdict(int)
            wait_job_counts = collections.defaultdict(int)

            action_trace = data.get('action_traces', [[]])[0] if 'action_traces' in data else []
            if action_trace:
                for entry in action_trace:
                    if entry.get('action_type') == 'schedule':
                        total_schedule_actions += 1
                    elif entry.get('action_type') == 'delay':
                        kind = entry.get('delay_kind')
                        val = int(entry.get('delay_value') or 0)
                        if kind == 'fixed':
                            total_delay_fixed_actions += 1
                            fixed_delay_counts[val] += 1
                        elif kind == 'wait':
                            total_delay_wait_actions += 1
                            wait_job_counts[val] += 1
            elif 'action_log_history' in data:
                action_log = data['action_log_history'][0]
                total_schedule_actions += action_log.get('schedule', 0)
                total_delay_fixed_actions += action_log.get('delay_fixed', 0)
                total_delay_wait_actions += action_log.get('delay_wait', 0)
                for idx, count in enumerate(action_log.get('delay_fixed_indices', [])):
                    fixed_delay_counts[config_dict['delay_time_list'][idx]] += count
                for idx, count in enumerate(action_log.get('delay_wait_indices', [])):
                    wait_job_counts[idx + 1] += count

            # --- Per-job metrics ---
            waits, responses, slowdowns = [], [], []
            episode_last_finish = 0.0
            utilization_events = collections.defaultdict(int)
            episode_carbon_emissions = 0.0
            episode_weighted_carbon_emissions = 0.0

            for job in jobs:
                wait = job.scheduled_time - job.submit_time
                finish = job.scheduled_time + job.run_time
                response = wait + job.run_time
                slowdown = (response / job.run_time) if job.run_time > 0 else float('inf')

                waits.append(wait)
                responses.append(response)
                slowdowns.append(slowdown)
                episode_last_finish = max(episode_last_finish, finish)

                # Carbon
                job_emissions = carbon_intensity_calculator.getCarbonEmissions(
                    job.power_usage, job.scheduled_time, finish
                )
                episode_carbon_emissions += job_emissions
                episode_weighted_carbon_emissions += job_emissions * getattr(job, 'carbon_consideration', 1.0)

                # Utilization (+ at start, - at finish)
                utilization_events[job.scheduled_time] += job.request_number_of_processors
                utilization_events[finish] -= job.request_number_of_processors

            # --- Utilization across [0, episode_last_finish] ---
            system_utilization = None
            if episode_last_finish > 0 and utilization_events:
                times = sorted(utilization_events.keys())
                current_procs = 0
                last_t = times[0]
                time_procs = 0
                for t in times:
                    if t > last_t:
                        time_procs += current_procs * (t - last_t)
                        last_t = t
                    current_procs += utilization_events[t]
                total_procs = config_dict.get('cluster_total_procs', 256)
                system_utilization = (time_procs / episode_last_finish) / total_procs

            # --- Compile results (now includes Validation Reward) ---
            reward_type = str(config_dict.get("reward_type", "")).lower()
            eta_value_raw = config_dict.get("eta", 0.0)
            try:
                eta_value = float(eta_value_raw)
            except (TypeError, ValueError):
                eta_value = 0.0
            if reward_type in {"wait_abs_ems", "wait_abs_ems_clip"} and eta_value == 0.0:
                validation_reward = -float(episode_carbon_emissions)
            else:
                validation_reward = env_reward

            avg_wait = np.mean(waits)
            processed_stats[checkpoint] = {
                "Validation Reward": validation_reward,
                "val_objective": val_objective(avg_wait,episode_carbon_emissions,eta=eta_value),
                "Avg Wait": float(avg_wait),
                "Max Wait": float(np.max(waits)),
                "Avg Response": float(np.mean(responses)),
                "Avg Slowdown": float(np.mean(slowdowns)),
                "Episode Duration": float(episode_last_finish),
                "Carbon Emissions": float(episode_carbon_emissions),
                "Weighted Carbon Emissions": float(episode_weighted_carbon_emissions),
            }
            if component_entry:
                wait_component_val = float(component_entry.get("wait", 0.0))
                carbon_component_val = float(component_entry.get("carbon", 0.0))
                total_component_val = float(
                    component_entry.get("total", wait_component_val + carbon_component_val)
                )
                processed_stats[checkpoint]["Reward Wait Component"] = wait_component_val
                processed_stats[checkpoint]["Reward Carbon Component"] = carbon_component_val
                processed_stats[checkpoint]["Reward Total Component"] = total_component_val
            if validation_reward != env_reward:
                processed_stats[checkpoint]["Env Reward"] = env_reward
            if system_utilization is not None:
                processed_stats[checkpoint]["System Utilization"] = float(system_utilization)

            total_actions = total_schedule_actions + total_delay_fixed_actions + total_delay_wait_actions
            if total_actions > 0:
                processed_stats[checkpoint]["Action Analysis"] = {
                    "Total Actions": int(total_actions),
                    "Schedule Action Percentage": 100.0 * total_schedule_actions / total_actions,
                    "Fixed Delay Percentage": 100.0 * total_delay_fixed_actions / total_actions,
                    "Wait Delay Percentage": 100.0 * total_delay_wait_actions / total_actions,
                    "Fixed Delays": {f"{t}s": int(c) for t, c in fixed_delay_counts.items()},
                    "Wait for Jobs": {f"{j} jobs": int(c) for j, c in wait_job_counts.items()},
                }

        return processed_stats



    @staticmethod
    def _action_log_from_trace(action_trace):
        """Create an action_log-style dict from a trace list."""
        fixed_counts = collections.defaultdict(int)
        wait_counts = collections.defaultdict(int)
        schedule = delay_fixed = delay_wait = 0
        for e in action_trace or []:
            if e.get('action_type') == 'schedule':
                schedule += 1
            elif e.get('action_type') == 'delay':
                kind = e.get('delay_kind')
                v = int(e.get('delay_value') or 0)
                if kind == 'fixed':
                    delay_fixed += 1
                    fixed_counts[v] += 1
                elif kind == 'wait':
                    delay_wait += 1
                    wait_counts[v] += 1
        return {
            'schedule': schedule,
            'delay_fixed': delay_fixed,
            'delay_wait': delay_wait,
            'fixed_delay_counts': dict(fixed_counts),
            'wait_job_counts': dict(wait_counts),
        }


    def _select_episode_trace(self, action_trace, episode_index: int | None = None):
        """Normalize action_trace input to a single episode trace list.

        Accepts either a single trace (list of dict-like entries) or a collection
        of traces as produced by validation stats (list of per-episode traces).
        """
        if action_trace is None:
            return []

        # Allow callers to pass the stats dict directly.
        if isinstance(action_trace, dict) and 'action_traces' in action_trace:
            return self._select_episode_trace(action_trace['action_traces'], episode_index=episode_index)

        # Convert numpy arrays or other sequence-like containers to lists.
        if hasattr(action_trace, 'tolist') and not isinstance(action_trace, list):
            return self._select_episode_trace(action_trace.tolist(), episode_index=episode_index)

        if isinstance(action_trace, (list, tuple)):
            action_list = list(action_trace)
            if not action_list:
                return []
            first = action_list[0]
            # Already a single trace (list of dict-like entries)
            if hasattr(first, 'get'):
                return action_list
            # A list of traces -> select requested episode (default first)
            if isinstance(first, (list, tuple)):
                idx = episode_index if episode_index is not None else 0
                if idx < 0 or idx >= len(action_list):
                    raise IndexError(f"Episode index {idx} out of range for {len(action_list)} traces")
                return self._select_episode_trace(action_list[idx], episode_index=None)

        raise TypeError(
            "action_trace must be a per-episode trace (list of dict-like entries) or a collection of traces."
        )


    def _compute_timeseries_from_trace(self, action_trace, mode='validation', rolling_window: int = 10):
        """
        Builds exact node utilization from schedule events in action_trace.
        Returns (usage_segments, delay_spans, ci_times, ci_values, wait_times_x, wait_rolling_avg, queue_times, queue_lengths) where:
          - usage_segments: list of {start, end, used_nodes}
          - delay_spans: list of (start, end) for shaded skipped periods
          - ci_times/ci_values: carbon intensity line
          - wait_times_x/wait_rolling_avg: job-schedule-time indexed rolling average wait (seconds)
          - queue_times/queue_lengths: per-step queue length at timestamp_after
        """
        if not action_trace:
            return [], [], [], [], [], [], [], []

        if not hasattr(action_trace[0], 'get'):
            raise TypeError(
                "Each action trace entry must provide a dict-like interface. Did you pass the list of traces instead of a single episode trace?"
            )

        # Carbon setup
        ci = CarbonIntensity(green_win_length=24, normalize=False)
        try:
            ci.set_mode(mode)
        except Exception:
            ci.set_mode('validation')

        def ci_at_time(t):
            return float(ci.intensity_at(t))

        # Build event list from schedule entries
        procs_per_node = max(1, int(self.config_dict.get('procs_per_node', 1)))
        events = []  # (time, delta_nodes)
        t_min = None
        t_max = None
        # For rolling wait-time series
        schedule_times = []
        wait_values = []
        # For queue length over time
        queue_times = []
        queue_lengths = []
        for e in action_trace:
            t_before = int(e.get('timestamp_before') or 0)
            t_after = int(e.get('timestamp_after') or t_before)
            t_min = t_before if t_min is None else min(t_min, t_before)
            t_max = t_after if t_max is None else max(t_max, t_after)
            q_after = e.get('queue_len_after')
            if q_after is not None:
                queue_times.append(t_after)
                try:
                    queue_lengths.append(int(q_after))
                except Exception:
                    pass
            if e.get('action_type') == 'schedule':
                start = t_before
                run_time = int(e.get('scheduled_job_run_time') or 0)
                end = start + run_time
                nodes = e.get('scheduled_job_nodes')
                if nodes is None:
                    procs = int(e.get('scheduled_job_procs') or 0)
                    # Approximate nodes if nodes not present
                    nodes = (procs + procs_per_node - 1) // procs_per_node
                nodes = int(nodes)
                if nodes > 0 and run_time > 0:
                    events.append((start, nodes))
                    events.append((end, -nodes))
                # Collect wait for rolling average if available
                w = e.get('scheduled_job_wait_time')
                if w is not None:
                    schedule_times.append(start)
                    wait_values.append(int(w))

        events.sort()

        # Build usage segments from events (piecewise constant)
        usage_segments = []
        used_nodes = 0
        if not events:
            # No schedules; flat zero usage over episode bounds
            if t_min is None or t_max is None or t_max <= t_min:
                return [], [], [], [], [], [], [], []
            usage_segments.append({'start': t_min, 'end': t_max, 'used_nodes': 0})
        else:
            last_time = events[0][0] if t_min is None else t_min
            # If first event happens after t_min, include initial zero segment
            if last_time > (t_min or last_time):
                usage_segments.append({'start': t_min, 'end': last_time, 'used_nodes': 0})
            idx = 0
            n = len(events)
            while idx < n:
                t = events[idx][0]
                if t > last_time:
                    usage_segments.append({'start': last_time, 'end': t, 'used_nodes': used_nodes})
                    last_time = t
                # Apply all deltas at time t
                while idx < n and events[idx][0] == t:
                    used_nodes += events[idx][1]
                    idx += 1
            # Tail segment to t_max if available
            if t_max is not None and t_max > last_time:
                usage_segments.append({'start': last_time, 'end': t_max, 'used_nodes': used_nodes})

        # Collect delay spans from trace
        delay_spans = []
        for e in action_trace:
            if e.get('action_type') == 'delay':
                s = int(e.get('timestamp_before') or 0)
                e_t = int(e.get('timestamp_after') or s)
                if e_t > s:
                    delay_spans.append((s, e_t))

        # Carbon intensity at boundaries
        t_start = 0
        t_end = action_trace[-1]["timestamp_after"]
        ci_times = np.arange(t_start,t_end,60)
        ci_values = [ci_at_time(t) for t in ci_times]

        # Rolling average of wait values aligned to schedule times
        wait_times_x = []
        wait_rolling_avg = []
        if schedule_times:
            # Sort by time to be safe
            pairs = sorted(zip(schedule_times, wait_values), key=lambda x: x[0])
            times_sorted, waits_sorted = zip(*pairs)
            window = max(1, int(rolling_window))
            csum = 0
            q = collections.deque()
            for t, w in zip(times_sorted, waits_sorted):
                q.append(w)
                csum += w
                if len(q) > window:
                    csum -= q.popleft()
                wait_times_x.append(t)
                wait_rolling_avg.append(csum / len(q))

        return usage_segments, delay_spans, ci_times, ci_values, wait_times_x, wait_rolling_avg, queue_times, queue_lengths

    def render_timeseries_plot(self, action_trace, name="timeseries", output_dir="renderings", mode='validation', rolling_window: int = 10, shade_delays: bool = True, max_delay_spans: int | None = None, debug: bool = False, save_png: bool = True, episode_index: int | None = None, display_carbon_itensity = True):
        """
        Renders a segment-based timeseries plot using the action_trace.
        Visualizes carbon intensity, used processors, rolling avg wait, queue length, and optional delay shading.

        Interactive display in notebooks uses mpl-interactions pan/zoom. In your
        notebook, enable the ipympl backend first:
          %matplotlib widget

        Returns (png_path_or_None, matplotlib_figure). A PNG is saved if requested.

        Args:
            action_trace: Either a single episode trace (list of dict entries) or a
                collection of traces such as stats['action_traces'].
            episode_index: Optional index when passing a collection of traces.
        """
        import time
        from typing import Optional, Tuple
        from IPython.display import display
        try:
            # panhandler + zoom_factory provide interaction helpers
            from mpl_interactions import panhandler, zoom_factory
        except Exception:
            panhandler = None
            zoom_factory = None
            if debug:
                print("[render] mpl_interactions unavailable; falling back to static Matplotlib figure.")

        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, f"{name}.png")

        selected_trace = self._select_episode_trace(action_trace, episode_index=episode_index)
        usage_segments, delay_spans, ci_times, ci_values, wait_x, wait_avg, q_times, q_lens = self._compute_timeseries_from_trace(selected_trace, mode=mode, rolling_window=rolling_window)
        if debug:
            print(f"[render] series: usage={len(usage_segments)}, delays={len(delay_spans)}, ci_points={len(ci_times)}, wait_points={len(wait_x)}, queue_points={len(q_times)}")
            if q_lens:
                print(f"[render] queue stats: min={min(q_lens)}, max={max(q_lens)}")

        # Build interactive Matplotlib figure (shown via ipympl widget backend)
        with plt.ioff():
            fig, ax_ci = plt.subplots(figsize=(18, 6))

        if display_carbon_itensity:
            ax_ci.set_title('Episode Timeseries Overview')
            ax_ci.set_xlabel('Time (s)')
            ci_line, = ax_ci.plot(ci_times, ci_values, color='seagreen', label='Carbon Intensity')
            ax_ci.set_ylabel('gCO2/kWh', color='seagreen')

        # Used nodes on first right axis as duration bars per segment
        ax_proc = ax_ci.twinx()
        proc_bars = []
        for seg in usage_segments:
            width = max(0, seg['end'] - seg['start'])
            if width == 0:
                continue
            bar = ax_proc.bar(seg['start'], seg['used_nodes'], width=width, align='edge', alpha=0.25, color='royalblue')
            proc_bars.append(bar)
        ax_proc.set_ylabel('Nodes', color='royalblue')

        # Add rolling wait and queue length lines on a third axis (offset on right)
        ax_queue = None
        queue_line = None
        if q_times and q_lens:
            ax_queue = ax_ci.twinx()
            ax_queue.spines['right'].set_position(('axes', 1.1))
            # Step-like appearance for queue (holds until next change)
            queue_line, = ax_queue.plot(q_times, q_lens, color='purple', drawstyle='steps-post', label='Queue Length')
            ax_queue.set_ylabel('Queue Length', color='purple')
            ax_queue.set_ylim(0, max(q_lens) + 1)

        # Shade skipped (delay) segments on the carbon axis
        added_label = False
        if shade_delays and delay_spans:
            spans_iter = delay_spans
            if isinstance(max_delay_spans, int) and max_delay_spans is not None and max_delay_spans >= 0:
                spans_iter = delay_spans[:max_delay_spans]
            for s, e in spans_iter:
                if e > s:
                    ax_ci.axvspan(s, e, color='gray', alpha=0.15, label='Skipped' if not added_label else None)
                    added_label = True

        # Legend
        labels = ['Carbon Intensity']
        handles = [ci_line]
        if proc_bars:
            handles.append(proc_bars[0])
            labels.append('Used Nodes')
        if queue_line is not None:
            labels.append('Queue Length')
            handles.append(queue_line)
        if added_label:
            from matplotlib.patches import Patch
            handles.append(Patch(facecolor='gray', alpha=0.15, label='Skipped'))
            labels.append('Skipped')
        ax_ci.legend(handles, labels, loc='upper right')

        use_plotly = panhandler is None or zoom_factory is None

        # Hook up interactions: scroll-zoom on primary axis; pan on figure
        if not use_plotly:
            _disconnect_zoom = zoom_factory(ax_ci)
            _pan_handler = panhandler(fig)

        # Save PNG if requested
        if save_png:
            fig.savefig(png_path, dpi=150)

        if use_plotly:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig_i = make_subplots(specs=[[{"secondary_y": True}]])

                # Carbon intensity on primary y
                fig_i.add_trace(
                    go.Scatter(x=ci_times, y=ci_values, name='Carbon Intensity', line=dict(color='seagreen')),
                    secondary_y=False,
                )

                # Queue length on secondary y (right), step line
                if q_times and q_lens:
                    fig_i.add_trace(
                        go.Scatter(x=q_times, y=q_lens, name='Queue Length', line=dict(color='purple'), line_shape='hv'),
                        secondary_y=True,
                    )

                # Used nodes as wide bars (on secondary y to avoid scale issues)
                if usage_segments:
                    x_vals = [seg['start'] for seg in usage_segments]
                    y_vals = [seg['used_nodes'] for seg in usage_segments]
                    widths = [max(0, seg['end'] - seg['start']) for seg in usage_segments]
                    fig_i.add_trace(
                        go.Bar(x=x_vals, y=y_vals, width=widths, name='Used Nodes', marker_color='royalblue', opacity=0.3),
                        secondary_y=True,
                    )

                # Shade delay spans
                if shade_delays and delay_spans:
                    spans_iter = delay_spans
                    if isinstance(max_delay_spans, int) and max_delay_spans is not None and max_delay_spans >= 0:
                        spans_iter = delay_spans[:max_delay_spans]
                    for s, e in spans_iter:
                        if e > s:
                            fig_i.add_shape(type="rect", x0=s, x1=e, y0=0, y1=1, xref='x', yref='paper', fillcolor='gray', opacity=0.15, line_width=0)

                fig_i.update_layout(
                    title_text='Episode Timeseries Overview',
                    barmode='overlay',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                fig_i.update_xaxes(title_text='Time (s)')
                fig_i.update_yaxes(title_text='gCO2/kWh', secondary_y=False)
                fig_i.update_yaxes(title_text='Queue / Nodes', secondary_y=True)

                if debug:
                    print(f"[render] Built Plotly figure in {time.time() - t0:.3f}s")
                return (png_path if save_png else None), fig_i
            except Exception as e:
                if debug:
                    print(f"[render] Plotly fallback failed: {e}")

        # Display interactive canvas in notebooks (ipympl path or Plotly fallback failure)
        try:
            display(fig.canvas if not use_plotly else fig)
        except Exception:
            # Fall back to plt.show() if display is unavailable
            plt.show()

        if debug:
            print(f"[render] Built interactive Matplotlib figure in {time.time() - t0:.3f}s")

        return (png_path if save_png else None), fig



def val_objective(avg_wait,total_carbon_emissions,eta):
    fcfs_wait_baseline = 6241.40
    best_carbon = 18596164.61

    """     return eta*(fcfs_wait_baseline/avg_wait)+(1-eta)*(best_carbon/total_carbon_emissions) """

    return eta*avg_wait/fcfs_wait_baseline + total_carbon_emissions/best_carbon * (1-eta)

    """

    worst carbon = 28000
    
    wait = 4990
    carbon = 280

    opt carbon = 185
    opt wait 6012 

    eta = 0.5

    6012 / 4990 * 0.5 + 185/285 * 0.5 = 1,3591615664
    0.6 + 0,7567567568
    
    6012 / 4990 * 0.5  + 280-185/280 * 0.5 = 0,3392857143

    4990/6012 * 0.5 + 285/185*0.5 =
    0,3320026613 + 0,7307692308


    17000/6012 * 0.5 + 270/185

    2,7419354839 * 0.5 + 1,4594594595
    
    """