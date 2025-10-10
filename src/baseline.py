import numpy as np
import time
import random
import configparser
import os
import ast
import abc

from src.hpc_env import HPCenv
from src.utils import get_config_as_dict

class Baseline(abc.ABC):
    def __init__(self, config_dict, env: HPCenv):
        self.config_dict = config_dict
        self.env = env
        assert self.config_dict is not None, "Config dict, did not parse"
        
    @abc.abstractmethod
    def run(self, seed):
        """
        The main scheduling logic for the baseline.
        Subclasses must implement this method.
        """
        pass

class PercentileBaseline(Baseline):
    def __init__(self,config_dict, env, percentile, mode):
        """
        Initializes the MedianBaseline, which schedules jobs based on carbon intensity.
        """
        super().__init__(config_dict,env)
        self.name = f"{percentile}-percentile Baseline"
        self.mode = mode
        self.percentile = percentile
        
    def run(self, seed=42, debug = False):
        self.env.reset(seed=seed, options={})
        terminated = False
        obs = self.env.build_observation()
        reward = 0
        step_count = 0
        
        assert len(self.env.job_queue) != 0, "NO jobs are start, will lead to error"
        
        while not terminated:
            # Replace all constants with config_dict keys
            queue_features_len = self.config_dict['max_queue_size'] * self.config_dict['job_feature']
            running_features_len = self.config_dict['run_win_length'] * self.config_dict['run_feature']
            carbon_start_idx = queue_features_len + running_features_len
            
            current_carbon_intensity = obs[carbon_start_idx]
            
            forecast_start_idx = carbon_start_idx + self.config_dict['green_feature_constant']
            carbon_forecast = obs[forecast_start_idx:]
            
            assert len(carbon_forecast) == (self.config_dict['green_forecast_length'] - 1) * self.config_dict['green_feature_pr_timeslot']
            assert (self.config_dict['green_feature_pr_timeslot'] == 1), "NOT IMPLEMENTED FOR NON multiple green feature pr timeslot"


            cutoffs = {'validation': {10: np.float64(49.23500000000001),
  25: np.float64(61.78025),
  50: np.float64(83.49033333333334),
   100: np.float64(-10000)
  },

 'test': {10: np.float64(24.151466666666668),
  25: np.float64(33.60491666666667),
  50: np.float64(51.55383333333333),
  100: np.float64(-10000)}}

            
            carbon_forecast_percentile = cutoffs[self.mode][self.percentile]

            if current_carbon_intensity < carbon_forecast_percentile:
                mask = self.env.valid_action_mask()
                job_mask = mask[:self.config_dict['max_queue_size']]
                
                if not job_mask.any():
                    
                    if mask[self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']]:
                        # Skip to next finished job
                        action = self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']
                        obs, rwd, terminated, truncated, info = self.env.step(action)
                    else:
                        # Delay 300 secunds
                        action = self.config_dict['max_queue_size']
                        obs, rwd, terminated, truncated, info = self.env.step(action)
                    reward += rwd
                else:
                    # Schedule the first job in queue
                    action = np.where(job_mask)[0][0]
                    
                    obs, rwd, terminated, truncated, info = self.env.step(action)
                    reward += rwd
            else:
                # Delay 300 sekunds
                action = self.config_dict['max_queue_size']
                obs, rwd, terminated, truncated, info = self.env.step(action)
            
            step_count += 1
            if debug:
                print("Step: ", step_count, " Reward: ", rwd, " Action: ", action)

 
        return reward, self.env.get_action_trace() 
class FCFSBaseline(Baseline):
    def __init__(self, config_dict, env):
        """
        Initializes the FCFSBaseline, which schedules jobs on a first-come, first-served basis.
        """
        super().__init__(config_dict, env)
        self.name = "FCFS Baseline"

    def run(self, seed=42, debug=False):
        """
        Runs the FCFS scheduling algorithm.

        The logic is as follows:
        1. Check if any jobs in the queue can be scheduled based on the valid action mask.
        2. If yes, schedule the one that arrived first (i.e., has the lowest index in the queue).
        3. If no schedulable jobs are available, the agent waits. It prioritizes skipping
           time until the next running job completes. If that is not possible, it takes a default
           small delay action.
        """
        self.env.reset(seed=seed, options={})
        terminated = False
        reward = 0
        step_count = 0

        assert len(self.env.job_queue) != 0, "Job queue is empty at the start, this may cause an error."
        while not terminated:
            # Get the mask of all valid actions from the environment
            mask = self.env.valid_action_mask()
            
            # The first part of the action mask corresponds to scheduling jobs from the queue
            job_mask = mask[:self.config_dict['max_queue_size']]
            
            # Check if there is any valid job to schedule
            if job_mask.any():
                # FCFS policy: find the index of the first 'True' value in the job mask,
                # which corresponds to the earliest arrived, schedulable job.
                action = np.where(job_mask)[0][0]
            else:
                # If no job can be scheduled, the agent must wait.
                # The index for the "skip to next event" action.
                skip_action_idx = self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']
                
                # Prioritize skipping to the next job completion event if it's a valid move.
                if mask[skip_action_idx]:
                    action = skip_action_idx
                else:
                    # Otherwise, take the default delay action (e.g., wait 300 seconds).
                    # This is typically the first action after the job actions.
                    action = self.config_dict['max_queue_size']

            # Execute the chosen action in the environment
            obs, rwd, terminated, truncated, info = self.env.step(action)
            reward += rwd
            
            step_count += 1
            if debug:
                print(f"Step: {step_count}, Action: {action}, Reward: {rwd:.2f}")
            
        return reward, self.env.get_action_trace() 

class FCFSEasyBackfillBaseline(Baseline):
    def __init__(self, config_dict, env, backfill_max_runtime=None):
        """
        FCFS with EASY-style backfilling:
        - Keep a reservation for the head-of-queue job (index 0).
        - If head can't start, allow backfilling with a later job only if it won't
          delay the head's earliest start (heuristic/safe check described below).
        """
        super().__init__(config_dict, env)
        self.name = "FCFS + EASY Backfilling Baseline"
        # Conservative cap used when we can't infer a precise safe window.
        self.backfill_max_runtime = (
            backfill_max_runtime
            if backfill_max_runtime is not None
            else self.config_dict.get("easy_backfill_max_runtime", 3600)  # 1h default
        )

    # ---------- Helpers ----------
    def _time_to_next_finish(self):
        """
        Best-effort estimate of the time until the next running job finishes.
        Tries env hooks if available; otherwise falls back to None.
        """
        # Env-provided helper (if you have one)
        if hasattr(self.env, "get_time_to_next_finish"):
            try:
                time_to_next_resource_free, resource = self.env._update_next_resource_release()

                return time_to_next_resource_free
            except Exception:
                pass

        # Inspect running jobs (common pattern)
        try:
            if hasattr(self.env, "running_jobs"):
                remaining = []
                for j in getattr(self.env, "running_jobs", []):
                    rem = None
                    for attr in ("remaining_time", "rem_time", "time_left"):
                        if hasattr(j, attr):
                            rem = getattr(j, attr)
                            break
                    if rem is not None:
                        remaining.append(rem)
                if remaining:
                    return float(min(remaining))
        except Exception:
            pass

        return None  # Unknown window

    def _get_job_runtime(self, job):
        """
        Extract a queued job's (requested) runtime in seconds.
        Supports multiple common attribute names.
        """
        for attr in ("run_time", "runtime", "duration", "req_time", "walltime", "requested_time"):
            if hasattr(job, attr):
                try:
                    val = getattr(job, attr)
                    if val is not None:
                        return float(val)
                except Exception:
                    continue
        return None

    # ---------- Policy ----------
    def run(self, seed=42, debug=False):
        """
        Algorithm:
        1) If the head-of-queue job (index 0) can start, start it (pure FCFS).
        2) Otherwise, consider backfill candidates among currently schedulable jobs (indices > 0).
           A candidate is "safe" if:
             - It can start now, AND
             - (a) its runtime <= time to next completion (if we can estimate), OR
             - (b) its runtime <= easy_backfill_max_runtime (conservative cap).
           Choose the earliest-arrived safe candidate (lowest index).
        3) If no safe candidate exists, skip to the next completion event if possible;
           otherwise take the default small delay action.
        """
        self.env.reset(seed=seed, options={})
        terminated = False
        reward = 0.0
        step_count = 0

        assert len(self.env.job_queue) != 0, "Job queue is empty at start."

        # Precompute action indices
        max_q = self.config_dict["max_queue_size"]
        delay_list_len = self.config_dict["delay_time_list_length"]
        skip_to_next_event_idx = max_q + delay_list_len      # "jump to next finish" action
        default_delay_idx = max_q                             # typically the small delay (e.g., 300s)

        while not terminated:
            mask = self.env.valid_action_mask()
            job_mask = mask[:max_q]

            # Case 1: head-of-queue can be scheduled now -> do it (pure FCFS)
            if job_mask.any() and job_mask[0]:
                action = 0

            else:
                # Case 2: head cannot start -> try safe backfilling
                # Identify schedulable jobs beyond the head
                schedulable_idxs = [i for i in np.where(job_mask)[0] if i != 0]

                backfill_action = None
                if schedulable_idxs:
                    safe_window = self._time_to_next_finish()  # may be None

                    for idx in schedulable_idxs:  # FCFS among backfillable
                        # sanity: ensure index maps to a real job
                        if idx >= len(self.env.job_queue):
                            continue
                        job = self.env.job_queue[idx]
                        rt = self._get_job_runtime(job)

                        # Decide safety
                        ok = False
                        if (safe_window is not None) and (rt is not None):
                            ok = (rt <= safe_window)
                        elif (rt is not None) and (self.backfill_max_runtime is not None):
                            ok = (rt <= float(self.backfill_max_runtime))

                        if ok:
                            backfill_action = idx
                            break

                if backfill_action is not None:
                    action = backfill_action
                else:
                    # Case 3: no safe backfill -> advance time
                    if skip_to_next_event_idx < len(mask) and mask[skip_to_next_event_idx]:
                        action = skip_to_next_event_idx
                    else:
                        action = default_delay_idx  # small delay fallback

            # Step the environment
            obs, rwd, terminated, truncated, info = self.env.step(action)
            reward += rwd
            step_count += 1

            if debug:
                print(f"Step {step_count:4d} | Action {action:3d} | Reward {rwd: .4f}")

        return reward, self.env.get_action_trace()
