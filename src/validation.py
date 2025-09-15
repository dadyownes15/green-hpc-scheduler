from src.baseline import Baseline, MedianBaseline, FCFSBaseline
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
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

from typing import List, Type
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path
class Validation():


    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """


    def validate_policy(self, n_eval_episodes, checkpoints, mode, debug = False):
        assert self.model_dir is not None

        self.mode = mode.lower()
        assert self.mode in ["validation", "test"]

        if debug:
            print("Validating policy on data from: ", self.mode)

        self.env = ActionMasker(HPCenv(config_dict=self.config_dict, mode=self.mode, debug=debug), action_mask_fn= mask_fn)

        stats_dict = {}
 
        for checkpoint in checkpoints:
            if debug:
                print("Initating checkpoint: ", checkpoint)
            base_model =   MaskablePPO("MlpPolicy", self.env)
            base_model.load(self.model_dir + "/logs/"+checkpoint)
            stats_dict[checkpoint] = {}
            stats_dict[checkpoint]["rewards"] = []
            stats_dict[checkpoint]["delay_history"] = []
            stats_dict[checkpoint]["job_scheduled_history"] = []
            stats_dict[checkpoint]["action_log_history"] = []
            
            for i in range(n_eval_episodes):
                    if debug and i % 10 == 0: 
                        print("Val episode: ", i)
                    model_reward, delay_history, job_scheduled_history, actions_log = self.evaluate_policy(seed=i, model=base_model)
                    stats_dict[checkpoint]['rewards'].append(model_reward)
                    stats_dict[checkpoint]['delay_history'].append(delay_history)
                    stats_dict[checkpoint]['job_scheduled_history'].append(job_scheduled_history)
                    stats_dict[checkpoint]['action_log_history'].append(actions_log)


        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        
        return self.process_metrics(stats_dict=stats_dict, carbon_intensity_calculator=carbon_intensity, config_dict=self.config_dict) 
 
    def load_dir(self,model_dir):
        self.model_dir = model_dir 
        config = configparser.ConfigParser()
        config_path = os.path.join(os.getcwd(), self.model_dir, 'config.json')

        try:
            with open(config_path, 'r') as f:
                self.config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please ensure it exists.")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file at '{config_path}' is not a valid JSON file. Please check its syntax.")

       
    def run_baselines(self, n_eval_episodes, mode, debug = False):

        baselines = [
            MedianBaseline(config_dict=self.config_dict, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug)),
            FCFSBaseline(config_dict=self.config_dict, env=HPCenv(config_dict=self.config_dict, mode=mode, debug=debug)) 
        ]
        
        stats_dict = {}
        for baseline in baselines:
            print("running baseline: ", baseline.name)
            stats_dict[baseline.name] = {}
            stats_dict[baseline.name]["rewards"] = []
            stats_dict[baseline.name]["delay_history"] = []
            stats_dict[baseline.name]["job_scheduled_history"] = []
            stats_dict[baseline.name]["action_log_history"] = []
    
            for i in range(n_eval_episodes): 
                reward, delay_history, job_scheduled_history, action_log_history = baseline.run(seed=i, debug=debug)
                stats_dict[baseline.name]['rewards'].append(reward)
                stats_dict[baseline.name]['delay_history'].append(delay_history)
                stats_dict[baseline.name]['job_scheduled_history'].append(job_scheduled_history)
                stats_dict[baseline.name]['action_log_history'].append(action_log_history)


        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        return self.process_metrics(stats_dict=stats_dict, carbon_intensity_calculator=carbon_intensity, config_dict=self.config_dict) 
    

    def deep_dive(self, seed, model):
       pass 

    def evaluate_policy(self,seed, model : MaskablePPO, debug = False):
        obs, _ = self.env.reset(seed=seed, options={})
        
        terminated = False
        total_reward = 0
        step_count = 0  # Add a counter
        while not terminated:
            action_masks = get_action_masks(self.env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic = True)
            dist = model.policy.get_distribution(obs=model.policy.features_extractor(model.policy.obs_to_tensor(obs)[0]),action_masks=action_masks)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

        delay_history = self.env.unwrapped.delay_history
        job_scheduled_history = self.env.unwrapped.scheduled_job_history
        action_log = self.env.unwrapped.action_log

        return total_reward, delay_history, job_scheduled_history, action_log

    def evaluate_policy_with_trace(self, seed, model: MaskablePPO, debug=False):
        """Runs a single episode and also returns the per-step action trace."""
        obs, _ = self.env.reset(seed=seed, options={})
        terminated = False
        total_reward = 0
        while not terminated:
            action_masks = get_action_masks(self.env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

        delay_history = self.env.unwrapped.delay_history
        job_scheduled_history = self.env.unwrapped.scheduled_job_history
        action_log = self.env.unwrapped.action_log
        action_trace = self.env.unwrapped.get_action_trace() if hasattr(self.env.unwrapped, 'get_action_trace') else []

        return total_reward, delay_history, job_scheduled_history, action_log, action_trace


    def render_input_model(self, model_path: str, seed: int, step_interval=1, name="model_rendering"):
        """
        Renders an episode of a trained model by generating plots for each step and compiling them into a video.
        """
        # 1. Instantiate the environment with rendering enabled
        self.config_dict['generate_rendering'] = True
        self.config_dict['name'] = name
        render_env = ActionMasker(HPCenv(workload_path=self.workload_path, config_dict=self.config_dict), action_mask_fn=mask_fn)

        # 2. Load the trained model
        model = MaskablePPO.load(model_path, env=render_env)

        # 3. Run the episode and render each step
        obs, _ = render_env.reset(seed=seed, options={})
        terminated = False
        step_count = 0

        while not terminated:
            if step_count % step_interval == 0:
                render_env.render(step_count=step_count)

            action_masks = get_action_masks(render_env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            step_count += 1

        # 4. Generate the video from the saved images
        video_gen = VideoGenerator(path=render_env.dir_path)
        video_gen.generate_video()
        print(f"Rendering complete. Video saved at {render_env.dir_path}/rendering.mp4")


    def process_metrics(self, stats_dict, carbon_intensity_calculator, config_dict):
        """
        Processes a dictionary of evaluation statistics and returns key performance metrics,
        including an analysis of the agent's action log.

        Args:
            stats_dict (dict): The dictionary containing evaluation results per checkpoin"median"t.
                              Expected structure:
                                {
                                    'checkpoint_name': {
                                        'rewards': [...],
                                        'delay_history': [[(start, end), ...], ...],
                                        'job_scheduled_history': [[Job, ...], ...],
                                        'action_log_history': [action_log_dict, ...] 
                                    }
                                }
            carbon_intensity_calculator (CarbonIntensity): An instance of the CarbonIntensity class
                                                            with the correct mode set.
            config_dict (dict): The configuration dictionary used to run the environment.
                                Must contain 'procs_per_node' and 'idle_power'.

        Returns:
            dict: A nested dictionary with calculated metrics for each checkpoint.
        """
        processed_stats = {}

        for checkpoint, data in stats_dict.items():
            processed_stats[checkpoint] = {}
            
            # Initialize lists to hold per-episode data
            all_wait_times = []
            all_response_times = []
            all_slowdowns = []
            all_carbon_emissions = []
            all_weighted_carbon_emissions = []
            all_utilization_data = []

            # Action Log Aggregation
            total_schedule_actions = 0
            total_delay_fixed_actions = 0
            total_delay_wait_actions = 0
            total_actions = 0
            
            # Initialize an aggregation dictionary for granular delay actions
            fixed_delay_counts = collections.defaultdict(int)
            wait_job_counts = collections.defaultdict(int)

            num_episodes = len(data.get('rewards', []))
            if num_episodes == 0:
                continue

            for i in range(num_episodes):
                job_scheduled_history = data['job_scheduled_history'][i]
                
                # Carbon and Utilization Calculations
                total_time_span = 0
                if job_scheduled_history:
                    last_job = job_scheduled_history[-1]
                    total_time_span = (last_job.scheduled_time + last_job.run_time)

                episode_carbon_emissions = 0
                episode_weighted_carbon_emissions = 0
                utilization_events = collections.defaultdict(int)
                
                # Action Log for this episode
                action_log = data['action_log_history'][i]
                total_schedule_actions += action_log['schedule']
                total_delay_fixed_actions += action_log['delay_fixed']
                total_delay_wait_actions += action_log['delay_wait']
                
                # Aggregate granular delay counts
                for idx, count in enumerate(action_log['delay_fixed_indices']):
                    delay_time = config_dict['delay_time_list'][idx]
                    fixed_delay_counts[delay_time] += count
                
                for idx, count in enumerate(action_log['delay_wait_indices']):
                    num_jobs = idx + 1
                    wait_job_counts[num_jobs] += count

                # Metrics for each job
                episode_wait_times = []
                episode_response_times = []
                episode_slowdowns = []

                for job in job_scheduled_history:
                    job_finish_time = job.scheduled_time + job.run_time
                    wait_time = job.scheduled_time - job.submit_time
                    response_time = wait_time + job.run_time
                    slowdown = response_time / job.run_time if job.run_time > 0 else float('inf')
                    
                    episode_wait_times.append(wait_time)
                    episode_response_times.append(response_time)
                    episode_slowdowns.append(slowdown)

                    job_emissions = carbon_intensity_calculator.getCarbonEmissions(job.power_usage, job.scheduled_time, job_finish_time)
                    episode_carbon_emissions += job_emissions
                    episode_weighted_carbon_emissions += job_emissions * job.carbon_consideration
                    
                    utilization_events[job.scheduled_time] += job.request_number_of_processors
                    utilization_events[job_finish_time] -= job.request_number_of_processors

                # Utilization calculation
                if total_time_span > 0:
                    times = sorted(utilization_events.keys())
                    current_procs = 0
                    last_time = 0
                    total_time_procs = 0
                    
                    for t in times:
                        duration = t - last_time
                        if duration > 0:
                            total_time_procs += current_procs * duration
                        current_procs += utilization_events[t]
                        last_time = t
                    
                    if total_time_span > last_time:
                        total_time_procs += current_procs * (total_time_span - last_time)

                    total_procs = 256
                    avg_utilization = (total_time_procs / total_time_span) / total_procs
                    all_utilization_data.append(avg_utilization)
                
                # Store per-episode results
                all_wait_times.extend(episode_wait_times)
                all_response_times.extend(episode_response_times)
                all_slowdowns.extend(episode_slowdowns)
                all_carbon_emissions.append(episode_carbon_emissions)
                all_weighted_carbon_emissions.append(episode_weighted_carbon_emissions)

            # Final Aggregation
            total_actions = total_schedule_actions + total_delay_fixed_actions + total_delay_wait_actions
            
            # Compile final metrics
            if all_wait_times:
                processed_stats[checkpoint]["Avg Wait"] = np.mean(all_wait_times)
                processed_stats[checkpoint]["Max Wait"] = np.max(all_wait_times)
                processed_stats[checkpoint]["Avg Response"] = np.mean(all_response_times)
                processed_stats[checkpoint]["Avg Slowdown"] = np.mean(all_slowdowns)
                processed_stats[checkpoint]["Carbon Emissions"] = np.mean(all_carbon_emissions)
                processed_stats[checkpoint]["Weighted Carbon Emissions"] = np.mean(all_weighted_carbon_emissions)
                processed_stats[checkpoint]["System Utilization"] = np.mean(all_utilization_data)

            if total_actions > 0:
                processed_stats[checkpoint]["Action Analysis"] = {
                    "Total Actions": total_actions,
                    "Schedule Action Percentage": (total_schedule_actions / total_actions) * 100,
                    "Fixed Delay Percentage": (total_delay_fixed_actions / total_actions) * 100,
                    "Wait Delay Percentage": (total_delay_wait_actions / total_actions) * 100,
                    "Fixed Delays": {f"{t}s": count for t, count in fixed_delay_counts.items()},
                    "Wait for Jobs": {f"{j} jobs": count for j, count in wait_job_counts.items()}
                }

        return processed_stats


    def _compute_timeseries(self, job_scheduled_history, mode='validation', step_seconds=None):
        """
        Computes time series for: carbon intensity, used processors, avg queue wait, new arrivals.
        Returns (times, ci_values, used_procs, avg_waits, new_arrivals, seconds_per_slot).
        """
        assert job_scheduled_history is not None and len(job_scheduled_history) > 0

        # Determine episode bounds
        start_time = min(j.submit_time for j in job_scheduled_history)
        end_time = max(j.scheduled_time + j.run_time for j in job_scheduled_history)

        # Carbon setup
        ci = CarbonIntensity(green_win_length=24, normalize=False)
        try:
            ci.set_mode(mode)
        except Exception:
            ci.set_mode('validation')
        seconds_per_slot = ci.seconds_per_slot if step_seconds is None else step_seconds
        assert seconds_per_slot > 0

        # Sampling grid
        n_slots = max(1, int((end_time - start_time) // seconds_per_slot) + 1)
        times = [start_time + i * seconds_per_slot for i in range(n_slots)]

        # Carbon intensity samples aligned similarly to env encoding
        episode_start_hour_offset = start_time // 3600
        ci_values = []
        for i, t in enumerate(times):
            slot_idx = int(t // seconds_per_slot)
            list_idx = (episode_start_hour_offset * (3600 // ci.seconds_per_slot) + slot_idx) % ci.total_slots
            ci_values.append(ci.carbonIntensityList[list_idx])

        # Used processors via event scan
        events = []
        for j in job_scheduled_history:
            events.append((j.scheduled_time, j.request_number_of_processors))
            events.append((j.scheduled_time + j.run_time, -j.request_number_of_processors))
        events.sort()
        used_procs = []
        current = 0
        e_idx = 0
        for t in times:
            # apply all events at or before t
            while e_idx < len(events) and events[e_idx][0] <= t:
                current += events[e_idx][1]
                e_idx += 1
            used_procs.append(current)

        # Average queue wait (event-driven) and new arrivals per slot
        # Build arrival counts per slot
        arrivals_by_slot = {}
        for j in job_scheduled_history:
            a_slot = int((j.submit_time - start_time) // seconds_per_slot)
            arrivals_by_slot[a_slot] = arrivals_by_slot.get(a_slot, 0) + 1

        # Prepare sorted events for arrivals/departures
        arrivals_sorted = sorted(job_scheduled_history, key=lambda j: j.submit_time)
        departures_sorted = sorted(job_scheduled_history, key=lambda j: j.scheduled_time)
        ai = di = 0
        waiting_count = 0
        waiting_submit_sum = 0.0
        waiting_present = set()

        avg_waits = []
        new_arrivals = []
        for t in times:
            # Remove any jobs that start running at or before t
            while di < len(departures_sorted) and departures_sorted[di].scheduled_time <= t:
                job = departures_sorted[di]
                if job.job_id in waiting_present:
                    waiting_present.remove(job.job_id)
                    waiting_count -= 1
                    waiting_submit_sum -= job.submit_time
                di += 1
            # Add any arrivals up to t (that are not already running at t)
            while ai < len(arrivals_sorted) and arrivals_sorted[ai].submit_time <= t:
                job = arrivals_sorted[ai]
                if job.scheduled_time > t and job.job_id not in waiting_present:
                    waiting_present.add(job.job_id)
                    waiting_count += 1
                    waiting_submit_sum += job.submit_time
                ai += 1
            # Average waiting time at t
            avg_waits.append(t - (waiting_submit_sum / waiting_count) if waiting_count > 0 else 0.0)
            # New arrivals in this slot index
            slot = int((t - start_time) // seconds_per_slot)
            new_arrivals.append(arrivals_by_slot.get(slot, 0))

        return times, ci_values, used_procs, avg_waits, new_arrivals, seconds_per_slot

    def render_timeseries_plot(self, job_scheduled_history, name="timeseries", output_dir="renderings", mode='validation', step_seconds=None):
        """
        Renders a static 4-line timeseries plot across the full episode timeframe.
        Lines: carbon intensity, used processors, avg wait in queue (s), new arrivals per slot.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{name}.png")

        times, ci_values, used_procs, avg_waits, new_arrivals, step = self._compute_timeseries(
            job_scheduled_history, mode=mode, step_seconds=step_seconds
        )

        # Build multi-axis plot with true units
        fig, ax_ci = plt.subplots(figsize=(18, 6))
        ax_ci.set_title('Episode Timeseries Overview')
        ax_ci.set_xlabel('Time (s)')
        ci_line, = ax_ci.plot(times, ci_values, color='seagreen', label='Carbon Intensity')
        ax_ci.set_ylabel('gCO2/kWh', color='seagreen')

        # Used processors on first right axis as histogram
        ax_proc = ax_ci.twinx()
        proc_bar = ax_proc.bar(times, used_procs, width=(step or 1), align='edge', alpha=0.25, color='royalblue', label='Used Processors')
        ax_proc.set_ylabel('Processors', color='royalblue')

        # Additional right axes for avg wait and arrivals
        ax_wait = ax_ci.twinx()
        ax_wait.spines["right"].set_position(("axes", 1.08))
        wait_line, = ax_wait.plot(times, avg_waits, color='darkorange', label='Avg Wait (s)')
        ax_wait.set_ylabel('Avg Wait (s)', color='darkorange')

        ax_arr = ax_ci.twinx()
        ax_arr.spines["right"].set_position(("axes", 1.16))
        arr_bar = ax_arr.bar(times, new_arrivals, width=(step or 1), align='edge', alpha=0.4, color='crimson', label='New Arrivals')
        ax_arr.set_ylabel('Arrivals per slot', color='crimson')

        # Compose legend
        lines = [ci_line, wait_line]
        labels = [l.get_label() for l in lines]
        labels += ['Used Processors', 'New Arrivals']
        lines += [proc_bar, arr_bar]
        ax_ci.legend(lines, labels, loc='upper right')

        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path



    def collect_traces(self, n_eval_episodes, checkpoint, mode, debug=False):
        """
        Validates a single checkpoint over N episodes, returning a list of
        dicts containing the episode traces and histories.
        """
        assert self.model_dir is not None
        self.mode = mode.lower()
        assert self.mode in ["validation", "test"]
        self.env = ActionMasker(HPCenv(config_dict=self.config_dict, mode=self.mode, debug=debug), action_mask_fn= mask_fn)

        model = MaskablePPO("MlpPolicy", self.env)
        model.load(self.model_dir + "/logs/" + checkpoint)

        episodes = []
        for i in range(n_eval_episodes):
            if debug and i % 10 == 0:
                print("Trace episode:", i)
            total_reward, delay_history, job_scheduled_history, action_log, action_trace = self.evaluate_policy_with_trace(seed=i, model=model, debug=debug)
            episodes.append({
                'seed': i,
                'reward': total_reward,
                'delay_history': delay_history,
                'job_scheduled_history': job_scheduled_history,
                'action_log': action_log,
                'action_trace': action_trace,
            })
        return episodes
