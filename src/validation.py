from src.baseline import Baseline, MedianBaseline
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

from typing import List, Type
class Validation():


    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """

    def __init__(self, model_dir) -> None:
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


    def validate_policy(self, n_eval_episodes, checkpoints, mode):
        self.mode = mode.lower()

        assert self.mode in ["validation", "testing"]

        self.env = ActionMasker(HPCenv(config_dict=self.config_dict, mode=self.mode), action_mask_fn= mask_fn)

        stats_dict = {}
 
        for checkpoint in checkpoints:
            base_model =   MaskablePPO("MlpPolicy", self.env)
            base_model.load(self.model_dir + "/logs/"+checkpoint)
            stats_dict[checkpoint] = {}
            stats_dict[checkpoint]["rewards"] = []
            stats_dict[checkpoint]["delay_history"] = []
            stats_dict[checkpoint]["job_scheduled_history"] = []
            stats_dict[checkpoint]["action_log_history"] = []
            
            for i in range(n_eval_episodes):
                    model_reward, delay_history, job_scheduled_history, actions_log = self.evaluate_policy(seed=i, model=base_model)
                    stats_dict[checkpoint]['rewards'].append(model_reward)
                    stats_dict[checkpoint]['delay_history'].append(delay_history)
                    stats_dict[checkpoint]['job_scheduled_history'].append(job_scheduled_history)
                    stats_dict[checkpoint]['action_log_history'].append(actions_log)


        carbon_intensity = CarbonIntensity(green_win_length=24, normalize=False)
        
        return self.process_metrics(stats_dict=stats_dict, carbon_intensity_calculator=carbon_intensity, config_dict=self.config_dict) 
 

    def run_baselines(self, n_eval_episodes):
        pass
    
    def deep_dive(self, seed, model):
       pass 

    def evaluate_policy(self,seed, model : MaskablePPO):
        obs, _ = self.env.reset(seed=seed, options={})
        
        terminated = False
        total_reward = 0
        step_count = 0  # Add a counter
        while not terminated:
            # Retrieve current action mask
            action_masks = get_action_masks(self.env)
            # --- DEBUGGING STEP ---
            # Print the mask and the number of valid actions
            num_valid_actions = sum(action_masks)
            if num_valid_actions <= 1 and step_count < 10: # Print for the first 10 steps
                print(f"Step {step_count}: Valid Actions = {num_valid_actions}, Mask = {action_masks}")
            # --------------------

            action, _states = model.predict(obs, action_masks=action_masks, deterministic = True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)

        delay_history = self.env.unwrapped.delay_history
        job_scheduled_history = self.env.unwrapped.scheduled_job_history
        action_log = self.env.unwrapped.action_log

        return total_reward, delay_history, job_scheduled_history, action_log


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
            stats_dict (dict): The dictionary containing evaluation results per checkpoint.
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