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
    def __init__(self, config: configparser.ConfigParser, env: HPCenv):
        self.config_dict = get_config_as_dict(config)
        self.env = env
        assert self.config_dict is not None, "Config dict, did not parse"
        
    @abc.abstractmethod
    def run(self, seed):
        """
        The main scheduling logic for the baseline.
        Subclasses must implement this method.
        """
        pass

class MedianBaseline(Baseline):
    def __init__(self,config, env):
        """
        Initializes the MedianBaseline, which schedules jobs based on carbon intensity.
        """
        super().__init__(config,env)
        self.name = "Median Baseline"
        
    def run(self, seed=42, debug = False):
        self.env.reset(seed=seed, options={})
        terminated = False
        obs = self.env.build_observation()
        reward = 0
        step_count = 0
        
        assert len(self.env.job_queue) != 0, "NO jobs are start, will lead to error"
        self.env.render(step_count=step_count)
        
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
            
            carbon_forecast_median = np.median(carbon_forecast)
            
            if current_carbon_intensity < carbon_forecast_median:
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
            self.env.render(step_count=step_count)
            
        return reward