import configparser
import os 

from src.carbon_intensity import CarbonIntensity
from src.job import Job 

class Reward():
    def __init__(self, config_dict) -> None:
        
        self.reward_type = config_dict['reward_type']
        self.bounded_slowdown_threshhold = config_dict['bounded_slowdown_threshhold']
        self.eta = config_dict['eta']

    def get_invalid_action_reward(self):
        return -10 

    def get_reward(self,scheduled_job : Job | None, carbon_intensity : CarbonIntensity, current_timestamp):
        reward = 0
        
        assert self.reward_type in ["CO2_direct", "delay_vs_now_reward"]

        if self.reward_type == "CO2_direct":
            print("CO2 Direct Reward")
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission = carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshhold, scheduled_job.run_time])

                reward = - (carbon_emission + bounded_slowdown*self.eta)

            else: 
                reward = 0

        if self.reward_type == "delay_vs_now_reward":
            print("Delay vs now reward")
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission_actual = carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                carbon_emission_initial = carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                
                carbon_ratio_reward = (carbon_emission_initial-carbon_emission_actual)/carbon_emission_initial

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshhold, scheduled_job.run_time])

                reward = carbon_ratio_reward # - bounded_slowdown*ETA
            else: 
                reward = 0
        return reward 

