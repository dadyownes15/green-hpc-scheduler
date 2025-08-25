import configparser
import os 

from src.carbon_intensity import CarbonIntensity
from src.job import Job 

config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config_file', 'config.ini')
config.read(config_path)

MIN_RUN_TIME_THRESHOLD= int(config.get('reward', 'bounded_slowdown_threshhold'))
ETA = float(config.get('reward', 'eta'))
REWARD_TYPE  = config.get('reward', 'reward_type')

class Reward():
    def __init__(self) -> None:
        self.reward_type = REWARD_TYPE 

    def get_invalid_action_reward(self):
        return -10 

    def get_reward(self,scheduled_job : Job | None, carbon_intensity : CarbonIntensity, current_timestamp):
        reward = 0
        assert self.reward_type == "CO2_direct"
        if self.reward_type == "CO2_direct":

            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission = carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([MIN_RUN_TIME_THRESHOLD, scheduled_job.run_time])

                reward = - (carbon_emission + bounded_slowdown*ETA)

            else: 
                reward = 0

        if self.reward_type == "carbon_ratio_reward":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission_actual = carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                carbon_emission_initial = carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                
                carbon_ratio_reward = (carbon_emission_initial-carbon_emission_actual)/carbon_emission_initial

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([MIN_RUN_TIME_THRESHOLD, scheduled_job.run_time])

                reward = - (carbon_ratio_reward + bounded_slowdown*ETA)
            else: 
                reward = 0
        return reward

