import gymnasium as gym
import numpy as np
import math
import os
import configparser
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from sb3_contrib.common.maskable.utils import get_action_masks

from src.hpc_env import HPCenv
from src.validation import Validation
from src.training import Train
from src.baseline import PercentileBaseline
from src.utils import mask_fn, get_config_as_dict
from src.carbon_intensity import CarbonIntensity

WORKLOAD_PATH = "data/workloads/training_workload.swf"

# Load config with explicit path and typed parsing
config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config_file', 'config.ini')
config.read(config_path)

config_dict = get_config_as_dict(config) 
print(config_dict)

seed_list = [2,3,4,5]
for seed in seed_list:

    print("Running on seed: ", seed)
    
    config_dict['seed'] = seed
    train = Train(config_dict=config_dict, workload_path=WORKLOAD_PATH, save_freq=config_dict['n_steps']*2)

    train.run(save_checkpoints=True)