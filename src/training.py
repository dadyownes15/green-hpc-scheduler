import gymnasium as gym
import numpy as np
import json
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
from src.utils import mask_fn
from src.hpc_env import HPCenv
from stable_baselines3.common.callbacks import CheckpointCallback


class Train():
    def __init__(self, config_dict, run_id, workload_path = None) -> None:
        self.config_dict = config_dict
        self.run_dir = "./" + run_id + "/"  
        # --- NEW LOGIC TO CREATE REPOSITORY AND SAVE CONFIG ---
        # Create the directory for the run if it doesn't already exist.
        # exist_ok=True prevents an error if the directory already exists.
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Define the path for the config file.
        config_path = os.path.join(self.run_dir, "config.json")
        
        # Save the config_dict as a human-readable JSON file.
        # This allows you to easily reference the settings used for this run.
        with open(config_path, 'w') as f:
            json.dump(self.config_dict, f, indent=4)
        
        print(f"Repository created at {self.run_dir} and config saved to {config_path}")

        # --- END OF NEW LOGIC ---
        self.env = Monitor(ActionMasker(HPCenv(workload_path=workload_path, config_dict=config_dict), mask_fn)) 

        policy_kwargs = dict(
            net_arch = dict(
                pi = self.config_dict['pi_nn'],
                vf = self.config_dict['vf_nn']
            )
        )

        self.model = MaskablePPO("MlpPolicy", self.env, verbose=1, tensorboard_log=self.run_dir,
                         gamma = self.config_dict['gamma'],
                         gae_lambda = self.config_dict['gae_lambda'],
                         batch_size = self.config_dict['batch_size'],
                         n_epochs = self.config_dict['n_epochs'],
                         n_steps = self.config_dict['n_steps'],
                         ent_coef = self.config_dict['ent_coef'],
                         policy_kwargs=policy_kwargs
                                  )
    
    def run(self, seed = 0, save_checkpoints = False):
        self.env.reset(seed=seed)

        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=self.run_dir + "/logs/",
            name_prefix="seed_"+str(seed),
            )
        self.model.learn(total_timesteps=self.config_dict['total_timesteps'], 
                         tb_log_name = "seed_" + str(seed), callback = checkpoint_callback)
        self.model.save(self.run_dir)