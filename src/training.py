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
from src.callbacks import ValidationCallback, StepInfoLoggerCallback
from src.utils import mask_fn, create_experiment_name
from src.hpc_env import HPCenv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from src.validation import Validation
from src.features import AttentionPoolFeaturesExtractor
import time
from src.utils import convert_numpy_types 
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class Train():
    def __init__(self, config_dict, workload_path = None, save_freq=500_000, trace_enabled = False) -> None:
        self.config_dict = config_dict
        self.save_freq = save_freq
        self.run_id = create_experiment_name(config=config_dict, workload_file=workload_path) 
        self.run_dir = "results/" + self.run_id + "/"  
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
   

        # Create vectorized env
        self.env = HPCenv(
                    mode="training",
                    config_dict=config_dict,
                    trace_enabled=trace_enabled
                )
        self.env= ActionMasker(self.env, mask_fn)  # mask invalid actions
        self.env = Monitor(self.env)  
        policy_kwargs = dict(
            net_arch=dict(
                pi=self.config_dict['pi_nn'],
                vf=self.config_dict['vf_nn']
            )
        )

        self.model = MaskablePPO("MlpPolicy", self.env, verbose=0, 
                         gamma = self.config_dict['gamma'],
                         gae_lambda = self.config_dict['gae_lambda'],
                         batch_size = self.config_dict['batch_size'],
                         n_epochs = self.config_dict['n_epochs'],
                         n_steps = self.config_dict['n_steps'],
                         ent_coef = self.config_dict['ent_coef'],
                         seed=self.config_dict['seed'],
                         learning_rate = self.config_dict['learning_rate'],
                         clip_range = self.config_dict['clip_range'],
                         clip_range_vf = self.config_dict['clip_range_vf'],
                         vf_coef=self.config_dict['vf_coef'],
                         policy_kwargs=policy_kwargs
                                  )
    
    def run(self, save_checkpoints = False):
        self.env.reset()
        checkpoint_callback = None
        validation_callback = None


        run_wandb =  wandb.init(
            project="green_scheduler",
            config=self.config_dict,
        )

        callbacks = [
            WandbCallback(
                model_save_path=f"models/{run_wandb.id}",
            )
        ]

        # Record per-step env info when tracing is enabled
        """callbacks.append(
            StepInfoLoggerCallback(
                save_dir=self.run_dir,
                run=run_wandb,
                filename="step_info.jsonl",
                flush_every=1000,
                wandb_sample_every=5000,
            )
        )"""
        ## I am doomed
 
        if save_checkpoints:
            name_prefix = "seed_" + str(self.config_dict['seed'])
            checkpoint_callback = CheckpointCallback(
                save_freq=self.save_freq,
                save_path=self.run_dir + "/logs/",
                name_prefix=name_prefix,
            )
            # Run validation on every checkpoint
            validation_callback = ValidationCallback(
                run_dir=self.run_dir,
                run=run_wandb,
                name_prefix=name_prefix,
                val_freq=self.save_freq,
                n_eval_episodes=1,
                verbose=1,
            )
            callbacks.append(checkpoint_callback)
            callbacks.append(validation_callback)
       

        self.model.learn(
            total_timesteps=self.config_dict['total_timesteps'],
            tb_log_name="seed_" + str(self.config_dict['seed']),
            callback=callbacks,
            log_interval=None,
            progress_bar=True
        )

        run_wandb.finish()
        self.model.save(self.run_dir)
