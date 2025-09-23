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
from src.utils import mask_fn, create_experiment_name
from src.hpc_env import HPCenv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from src.validation import Validation
from src.features import AttentionPoolFeaturesExtractor
import time
from src.utils import convert_numpy_types
import wandb
from wandb.integration.sb3 import WandbCallback

class RewardLoggingCallback(BaseCallback):
    """Logs separate reward components to TensorBoard from env infos.

    It expects info keys populated by HPCenv.step:
      - 'reward_carbon'
      - 'reward_wait_schedule'
      - 'reward_delay_wait'
      - 'reward_total'
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        # Support vectorized envs (list of dicts)
        if isinstance(infos, dict):
            infos = [infos]
        # Aggregate simple means across envs for logging
        if infos:
            keys = [
                ("reward/carbon", "reward_carbon"),
                ("reward/wait_schedule", "reward_wait_schedule"),
                ("reward/delay_wait", "reward_delay_wait"),
                ("reward/total", "reward_total"),
                ("queue/len_after", "queue_len_after"),
            ]
            for log_key, info_key in keys:
                vals = [i.get(info_key) for i in infos if info_key in i]
                if vals:
                    try:
                        self.logger.record(log_key, float(np.mean(vals)))
                    except Exception:
                        # Be resilient to any type issues
                        pass
        return True

class ValidationCallback(BaseCallback):
    """
    Runs validation on the latest checkpoint at a fixed frequency using
    Validation.validate_policy in "validation" mode.

    It expects checkpoints to be saved with Stable-Baselines3's CheckpointCallback
    naming convention: {name_prefix}_{num_timesteps}_steps under `<run_dir>/logs/`.
    """

    def __init__(self, run_dir: str, name_prefix: str, val_freq: int = 500000, n_eval_episodes: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir.rstrip("/")
        self.name_prefix = name_prefix
        self.val_freq = int(val_freq)
        self.n_eval_episodes = int(n_eval_episodes)

    def _on_step(self) -> bool:
        # Trigger validation right after a checkpoint save frequency
        if self.num_timesteps > 0 and (self.num_timesteps % self.val_freq == 0):
            ckpt_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            ckpt_dir = os.path.join(self.run_dir, "logs")
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            # Small wait to ensure checkpoint file is fully written to disk
            # (Callback order should already make this safe when used after CheckpointCallback.)
            for _ in range(3):
                if os.path.exists(ckpt_path) or os.path.exists(ckpt_path + ".zip"):
                    break
                time.sleep(0.1)

            if self.verbose:
                print(f"[ValidationCallback] Running validation for checkpoint: {ckpt_name}")

            try:
                validator = Validation()
                validator.load_dir(self.run_dir)
                results = validator.validate_policy(
                    n_eval_episodes=self.n_eval_episodes,
                    checkpoints=[ckpt_name],
                    mode="validation",
                    debug=False,
                )

                results = convert_numpy_types(results)
                # Persist/append results to a JSON file for later inspection
                results_path = os.path.join(self.run_dir, "validation_metrics.json")
                if os.path.exists(results_path):
                    try:
                        with open(results_path, "r") as f:
                            existing = json.load(f)
                    except Exception:
                        existing = {}
                else:
                    existing = {}
                existing.update(results or {})
                with open(results_path, "w") as f:
                    json.dump(existing, f, indent=2)

                if self.verbose:
                    print(f"[ValidationCallback] Validation metrics saved to {results_path}")
            except Exception as e:
                print(f"[ValidationCallback] Validation failed for {ckpt_name}: {e}")

        return True


class Train():
    def __init__(self, config_dict, workload_path = None, save_freq=500_000, trace_enabled = False) -> None:
        self.config_dict = config_dict
        self.save_freq = save_freq
        self.run_id = create_experiment_name(config=config_dict, workload_file=workload_path) 
        self.run_dir = "./results/" + self.run_id + "/"  
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
        self.env = Monitor(ActionMasker(HPCenv(mode="training", config_dict=config_dict, trace_enabled=trace_enabled), mask_fn)) 


        policy_kwargs = dict(
            net_arch=dict(
                pi=self.config_dict['pi_nn'],
                vf=self.config_dict['vf_nn']
            )
        )

        self.model = MaskablePPO("MlpPolicy", self.env, verbose=1, tensorboard_log=self.run_dir,
                         gamma = self.config_dict['gamma'],
                         gae_lambda = self.config_dict['gae_lambda'],
                         batch_size = self.config_dict['batch_size'],
                         n_epochs = self.config_dict['n_epochs'],
                         n_steps = self.config_dict['n_steps'],
                         ent_coef = self.config_dict['ent_coef'],
                         seed=self.config_dict['seed'],
                         learning_rate = self.config_dict['learning_rate'],
                         clip_range = self.config_dict['clip_range'],
                         policy_kwargs=policy_kwargs
                                  )
    
    def run(self, save_checkpoints = False):
        self.env.reset()
        checkpoint_callback = None
        validation_callback = None


        run_wandb =  wandb.init(
            project="green_scheduler",
            config=self.config_dict,
            sync_tensorboard=True,
        )

        callbacks = [WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run_wandb.id}",
        verbose=2,
    ),] 
 
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
                name_prefix=name_prefix,
                val_freq=self.save_freq,
                n_eval_episodes=int(self.config_dict.get('n_eval_episodes', 3)),
                verbose=1,
            )
            callbacks.append(validation_callback)
            callbacks.append(checkpoint_callback)
       

        self.model.learn(
            total_timesteps=self.config_dict['total_timesteps'],
            tb_log_name="seed_" + str(self.config_dict['seed']),
            callback=callback,
        )

        run_wandb.finish()
        self.model.save(self.run_dir)
