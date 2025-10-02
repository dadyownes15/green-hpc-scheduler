# train_sweep.py
import configparser
from copy import deepcopy
import argparse
import json
import os
import yaml
import wandb
from sb3_contrib import MaskablePPO
from wandb.integration.sb3 import WandbCallback
from src.hpc_env import HPCenv
from src.utils import create_experiment_name, mask_fn, get_config_as_dict
from src.callbacks import SweepCallBack, ValidationCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.callbacks import CheckpointCallback
"""
Utilities and sweep merge helpers
"""


def _to_int_list(x):
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, str):
        # support "1024,512"
        return [int(v.strip()) for v in x.split(",")]
    return x

def make_multiple(n, base):
    """Snap n to the nearest higher multiple of base (keeps SB3 PPO happy)."""
    if base <= 0:
        return n
    return ((n + base - 1) // base) * base

def merge_overrides(base_cfg: dict, overrides: dict) -> dict:
    """
    Merge a sweep-provided overrides dict into the base config dict.

    - Applies key aliases (e.g., delay_list -> delay_time_list)
    - Restricts to a safe set of overridable keys
    - Normalizes types (lists to int, numeric casts)
    - Enforces PPO constraints (n_steps multiple of batch_size)
    - Recomputes derived fields (delay_time_list_length)
    """
    cfg = deepcopy(base_cfg)

    for k, v in (overrides or {}).items():
        # Only merge keys that exist in the base config
        if k not in cfg:
            continue
        base_v = cfg[k]

        # Try to coerce to the same type as base config
        try:
            if isinstance(base_v, (list, tuple)):
                v = _to_int_list(v)
            elif isinstance(base_v, int):
                v = int(v)
            elif isinstance(base_v, float):
                v = float(v)
            else:
                # leave as-is for strings/booleans/other
                pass
        except Exception:
            # If casting fails, keep the override as-is
            pass

        cfg[k] = v

    # Safety: ensure n_steps is a multiple of batch_size for PPO
    bs = int(cfg["batch_size"]) if "batch_size" in cfg else 64
    if "n_steps" in cfg:
        cfg["n_steps"] = make_multiple(int(cfg["n_steps"]), bs)

    # Derived: delay_time_list_length
    if "delay_time_list" in cfg and isinstance(cfg["delay_time_list"], (list, tuple)):
        cfg["delay_time_list_length"] = len(cfg["delay_time_list"])

    return cfg

def build_policy_kwargs(cfg: dict) -> dict:
    return dict(net_arch=dict(
        pi=_to_int_list(cfg["pi_nn"]),
        vf=_to_int_list(cfg["vf_nn"]),
    ))

def train():
    with wandb.init(project="green_scheduler") as run:
        save_freq = config_dict["n_steps"]
        sweep_overrides = dict(run.config)
        print(sweep_overrides)
        # Merge the sweep-chosen params into the base config from file
        cfg = merge_overrides(config_dict, sweep_overrides)

        # log the effective merged config so the run is fully reproducible
        wandb.config.update(cfg, allow_val_change=True)
        
        run_id = create_experiment_name(config=config_dict, workload_file=None) 
        run_dir = "results/" + run_id +  "/"  
        # --- NEW LOGIC TO CREATE REPOSITORY AND SAVE CONFIG ---
        # Create the directory for the run if it doesn't already exist.
        # exist_ok=True prevents an error if the directory already exists.
        os.makedirs(run_dir, exist_ok=True)
        
        # Define the path for the config file.
        config_path = os.path.join(run_dir, "config.json")
           
        # Save the config_dict as a human-readable JSON file.
        # This allows you to easily reference the settings used for this run.
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)


        print(f"Repository created at {run_dir} and config saved to {config_path}")
   
        env = ActionMasker(HPCenv(mode="training", config_dict=cfg), mask_fn)

        policy_kwargs = build_policy_kwargs(cfg)

        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            n_steps=cfg["n_steps"],
            ent_coef=cfg["ent_coef"],
            seed=cfg["seed"],
            learning_rate=cfg["learning_rate"],
            clip_range=cfg["clip_range"],
            policy_kwargs=policy_kwargs,
        )

        wandb_cb = WandbCallback(
            gradient_save_freq=0,              # set >0 to log gradients
            model_save_path=f"models/{run.name}",
            model_save_freq=0,                 # set >0 to checkpoint periodically
            verbose=0,
        )


        model.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=[wandb_cb, 
            CheckpointCallback(
                save_freq=save_freq,
                save_path=run_dir + "/logs/" + cfg["seed"],
                name_prefix=run.sweep_id,
            ),
            ValidationCallback(run=run, run_dir=run_dir, name_prefix=run.sweep_id,val_freq=save_freq )],
            progress_bar=False,
            log_interval=None,
        )

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=None,
                        help="Path to sweep.yaml to create+run a sweep agent")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of runs for the agent")
    args = parser.parse_args()

        # one sweep run
    
    # Load config with explicit path and typed parsing
    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), 'config_file', 'config.ini')
    config.read(config_path) 
    config_dict = get_config_as_dict(config=config)
    #print(config_dict)
    with open(args.sweep, "r") as f:
        sweep_cfg = yaml.safe_load(f)
        print(sweep_cfg)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project="green_scheduler")
    wandb.agent(sweep_id, function=train, count=args.count)
    
