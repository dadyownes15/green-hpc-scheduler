# train_sweep.py
from copy import deepcopy
import argparse
import yaml
import wandb
from sb3_contrib import MaskablePPO
from wandb.integration.sb3 import WandbCallback
from src.hpc_env import HPCenv
from src.utils import mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
# --- Base config (flat), exactly as your env expects ---
BASE_CFG = {
    # power settings
    "use_constant_power": True,
    "constant_power_per_processor": 500,
    "procs_per_node": 1,
    "idle_power": 15,
    "carbon_year": 2021,
    "custom_intensity": False,
    # architecture
    "green_forecast_length": 24,
    "max_queue_size": 256,
    "run_win_length": 64,
    "delay_time_list": [300, 600, 1200, 1800, 2400, 3000, 3600],
    "delay_time_list_length": 3,
    "max_wait_n_jobs": 3,
    "job_feature": 5,
    "run_feature": 2,
    "green_feature_pr_timeslot": 1,
    "green_feature_constant": 8,
    # training
    "episode_length": 3200,        # LOCKED
    "gamma": 0.99,
    "gae_lambda": 0.97,
    "batch_size": 2048,
    "seed": 2,
    "n_epochs": 1,
    "pi_nn": [4000, 1000],
    "vf_nn": [4000, 1000],
    "n_steps": 16538,
    "total_timesteps": 300_000,
    "ent_coef": 0.01,
    "learning_rate": 3e-4,
    "clip_range": 0.1,
    # reward
    "base_line_wait_carbon_penality": 0.01,
    "bounded_slowdown_threshold": 10,
    "alpha": 0,
    "eta": 10,
    "reward_type": "wait_relative_ems",
    # normalization constants
    "max_power": 19000,
    "max_green": 19000,
    "max_wait_time": 200000,
    "max_run_time": 162754,
    "max_requested_processors": 256,
}

# which keys are allowed to change in sweeps
TRAINING_KEYS = {
    "gamma", "gae_lambda", "batch_size", "seed", "n_epochs",
    "pi_nn", "vf_nn", "n_steps", "total_timesteps",
    "ent_coef", "learning_rate", "clip_range",
    # "episode_length" is deliberately excluded below (locked)
}
LOCKED_KEYS = {"episode_length"}

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
    cfg = deepcopy(base_cfg)
    # apply only allowed training overrides, keep everything else identical
    for k, v in overrides.items():
        if k in TRAINING_KEYS:
            if k in {"pi_nn", "vf_nn"}:
                v = _to_int_list(v)
            cfg[k] = v
    # hard lock episode_length
    cfg["episode_length"] = base_cfg["episode_length"]

    # Safety: ensure n_steps is a multiple of batch_size for PPO
    bs = int(cfg["batch_size"])
    cfg["n_steps"] = make_multiple(int(cfg["n_steps"]), bs)
    return cfg

def build_policy_kwargs(cfg: dict) -> dict:
    return dict(net_arch=dict(
        pi=_to_int_list(cfg["pi_nn"]),
        vf=_to_int_list(cfg["vf_nn"]),
    ))

def train():
    # one sweep run
    with wandb.init(project="green_scheduler") as run:
        overrides = dict(run.config)
        cfg = merge_overrides(BASE_CFG, overrides)

        # log the effective merged config so the run is fully reproducible
        wandb.config.update(cfg, allow_val_change=True)

        env = Monitor(ActionMasker(HPCenv(mode="training", config_dict=cfg), mask_fn)) 

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
            verbose=2,
        )

        model.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=wandb_cb,
            progress_bar=True,
        )

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=None,
                        help="Path to sweep.yaml to create+run a sweep agent")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of runs for the agent")
    args = parser.parse_args()

    if args.sweep:
        with open(args.sweep, "r") as f:
            sweep_cfg = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_cfg, project="green_scheduler")
        wandb.agent(sweep_id, function=train, count=args.count)
    else:
        # single run (no sweep): just use BASE_CFG
        with wandb.init(project="green_scheduler", config=BASE_CFG):
            train()
