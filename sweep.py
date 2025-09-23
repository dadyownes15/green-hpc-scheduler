# train_sweep.py
from copy import deepcopy
import wandb
from sb3_contrib import MaskablePPO
from wandb.integration.sb3 import WandbCallback
from src.hpc_env import HPCenv

# --- 2.1 Base config (exactly your values) ---
BASE_CFG = {
    "power settings": {
        "use_constant_power": True,
        "constant_power_per_processor": 500,
        "procs_per_node": 1,
        "idle_power": 15,
        "carbon_year": 2021,
        "custom_intensity": False,
    },
    "architecture": {
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
    },
    "training": {
        "episode_length": 3200,   # must remain fixed
        "gamma": 0.99,
        "gae_lambda": 0.97,
        "batch_size": 2048,
        "seed": 2,
        "n_epochs": 1,
        "pi_nn": [4000, 1000],
        "vf_nn": [4000, 1000],
        "n_steps": 16538,
        "total_timesteps": 1_000_000,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "clip_range": 0.1,
    },
    "reward": {
        "base_line_wait_carbon_penality": 0.01,
        "bounded_slowdown_threshold": 10,
        "alpha": 0,
        "eta": 10,
        "reward_type": "wait_abs_em",
    },
    "normalization constants": {
        "max_power": 19000,
        "max_green": 19000,
        "max_wait_time": 200000,
        "max_run_time": 162754,
        "max_requested_processors": 256,
    },
}

# --- 2.2 Helpers: merge W&B overrides into nested "training" ---
def merge_training_overrides(base_cfg: dict, overrides: dict) -> dict:
    """
    Returns a deep copy of base_cfg with training overrides applied.
    Works with either flat keys like 'gamma' or dotted keys like 'training.gamma'.
    Never lets 'episode_length' change.
    """
    cfg = deepcopy(base_cfg)

    # Accept both flat and dotted keys from the sweep
    for k, v in overrides.items():
        if k.startswith("training."):
            subk = k.split(".", 1)[1]
            if subk != "episode_length":
                cfg["training"][subk] = v
        elif k in cfg.get("training", {}):
            if k != "episode_length":
                cfg["training"][k] = v
        else:
            # ignore non-training keys in the sweep to respect "only vary training"
            pass

    # hard lock episode_length to the base value
    cfg["training"]["episode_length"] = base_cfg["training"]["episode_length"]
    return cfg


def build_policy_kwargs(training_cfg: dict) -> dict:
    # Make sure lists come through as lists of ints
    pi_layers = [int(x) for x in training_cfg["pi_nn"]]
    vf_layers = [int(x) for x in training_cfg["vf_nn"]]
    return dict(net_arch=dict(pi=pi_layers, vf=vf_layers))


def make_env(full_cfg: dict):
    # Your env depends on the full nested config
    # If HPCenv takes a seed, set it here as well
    env = HPCenv(config_dict=full_cfg, mode="training")
    return env


def train():
    """
    This function is called by the W&B agent for each sweep run.
    It reads wandb.config (training overrides), merges with BASE_CFG, builds the env,
    creates the PPO model, and learns for total_timesteps.
    """
    with wandb.init(project="green_scheduler") as run:
        # 1) Merge the sweep overrides into the base config
        overrides = dict(run.config)  # W&B provides a ReadOnlyConfig, convert to dict
        full_cfg = merge_training_overrides(BASE_CFG, overrides)
        tr = full_cfg["training"]

        # 2) Build env
        env = make_env(full_cfg)

        # 3) Policy kwargs from training config
        policy_kwargs = build_policy_kwargs(tr)

        # 4) Create model (note: use values from nested training section)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=tr["gamma"],
            gae_lambda=tr["gae_lambda"],
            batch_size=tr["batch_size"],
            n_epochs=tr["n_epochs"],
            n_steps=tr["n_steps"],
            ent_coef=tr["ent_coef"],
            seed=tr["seed"],
            learning_rate=tr["learning_rate"],
            clip_range=tr["clip_range"],
            policy_kwargs=policy_kwargs,
        )

        # 5) W&B callback
        wandb_cb = WandbCallback(
            gradient_save_freq=100,          # log gradients periodically
            model_save_path=f"models/{run.name}",
            model_save_freq=0,               # set >0 to save every N steps
            verbose=2,
        )

        # 6) Train
        model.learn(
            total_timesteps=tr["total_timesteps"],
            callback=wandb_cb,
            progress_bar=True,
        )

        env.close()


if __name__ == "__main__":
    # Two ways to run:
    # A) Plain run (no sweep), uses BASE_CFG as defaults in the W&B run:
    #    wandb.init(config=BASE_CFG["training"]) + train()
    #
    # B) Sweep agent (recommended):
    #    1) sweep_id = wandb.sweep("sweep.yaml")
    #    2) wandb.agent(sweep_id, function=train, count=20)
    #
    # Here we implement B) if you choose to run agents from this script.

    import argparse, yaml, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=None,
                        help="Path to sweep.yaml. If provided, will create the sweep and start an agent.")
    parser.add_argument("--count", type=int, default=20, help="Number of runs for the agent.")
    args = parser.parse_args()

    if args.sweep is None:
        # single run (no sweep): start a W&B run with BASE training defaults
        with wandb.init(project="green_scheduler", config=BASE_CFG["training"]):
            train()
    else:
        with open(args.sweep, "r") as f:
            sweep_cfg = yaml.safe_load(f)

        # If your sweep uses dotted keys like training.gamma,
        # it is fine. Our merge helper supports both styles.
        sweep_id = wandb.sweep(sweep=sweep_cfg, project="green_scheduler")
        wandb.agent(sweep_id, function=train, count=args.count)
