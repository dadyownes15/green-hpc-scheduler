import configparser
import os
from pathlib import Path

import wandb
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from src.callbacks import BestValidationCallback
from src.hpc_env import HPCenv
from src.utils import create_experiment_name, get_config_as_dict, mask_fn


def main() -> None:
    config = configparser.ConfigParser()
    config_path = Path.cwd() / "config_file" / "config.ini"
    if not config.read(config_path):
        raise FileNotFoundError(f"Unable to read config file at '{config_path}'.")
    config_dict = get_config_as_dict(config)

    n_envs = int(config_dict.get("n_envs", 16))
    batch_size = int(config_dict.get("batch_size", 64))
    n_steps = int(config_dict.get("n_steps", 2048))
    rollout_steps = max(1, n_steps // n_envs)
    n_steps = rollout_steps * n_envs
    config_dict["n_steps"] = n_steps  # keep config aligned with actual rollout size
    log_interval = int(config_dict.get("log_interval", 10))

    run_name = create_experiment_name(config=config_dict, workload_file=None)
    project_name = (
        config_dict.get("wandb_project")
        or os.environ.get("WANDB_PROJECT")
        or "green_scheduler"
    )
    wandb_dir = config_dict.get("wandb_dir") or os.environ.get("WANDB_DIR")

    results_dir = Path("results") / run_name
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        HPCenv,
        n_envs=n_envs,
        env_kwargs=dict(config_dict=config_dict, mode="training"),
        wrapper_class=ActionMasker,
        wrapper_kwargs=dict(action_mask_fn=mask_fn),
        vec_env_cls=SubprocVecEnv,
        seed=config_dict.get("seed", 42),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=config_dict.get("n_epochs", 10),
        gamma=config_dict.get("gamma", 0.99),
        gae_lambda=config_dict.get("gae_lambda", 0.95),
        ent_coef=config_dict.get("ent_coef", 0.0),
        vf_coef=config_dict.get("vf_coef", 0.5),
        clip_range=config_dict.get("clip_range", 0.2),
        clip_range_vf=config_dict.get("clip_range_vf", 1.0),
        learning_rate=config_dict.get("learning_rate", 3e-4),
        verbose=1,
        seed=config_dict.get("seed", 42),
    )

    # Keep CSV/stdout logs only; skip tensorboard per request
    logger = configure(str(logs_dir), ["stdout", "csv"])
    model.set_logger(logger)

    wandb_init_kwargs = {
        "project": project_name,
        "config": config_dict,
    }
    if wandb_dir is not None:
        wandb_init_kwargs["dir"] = wandb_dir

    best_model_path = results_dir / "best_model.zip"
    final_model_path = results_dir / "final_model.zip"

    with wandb.init(**wandb_init_kwargs) as run:
        run.name = run_name
        wandb_cb = WandbCallback(
            gradient_save_freq=0,
            model_save_freq=0,
            model_save_path=None,
            verbose=0,
        )
        best_cb = BestValidationCallback(
            config_dict=config_dict,
            save_path=str(best_model_path),
            eval_freq=int(config_dict.get("validation_freq", config_dict["n_steps"])),
            n_eval_episodes=int(config_dict.get("validation_episodes", 1)),
            run=run,
        )

        model.learn(
            total_timesteps=int(config_dict.get("total_timesteps", 1_000_000)),
            callback=[wandb_cb, best_cb],
            log_interval=log_interval,
        )

        model.save(str(final_model_path))
        env.close()

        run.summary["final_model_path"] = str(final_model_path)
        run.summary["best_model_path"] = str(best_model_path)
        run.summary["best_validation_reward"] = best_cb.best_score


if __name__ == "__main__":
    main()
