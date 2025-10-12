import json
import os
from pathlib import Path

import wandb
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

from src.callbacks import BestValidationCallback, WandbTrainingMetricsCallback
from src.hpc_env import HPCenv
from src.utils import create_experiment_name, mask_fn


def _format_suffix_value(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _build_wandb_run_name(base_name: str, config: dict) -> str:
    suffix_parts = []
    seed = config.get("seed")
    if seed is not None:
        suffix_parts.append(f"seed{seed}")
    eta = config.get("eta")
    if eta is not None:
        suffix_parts.append(f"eta{_format_suffix_value(eta)}")
    if not suffix_parts:
        return base_name
    return f"{base_name}__{'_'.join(map(str, suffix_parts))}"


class Train:
    def __init__(
        self,
        config_dict,
        workload_path=None,
        save_freq: int = 500_000,
        trace_enabled: bool = False,
    ) -> None:
        self.config_dict = dict(config_dict)
        self.save_freq = int(save_freq)
        self.run_id = create_experiment_name(config=self.config_dict, workload_file=workload_path)
        self.run_path = Path("results") / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.run_dir = str(self.run_path)

        config_path = self.run_path / "config.json"
        with config_path.open("w") as f:
            json.dump(self.config_dict, f, indent=4)

        print(f"Repository created at {self.run_path} and config saved to {config_path}")

        # Resolve rollout configuration for vectorised training
        self.n_envs = int(self.config_dict.get("n_envs", 1))
        base_n_steps = int(self.config_dict.get("n_steps", 2048))
        self.rollout_steps = max(1, base_n_steps // self.n_envs)
        self.config_dict["n_envs"] = self.n_envs
        self.config_dict["n_steps"] = self.rollout_steps * self.n_envs
        self.log_interval = int(self.config_dict.get("log_interval", 1))

        vec_env_cls = SubprocVecEnv if self.n_envs > 1 else DummyVecEnv
        env_kwargs = dict(
            config_dict=self.config_dict,
            mode="training",
            trace_enabled=trace_enabled,
        )
        wrapper_kwargs = dict(action_mask_fn=mask_fn)
        env = make_vec_env(
            HPCenv,
            n_envs=self.n_envs,
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs=wrapper_kwargs,
            vec_env_cls=vec_env_cls,
            seed=self.config_dict.get("seed"),
        )

        monitor_dir = self.run_path / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        self.env = VecMonitor(env, filename=str(monitor_dir / "monitor.csv"))

        self.logger_dir = self.run_path / "training_logs"
        self.logger_dir.mkdir(parents=True, exist_ok=True)
        logger = configure(str(self.logger_dir), ["stdout", "csv", "tensorboard"])

        policy_kwargs = dict(
            net_arch=dict(
                pi=self.config_dict["pi_nn"],
                vf=self.config_dict["vf_nn"],
            )
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            gamma=self.config_dict["gamma"],
            gae_lambda=self.config_dict["gae_lambda"],
            batch_size=self.config_dict["batch_size"],
            n_epochs=self.config_dict["n_epochs"],
            n_steps=self.rollout_steps,
            ent_coef=self.config_dict["ent_coef"],
            seed=self.config_dict["seed"],
            learning_rate=self.config_dict["learning_rate"],
            clip_range=self.config_dict["clip_range"],
            clip_range_vf=self.config_dict["clip_range_vf"],
            vf_coef=self.config_dict["vf_coef"],
            policy_kwargs=policy_kwargs,
        )
        self.model.set_logger(logger)

    def run(
        self,
        save_checkpoints: bool = False,
        save_validation_logs: bool = False,
        validation_log_path: str | Path | None = None,
    ):
        self.env.reset()
        run_name = _build_wandb_run_name(self.run_id, self.config_dict)

        project_name = (
            self.config_dict.get("wandb_project")
            or os.environ.get("WANDB_PROJECT")
            or "green_scheduler"
        )
        wandb_init_kwargs = {
            "project": project_name,
            "config": self.config_dict,
            "name": run_name,
        }
        wandb_dir = self.config_dict.get("wandb_dir") or os.environ.get("WANDB_DIR")
        if wandb_dir:
            wandb_init_kwargs["dir"] = wandb_dir

        run_wandb = wandb.init(**wandb_init_kwargs)

        callbacks = [
            WandbCallback(
                model_save_path=None,
                gradient_save_freq=0,
                model_save_freq=0,
                verbose=0,
            ),
            WandbTrainingMetricsCallback(run=run_wandb),
        ]


 
        best_callback = None
        if save_checkpoints:
            eval_freq = int(
                self.config_dict.get(
                    "validation_freq",
                    self.save_freq,
                )
            )
            best_model_path = self.run_path / "best_model.zip"
            best_callback = BestValidationCallback(
                config_dict=self.config_dict,
                save_path=str(best_model_path),
                eval_freq=eval_freq,
                n_eval_episodes=int(self.config_dict.get("validation_episodes", 1)),
                metric="Validation Reward",
                run=run_wandb,
                seed_label=None,
                verbose=1,
            )
            callbacks.append(best_callback)
       

        self.model.learn(
            total_timesteps=self.config_dict['total_timesteps'],
            callback=callbacks,
            log_interval=self.log_interval,
        )

        if best_callback is not None:
            if best_callback.save_path.exists():
                run_wandb.summary["best_model_path"] = str(best_callback.save_path)
                run_wandb.summary["best_validation_score"] = best_callback.best_score
            run_wandb.summary["validation_frequency"] = best_callback.eval_freq

        run_wandb.finish()
        self.env.close()
