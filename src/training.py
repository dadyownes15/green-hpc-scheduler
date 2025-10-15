import json
import os
from pathlib import Path

import wandb
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from wandb.integration.sb3 import WandbCallback
import numpy as np
from src.callbacks import BestValidationCallback, WandbTrainingMetricsCallback
from src.hpc_env import HPCenv
from src.utils import (
    create_experiment_name,
    generate_unique_run_suffix,
    mask_fn,
)


def _format_suffix_value(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _build_wandb_run_name(base_name: str, config: dict, suffix: str | None = None) -> str:
    suffix_parts = []
    seed = config.get("seed")
    if seed is not None:
        suffix_parts.append(f"seed{seed}")
    eta = config.get("eta")
    if eta is not None:
        suffix_parts.append(f"eta{_format_suffix_value(eta)}")
    if suffix:
        suffix_parts.append(str(suffix))
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
        self.base_run_name = create_experiment_name(config=self.config_dict, workload_file=workload_path)
        self.run_suffix = generate_unique_run_suffix()
        self.run_id = f"{self.base_run_name}__{self.run_suffix}"
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

        # Normalize obs + rewards (stable training)
        self.env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config_dict["gamma"],  # keep consistent with PPO
        )

        # Monitor AFTER normalization so logs reflect what PPO sees
        monitor_dir = self.run_path / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        self.env = VecMonitor(self.env, filename=str(monitor_dir / "monitor.csv"))

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
    def _debug_rollout_buffer(self, buf, n_samples: int = 5):

        buf = buf
        print(f"\nRolloutBuffer debug:")
        print(f" - buffer size: {buf.buffer_size}")
        print(f" - num_envs: {buf.n_envs}")
        print(f" - obs shape: {buf.observations.shape}")
        print(f" - actions shape: {buf.actions.shape}")

        if hasattr(buf, "action_masks"):
            mask = buf.action_masks
            print(f" - action_masks shape: {mask.shape}")
        else:
            print(" - No action_masks found on buffer")
            return

        # === Handle both 2D and 3D cases ===
        if mask.ndim == 3:
            # (buffer_size, n_envs, n_actions)
            all_false_per_env = (~mask.any(axis=2)).sum(axis=0)
            total_all_false = all_false_per_env.sum()
            print(f"\nTotal all-false masks: {total_all_false}")
            for env_id, n in enumerate(all_false_per_env):
                if n > 0:
                    print(f" ⚠️ Env {env_id} had {n} timesteps with all actions masked")

        elif mask.ndim == 2:
            # (buffer_size, n_actions)
            total_all_false = (~mask.any(axis=1)).sum()
            print(f"\nTotal all-false masks: {total_all_false}")
            if total_all_false > 0:
                bad_idx = np.where(~mask.any(axis=1))[0]
                print(f" ⚠️ Indices with all-false masks: {bad_idx[:20]}...")

        else:
            print(f"Unexpected mask dimensions: {mask.ndim}")

        # === Print a few random samples ===
        idxs = np.random.choice(buf.buffer_size, size=min(n_samples, buf.buffer_size), replace=False)
        for i in idxs:
            print(f"\n--- Sample {i} ---")
            m = mask[i]
            valid = m.sum()
            print(f"mask valid actions: {valid}/{m.size}")
            if valid == 0:
                print("⚠️ ALL ACTIONS MASKED")
                print(m.astype(int))
            print("action:", buf.actions[i])
            print("value:", buf.values[i])
            print("reward:", buf.rewards[i])

    def _probe_logits_from_buffer(self):
        import numpy as np
        import torch

        policy = self.model.policy
        buf = self.model.rollout_buffer

        # to torch
        obs = torch.as_tensor(buf.observations, device=policy.device, dtype=torch.float32)
        masks_np = getattr(buf, "action_masks", None)
        if masks_np is None:
            print("[Probe] buffer has no action_masks")
            return
        masks = torch.as_tensor(masks_np, device=policy.device, dtype=torch.bool)

        # ==== Forward pass w/o private methods ====
        # 1) extract features
        features = policy.extract_features(obs)
        # 2) get actor/critic latents
        if hasattr(policy, "mlp_extractor"):
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            # some policies name it differently; fall back
            latent_pi = features
            latent_vf = features

        # 3) raw action logits from actor head
        logits = policy.action_net(latent_pi)  # shape: (batch, n_actions)

        # Basic stats on raw logits
        def tstats(t, name):
            finite = torch.isfinite(t)
            print(f"{name}: shape={tuple(t.shape)}, "
                f"min={torch.nan_to_num(t).min().item():.3e}, "
                f"max={torch.nan_to_num(t).max().item():.3e}, "
                f"any_nan={torch.isnan(t).any().item()}, "
                f"any_inf={(~finite).any().item()}")

        print("\n[Probe] Raw action logits BEFORE masking")
        tstats(logits, "logits")

        # ==== Apply masks and compute probs manually (stable) ====
        # guard against any row being fully false (you already checked, but keep safe)
        row_any = masks.any(dim=-1)
        if not torch.all(row_any):
            bad = torch.where(~row_any)[0]
            print(f"[Probe] Found {bad.numel()} rows with all-false masks (unexpected): {bad[:10].tolist()}")

        # masked logits: set invalid actions to -inf, then stabilize before softmax
        masked_logits = logits.clone()
        masked_logits[~masks] = float("-inf")
        # stabilize: subtract max over VALID entries only
        # (if a row is all -inf, max is -inf; we protected above)
        row_max = torch.amax(masked_logits, dim=-1, keepdim=True)
        masked_logits = masked_logits - row_max

        probs = torch.softmax(masked_logits, dim=-1)

        print("\n[Probe] Probs AFTER masking")
        tstats(probs, "probs")

        # ==== Check simplex ====
        row_sum = probs.sum(dim=-1)
        nonneg = (probs >= 0).all(dim=-1)
        finite = torch.isfinite(probs).all(dim=-1)
        ok_sum = torch.isfinite(row_sum) & ((row_sum - 1.0).abs() < 1e-5)

        bad_idx = torch.where(~(nonneg & finite & ok_sum))[0]
        print(f"[Probe] bad rows: {bad_idx.numel()} / {probs.shape[0]}")
        if bad_idx.numel() > 0:
            for i in bad_idx[:10].tolist():
                rs = row_sum[i].item()
                any_nan = torch.isnan(probs[i]).any().item()
                any_inf = (~torch.isfinite(probs[i])).any().item()
                vmin = torch.nan_to_num(probs[i]).min().item()
                vmax = torch.nan_to_num(probs[i]).max().item()
                valid_count = masks[i].sum().item()
                print(f"  idx {i}: sum={rs:.8f}, any_nan={any_nan}, any_inf={any_inf}, "
                    f"min={vmin:.3e}, max={vmax:.3e}, valid_actions={valid_count}")

    def run(
        self,
        save_checkpoints: bool = False,
        save_validation_logs: bool = False,
        validation_log_path: str | Path | None = None,
    ):
        self.env.reset()
        run_name = _build_wandb_run_name(self.base_run_name, self.config_dict, suffix=self.run_suffix)

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
        """ try: 
            self.model.learn(
                total_timesteps=self.config_dict['total_timesteps'],
                callback=callbacks,
                log_interval=self.log_interval,
            )
        except:
            print("\n===== EXCEPTION DURING TRAINING =====")
            print("\n===== DEBUGGING ROLLOUT BUFFER =====")
            self._debug_rollout_buffer(buf=self.model.rollout_buffer)
            self._probe_logits_from_buffer()"""
        
        
        run_wandb.summary["run_base_name"] = self.base_run_name
        run_wandb.summary["run_suffix"] = self.run_suffix
        run_wandb.summary["run_id"] = self.run_id
        run_wandb.summary["run_path"] = str(self.run_path)

        if best_callback is not None:
            if best_callback.save_path.exists():
                run_wandb.summary["best_model_path"] = str(best_callback.save_path)
                run_wandb.summary["best_validation_score"] = best_callback.best_score
            run_wandb.summary["validation_frequency"] = best_callback.eval_freq

        run_wandb.finish()
        self.env.close()
