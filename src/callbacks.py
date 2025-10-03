
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import time
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.validation import Validation
from src.utils import convert_numpy_types
import os
from pathlib import Path


class ValidationCallback(BaseCallback):
    """
    Runs validation on the latest checkpoint at a fixed frequency using
    Validation.validate_policy in "validation" mode.

    It expects checkpoints to be saved with Stable-Baselines3's CheckpointCallback
    naming convention: {name_prefix}_{num_timesteps}_steps under `<run_dir>/logs/`.
    """

    def __init__(
        self,
        run_dir: str,
        name_prefix: str,
        run,
        val_freq: int = 500000,
        n_eval_episodes: int = 1,
        verbose: int = 0,
        model_save_dir: str | Path = "logs",
        save_results: bool = False,
        results_path: str | Path | None = None,
    ):
        super().__init__(verbose)
        self.run_dir = run_dir.rstrip("/")
        self._run_path = Path(self.run_dir).expanduser().resolve(strict=False)
        self.name_prefix = name_prefix
        self.val_freq = int(val_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.run = run
        self.model_save_dir = Path(model_save_dir)
        self.save_results = bool(save_results)
        self.results_path = (
            self._resolve_results_path(results_path)
            if self.save_results
            else None
        )

    def _on_step(self) -> bool:
        # Trigger validation right after a checkpoint save frequency
        if self.num_timesteps > 0 and (self.num_timesteps % self.val_freq == 0):
            ckpt_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            ckpt_dir = self._resolve_checkpoint_dir()
            ckpt_path = ckpt_dir / ckpt_name

            # Small wait to ensure checkpoint file is fully written to disk
            # (Callback order should already make this safe when used after CheckpointCallback.)
            for _ in range(3):
                if ckpt_path.exists() or ckpt_path.with_suffix(".zip").exists():
                    break
                time.sleep(0.1)

            if self.verbose:
                print(f"[ValidationCallback] Running validation for checkpoint: {ckpt_name}")

            try:
                validator = Validation()
                validator.load_dir(self.run_dir)

    

                results, _ = validator.validate_policy(
                    n_eval_episodes=self.n_eval_episodes,
                    checkpoints=[ckpt_name],
                    mode="validation",
                    debug=False,
                    checkpoint_dir=self.model_save_dir,
                )
                ckpt_results = results[ckpt_name]
                print(ckpt_results)
                self.run.log(results[ckpt_name])
                if self.save_results and self.results_path is not None:
                    self._write_results(ckpt_name, ckpt_results)

    
            except Exception as e:
                print(f"[ValidationCallback] Validation failed for {ckpt_name}: {e}")

        return True

    def _resolve_checkpoint_dir(self) -> Path:
        if self.model_save_dir.is_absolute():
            candidates = [self.model_save_dir]
        else:
            candidates = [self._run_path / self.model_save_dir, self.model_save_dir]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_results_path(self, provided: str | Path | None) -> Path:
        if provided is None:
            return self._run_path / "validation_results.jsonl"

        path = Path(provided).expanduser()
        if path.is_absolute():
            return path
        return (self._run_path / path).absolute()

    def _write_results(self, checkpoint_name: str, checkpoint_results: Dict[str, float]) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(self._run_path),
            "checkpoint": checkpoint_name,
            "metrics": convert_numpy_types(checkpoint_results),
        }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    
    def on_rollout_end(self) -> None:
        log_dict = self.model.logger.name_to_value
        self.run.log(log_dict)

        return super().on_rollout_end()


class debugCallback(BaseCallback):
    def __init__(self, run=None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run = run
        self._episode_sums: Dict[int, Dict[str, float]] = {}
        self._rollout_episode_records: List[Dict[str, float]] = []

    def _init_callback(self) -> None:
        self._ensure_episode_sums()

    def _ensure_episode_sums(self) -> None:
        if not hasattr(self, 'model'):
            return
        env = self.training_env
        self._episode_sums = {
            idx: {'reward_total': 0.0, 'reward_wait': 0.0, 'reward_carbon': 0.0, 'steps': 0.0}
            for idx in range(getattr(env, 'num_envs', 1))
        }

    def _on_training_start(self) -> None:
        self._ensure_episode_sums()

    def _on_rollout_start(self) -> None:
        if not self._episode_sums:
            self._ensure_episode_sums()
        self._rollout_episode_records = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])

        for idx, info in enumerate(infos):
            info = info or {}
            sums = self._episode_sums.setdefault(
                idx,
                {'reward_total': 0.0, 'reward_wait': 0.0, 'reward_carbon': 0.0, 'steps': 0.0},
            )

            reward_total = info.get('reward_total')
            if reward_total is None and idx < len(rewards):
                reward_total = float(rewards[idx])

            sums['reward_total'] += float(reward_total or 0.0)
            sums['reward_wait'] += float(info.get('reward_wait', 0.0))
            sums['reward_carbon'] += float(info.get('reward_carbon', 0.0))
            sums['steps'] += 1.0

            if idx < len(dones) and dones[idx]:
                episode_info = info.get('episode', {})
                episode_metrics = info.get('episode_metrics', {})
                record = {
                    'ep_mean': self._safe_div(sums['reward_total'], sums['steps']),
                    'ep_mean_wait': self._safe_div(sums['reward_wait'], sums['steps']),
                    'ep_mean_carbon_reward': self._safe_div(sums['reward_carbon'], sums['steps']),
                    'episode_return': float(episode_info.get('r', sums['reward_total'])),
                    'avg_wait': float(episode_metrics.get('avg_wait', 0.0)),
                    'avg_emissions': float(episode_metrics.get('avg_emissions', 0.0)),
                    'span_seconds': float(episode_metrics.get('span_seconds', 0.0)),
                    'job_count': float(episode_metrics.get('job_count', 0.0)),
                }
                self._rollout_episode_records.append(record)
                sums.update({'reward_total': 0.0, 'reward_wait': 0.0, 'reward_carbon': 0.0, 'steps': 0.0})

        return True

    def _on_rollout_end(self) -> None:
        payload = self._build_log_payload()
        if payload and self.run is not None:
            self.run.log(payload, step=self.model.num_timesteps)
        if self.verbose and payload:
            print(f"[debugCallback] {payload}")
        self._rollout_episode_records.clear()

    def _build_log_payload(self) -> Dict[str, float]:
        episodes = self._rollout_episode_records or self._collect_partial_episode_metrics()
        if not episodes:
            return {}

        log_data: Dict[str, float] = {}

        def add_mean(key: str, wandb_key: str) -> None:
            values = [entry.get(key) for entry in episodes if self._is_finite(entry.get(key))]
            if values:
                log_data[wandb_key] = float(np.mean(values))

        add_mean('ep_mean', 'debug/avg_ep_mean')
        add_mean('ep_mean_carbon_reward', 'debug/avg_ep_mean_carbon_reward')
        add_mean('ep_mean_wait', 'debug/avg_ep_mean_wait')
        add_mean('avg_wait', 'debug/avg_wait_seconds')
        add_mean('avg_emissions', 'debug/avg_emissions')
        add_mean('span_seconds', 'debug/span_seconds')
        add_mean('job_count', 'debug/job_count')

        log_data['debug/episodes_in_rollout'] = float(len(self._rollout_episode_records))
        return log_data

    def _collect_partial_episode_metrics(self) -> List[Dict[str, float]]:
        partials: List[Dict[str, float]] = []
        for idx, sums in self._episode_sums.items():
            env_metrics = self._compute_env_metrics(idx)
            partials.append(
                {
                    'ep_mean': self._safe_div(sums['reward_total'], sums['steps']),
                    'ep_mean_wait': self._safe_div(sums['reward_wait'], sums['steps']),
                    'ep_mean_carbon_reward': self._safe_div(sums['reward_carbon'], sums['steps']),
                    **env_metrics,
                }
            )
        return partials

    def _compute_env_metrics(self, index: int) -> Dict[str, float]:
        try:
            env = self.training_env.envs[index]
        except Exception:
            return {'avg_wait': 0.0, 'avg_emissions': 0.0, 'span_seconds': 0.0, 'job_count': 0.0}

        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        try:
            metrics = base_env._compute_episode_metrics()
        except Exception:
            metrics = {}

        return {
            'avg_wait': float(metrics.get('avg_wait', 0.0)),
            'avg_emissions': float(metrics.get('avg_emissions', 0.0)),
            'span_seconds': float(metrics.get('span_seconds', 0.0)),
            'job_count': float(metrics.get('job_count', 0.0)),
        }

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator:
            return float(numerator) / float(denominator)
        return 0.0

    @staticmethod
    def _is_finite(value: Optional[float]) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

class StepInfoLoggerCallback(BaseCallback):
    """
    Capture `info` dicts returned by the env at every step
    and append them to a JSONL file for later analysis.

    Notes:
    - Works with vectorized envs (records one line per env having non-empty info).
    - Keeps a small in-memory buffer and flushes to disk every `flush_every` steps
      to avoid excessive I/O on HPC.
    - Optionally logs a sampled subset to Weights & Biases via `run.log`.
    """

    def __init__(
        self,
        save_dir: str,
        run=None,
        filename: str = "step_info.jsonl",
        flush_every: int = 1000,
        wandb_sample_every: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = save_dir.rstrip("/")
        self.run = run
        self.filename = filename
        self.flush_every = int(max(1, flush_every))
        self.wandb_sample_every = int(wandb_sample_every) if wandb_sample_every else None
        self._buffer: List[Dict[str, Any]] = []
        self._path = os.path.join(self.save_dir, self.filename)

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        # Create file if missing; keep appending across resumes
        if not os.path.exists(self._path):
            with open(self._path, "w") as f:
                pass

    def _flush(self) -> None:
        if not self._buffer:
            return
        def _to_serializable(obj):
            try:
                import numpy as _np  # local import to avoid global if unused
            except Exception:
                _np = None

            # Numpy scalars
            if _np is not None and isinstance(obj, (
                _np.integer,
                _np.floating,
                _np.bool_,
            )):
                return obj.item()

            # Numpy arrays
            if _np is not None and isinstance(obj, _np.ndarray):
                return obj.tolist()

            # General fallback for objects with a __dict__ of simple fields
            if hasattr(obj, "__dict__"):
                return {k: _to_serializable(v) for k, v in obj.__dict__.items()}

            # Let json handle (may raise TypeError which json will catch upstream)
            return obj

        with open(self._path, "a") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, default=_to_serializable) + "\n")
        self._buffer.clear()

    def _maybe_log_wandb(self, rec: Dict[str, Any]) -> None:
        if self.run is None or self.wandb_sample_every is None:
            return
        if (self.num_timesteps % self.wandb_sample_every) == 0:
            # Flatten a few top-level fields for readability
            payload = {k: v for k, v in rec.items() if k in ("timestep", "env_index")}
            info = rec.get("info", {})
            for k, v in info.items():
                payload[f"info/{k}"] = v
            self.run.log(payload)

    def _on_step(self) -> bool:
        # SB3 passes per-env infos as a list in self.locals["infos"].
        infos = self.locals.get("infos")
        if infos is None:
            return True
        # Record one entry per env that returned a non-empty info
        for idx, info in enumerate(infos):
            if not isinstance(info, dict) or not info:
                continue

            rec: Dict[str, Any] = {
                "timestep": int(self.num_timesteps),
                "env_index": int(idx),
                "info": info,
            }

            self._buffer.append(rec)
            self._maybe_log_wandb(rec)

        # Flush periodically to keep memory bounded
        if (self.num_timesteps % self.flush_every) == 0:
            self._flush()

        return True

    def _on_rollout_end(self) -> None:
        # Ensure recent entries are persisted between rollouts
        self._flush()

    def _on_training_end(self) -> None:
        # Final flush on training complete
        self._flush()
