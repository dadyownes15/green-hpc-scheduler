
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import time
import json
from typing import List, Dict, Any, Optional

from src.validation import Validation
import os

class SweepCallBack(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,run, config_dict, verbose = 0, val_freq = 100_000):
        self.run = run
        self.config_dict = config_dict
        self.val_freq = val_freq
        self.steps_count = 0
        self.roll_out_count = 0
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: If the callback returns False, training is aborted early.
        """
        self.steps_count += 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        log_dict = self.model.logger.name_to_value
        print("Roll out end: ", log_dict)
        self.run.log(log_dict)
    


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        val = Validation()
        val.load_dir(config_dict=self.config_dict)
        results, _ = val.validate_model(1, self.model, "validation")
        self.run.log(results)



class ValidationCallback(BaseCallback):
    """
    Runs validation on the latest checkpoint at a fixed frequency using
    Validation.validate_policy in "validation" mode.

    It expects checkpoints to be saved with Stable-Baselines3's CheckpointCallback
    naming convention: {name_prefix}_{num_timesteps}_steps under `<run_dir>/logs/`.
    """

    def __init__(self, run_dir: str, name_prefix: str, run, val_freq: int = 500000, n_eval_episodes: int = 1, verbose: int = 0, ):
        super().__init__(verbose)
        self.run_dir = run_dir.rstrip("/")
        self.name_prefix = name_prefix
        self.val_freq = int(val_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.run = run

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
                results, _ = validator.validate_policy(
                    n_eval_episodes=self.n_eval_episodes,
                    checkpoints=[ckpt_name],
                    mode="validation",
                    debug=False,
                )
                print(results[ckpt_name])
                self.run.log(results[ckpt_name])

    
            except Exception as e:
                print(f"[ValidationCallback] Validation failed for {ckpt_name}: {e}")

        return True
    
    def on_rollout_end(self) -> None:
        log_dict = self.model.logger.name_to_value
        self.run.log(log_dict)

        return super().on_rollout_end()


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
