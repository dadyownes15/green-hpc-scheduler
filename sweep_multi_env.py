import argparse
import configparser
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import wandb
import yaml
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from src.callbacks import BestValidationCallback
from src.hpc_env import HPCenv
from src.utils import (
    create_experiment_name,
    generate_unique_run_suffix,
    get_config_as_dict,
    mask_fn,
)

CONFIG_DICT: Dict[str, Any] = {}


def _to_int_list(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, str):
        return [int(v.strip()) for v in x.split(",") if v.strip()]
    return x


def make_multiple(n: int, base: int) -> int:
    if base <= 0:
        return max(1, int(n))
    n = int(n)
    return ((n + base - 1) // base) * base


def _parse_optional(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def _ensure_seed_sequence(seeds: Any) -> List[int]:
    if isinstance(seeds, (list, tuple)):
        return [int(s) for s in seeds]
    if isinstance(seeds, str):
        parts = [p.strip() for p in seeds.split(",") if p.strip()]
        return [int(p) for p in parts] if parts else []
    if seeds is None:
        return []
    return [int(seeds)]


def merge_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)

    for key, value in (overrides or {}).items():
        if key not in cfg:
            continue
        base_value = cfg[key]
        try:
            if isinstance(base_value, (list, tuple)):
                value = _to_int_list(value)
            elif isinstance(base_value, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "y")
                else:
                    value = bool(value)
            elif isinstance(base_value, int):
                try:
                    float_value = float(value)
                except (TypeError, ValueError):
                    value = int(value)
                else:
                    if float_value.is_integer():
                        value = int(float_value)
                    else:
                        value = float_value
            elif isinstance(base_value, float):
                value = float(value)
        except Exception:
            pass
        cfg[key] = value

    # PPO safety constraints
    batch_size = int(cfg.get("batch_size", 64))
    cfg["n_steps"] = make_multiple(int(cfg.get("n_steps", 2048)), batch_size)

    n_envs = max(1, int(cfg.get("n_envs", 1)))
    cfg["n_envs"] = n_envs
    cfg["n_steps"] = make_multiple(int(cfg["n_steps"]), n_envs)

    if "delay_time_list" in cfg and isinstance(cfg["delay_time_list"], (list, tuple)):
        cfg["delay_time_list_length"] = len(cfg["delay_time_list"])

    return cfg


def build_policy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "net_arch": {
            "pi": _to_int_list(cfg["pi_nn"]),
            "vf": _to_int_list(cfg["vf_nn"]),
        }
    }


def _format_suffix_value(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _build_wandb_run_name(base_name: str, cfg: Dict[str, Any], suffix: str | None = None) -> str:
    suffix_parts: List[str] = []
    seeds = _ensure_seed_sequence(cfg.get("sweep_seeds")) or [cfg.get("seed")]
    if seeds:
        suffix_parts.append("seeds" + "-".join(str(s) for s in seeds))
    eta = cfg.get("eta")
    if eta is not None:
        suffix_parts.append(f"eta{_format_suffix_value(eta)}")
    if suffix:
        suffix_parts.append(str(suffix))
    if not suffix_parts:
        return base_name
    return f"{base_name}__{'_'.join(suffix_parts)}"


def _save_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(config, handle, indent=4)


def _run_single_seed(
    cfg: Dict[str, Any],
    run_path: Path,
    run,
    base_run_name: str,
) -> float:
    seed = int(cfg.get("seed", 0))
    seed_dir = run_path / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _save_config(cfg, seed_dir / "config.json")

    n_envs = 8
    rollout_steps = max(1, int(cfg["n_steps"]) // n_envs)

    env = make_vec_env(
    HPCenv,
        n_envs=cfg["n_envs"],
        env_kwargs=dict(config_dict=cfg, mode="training"),
        wrapper_class=ActionMasker,
        wrapper_kwargs=dict(action_mask_fn=mask_fn),
        vec_env_cls=SubprocVecEnv,
        seed=cfg["seed"],
    )

    # Normalize obs + rewards (stable training)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg["gamma"],  # keep consistent with PPO
    )

    policy_kwargs = build_policy_kwargs(cfg)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        n_steps=rollout_steps,
        ent_coef=cfg["ent_coef"],
        seed=seed,
        learning_rate=cfg["learning_rate"],
        clip_range=cfg["clip_range"],
        clip_range_vf=_parse_optional(cfg.get("clip_range_vf")),
        normalize_advantage=cfg.get("normalize_advantage", True),
        vf_coef=cfg["vf_coef"],
        policy_kwargs=policy_kwargs,
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_freq=0,
        model_save_path=None,
        verbose=0,
    )

    best_cb = BestValidationCallback(
        config_dict=cfg,
        save_path=seed_dir / "best_model.zip",
        eval_freq=cfg.get("validation_freq", cfg["n_steps"]),
        n_eval_episodes=cfg.get("validation_episodes", 1),
        run=run,
        seed_label=f"seed_{seed}",
    )

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=[wandb_cb, best_cb],
        progress_bar=False,
        log_interval=1,
    )

    env.close()
    run.summary[f"seed_{seed}_best_validation_reward"] = best_cb.best_score
    return best_cb.best_score


def train():
    with wandb.init(project="green_scheduler") as run:
        sweep_overrides = dict(run.config)
        cfg = merge_overrides(CONFIG_DICT, sweep_overrides)

        base_run_id = create_experiment_name(config=cfg, workload_file=None)
        run_suffix = generate_unique_run_suffix()
        run_id = f"{base_run_id}__{run_suffix}"
        run.name = _build_wandb_run_name(base_run_id, cfg, suffix=run_suffix)
        run_path = Path("results") / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        _save_config(cfg, run_path / "config.json")

        wandb.config.update(cfg, allow_val_change=True)
        run.summary["run_base_name"] = base_run_id
        run.summary["run_suffix"] = run_suffix
        run.summary["run_id"] = run_id
        run.summary["run_path"] = str(run_path)

        seeds = _ensure_seed_sequence(cfg.get("sweep_seeds"))
        if not seeds:
            seeds = [cfg.get("seed", 0)]

        best_scores: List[float] = []
        for seed in seeds:
            seed_cfg = deepcopy(cfg)
            seed_cfg["seed"] = int(seed)
            score = _run_single_seed(seed_cfg, run_path, run, run_id)
            if np.isfinite(score):
                best_scores.append(score)

        if best_scores:
            mean_score = float(np.mean(best_scores))
            std_score = float(np.std(best_scores)) if len(best_scores) > 1 else 0.0
            run.log({"objective": mean_score})
            run.summary["objective"] = mean_score
            run.summary["objective_std"] = std_score
        else:
            run.log({"objective": float("nan")})
            run.summary["objective"] = float("nan")
            run.summary["objective_std"] = float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Path to sweep.yaml specification.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of agents to execute for the sweep.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override the eta value used during the sweep.",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), "config_file", "config.ini")
    config.read(config_path)
    CONFIG_DICT = get_config_as_dict(config=config)
    if args.eta is not None:
        CONFIG_DICT["eta"] = float(args.eta)

    with open(args.sweep, "r") as handle:
        sweep_cfg = yaml.safe_load(handle)

    sweep_id = wandb.sweep(sweep=sweep_cfg, project="green_scheduler")
    wandb.agent(sweep_id, function=train, count=args.count)
