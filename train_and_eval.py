import argparse
import configparser
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import wandb
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from src.callbacks import BestValidationCallback
from src.hpc_env import HPCenv
from src.utils import convert_numpy_types, generate_unique_run_suffix, get_config_as_dict, mask_fn
from src.validation import Validation


warnings.filterwarnings(
    "ignore",
    message="The 'repr' attribute with value False was provided to the `Field()` function",
)
warnings.filterwarnings(
    "ignore",
    message="The 'frozen' attribute with value True was provided to the `Field()` function",
)

def _to_int_list(values: Any) -> Any:
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    if isinstance(values, str):
        parts = [p.strip() for p in values.split(",") if p.strip()]
        return [int(p) for p in parts]
    return values


def _format_suffix_value(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _build_wandb_run_name(cfg: Dict[str, Any], suffix: str | None = None) -> str:
    eta = cfg.get("eta")
    seed = cfg.get("seed")
    user_ci = cfg.get("user_ci")
    parts = []
    if user_ci is not None:
        parts.append(f"ci{_format_suffix_value(user_ci)}")
    eta_part = f"eta{_format_suffix_value(eta)}" if eta is not None else "etaNA"
    parts.append(eta_part)
    if seed is not None:
        parts.append(f"seed{seed}")
    if suffix:
        parts.append(str(suffix))
    return "_".join(parts)


def _save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(cfg, handle, indent=4)


def _ensure_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return max(1, int(value))
    value = int(value)
    return ((value + multiple_of - 1) // multiple_of) * multiple_of


def _build_policy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "net_arch": {
            "pi": _to_int_list(cfg["pi_nn"]),
            "vf": _to_int_list(cfg["vf_nn"]),
        }
    }


def _parse_optional(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def _load_config(path: str) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"Unable to read config file at '{path}'.")
    return get_config_as_dict(parser)


def _apply_overrides(cfg: Dict[str, Any], overrides: Iterable[tuple[str, Optional[Any]]]) -> Dict[str, Any]:
    updated = dict(cfg)
    for key, value in overrides:
        if value is None or key not in updated:
            continue
        base_value = updated[key]
        try:
            if isinstance(base_value, bool):
                if isinstance(value, str):
                    updated[key] = value.lower() in {"true", "1", "yes", "y"}
                else:
                    updated[key] = bool(value)
            elif isinstance(base_value, int):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    updated[key] = value
                else:
                    if numeric_value.is_integer():
                        updated[key] = int(numeric_value)
                    else:
                        updated[key] = numeric_value
            elif isinstance(base_value, float):
                updated[key] = float(value)
            elif isinstance(base_value, (list, tuple)):
                updated[key] = _to_int_list(value)
            else:
                updated[key] = value
        except Exception:
            updated[key] = value
    if "delay_time_list" in updated and isinstance(updated["delay_time_list"], (list, tuple)):
        updated["delay_time_list_length"] = len(updated["delay_time_list"])
    return updated


def _determine_seeds(cfg: Dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.seed_list:
        seeds = []
        for item in args.seed_list.split(","):
            item = item.strip()
            if not item:
                continue
            seeds.append(int(item))
        if not seeds:
            raise ValueError("Provided --seed-list is empty after parsing.")
        return seeds

    num_seeds = max(1, int(getattr(args, "seeds", 1)))
    if num_seeds == 1:
        return [int(cfg.get("seed", args.seed or 0))]
    return list(range(1, num_seeds + 1))


def _evaluate_best_checkpoint(seed_dir: Path, model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        return {}

    try:
        evaluator = Validation().load_dir(model_dir=str(seed_dir))
        metrics, _ = evaluator.validate_policy(
            n_eval_episodes=1,
            checkpoints=[model_path.name],
            mode="test",
            debug=False,
            checkpoint_dir=model_path.parent,
        )

        print("Metrics from the fina test of train and eval: ", metrics)
    except Exception as exc:
        print(f"[train_and_eval] Test evaluation failed for '{model_path}': {exc}")
        return {}

    return metrics.get(model_path.name, {})


def train_and_eval(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    overrides = [
        ("eta", args.eta),
        ("seed", args.seed),
        ("total_timesteps", args.total_timesteps),
        ("validation_freq", args.validation_freq),
        ("validation_episodes", args.validation_episodes),
        ("learning_rate", args.learning_rate),
        ("batch_size", args.batch_size),
        ("n_steps", args.n_steps),
        ("user_ci", args.user_ci),
    ]
    base_cfg = _apply_overrides(cfg, overrides)

    seeds_to_run = _determine_seeds(base_cfg, args)
    session_suffix = generate_unique_run_suffix()

    run_root = Path("results") / "train_and_eval" / f"session_{session_suffix}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[train_and_eval] Outputs will be written to {run_root}")

    user_ci_value = base_cfg.get("user_ci", "disabled") or "disabled"
    user_ci_suffix = str(_format_suffix_value(user_ci_value)).replace(" ", "_").replace("/", "_")
    user_ci_dir = run_root / f"user_ci_{user_ci_suffix}"
    user_ci_dir.mkdir(parents=True, exist_ok=True)

    eta_value = base_cfg.get("eta")
    eta_suffix = _format_suffix_value(eta_value) if eta_value is not None else "NA"
    eta_dir = user_ci_dir / f"eta_{eta_suffix}"
    eta_dir.mkdir(parents=True, exist_ok=True)
    _save_config(base_cfg, eta_dir / "config.json")
    eval_log_path = eta_dir / "test_eval_log.jsonl"

    for seed in seeds_to_run:
        cfg_seed = dict(base_cfg)
        cfg_seed["seed"] = int(seed)

        n_envs = args.n_envs if args.n_envs is not None else cfg_seed.get("n_envs", 8)
        cfg_seed["n_envs"] = max(1, int(n_envs))

        batch_size = int(cfg_seed.get("batch_size", 64))
        cfg_seed["batch_size"] = batch_size
        cfg_seed["n_steps"] = _ensure_multiple(int(cfg_seed.get("n_steps", 2048)), batch_size)
        rollout_steps = max(1, cfg_seed["n_steps"] // cfg_seed["n_envs"])
        cfg_seed["n_steps"] = rollout_steps * cfg_seed["n_envs"]

        seed_dir = eta_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _save_config(cfg_seed, seed_dir / "config.json")

        if args.run_name:
            base_run_name = args.run_name
            if len(seeds_to_run) > 1:
                base_run_name = f"{base_run_name}__seed{seed}"
            run_name = f"{base_run_name}__{session_suffix}"
        else:
            run_name = _build_wandb_run_name(cfg_seed, suffix=session_suffix)

        wandb_kwargs = {
            "project": "green_hpc_scheduler_v2",
            "config": cfg_seed,
            "name": run_name,
            "sync_tensorboard": True,
            "tags": [],
        }
        if args.wandb_dir is not None:
            wandb_kwargs["dir"] = args.wandb_dir

        with wandb.init(
            
            **wandb_kwargs
                        
                        ) as run:
            run.name = run_name

            env = make_vec_env(
                HPCenv,
                n_envs=cfg_seed["n_envs"],
                env_kwargs=dict(config_dict=cfg_seed, mode="training"),
                wrapper_class=ActionMasker,
                wrapper_kwargs=dict(action_mask_fn=mask_fn),
                vec_env_cls=SubprocVecEnv,
                seed=cfg_seed["seed"],
            )

            env = VecNormalize(venv=env)

            policy_kwargs = _build_policy_kwargs(cfg_seed)
            tensorboard_dir = seed_dir / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                gamma=cfg_seed["gamma"],
                gae_lambda=cfg_seed["gae_lambda"],
                batch_size=cfg_seed["batch_size"],
                n_epochs=cfg_seed["n_epochs"],
                n_steps=rollout_steps,
                ent_coef=cfg_seed["ent_coef"],
                seed=cfg_seed["seed"],
                learning_rate=cfg_seed["learning_rate"],
                clip_range=cfg_seed["clip_range"],
                clip_range_vf=_parse_optional(cfg_seed.get("clip_range_vf")),
                normalize_advantage=cfg_seed.get("normalize_advantage", True),
                vf_coef=cfg_seed["vf_coef"],
                max_grad_norm=cfg_seed.get("max_grad_norm", 0.5),
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(tensorboard_dir),
            )

            wandb_cb = WandbCallback(
                gradient_save_freq=0,
                model_save_freq=0,
                model_save_path=None,
                verbose=0,
            )
            best_model_path = seed_dir / "best_model.zip"
            best_cb = BestValidationCallback(
                config_dict=cfg_seed,
                save_path=best_model_path,
                eval_freq=max(1, cfg_seed["n_steps"]*2),
                greater_is_better=False,
                n_eval_episodes=cfg_seed.get("validation_episodes", 1),
                run=run,
                seed_label=None,
            )

            model.learn(
                total_timesteps=int(cfg_seed["total_timesteps"]),
                callback=[wandb_cb, best_cb],
                progress_bar=False,
                log_interval=1,
            )
            env.close()

            best_score = best_cb.best_score
            run.summary["session_suffix"] = session_suffix
            run.summary["output_root"] = str(run_root)
            run.summary["best_validation_reward"] = best_score if math.isfinite(best_score) else float("nan")
            run.summary["best_model_path"] = str(best_model_path)

            test_metrics = _evaluate_best_checkpoint(seed_dir, best_model_path)
            if test_metrics:
                test_metrics = convert_numpy_types(test_metrics)
                run.summary["test_eval_metrics"] = test_metrics

            log_payload: Dict[str, Any] = {
                "seed": seed,
                "eta": cfg_seed.get("eta"),
                "session_suffix": session_suffix,
                "best_model_path": str(best_model_path),
                "best_validation_reward": best_score,
                "test_eval_metrics": test_metrics,
            }
            with eval_log_path.open("a", encoding="utf-8") as handle:
                json.dump(convert_numpy_types(log_payload), handle)
                handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO across seeds with validation and test eval.")
    parser.add_argument("--config", type=str, default=os.path.join("config_file", "config.ini"))
    parser.add_argument("--project", type=str, default="green_scheduler")
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-list", type=str, default=None, help="Comma separated list of seeds to run. Overrides --seed/--seeds.")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--validation-freq", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to train over, starting at seed 1.")
    parser.add_argument("--user-ci", type=str, default=None, help="Override user_ci distribution (e.g., disabled, uniform, two_bins, exp_drop).")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_eval(parse_args())
