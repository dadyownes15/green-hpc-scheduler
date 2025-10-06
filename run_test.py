#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils import convert_numpy_types, create_experiment_name, get_config_as_dict
from src.validation import Validation

WORKLOAD_PATH = Path("data/workloads/training_workload.swf")
DEFAULT_STEPS_PER_CHECKPOINT = 2

SEED_FILE_PATTERN = re.compile(r"^seed_(?P<seed>\d+)_(?P<timesteps>\d+)_steps$")
MODEL_FILE_PATTERN = re.compile(r"^(?:model|seed_\d+)_(?P<timesteps>\d+)_steps$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best checkpoints on the test split.")
    parser.add_argument(
        "--best-csv",
        type=Path,
        default=Path("best_in_val.csv"),
        help="CSV file listing best validation checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config_file") / "config.ini",
        help="Path to the base config used for training runs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory that contains trained model runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "test_metrics.csv",
        help="Destination CSV for aggregated test metrics.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to run per checkpoint.",
    )
    parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=DEFAULT_STEPS_PER_CHECKPOINT,
        help="Number of CSV step increments that correspond to one saved checkpoint.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging inside the validation loop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the checkpoints that would be evaluated without running them.",
    )
    return parser.parse_args()


def load_base_config(config_path: Path) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    parser.read(config_path)
    return get_config_as_dict(parser)


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            pass
    return payload


def read_best_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Best-in-validation CSV not found: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"eta", "seed", "steps"}
        if not required.issubset(reader.fieldnames or set()):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for raw in reader:
            try:
                eta = float(raw["eta"])
                seed = int(float(raw["seed"]))
                step_value = int(float(raw["steps"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {raw}") from exc
            rows.append({
                "eta": eta,
                "seed": seed,
                "step_value": step_value,
                "raw": raw,
            })
    return rows


def expected_run_dir(base_config: Dict[str, Any], eta: float, results_root: Path) -> Path:
    config_clone = dict(base_config)
    config_clone["eta"] = eta
    run_name = create_experiment_name(config_clone, str(WORKLOAD_PATH))
    return results_root / run_name


def scan_candidates(results_root: Path, eta: float) -> List[Tuple[Path, Dict[str, Any]]]:
    candidates: List[Tuple[Path, Dict[str, Any]]] = []
    for cfg_path in results_root.rglob("config.json"):
        try:
            data = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        eta_val = data.get("eta")
        if isinstance(eta_val, str):
            try:
                eta_val = float(eta_val)
            except ValueError:
                continue
        if eta_val is None:
            continue
        if abs(float(eta_val) - eta) > 1e-6:
            continue
        candidates.append((cfg_path.parent, data))
    return candidates


def resolve_model_dir(base_config: Dict[str, Any], eta: float, results_root: Path) -> Path:
    primary = expected_run_dir(base_config, eta, results_root)
    if primary.exists():
        return primary
    candidates = scan_candidates(results_root, eta)
    if not candidates:
        raise FileNotFoundError(f"No trained run found for eta={eta}")
    if len(candidates) == 1:
        return candidates[0][0]
    target_batch = base_config.get("batch_size")
    target_reward = base_config.get("reward_type")
    filtered = [c for c in candidates if c[1].get("batch_size") == target_batch and c[1].get("reward_type") == target_reward]
    if filtered:
        candidates = filtered
    candidates.sort(key=lambda item: str(item[0]))
    return candidates[0][0]


def parse_checkpoint_filename(filename: str, parent_seed: Optional[int]) -> Optional[Tuple[Optional[int], int]]:
    stem = filename[:-4] if filename.endswith(".zip") else filename
    match = SEED_FILE_PATTERN.match(stem)
    if match:
        return int(match.group("seed")), int(match.group("timesteps"))
    match = MODEL_FILE_PATTERN.match(stem)
    if match:
        return parent_seed, int(match.group("timesteps"))
    return None


def _infer_seed_from_name(name: str) -> Optional[int]:
    if name.isdigit():
        return int(name)
    if name.startswith("seed_"):
        try:
            return int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def collect_seed_checkpoints(run_dir: Path, seed: int) -> List[Tuple[int, Path]]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory missing under {run_dir}")

    checkpoints: Dict[int, Path] = {}

    def consider(path: Path, parent_seed: Optional[int]) -> None:
        parsed = parse_checkpoint_filename(path.name, parent_seed)
        if parsed is None:
            return
        file_seed, timesteps = parsed
        if file_seed is None:
            file_seed = parent_seed
        if file_seed != seed:
            return
        checkpoints.setdefault(timesteps, path)

    for child in logs_dir.iterdir():
        if child.is_file():
            consider(child, None)
        elif child.is_dir():
            inferred_seed = _infer_seed_from_name(child.name)
            for file_path in child.iterdir():
                if file_path.is_file():
                    consider(file_path, inferred_seed)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints for seed={seed} in {logs_dir}")

    return sorted(checkpoints.items(), key=lambda item: item[0])


def infer_n_steps(run_config: Dict[str, Any], base_config: Dict[str, Any], checkpoints: Sequence[Tuple[int, Path]]) -> Optional[int]:
    for source in (run_config, base_config):
        try:
            candidate = int(source.get("n_steps"))  # type: ignore[arg-type]
        except (TypeError, ValueError, AttributeError):
            candidate = None
        if candidate and candidate > 0:
            return candidate
    timesteps = sorted({ts for ts, _ in checkpoints})
    diffs = [b - a for a, b in zip(timesteps, timesteps[1:]) if b > a]
    if diffs:
        return min(diffs)
    return timesteps[0] if timesteps else None


def select_checkpoint(
    checkpoints: Sequence[Tuple[int, Path]],
    step_value: int,
    n_steps: Optional[int],
    steps_per_checkpoint: int,
) -> Tuple[Path, int, bool, Optional[int], int, int]:
    stride = max(steps_per_checkpoint, 1)
    bucket_index = max(step_value // stride, 0)
    selected_index = min(bucket_index, len(checkpoints) - 1)
    timesteps, path = checkpoints[selected_index]
    expected_timesteps = None
    if n_steps is not None and n_steps > 0:
        expected_timesteps = n_steps * (bucket_index + 1)
    exact_match = expected_timesteps == timesteps if expected_timesteps is not None else False
    return path, timesteps, bool(exact_match), expected_timesteps, bucket_index, selected_index

def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(convert_numpy_types(value), sort_keys=True)
        else:
            flat[key] = value
    return flat


def evaluate_checkpoint(
    run_dir: Path,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    episodes: int,
    mode: str,
    debug: bool,
) -> Dict[str, Any]:
    validator = Validation()
    validator.load_dir(str(run_dir))
    try:
        relative_dir = checkpoint_dir.relative_to(run_dir)
    except ValueError:
        relative_dir = checkpoint_dir
    processed, _ = validator.validate_policy(
        n_eval_episodes=episodes,
        checkpoints=[checkpoint_path.name],
        mode=mode,
        debug=debug,
        checkpoint_dir=str(relative_dir),
    )
    return processed[checkpoint_path.name]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_results(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    rows = list(rows)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    base_config = load_base_config(args.config)
    best_rows = read_best_rows(args.best_csv)
    results: List[Dict[str, Any]] = []

    for entry in best_rows:
        eta = entry["eta"]
        seed = entry["seed"]
        step_value = entry["step_value"]
        run_info: Dict[str, Any] = {
            "eta": eta,
            "seed": seed,
            "step_value": step_value,
        }

        try:
            run_dir = resolve_model_dir(base_config, eta, args.results_root)
        except FileNotFoundError as exc:
            run_info["status"] = "missing_run"
            run_info["error"] = str(exc)
            results.append(run_info)
            print(f"[SKIP] {run_info}: {exc}")
            continue

        run_info["model_dir"] = str(run_dir)

        run_config = load_run_config(run_dir)

        try:
            checkpoints = collect_seed_checkpoints(run_dir, seed)
        except FileNotFoundError as exc:
            run_info.update({
                "status": "missing_seed",
                "error": str(exc),
            })
            results.append(run_info)
            print(f"[SKIP] {run_info}: {exc}")
            continue

        n_steps = infer_n_steps(run_config, base_config, checkpoints)

        selection = select_checkpoint(
            checkpoints=checkpoints,
            step_value=step_value,
            n_steps=n_steps,
            steps_per_checkpoint=args.steps_per_checkpoint,
        )
        checkpoint_file, timesteps, exact_match, expected_ts, bucket_index, selected_index = selection
        checkpoint_dir = checkpoint_file.parent
        run_info.update({
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_name": checkpoint_file.name,
            "timesteps": timesteps,
            "timesteps_match": "exact" if exact_match else "approximate",
        })
        if expected_ts is not None:
            run_info["expected_timesteps"] = expected_ts
        elif n_steps is not None:
            run_info["inferred_n_steps"] = n_steps
        if bucket_index != selected_index:
            run_info["bucket_index"] = bucket_index
            run_info["selected_index"] = selected_index

        if args.dry_run:
            run_info["status"] = "pending"
            results.append(run_info)
            print(f"[DRY] Would evaluate {checkpoint_file}")
            continue

        try:
            metrics = evaluate_checkpoint(
                run_dir=run_dir,
                checkpoint_path=checkpoint_file,
                checkpoint_dir=checkpoint_dir,
                episodes=args.episodes,
                mode=args.mode,
                debug=args.debug,
            )
        except Exception as exc:  # noqa: BLE001
            run_info.update({
                "status": "error",
                "error": str(exc),
            })
            results.append(run_info)
            print(f"[FAIL] {checkpoint_file}: {exc}")
            continue

        flat_metrics = flatten_metrics(metrics)
        run_info.update(flat_metrics)
        run_info["status"] = "ok"
        results.append(run_info)
        print(f"[OK] eta={eta} seed={seed} step={step_value} -> {checkpoint_file.name}")

    write_results(results, args.output)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
