#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils import create_experiment_name, get_config_as_dict, convert_numpy_types
from src.validation import Validation

WORKLOAD_PATH = Path("data/workloads/training_workload.swf")
CHECKPOINT_PATTERN = re.compile(r"(?:model|seed_\\d+)_(\\d+)_steps(?:\\.zip)?$")


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
                step_index = int(float(raw["steps"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {csv_path}: {raw}") from exc
            rows.append({
                "eta": eta,
                "seed": seed,
                "step_index": step_index,
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
    # Prefer directories with matching batch size and reward type.
    target_batch = base_config.get("batch_size")
    target_reward = base_config.get("reward_type")
    filtered = [c for c in candidates if c[1].get("batch_size") == target_batch and c[1].get("reward_type") == target_reward]
    if filtered:
        candidates = filtered
    candidates.sort(key=lambda item: str(item[0]))
    return candidates[0][0]


def resolve_seed_dir(run_dir: Path, seed: int) -> Path:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory missing under {run_dir}")
    direct = logs_dir / str(seed)
    if direct.exists():
        return direct
    prefixed = logs_dir / f"seed_{seed}"
    if prefixed.exists():
        return prefixed
    matches = [p for p in logs_dir.iterdir() if p.is_dir() and re.fullmatch(r"seed_?%d" % seed, p.name)]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No checkpoints folder for seed={seed} in {logs_dir}")


def list_checkpoints(seed_dir: Path) -> List[Tuple[int, Path]]:
    candidates: List[Tuple[int, Path]] = []
    if not seed_dir.exists():
        return candidates
    for item in seed_dir.iterdir():
        if not item.is_file():
            continue
        match = CHECKPOINT_PATTERN.search(item.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), item))
    candidates.sort(key=lambda entry: entry[0])
    return candidates


def pick_checkpoint(seed_dir: Path, step_index: int) -> Tuple[Path, int]:
    checkpoints = list_checkpoints(seed_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir}")
    if step_index < 0 or step_index >= len(checkpoints):
        raise IndexError(f"Requested step index {step_index} but only {len(checkpoints)} checkpoints available in {seed_dir}")
    return checkpoints[step_index][1], checkpoints[step_index][0]


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
        step_index = entry["step_index"]
        run_info = {
            "eta": eta,
            "seed": seed,
            "step_index": step_index,
        }
        try:
            run_dir = resolve_model_dir(base_config, eta, args.results_root)
        except FileNotFoundError as exc:
            run_info["status"] = "missing_run"
            run_info["error"] = str(exc)
            results.append(run_info)
            print(f"[SKIP] {run_info}: {exc}")
            continue

        try:
            seed_dir = resolve_seed_dir(run_dir, seed)
        except FileNotFoundError as exc:
            run_info.update({
                "status": "missing_seed",
                "model_dir": str(run_dir),
                "error": str(exc),
            })
            results.append(run_info)
            print(f"[SKIP] {run_info}: {exc}")
            continue

        try:
            checkpoint_file, timestep = pick_checkpoint(seed_dir, step_index)
        except (FileNotFoundError, IndexError) as exc:
            run_info.update({
                "status": "missing_checkpoint",
                "model_dir": str(run_dir),
                "checkpoint_dir": str(seed_dir),
                "error": str(exc),
            })
            results.append(run_info)
            print(f"[SKIP] {run_info}: {exc}")
            continue

        run_info.update({
            "model_dir": str(run_dir),
            "checkpoint_dir": str(seed_dir),
            "checkpoint_name": checkpoint_file.name,
            "timesteps": timestep,
        })

        if args.dry_run:
            run_info["status"] = "pending"
            results.append(run_info)
            print(f"[DRY] Would evaluate {checkpoint_file}")
            continue

        try:
            metrics = evaluate_checkpoint(
                run_dir=run_dir,
                checkpoint_path=checkpoint_file,
                checkpoint_dir=seed_dir,
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
        print(f"[OK] eta={eta} seed={seed} step={step_index} -> {checkpoint_file.name}")

    write_results(results, args.output)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
