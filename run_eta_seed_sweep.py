"""Run training sweeps across multiple seeds and eta values."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import configparser

from src.training import Train
from src.utils import get_config_as_dict


def find_project_root(start: Path) -> Path:
    for directory in [start, *start.parents]:
        if (directory / "config_file" / "config.ini").exists():
            return directory
    raise FileNotFoundError(
        "Could not locate project root containing 'config_file/config.ini'."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DEFAULT_WORKLOAD = PROJECT_ROOT / "data" / "workloads" / "4h_mean" / "training_workload.swf"
CONFIG_PATH = PROJECT_ROOT / "config_file" / "config.ini"
ETA_VALUES = [0,0.25,0.5,0.75,1]
SEED_VALUES = list(range(1, 6))


def load_base_config(config_path: Path) -> dict:
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"Failed to read configuration file at '{config_path}'.")
    return get_config_as_dict(config)


def main() -> None:
    base_cfg = load_base_config(CONFIG_PATH)
    workload_path = DEFAULT_WORKLOAD
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found at '{workload_path}'.")

    for eta in ETA_VALUES:
        for seed in SEED_VALUES:
            run_cfg = deepcopy(base_cfg)
            run_cfg["eta"] = eta
            run_cfg["seed"] = seed

            print("=" * 80)
            print(f"Starting training run | eta={eta:.1f} | seed={seed}")
            trainer = Train(
                config_dict=run_cfg,
                workload_path=str(workload_path),
                save_freq=run_cfg["n_steps"],
            )
            try:
                trainer.run(save_checkpoints=True)
            except Exception as exc:  # noqa: BLE001 - continue sweep even on failure
                print(f"Run failed for eta={eta:.1f}, seed={seed}: {exc}")


if __name__ == "__main__":
    main()
