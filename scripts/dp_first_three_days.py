#!/usr/bin/env python3

import argparse
import configparser
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dp_baseline import DynamicProgrammingBaseline
from src.hpc_env import HPCenv
from src.utils import get_config_as_dict
from src.validation import Validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an approximate optimal schedule for the first three days.")
    parser.add_argument("--config", type=Path, default=Path("config_file/config.ini"), help="Path to config.ini.")
    parser.add_argument("--mode", type=str, default="test", choices=["training", "validation", "test"], help="Dataset split.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for env reset.")
    parser.add_argument("--horizon-hours", type=float, default=72.0, help="Planning horizon in hours.")
    parser.add_argument("--beam-width", type=int, default=12, help="Beam width for DP approximation.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of queue jobs evaluated per step.")
    parser.add_argument("--max-iters", type=int, default=1500, help="Maximum DP iterations.")
    parser.add_argument("--no-plot", action="store_true", help="Skip rendering the timeseries plot.")
    parser.add_argument("--output-name", type=str, default="dp_first_three_days", help="Prefix for rendered artefacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    config_dict = get_config_as_dict(config)
    config_dict.setdefault("custom_intensity", False)

    horizon_seconds = int(args.horizon_hours * 3600)

    env = HPCenv(config_dict=config_dict, mode=args.mode, trace_enabled=True)
    dp_baseline = DynamicProgrammingBaseline(
        config_dict=config_dict,
        env=env,
        horizon_seconds=horizon_seconds,
        beam_width=args.beam_width,
        schedule_top_k=args.top_k,
        max_iterations=args.max_iters,
    )

    reward, components, trace = dp_baseline.run(seed=args.seed, debug=False)
    print(f"Total reward: {reward:.3f}")
    print(f"Reward components: {components}")
    print(f"Planned actions: {len(trace)} steps logged (trimmed to horizon).")

    if args.no_plot:
        return

    validator = Validation()
    validator.load_dir(config_dict=config_dict)
    validator.mode = args.mode
    validator.render_timeseries_plot(
        trace,
        name=args.output_name,
        mode=args.mode,
        start_time=0,
        end_time=horizon_seconds,
        calendar_split=args.mode,
        save_png=True,
    )
    print(f"Timeseries plot saved to renderings/{args.output_name}.png")


if __name__ == "__main__":
    main()
