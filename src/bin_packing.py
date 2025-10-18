from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.utils import get_config_as_dict
from src.workloads import Workloads


@dataclass(frozen=True)
class BinPackingResult:
    """Lightweight container for best-fit bin packing results."""

    bin_capacity: int
    total_bins_used: int
    total_jobs: int
    total_requested_processors: int
    assignments: Sequence[int]
    remaining_capacity_per_bin: Sequence[int]

    @property
    def average_bin_utilization(self) -> float:
        if not self.remaining_capacity_per_bin:
            return 0.0
        used_capacity = self.total_bins_used * self.bin_capacity - sum(self.remaining_capacity_per_bin)
        return used_capacity / (self.total_bins_used * self.bin_capacity)

    def to_summary_dict(self) -> Dict[str, float]:
        """Return a compact summary that is convenient for logs/tests."""
        return {
            "bin_capacity": float(self.bin_capacity),
            "total_bins_used": float(self.total_bins_used),
            "total_jobs": float(self.total_jobs),
            "total_requested_processors": float(self.total_requested_processors),
            "avg_bin_utilization": float(self.average_bin_utilization),
        }


def _best_fit_assignments(job_sizes: Iterable[int], bin_capacity: int) -> Tuple[List[int], List[int]]:
    """
    Run the best-fit bin packing algorithm for a sequence of job sizes.

    Returns:
        assignments: list mapping job index -> bin index.
        remaining_capacity: list of remaining capacity for each bin.
    """
    assignments: List[int] = []
    remaining_capacity: List[int] = []

    for size in job_sizes:
        if size <= 0:
            assignments.append(-1)
            continue
        if size > bin_capacity:
            raise ValueError(f"Job size {size} exceeds bin capacity {bin_capacity}.")

        best_bin = -1
        best_space = bin_capacity + 1

        for idx, space_left in enumerate(remaining_capacity):
            if size <= space_left:
                new_space = space_left - size
                if new_space < best_space:
                    best_space = new_space
                    best_bin = idx

        if best_bin == -1:
            best_bin = len(remaining_capacity)
            remaining_capacity.append(bin_capacity - size)
        else:
            remaining_capacity[best_bin] -= size

        assignments.append(best_bin)

    return assignments, remaining_capacity


def best_fit_bin_packing_for_workload(
    workload_path: str | Path,
    *,
    config_path: str | Path = "config_file/config.ini",
) -> BinPackingResult:
    """
    Execute best-fit bin packing on a workload file.

    Args:
        workload_path: Path to a workload in SWF format.
        config_path: Path to the scheduler configuration file.

    Returns:
        BinPackingResult with assignments and summary statistics.
    """
    workload_path = Path(workload_path)
    config_path = Path(config_path)

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload file not found: {workload_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)
    config_dict = get_config_as_dict(parser)

    loads = Workloads(str(workload_path), config_dict=config_dict)

    bin_capacity = int(loads.max_nodes or loads.max_procs)
    if bin_capacity <= 0:
        raise ValueError("Bin capacity must be positive based on workload metadata.")

    job_sizes = [job.request_number_of_processors for job in loads.loaded_jobs]
    assignments, remaining_capacity = _best_fit_assignments(job_sizes, bin_capacity)

    total_requested = sum(max(0, size) for size in job_sizes)

    return BinPackingResult(
        bin_capacity=bin_capacity,
        total_bins_used=len(remaining_capacity),
        total_jobs=len(job_sizes),
        total_requested_processors=total_requested,
        assignments=assignments,
        remaining_capacity_per_bin=remaining_capacity,
    )


def best_fit_on_test_workload() -> BinPackingResult:
    """
    Helper that runs best-fit bin packing on the interactive test workload.
    """
    test_workload = Path("data/workloads/interactive/test_workload.swf")
    return best_fit_bin_packing_for_workload(test_workload)
