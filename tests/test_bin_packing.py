import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bin_packing import best_fit_on_test_workload


def test_best_fit_on_test_workload_summary():
    result = best_fit_on_test_workload()
    summary = result.to_summary_dict()

    assert summary["bin_capacity"] == 256.0
    assert summary["total_jobs"] == float(len(result.assignments))
    assert result.total_bins_used == len(result.remaining_capacity_per_bin)
    assert result.total_bins_used == 1480
    assert summary["total_requested_processors"] == 378707.0
    assert result.assignments.count(-1) == 0
