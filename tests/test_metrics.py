import copy
import configparser
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.carbon_intensity import CarbonIntensity
from src.metrics import compute_average_wait, compute_carbon_emissions
from src.utils import get_config_as_dict
from src.workloads import Workloads


def _load_sample_jobs(count: int = 3):
    config = configparser.ConfigParser()
    config.read(Path("config_file") / "config.ini")
    config_dict = get_config_as_dict(config)
    workloads = Workloads("data/workloads/interactive/test_workload.swf", config_dict=config_dict)
    jobs = []
    for idx, job in enumerate(workloads.loaded_jobs[:count]):
        job_copy = copy.deepcopy(job)
        job_copy.scheduled_time = job_copy.submit_time + (idx + 1) * 100
        jobs.append(job_copy)
    return jobs, config_dict


def test_compute_average_wait_matches_manual_mean():
    jobs, _ = _load_sample_jobs()
    waits = [job.scheduled_time - job.submit_time for job in jobs]

    assert waits  # sanity
    expected_avg = sum(waits) / len(waits)

    avg_wait = compute_average_wait(jobs)
    assert avg_wait == pytest.approx(expected_avg)


def test_compute_carbon_emissions_matches_manual_sum():
    jobs, config_dict = _load_sample_jobs()

    carbon = CarbonIntensity(
        green_win_length=config_dict["green_forecast_length"],
        custom_intensity=config_dict["custom_intensity"],
        normalize=False,
    )
    carbon.set_mode("test")

    total, weighted = compute_carbon_emissions(jobs, carbon)

    manual_total = 0.0
    manual_weighted = 0.0
    for job in jobs:
        start = float(job.scheduled_time)
        end = start + float(job.run_time)
        emissions = carbon.getCarbonEmissions(float(job.power_usage), start, end)
        manual_total += emissions
        manual_weighted += emissions * float(getattr(job, "carbon_consideration", 1.0))

    assert total == pytest.approx(manual_total)
    assert weighted == pytest.approx(manual_weighted)


def test_metrics_skip_unscheduled_jobs():
    jobs, _ = _load_sample_jobs()
    unscheduled = copy.deepcopy(jobs[0])
    unscheduled.scheduled_time = -1

    avg_wait = compute_average_wait([unscheduled])
    assert avg_wait == 0.0
