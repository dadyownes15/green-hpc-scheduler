from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.carbon_intensity import CarbonIntensity
    from src.job import Job


def _iter_jobs(job_history: Iterable["Job"] | None) -> Iterable["Job"]:
    """Yield jobs from potentially None input while skipping null entries."""
    if job_history is None:
        return ()
    return (job for job in job_history if job is not None)


def collect_wait_times(job_history: Iterable["Job"] | None) -> List[float]:
    """
    Extract raw wait times (seconds) for all scheduled jobs in the history.
    Jobs without a scheduled time are ignored.
    """
    waits: List[float] = []
    for job in _iter_jobs(job_history):
        scheduled_time = getattr(job, "scheduled_time", -1)
        submit_time = getattr(job, "submit_time", None)
        if scheduled_time is None or submit_time is None or scheduled_time < 0:
            continue
        waits.append(float(scheduled_time - submit_time))
    return waits


def compute_average_wait(job_history: Iterable["Job"] | None) -> float:
    """Return the mean wait time in seconds across all scheduled jobs."""
    waits = collect_wait_times(job_history)
    if not waits:
        return 0.0
    return float(sum(waits) / len(waits))


def compute_carbon_emissions(
    job_history: Sequence["Job"] | None,
    carbon_intensity_calculator: "CarbonIntensity",
) -> Tuple[float, float]:
    """
    Calculate total and carbon-consideration-weighted emissions for the episode.

    Returns:
        (total_emissions, weighted_emissions)
    """
    if job_history is None:
        return 0.0, 0.0

    total_emissions = 0.0
    weighted_emissions = 0.0

    for job in _iter_jobs(job_history):
        scheduled_time = getattr(job, "scheduled_time", -1)
        run_time = getattr(job, "run_time", 0)
        power_usage = getattr(job, "power_usage", -1)

        if scheduled_time is None or scheduled_time < 0:
            continue
        if run_time is None or run_time <= 0:
            continue
        if power_usage is None or power_usage <= 0:
            continue

        end_time = scheduled_time + run_time
        job_emissions = carbon_intensity_calculator.getCarbonEmissions(
            float(power_usage),
            float(scheduled_time),
            float(end_time),
        )
        total_emissions += job_emissions
        weighted_factor = float(getattr(job, "carbon_consideration", 1.0))
        weighted_emissions += job_emissions * weighted_factor

    return total_emissions, weighted_emissions
