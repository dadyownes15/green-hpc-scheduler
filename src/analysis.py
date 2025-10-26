from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.metrics import collect_wait_times
from src.carbon_intensity import CarbonIntensity
from src.viz_style import PALETTE, finalize, use_house_style

_PROCESSOR_BUCKET_LABELS = ["1", "2-3", "4-7", "8-31", "32-127", "128-255", "256+"]


def _processor_bucket(value: float) -> int:
    v = float(value)
    if v <= 1:
        return 0
    if v <= 3:
        return 1
    if v <= 7:
        return 2
    if v <= 31:
        return 3
    if v <= 127:
        return 4
    if v <= 255:
        return 5
    return 6


def _processor_bucket_edges(num_buckets: int) -> np.ndarray:
    return np.arange(num_buckets + 1, dtype=float) - 0.5


def _compute_wait_bin_edges(
    waits: Sequence[float],
    *,
    min_bins: int = 4,
    max_bins: int = 8,
) -> np.ndarray:
    arr = np.asarray(waits, dtype=float)
    if arr.size == 0:
        return np.array([0.0, 1.0])

    arr = arr[arr >= 0.0]
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)

    candidate_percentiles = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    edges = np.quantile(arr, candidate_percentiles)
    edges = np.unique(np.round(edges, decimals=6))

    if edges.size < 2:
        single = edges[0]
        return np.array([single - 0.5, single + 0.5])

    while edges.size - 1 < min_bins:
        edges = np.linspace(edges[0], edges[-1], edges.size + 1)
        edges = np.unique(np.round(edges, decimals=6))

    if edges.size - 1 > max_bins:
        idx = np.linspace(0, edges.size - 1, max_bins + 1).round().astype(int)
        edges = edges[idx]
        edges = np.unique(edges)

    if edges.size < 2:
        edges = np.array([arr.min(), arr.max()])
        if np.isclose(edges[0], edges[1]):
            edges[1] = edges[0] + 1.0

    return edges


def _format_wait_labels(edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            labels.append(f"{int(round(lo))}+")
        else:
            labels.append(f"{int(round(lo))}–{int(round(hi))}")
    return labels


def _collect_waits_and_sizes(
    raw_stats: Mapping[str, Any],
    *,
    size_metric: str,
    key: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if key is not None:
        raw_stats = raw_stats.get(key) or {}

    histories = raw_stats.get("job_scheduled_history") or []
    waits: list[float] = []
    sizes: list[float] = []
    for history in histories:
        for job in history or []:
            if job is None:
                continue
            scheduled_time = getattr(job, "scheduled_time", None)
            submit_time = getattr(job, "submit_time", None)
            if scheduled_time is None or submit_time is None or scheduled_time < 0:
                continue
            waits.append(float(scheduled_time - submit_time))
            if size_metric == "compute_hours":
                sizes.append(float(getattr(job, "compute_hours", 0.0)))
            else:
                sizes.append(float(getattr(job, "request_number_of_processors", 0.0)))
    return np.asarray(waits, dtype=float), np.asarray(sizes, dtype=float)


def _resolve_range(values: np.ndarray, *, value_range: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
    elif values.size:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        if math.isclose(lo, hi):
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    return lo, hi


def _resolve_compute_hours_edges(
    sizes: np.ndarray,
    *,
    bins: int | Sequence[float] | str,
    value_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    if sizes.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    xr = _resolve_range(sizes, value_range=value_range)

    if isinstance(bins, str):
        edges = np.asarray(np.histogram_bin_edges(sizes, bins=bins, range=xr), dtype=float)
    elif isinstance(bins, int):
        edges = np.linspace(xr[0], xr[1], int(bins) + 1, dtype=float)
    else:
        edges = np.asarray(bins, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("Processor bin edges must be a 1-D sequence with at least two values.")

    return edges

__all__ = [
    "wait_time_distribution",
    "plot_wait_time_distributions",
    "plot_wait_time_boxplot",
    "build_wait_size_heatmap",
    "plot_wait_size_heatmaps",
    "evaluate_models",
    "evaluate_baselines",
    "wait_time_distributions_vs_baseline",
    "plot_wait_time_distributions_for_seeds",
    "plot_wait_time_boxplots_for_seeds",
    "build_wait_size_heatmap_for_seed",
    "build_wait_size_heatmap_for_fcfs",
    "build_wait_size_heatmaps_for_seed_vs_fcfs",
    "plot_wait_size_heatmap",
    "carbon_intensity_distribution",
]


# ---------------------------------------------------------------------------
# Carbon intensity utilities


def carbon_intensity_distribution(
    action_trace: Sequence[Mapping[str, Any]],
    mode: str,
    *,
    bins: int | Sequence[float] | str = "fd",
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Build a histogram-style payload capturing carbon intensities when jobs start.

    Args:
        action_trace: Per-episode action trace where schedule actions contain job start times.
        mode: Carbon intensity split, one of {"training", "validation", "test"}.
        bins: Histogram bin specification forwarded to NumPy.
        normalize: If True, return a density histogram instead of raw counts.
        value_range: Optional clamp for histogram calculations.

    Returns:
        Dictionary with bin edges, histogram counts, CDF data, raw intensities, summary stats, and metadata.
    """
    if not isinstance(mode, str):
        raise TypeError("mode must be a string identifying the carbon intensity split.")
    mode_normalized = mode.lower()

    ci = CarbonIntensity(green_win_length=24, normalize=False)
    try:
        ci.set_mode(mode_normalized)
    except ValueError as exc:
        raise ValueError("mode must be one of {'training', 'validation', 'test'}.") from exc

    intensities: list[float] = []
    for entry in action_trace or []:
        if not hasattr(entry, "get"):
            raise TypeError("Each action trace entry must provide dict-like access via .get().")
        if entry.get("action_type") != "schedule":
            continue
        start_time = entry.get("timestamp_before")
        if start_time is None:
            continue
        try:
            start_seconds = float(start_time)
        except (TypeError, ValueError):
            continue
        intensities.append(float(ci.intensity_at(start_seconds)))

    arr = np.asarray(intensities, dtype=float)

    def _serialize_bins(spec: Any) -> Any:
        if isinstance(spec, (list, tuple, np.ndarray)):
            return [float(b) for b in np.asarray(spec, dtype=float)]
        return spec

    def _summary(values: np.ndarray) -> dict[str, float | int]:
        if values.size == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
                "p95": 0.0,
            }
        return {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }

    def _cdf(values: np.ndarray) -> tuple[list[float], list[float]]:
        if values.size == 0:
            return [], []
        sorted_vals = np.sort(values)
        n = sorted_vals.size
        y_vals = np.arange(1, n + 1, dtype=float) / float(n)
        return sorted_vals.tolist(), y_vals.tolist()

    bins_meta = _serialize_bins(bins)

    histogram_range: Optional[Tuple[float, float]]
    if value_range is not None:
        histogram_range = (float(value_range[0]), float(value_range[1]))
    elif arr.size:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if math.isclose(lo, hi):
            hi = lo + 1.0
        histogram_range = (lo, hi)
    else:
        histogram_range = None

    if arr.size == 0:
        if histogram_range is not None:
            bin_edges = np.histogram_bin_edges(
                np.linspace(histogram_range[0], histogram_range[1], 2, dtype=float),
                bins=bins,
                range=histogram_range,
            )
            if bin_edges.size <= 1:
                center = histogram_range[0]
                bin_edges = np.array([center - 0.5, center + 0.5], dtype=float)
            counts = np.zeros(bin_edges.size - 1, dtype=float)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        else:
            bin_edges = np.array([], dtype=float)
            counts = np.array([], dtype=float)
            bin_centers = np.array([], dtype=float)
        cdf_x, cdf_y = [], []
    else:
        hist_kwargs: dict[str, Any] = {"bins": bins}
        if histogram_range is not None:
            hist_kwargs["range"] = histogram_range
        bin_edges = np.histogram_bin_edges(arr, **hist_kwargs)
        if bin_edges.size <= 1:
            center = float(arr[0])
            bin_edges = np.array([center - 0.5, center + 0.5], dtype=float)
        counts, _ = np.histogram(arr, bins=bin_edges, density=normalize)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        if normalize:
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        cdf_x, cdf_y = _cdf(arr)

    payload = {
        "bin_edges": bin_edges.tolist(),
        "hist_counts": counts.tolist(),
        "bin_centers": bin_centers.tolist(),
        "cdf_x": cdf_x,
        "cdf_y": cdf_y,
        "intensities": arr.tolist(),
        "summary": _summary(arr),
        "metadata": {
            "mode": mode_normalized,
            "bins": bins_meta,
            "density": bool(normalize),
            "range": list(histogram_range) if histogram_range else None,
        },
    }
    return payload


# ---------------------------------------------------------------------------
# Wait-time distribution utilities


def wait_time_distribution(
    model_stats: Mapping[str, Mapping[str, list]],
    baseline_stats: Mapping[str, Mapping[str, list]],
    *,
    model_key: Optional[str] = None,
    baseline_key: Optional[str] = None,
    bins: int | Sequence[float] | str = "fd",
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute comparable wait-time distributions for a model and a baseline.

    Args:
        model_stats: Stats mapping for the model (typically output from Validation.validate_model).
        baseline_stats: Stats mapping for the baseline (typically from Validation.run_baselines).
        model_key: Optional key selector inside ``model_stats``. Defaults to the first entry.
        baseline_key: Optional key selector inside ``baseline_stats``. Defaults to the first entry.
        bins: Histogram bin specification passed to NumPy (integer, sequence, or string such as ``"fd"``).
        normalize: If True, use probability densities when building histograms.
        value_range: Optional (min, max) used to clamp the histogram domain.

    Returns:
        Dictionary containing shared bin edges, per-policy histogram/PDF payloads, CDF curves,
        summary statistics, and metadata describing the histogram configuration.
    """

    def _resolve_entry(
        stats_map: Mapping[str, Mapping[str, list]],
        name: str,
        preferred_key: Optional[str],
    ) -> tuple[str, Mapping[str, list]]:
        if not stats_map:
            raise ValueError(f"{name} stats are empty; run evaluation before building distributions.")
        candidate_key = preferred_key or next(iter(stats_map.keys()))
        if candidate_key not in stats_map:
            available = ", ".join(str(k) for k in stats_map.keys())
            raise KeyError(f"{name} '{candidate_key}' not found. Available keys: {available}")
        return candidate_key, stats_map[candidate_key]

    def _collect_waits(stats_entry: Mapping[str, list]) -> list[float]:
        histories = stats_entry.get("job_scheduled_history") or []
        waits: list[float] = []
        for history in histories:
            waits.extend(collect_wait_times(history))
        return waits

    def _serialize_bins(spec: Any) -> Any:
        if isinstance(spec, (list, tuple, np.ndarray)):
            return [float(b) for b in np.asarray(spec, dtype=float)]
        return spec

    bins_meta = _serialize_bins(bins)
    model_id, model_entry = _resolve_entry(model_stats, "model", model_key)
    baseline_id, baseline_entry = _resolve_entry(baseline_stats, "baseline", baseline_key)

    model_waits = _collect_waits(model_entry)
    baseline_waits = _collect_waits(baseline_entry)
    combined_waits = model_waits + baseline_waits

    if value_range is not None:
        histogram_range = (float(value_range[0]), float(value_range[1]))
    elif combined_waits:
        min_wait = float(min(combined_waits))
        max_wait = float(max(combined_waits))
        if math.isclose(min_wait, max_wait):
            max_wait = min_wait + 1.0
        histogram_range = (min_wait, max_wait)
    else:
        histogram_range = None

    if not histogram_range:
        empty_payload = {
            "wait_times": [],
            "hist_counts": [],
            "bin_centers": [],
            "cdf_x": [],
            "cdf_y": [],
            "summary": {"count": 0},
        }
        return {
            "bin_edges": [],
            model_id: empty_payload,
            baseline_id: empty_payload,
            "metadata": {
                "bins": bins_meta,
                "density": True,
                "input_normalize": bool(normalize),
                "range": None,
            },
        }

    hist_kwargs: dict[str, Any] = {"bins": bins}
    if histogram_range is not None:
        hist_kwargs["range"] = histogram_range

    bin_edges = np.histogram_bin_edges(combined_waits, **hist_kwargs)
    if len(bin_edges) <= 1:
        center = histogram_range[0]
        bin_edges = np.array([center - 0.5, center + 0.5], dtype=float)

    counts_model, _ = np.histogram(model_waits, bins=bin_edges, density=normalize)
    counts_baseline, _ = np.histogram(baseline_waits, bins=bin_edges, density=normalize)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def _summary(wait_values: Sequence[float]) -> dict[str, float | int]:
        if not wait_values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "zero_fraction": 0.0,
            }
        arr = np.asarray(wait_values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "zero_fraction": float(np.mean(arr <= 1e-6)),
        }

    def _cdf(wait_values: Sequence[float]) -> tuple[list[float], list[float]]:
        if not wait_values:
            return [], []
        arr = np.sort(np.asarray(wait_values, dtype=float))
        n = arr.size
        y_vals = np.arange(1, n + 1, dtype=float) / float(n)
        return arr.tolist(), y_vals.tolist()

    counts_model = np.asarray(counts_model, dtype=float)
    counts_baseline = np.asarray(counts_baseline, dtype=float)

    return {
        "bin_edges": bin_edges.tolist(),
        model_id: {
            "wait_times": model_waits,
            "hist_counts": counts_model.tolist(),
            "bin_centers": bin_centers.tolist(),
            "cdf_x": _cdf(model_waits)[0],
            "cdf_y": _cdf(model_waits)[1],
            "summary": _summary(model_waits),
        },
        baseline_id: {
            "wait_times": baseline_waits,
            "hist_counts": counts_baseline.tolist(),
            "bin_centers": bin_centers.tolist(),
            "cdf_x": _cdf(baseline_waits)[0],
            "cdf_y": _cdf(baseline_waits)[1],
            "summary": _summary(baseline_waits),
        },
        "metadata": {
            "bins": bins_meta,
            "density": True,
            "input_normalize": bool(normalize),
            "range": list(histogram_range) if histogram_range else None,
        },
    }


def plot_wait_time_distributions(
    series: Sequence[dict[str, Any]],
    *,
    kind: str = "pdf",
    figsize: tuple[float, float] = (10.0, 6.0),
    alpha: float = 0.45,
    linewidth: float = 2.0,
    show: bool = False,
    save_path: str | Path | None = None,
    ax_hist: Optional[Any] = None,
    ax_cdf: Optional[Any] = None,
) -> tuple[Any, Optional[Any]]:
    """
    Plot wait-time distributions with optional histogram, PDF, and CDF overlays.

    Args:
        series: Sequence of mappings with keys:
            - ``distribution``: output from :func:`wait_time_distribution`.
            - ``key``: identifier inside the distribution (e.g. model name).
            - ``label`` (optional): legend label override.
        kind: One or more of ``"hist"``, ``"pdf"``, ``"cdf"`` joined with ``+``.
        figsize: Figure size when creating new axes.
        alpha: Bar transparency for histogram overlays.
        linewidth: Line width for PDF/CDF plots.
        show: If True, call ``plt.show()`` before returning.
        save_path: Optional path for saving the figure.
        ax_hist: Optional Matplotlib axis to reuse for histograms/PDFs.
        ax_cdf: Optional Matplotlib axis to reuse for CDFs.

    Returns:
        Tuple ``(ax_hist, ax_cdf_or_None)`` for further customization.
    """
    requested_kinds = {k.strip() for k in str(kind).lower().split("+") if k.strip()}
    if not requested_kinds:
        requested_kinds = {"pdf"}
    valid_kinds = {"hist", "cdf", "pdf"}
    unknown = requested_kinds - valid_kinds
    if unknown:
        raise ValueError(f"Unsupported kind(s): {unknown}. Valid options are {valid_kinds}.")
    if not series:
        raise ValueError("series is empty; provide at least one distribution/key pair.")

    plot_hist = "hist" in requested_kinds
    plot_cdf = "cdf" in requested_kinds
    plot_pdf = "pdf" in requested_kinds

    use_house_style()

    created_fig = None
    if ax_hist is None:
        created_fig, ax_hist = plt.subplots(figsize=figsize)
    if plot_cdf and ax_cdf is None:
        ax_cdf = ax_hist.twinx()

    hist_handles: list[Any] = []
    cdf_handles: list[Any] = []
    pdf_handles: list[Any] = []

    for entry in series:
        dist = entry.get("distribution")
        key = entry.get("key")
        if dist is None or key is None:
            raise KeyError("Each series entry must include 'distribution' and 'key'.")
        label = entry.get("label") or str(key)
        metadata = dist.get("metadata", {})
        bins = dist.get("bin_edges")
        payload = dist.get(key)
        if bins is None or payload is None:
            available = ", ".join(k for k in dist.keys() if k not in {"metadata", "bin_edges"})
            raise KeyError(f"Series '{key}' not found. Available: {available or 'none'}.")

        waits = payload.get("wait_times", []) or []
        if not waits:
            continue

        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"Invalid bin edges for series '{label}'.")

        centers = np.asarray(payload.get("bin_centers") or (edges[:-1] + edges[1:]) / 2.0, dtype=float)

        if plot_hist:
            counts = payload.get("hist_counts")
            if counts:
                counts_arr = np.asarray(counts, dtype=float)
                if counts_arr.size != edges.size - 1:
                    raise ValueError(f"hist_counts length mismatch for series '{label}'.")
            else:
                counts_arr, _ = np.histogram(
                    waits,
                    bins=edges,
                    density=bool(metadata.get("density", True)),
                )

            bar = ax_hist.bar(
                edges[:-1],
                counts_arr,
                width=np.diff(edges),
                align="edge",
                alpha=alpha,
                label=label,
                edgecolor="none",
            )
            if bar:
                hist_handles.append(bar[0])

        if plot_pdf:
            counts = payload.get("hist_counts") or []
            if counts:
                counts_arr = np.asarray(counts, dtype=float)
                if counts_arr.size != centers.size:
                    raise ValueError(f"PDF counts length mismatch for series '{label}'.")
                line, = ax_hist.plot(
                    centers,
                    counts_arr,
                    linewidth=linewidth,
                    label=f"{label} PDF",
                )
                pdf_handles.append(line)

        if plot_cdf and ax_cdf is not None:
            cdf_x = payload.get("cdf_x") or []
            cdf_y = payload.get("cdf_y") or []
            if not cdf_x or not cdf_y:
                sorted_waits = np.sort(np.asarray(waits, dtype=float))
                n = sorted_waits.size
                cdf_x = sorted_waits.tolist()
                cdf_y = (np.arange(1, n + 1, dtype=float) / float(n)).tolist()
            line, = ax_cdf.plot(cdf_x, cdf_y, linewidth=linewidth, label=f"{label} CDF")
            cdf_handles.append(line)

    if hist_handles or pdf_handles:
        ax_hist.set_xlabel("Wait Time (s)")
        ax_hist.set_ylabel("Probability Density")

    if plot_cdf and ax_cdf is not None:
        ax_cdf.set_ylabel("Cumulative Probability")

    handles: list[Any] = []
    labels: list[str] = []
    for h in hist_handles:
        handles.append(h)
        labels.append(h.get_label())
    for h in pdf_handles:
        handles.append(h)
        labels.append(h.get_label())
    for h in cdf_handles:
        handles.append(h)
        labels.append(h.get_label())
    if handles:
        ax_hist.legend(handles, labels)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    finalize(ax_hist, outfile=str(save_path) if save_path else None)
    if show:
        plt.show()

    return ax_hist, ax_cdf if plot_cdf else None


def plot_wait_time_boxplot(
    series: Sequence[dict[str, Any]],
    *,
    figsize: tuple[float, float] = (8.0, 5.0),
    whis: tuple[float, float] | float = (5, 95),
    showfliers: bool = False,
    log_scale: bool = True,
    top_n_points: int = 5,
    point_size: float = 18.0,
    point_alpha: float = 0.9,
    show: bool = False,
    save_path: str | Path | None = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot a boxplot for multiple wait-time series taken from distribution payloads.
    """
    if not series:
        raise ValueError("series is empty; provide at least one distribution/key pair.")

    use_house_style()

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    data: list[list[float]] = []
    labels: list[str] = []
    top_points: list[list[float]] = []
    for entry in series:
        dist = entry.get("distribution")
        key = entry.get("key")
        if dist is None or key is None:
            raise KeyError("Each series entry must include 'distribution' and 'key'.")
        payload = dist.get(key)
        if not payload:
            continue
        waits = list(payload.get("wait_times") or [])
        if not waits:
            continue
        data.append(waits)
        labels.append(entry.get("label") or str(key))
        if top_n_points and top_n_points > 0:
            sorted_w = sorted(waits)
            top_points.append(sorted_w[-top_n_points:] if len(sorted_w) >= top_n_points else sorted_w)
        else:
            top_points.append([])

    if not data:
        raise ValueError("No wait-time samples found in provided series.")

    bp = ax.boxplot(
        data,
        labels=labels,
        whis=whis,
        showfliers=showfliers,
        patch_artist=True,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.5)

    ax.set_ylabel("Wait Time (s)")
    if log_scale:
        ax.set_yscale("log")

    if any(top_points):
        x_positions = range(1, len(data) + 1)
        for xi, pts in zip(x_positions, top_points):
            if not pts:
                continue
            xs = [xi + (random.random() - 0.5) * 0.06 for _ in pts]
            ax.scatter(xs, pts, s=point_size, c="black", alpha=point_alpha, zorder=3, label=None)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    finalize(ax, outfile=str(save_path) if save_path else None)
    if show:
        plt.show()

    return ax
import json
import math
import os
from typing import Any, Dict, List, Tuple, Union


def _load_json_or_jsonl(path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Loads either:
    - normal JSON (entire file is an object or list), OR
    - JSONL / NDJSON (one JSON object per line).

    Heuristic:
    - if extension ends with .jsonl -> parse line by line
    - else -> try normal json.load
    """
    _, ext = os.path.splitext(path)

    if ext.lower() == ".jsonl":
        runs: List[Dict[str, Any]] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                runs.append(json.loads(line))
        return runs

    # default: assume .json
    with open(path, "r") as f:
        return json.load(f)


def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary.
    Keys are joined with dots, e.g. 'Action Analysis.Total Actions'.
    Example:
        {"Action Analysis": {"Total Actions": 5}}
        -> {"Action Analysis.Total Actions": 5}
    Lists are skipped.
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            # Not aggregating lists here.
            continue
        else:
            flat[new_key] = v
    return flat


def collect_metrics(runs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Builds {metric_name: [values across runs]}.

    Rules:
    - Include `best_validation_reward` from top level.
    - Dive into run["test_eval_metrics"] recursively.
    - Rename "Validation Reward" -> "Objective".
    - Only keep numeric values.
    """
    metrics_across_seeds: Dict[str, List[float]] = {}

    for run in runs:
        # top level metric
        if "best_validation_reward" in run:
            metrics_across_seeds.setdefault("best_validation_reward", []).append(
                float(run["best_validation_reward"])
            )

        # test_eval_metrics block
        tem = run.get("test_eval_metrics", {})
        flat = flatten_dict(tem)

        # rename "Validation Reward" to "Objective"
        renamed_flat: Dict[str, Any] = {}
        for k, v in flat.items():
            if k == "Validation Reward":
                renamed_flat["Objective"] = v
            else:
                renamed_flat[k] = v

        # keep numeric values
        for k, v in renamed_flat.items():
            if isinstance(v, (int, float)):
                metrics_across_seeds.setdefault(k, []).append(float(v))

    return metrics_across_seeds


def mean_std(vals: List[float]) -> Tuple[float, float]:
    """
    Returns (mean, sample_std). If only one value, std = 0.0.
    """
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")

    mean_val = sum(vals) / n

    if n == 1:
        return mean_val, 0.0

    var = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
    std_val = math.sqrt(var)
    return mean_val, std_val


def format_number(x: float) -> str:
    """
    Pretty-print a float for console output.
    - large numbers -> scientific
    - medium numbers -> 3 decimals
    - small numbers -> strip trailing zeros
    """
    if math.isnan(x):
        return "nan"

    ax = abs(x)
    if ax >= 1e6:
        return f"{x:.3e}"
    elif ax >= 1e3:
        return f"{x:.3f}"
    else:
        s = f"{x:.6f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s


def summarize_runs(runs: List[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    """
    - Collects metrics across runs
    - Computes mean/std per metric
    - Returns sorted list of (metric_name, mean, std)
    """
    metrics = collect_metrics(runs)

    summary_rows: List[Tuple[str, float, float]] = []
    for metric_name, values in metrics.items():
        m, s = mean_std(values)
        summary_rows.append((metric_name, m, s))

    summary_rows.sort(key=lambda x: x[0].lower())
    return summary_rows


def print_results(summary_rows: List[Tuple[str, float, float]]) -> None:
    """
    Print table to stdout (not LaTeX yet).
    """
    print(f"{'Metric':60s} | {'Mean':>15s} | {'Std':>15s}")
    print("-" * 96)
    for metric_name, m, s in summary_rows:
        print(
            f"{metric_name:60s} | "
            f"{format_number(m):>15s} | "
            f"{format_number(s):>15s}"
        )


def summarize_file(path: str) -> None:
    """
    Public entry point:
    - Load .json or .jsonl
    - Normalize to list[runs]
    - Compute stats
    - Print table
    """
    data = _load_json_or_jsonl(path)

    # normalize into list of runs
    if isinstance(data, dict):
        runs = [data]
    elif isinstance(data, list):
        runs = data
    else:
        raise ValueError(
            "Parsed file must be either a dict (single run) or list[dict] (multiple runs)"
        )

    summary_rows = summarize_runs(runs)
    print_results(summary_rows)


