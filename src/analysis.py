from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.metrics import collect_wait_times
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
]


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


# ---------------------------------------------------------------------------
# Wait-time vs job-size heatmap utilities


def build_wait_size_heatmap(
    raw_stats: Mapping[str, Any],
    *,
    key: Optional[str] = None,
    size_metric: str = "procs",
    x_bins: int | Sequence[float] | str = 40,
    y_bins: int | Sequence[float] | str = "fd",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    processor_edges: Optional[Sequence[float]] = None,
    wait_edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Build a correlation heatmap payload between processor-size bins and wait-time bins.

    The computation constructs indicator variables for each processor bin and wait bin, then
    reports the Pearson correlation (phi coefficient) for every processor/wait bin pair.
    """
    waits_arr, sizes_arr = _collect_waits_and_sizes(
        raw_stats,
        size_metric=size_metric,
        key=key,
    )

    if waits_arr.size == 0 or sizes_arr.size == 0:
        return {
            "correlation_matrix": [],
            "count": 0,
            "processor_edges": [],
            "wait_edges": [],
            "processor_labels": [],
            "wait_labels": [],
            "x_label": "Wait Time Bins",
            "y_label": "Processor Size Bins" if size_metric != "compute_hours" else "Compute Hours Bins",
            "value_units": "correlation",
        }

    # Determine processor bin edges and labels.
    if size_metric == "compute_hours":
        if processor_edges is not None:
            proc_edges = np.asarray(processor_edges, dtype=float)
        else:
            proc_edges = _resolve_compute_hours_edges(
                sizes_arr,
                bins=x_bins,
                value_range=x_range,
            )
        proc_labels = [
            f"{int(round(lo))}+" if hi == proc_edges[-1] else f"{int(round(lo))}–{int(round(hi))}"
            for lo, hi in zip(proc_edges[:-1], proc_edges[1:])
        ]
        proc_values = sizes_arr
    else:
        if processor_edges is not None:
            proc_edges = np.asarray(processor_edges, dtype=float)
        else:
            proc_edges = _processor_bucket_edges(len(_PROCESSOR_BUCKET_LABELS))
        num_labels = proc_edges.size - 1
        proc_labels = list(_PROCESSOR_BUCKET_LABELS[:num_labels])
        proc_values = np.array([_processor_bucket(s) for s in sizes_arr], dtype=float)

    if proc_edges.ndim != 1 or proc_edges.size < 2:
        raise ValueError("processor_edges must be a 1-D array with at least two values.")

    # Determine wait-time bin edges and labels.
    if wait_edges is not None:
        wait_edges_arr = np.asarray(wait_edges, dtype=float)
    else:
        wait_edges_arr = _compute_wait_bin_edges(waits_arr, min_bins=6, max_bins=8)
    if wait_edges_arr.ndim != 1 or wait_edges_arr.size < 2:
        raise ValueError("wait_edges must be a 1-D array with at least two values.")

    wait_labels = _format_wait_labels(wait_edges_arr)

    counts, _, _ = np.histogram2d(
        proc_values,
        waits_arr,
        bins=(proc_edges, wait_edges_arr),
        density=False,
    )
    total = float(np.sum(counts))
    if not total:
        return {
            "correlation_matrix": [],
            "count": 0,
            "processor_edges": proc_edges.tolist(),
            "wait_edges": wait_edges_arr.tolist(),
            "processor_labels": proc_labels,
            "wait_labels": wait_labels,
            "x_label": "Wait Time Bins",
            "y_label": "Processor Size Bins" if size_metric != "compute_hours" else "Compute Hours Bins",
            "value_units": "correlation",
        }

    proc_probs = np.sum(counts, axis=1) / total
    wait_probs = np.sum(counts, axis=0) / total

    correlations = np.zeros_like(counts, dtype=float)
    for i in range(counts.shape[0]):
        p_i = proc_probs[i]
        if p_i <= 0.0 or p_i >= 1.0:
            continue
        for j in range(counts.shape[1]):
            p_j = wait_probs[j]
            if p_j <= 0.0 or p_j >= 1.0:
                continue
            p_ij = counts[i, j] / total
            expected = p_i * p_j
            denom = math.sqrt(p_i * (1.0 - p_i) * p_j * (1.0 - p_j))
            if denom > 0:
                correlations[i, j] = (p_ij - expected) / denom

    correlations = np.clip(correlations, -1.0, 1.0)

    y_axis_label = "Compute Hours Bins" if size_metric == "compute_hours" else "Processor Size Bins"

    return {
        "correlation_matrix": correlations.tolist(),
        "counts": counts.tolist(),
        "processor_edges": proc_edges.tolist(),
        "wait_edges": wait_edges_arr.tolist(),
        "processor_labels": proc_labels,
        "wait_labels": wait_labels,
        "count": int(waits_arr.size),
        "x_label": "Wait Time Bins",
        "y_label": y_axis_label,
        "value_units": "correlation",
    }


def plot_wait_size_heatmaps(
    heatmaps: Sequence[Mapping[str, Any]],
    *,
    labels: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    cmap: str = "coolwarm",
    log_color: bool = False,
    share_colorbar: bool = True,
    title: Optional[str] = None,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Any:
    """
    Plot one or more correlation heatmaps side-by-side.

    Note:
        ``log_color`` is ignored for correlation plots but retained for API compatibility.
    """
    if not heatmaps:
        raise ValueError("heatmaps is empty.")
    use_house_style()
    sns.set_theme(style="white")

    n = len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    shared_colorbar = None
    for i, (ax, hm) in enumerate(zip(axes, heatmaps)):
        corr_payload = hm.get("correlation_matrix") or hm.get("correlations")
        if corr_payload is None:
            raise ValueError("heatmap payload missing 'correlation_matrix'.")

        corr_array = np.asarray(corr_payload, dtype=float)
        if corr_array.size == 0:
            ax.set_visible(False)
            continue

        proc_labels = list(hm.get("processor_labels") or hm.get("y_tick_labels") or [])
        wait_labels = list(hm.get("wait_labels") or hm.get("x_tick_labels") or [])
        num_proc, num_wait = corr_array.shape
        if proc_labels and len(proc_labels) != num_proc:
            raise ValueError("processor_labels length does not match correlation matrix rows.")
        if wait_labels and len(wait_labels) != num_wait:
            raise ValueError("wait_labels length does not match correlation matrix columns.")
        if not proc_labels:
            proc_labels = [f"Bin {i}" for i in range(num_proc)]
        if not wait_labels:
            wait_labels = [f"Bin {j}" for j in range(num_wait)]

        show_colorbar = share_colorbar and shared_colorbar is None
        cbar_kws = {"ticks": np.linspace(-1.0, 1.0, 5), "label": "Correlation"} if show_colorbar else None
        heat = sns.heatmap(
            corr_array,
            ax=ax,
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 9},
            cbar=show_colorbar,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=wait_labels,
            yticklabels=proc_labels,
            cbar_kws=cbar_kws,
        )
        if show_colorbar:
            shared_colorbar = heat.collections[0].colorbar
            shared_colorbar.ax.tick_params(length=0)

        ax.set_xticklabels(wait_labels, rotation=45, ha="right")
        ax.set_yticklabels(proc_labels, rotation=0)
        ax.tick_params(length=0)
        ax.set_xlabel(hm.get("x_label", "Wait Time Bins"))
        ax.set_ylabel(hm.get("y_label", "Processor Bins"))
        ax.grid(False)

        if labels and i < len(labels):
            ax.annotate(
                labels[i],
                xy=(0.5, 1.02),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    if title:
        fig.suptitle(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    first_ax = next((ax for ax in axes if ax.get_visible()), axes[0])
    finalize(first_ax, outfile=str(save_path) if save_path else None)
    if show:
        plt.show()
    return axes


# ---------------------------------------------------------------------------
# Convenience helpers that operate on Validation outputs


def _validate_dir(train_eval_dir: str | Path) -> Path:
    resolved = Path(train_eval_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Directory '{resolved}' does not exist.")
    return resolved


def _init_validation(train_eval_dir: str | Path):
    from src.validation import Validation  # Local import to avoid circular dependency

    validator = Validation()
    validator.load_dir(str(_validate_dir(train_eval_dir)))
    return validator


def evaluate_models(
    train_eval_dir: str | Path,
    seeds: Sequence[int],
    *,
    mode: str = "test",
    n_eval_episodes: int = 1,
    debug: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate trained models for the provided seeds and return their stats.
    """
    directory = _validate_dir(train_eval_dir)
    results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        model_path = directory / f"seed_{seed}" / "best_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'.")

        validator = _init_validation(directory)

        from sb3_contrib.ppo_mask import MaskablePPO

        model = MaskablePPO.load(str(model_path))
        processed, raw = validator.validate_model(
            n_eval_episodes,
            model,
            mode=mode,
            debug=debug,
        )
        results[int(seed)] = {
            "processed": processed,
            "raw": raw,
            "model_path": str(model_path),
        }

    return results


def evaluate_baselines(
    train_eval_dir: str | Path,
    *,
    mode: str = "test",
    n_eval_episodes: int = 1,
    include_percentiles: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run the configured baselines and return their processed/raw statistics.
    """
    validator = _init_validation(train_eval_dir)
    processed, raw = validator.run_baselines(
        n_eval_episodes=n_eval_episodes,
        mode=mode,
        debug=debug,
        run_percentile=include_percentiles,
    )
    return {"processed": processed, "raw": raw}


def wait_time_distributions_vs_baseline(
    train_eval_dir: str | Path,
    model_stats: Mapping[int, Mapping[str, Any]],
    baseline_raw_stats: Mapping[str, Any],
    *,
    baseline_key: str = "FCFS Baseline",
    bins: int | Sequence[float] | str = "fd",
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Build wait-time distributions comparing each model seed against a baseline.
    """
    validator = _init_validation(train_eval_dir)
    distributions: Dict[int, Dict[str, Any]] = {}

    for seed, stats in model_stats.items():
        raw_stats = stats.get("raw")
        if not raw_stats:
            raise ValueError(f"No raw stats found for seed {seed}.")
        distribution = wait_time_distribution(
            model_stats=raw_stats,
            baseline_stats=baseline_raw_stats,
            model_key="model",
            baseline_key=baseline_key,
            bins=bins,
            normalize=normalize,
            value_range=value_range,
        )
        distributions[int(seed)] = distribution

    return distributions


def plot_wait_time_distributions_for_seeds(
    train_eval_dir: str | Path,
    distributions: Mapping[int, Dict[str, Any]],
    *,
    baseline_key: str = "FCFS Baseline",
    baseline_label: str | None = None,
    kind: str = "pdf",
    figsize: tuple[float, float] = (10.0, 6.0),
    alpha: float = 0.45,
    linewidth: float = 2.0,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Convenience wrapper for plotting distributions across model seeds plus a baseline.
    """
    if not distributions:
        raise ValueError("distributions is empty; nothing to plot.")

    series = []
    for seed, distribution in distributions.items():
        series.append(
            {
                "distribution": distribution,
                "key": "model",
                "label": f"Model (seed {seed})",
            }
        )

    first_distribution = next(iter(distributions.values()))
    series.append(
        {
            "distribution": first_distribution,
            "key": baseline_key,
            "label": baseline_label or baseline_key,
        }
    )

    return plot_wait_time_distributions(
        series,
        kind=kind,
        figsize=figsize,
        alpha=alpha,
        linewidth=linewidth,
        show=show,
        save_path=save_path,
    )


def plot_wait_time_boxplots_for_seeds(
    train_eval_dir: str | Path,
    distributions: Mapping[int, Dict[str, Any]],
    *,
    baseline_key: str = "FCFS Baseline",
    baseline_label: str | None = None,
    model_label: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    whis: tuple[float, float] | float = (5, 95),
    showfliers: bool = False,
    log_scale: bool = True,
    top_n_points: int = 5,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Any:
    """
    Plot a boxplot comparing each model seed against the chosen baseline.
    """
    if not distributions:
        raise ValueError("distributions is empty; nothing to plot.")

    series = []
    for seed, distribution in distributions.items():
        label = f"{model_label} (seed {seed})" if model_label else f"Model (seed {seed})"
        series.append(
            {
                "distribution": distribution,
                "key": "model",
                "label": label,
            }
        )

    first_distribution = next(iter(distributions.values()))
    series.append(
        {
            "distribution": first_distribution,
            "key": baseline_key,
            "label": baseline_label or baseline_key,
        }
    )

    return plot_wait_time_boxplot(
        series,
        figsize=figsize,
        whis=whis,
        showfliers=showfliers,
        log_scale=log_scale,
        top_n_points=top_n_points,
        show=show,
        save_path=save_path,
    )


def build_wait_size_heatmap_for_seed(
    train_eval_dir: str | Path,
    model_eval: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
    size_metric: str = "procs",
    x_bins: int | Sequence[float] | str = 40,
    y_bins: int | Sequence[float] | str = "fd",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    processor_edges: Optional[Sequence[float]] = None,
    wait_edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Build a single wait-vs-size heatmap payload for a model seed.
    """
    validator = _init_validation(train_eval_dir)
    seed_raw = model_eval.get(int(seed), {}).get("raw")
    if not seed_raw:
        raise ValueError(f"No raw stats for seed {seed}.")
    return build_wait_size_heatmap(
        seed_raw,
        key="model",
        size_metric=size_metric,
        x_bins=x_bins,
        y_bins=y_bins,
        x_range=x_range,
        y_range=y_range,
        density=density,
        processor_edges=processor_edges,
        wait_edges=wait_edges,
    )


def build_wait_size_heatmap_for_fcfs(
    train_eval_dir: str | Path,
    baseline_eval: Mapping[str, Any],
    *,
    size_metric: str = "procs",
    x_bins: int | Sequence[float] | str = 40,
    y_bins: int | Sequence[float] | str = "fd",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    processor_edges: Optional[Sequence[float]] = None,
    wait_edges: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Build a single wait-vs-size heatmap payload for the FCFS baseline.
    """
    _ = _init_validation(train_eval_dir)  # ensure config loaded for consistency
    return build_wait_size_heatmap(
        baseline_eval.get("raw", baseline_eval),
        key="FCFS Baseline",
        size_metric=size_metric,
        x_bins=x_bins,
        y_bins=y_bins,
        x_range=x_range,
        y_range=y_range,
        density=density,
        processor_edges=processor_edges,
        wait_edges=wait_edges,
    )


def build_wait_size_heatmaps_for_seed_vs_fcfs(
    train_eval_dir: str | Path,
    model_eval: Mapping[int, Mapping[str, Any]],
    baseline_eval: Mapping[str, Any],
    *,
    seed: int,
    baseline_key: str = "FCFS Baseline",
    size_metric: str = "procs",
    x_bins: int | Sequence[float] | str = 40,
    y_bins: int | Sequence[float] | str = "fd",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    processor_edges: Optional[Sequence[float]] = None,
    wait_edges: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build model and FCFS heatmaps using shared processor and wait-time bins.
    """
    _ = _init_validation(train_eval_dir)

    seed_raw = model_eval.get(int(seed), {}).get("raw")
    if not seed_raw:
        raise ValueError(f"No raw stats for seed {seed}.")

    baseline_root = baseline_eval.get("raw", baseline_eval)
    baseline_raw = baseline_root.get(baseline_key)
    if not baseline_raw:
        available = ", ".join(str(k) for k in baseline_root.keys())
        raise ValueError(f"No raw stats found for baseline '{baseline_key}'. Available: {available}")

    # Build FCFS first to establish reference bins.
    fcfs_heatmap = build_wait_size_heatmap(
        baseline_root,
        key=baseline_key,
        size_metric=size_metric,
        x_bins=x_bins,
        y_bins=y_bins,
        x_range=x_range,
        y_range=y_range,
        density=density,
        processor_edges=processor_edges,
        wait_edges=wait_edges,
    )

    proc_edges = fcfs_heatmap.get("processor_edges") or None
    wait_edges_arr = fcfs_heatmap.get("wait_edges") or None

    model_heatmap = build_wait_size_heatmap(
        seed_raw,
        key="model",
        size_metric=size_metric,
        x_bins=x_bins,
        y_bins=y_bins,
        x_range=x_range,
        y_range=y_range,
        density=density,
        processor_edges=proc_edges,
        wait_edges=wait_edges_arr,
    )

    return model_heatmap, fcfs_heatmap


def plot_wait_size_heatmap(
    heatmap_payload: Mapping[str, Any],
    *,
    label: str | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    cmap: str = "coolwarm",
    log_color: bool = False,
    show: bool = False,
    save_path: str | Path | None = None,
) -> Any:
    """
    Plot a single correlation heatmap payload.
    """
    axes = plot_wait_size_heatmaps(
        [heatmap_payload],
        labels=[label] if label else None,
        figsize=figsize,
        cmap=cmap,
        log_color=log_color,
        share_colorbar=False,
        show=show,
        save_path=save_path,
    )
    return axes[0] if isinstance(axes, (list, tuple)) else axes
