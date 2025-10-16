# scripts/plot_scheduler_viz.py
import os
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# -----------------------------
# Helpers
# -----------------------------
def _fd_bins(x: np.ndarray, min_bins: int = 10) -> int | str:
    """Freedman–Diaconis bin rule with a sensible fallback."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return "auto"
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return "auto"
    bin_w = 2 * iqr * (x.size ** (-1.0 / 3.0))
    if bin_w <= 0:
        return "auto"
    bins = int(np.ceil((x.max() - x.min()) / bin_w))
    return max(min_bins, bins)


def _flatten_action_traces(container: Any) -> List[Dict[str, Any]]:
    """
    Robustly pull out the list of action dicts from various shapes:
    - ( {'model': {...}}, {'model': {'action_traces': [[...]]}} )
    - {'model': {'action_traces': [[...]]}}
    - {'action_traces': [[...]]}
    - or directly [[...]] / [...]
    """
    # try several access paths
    cand = container
    if isinstance(cand, tuple) or isinstance(cand, list):
        # look for dicts that have 'model' or 'action_traces'
        for item in cand:
            if isinstance(item, dict) and ("model" in item or "action_traces" in item):
                cand = item
                break

    if isinstance(cand, dict) and "model" in cand:
        cand = cand["model"]

    if isinstance(cand, dict) and "action_traces" in cand:
        traces = cand["action_traces"]
    else:
        traces = cand

    # traces can be [[...]] or [...]
    if isinstance(traces, list) and len(traces) > 0 and isinstance(traces[0], list):
        # multiple episodes -> flatten
        actions = [a for episode in traces for a in episode]
    elif isinstance(traces, list):
        actions = traces
    else:
        raise ValueError("Could not locate action_traces in the provided data structure.")

    # keep only dict actions
    actions = [a for a in actions if isinstance(a, dict)]
    return actions


def _extract_schedule_rows(actions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for a in actions:
        if a.get("action_type") == "schedule":
            wt = a.get("scheduled_job_wait_time", None)
            nodes = a.get("scheduled_job_nodes", None)
            rt = a.get("scheduled_job_run_time", None)
            ts = a.get("timestamp_before", None)
            jid = a.get("scheduled_job_id", None)
            if wt is None or nodes is None:
                continue
            rows.append(
                {
                    "job_id": jid,
                    "timestamp": ts,
                    "wait_time_s": float(wt),
                    "wait_time_h": float(wt) / 3600.0,
                    "nodes": int(nodes),
                    "run_time_s": float(rt) if rt is not None else np.nan,
                    "run_time_h": (float(rt) / 3600.0) if rt is not None else np.nan,
                }
            )
    if not rows:
        raise ValueError("No schedule actions with wait_time and nodes found.")
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Filter out any negative or NaN wait times just in case
    df = df[(df["wait_time_s"] >= 0) & np.isfinite(df["wait_time_s"]) & np.isfinite(df["nodes"])]
    return df


# -----------------------------
# Plotters
# -----------------------------
def plot_wait_hist(df: pd.DataFrame, outdir: Path):
    xh = df["wait_time_h"].values
    bins = _fd_bins(xh)
    plt.figure(figsize=(7, 4.5))
    plt.hist(xh, bins=bins, density=True)
    plt.xlabel("Wait time (hours)")
    plt.ylabel("Density")
    plt.title("Distribution of Wait Time (Histogram)")
    plt.grid(True, alpha=0.3, linestyle=":")
    out = outdir / "wait_time_hist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_wait_ecdf(df: pd.DataFrame, outdir: Path):
    x = np.sort(df["wait_time_h"].values)
    y = np.arange(1, x.size + 1) / x.size
    plt.figure(figsize=(7, 4.5))
    plt.step(x, y, where="post")
    plt.xlabel("Wait time (hours)")
    plt.ylabel("ECDF")
    plt.title("Wait Time ECDF")
    plt.grid(True, alpha=0.3, linestyle=":")
    out = outdir / "wait_time_ecdf.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_wait_nodes_heatmap(df: pd.DataFrame, outdir: Path, log_counts: bool = True):
    x = df["wait_time_h"].values
    y = df["nodes"].values

    x_bins = _fd_bins(x, min_bins=20)
    # Integer bin edges for nodes (center bins on integers)
    y_min, y_max = int(np.nanmin(y)), int(np.nanmax(y))
    y_edges = np.arange(y_min - 0.5, y_max + 1.5, 1)

    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_edges])

    plt.figure(figsize=(8, 5))
    if log_counts:
        # Add +1 to avoid log(0) if needed
        plt.pcolormesh(xedges, yedges, (H.T + 1e-9), norm=LogNorm())
    else:
        plt.pcolormesh(xedges, yedges, H.T)

    cbar = plt.colorbar()
    cbar.set_label("Count")
    plt.xlabel("Wait time (hours)")
    plt.ylabel("Requested nodes")
    plt.title("Density Heatmap: Wait Time vs Requested Nodes")
    plt.tight_layout()
    out = outdir / "wait_vs_nodes_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()


def plot_cov_corr_heatmap(df: pd.DataFrame, outdir: Path):
    X = df[["wait_time_h", "nodes"]].to_numpy()
    cov = np.cov(X.T)
    corr = np.corrcoef(X.T)

    # Covariance heatmap
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cov, interpolation="nearest")
    plt.xticks([0, 1], ["wait_h", "nodes"])
    plt.yticks([0, 1], ["wait_h", "nodes"])
    plt.title("Covariance Matrix")
    plt.colorbar()
    # Annotate values
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "covariance_heatmap.png", dpi=160)
    plt.close()

    # Correlation heatmap (often easier to read)
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(corr, vmin=-1, vmax=1, interpolation="nearest")
    plt.xticks([0, 1], ["wait_h", "nodes"])
    plt.yticks([0, 1], ["wait_h", "nodes"])
    plt.title("Correlation Matrix")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_summary(df: pd.DataFrame, outdir: Path):
    summary = {
        "n_jobs": int(df.shape[0]),
        "wait_time_hours": {
            "mean": float(df["wait_time_h"].mean()),
            "median": float(df["wait_time_h"].median()),
            "p95": float(df["wait_time_h"].quantile(0.95)),
            "max": float(df["wait_time_h"].max()),
        },
        "nodes": {
            "mean": float(df["nodes"].mean()),
            "median": float(df["nodes"].median()),
            "max": int(df["nodes"].max()),
        },
        "pearson_r(wait_h,nodes)": float(np.corrcoef(df["wait_time_h"], df["nodes"])[0, 1]),
    }
    with open(outdir / "wait_nodes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# -----------------------------
# Public APIs
# -----------------------------
def plot_from_result(result_like: Any, outdir: str | Path = "figs") -> Path:
    """
    Use this when you have the in-memory Python object (your pasted structure).
    Example:
        from scripts.plot_scheduler_viz import plot_from_result
        plot_from_result(result)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    actions = _flatten_action_traces(result_like)
    df = _extract_schedule_rows(actions)

    # Save extracted data
    df.to_csv(outdir / "scheduled_jobs_wait_nodes.csv", index=False)

    # Plots
    plot_wait_hist(df, outdir)
    plot_wait_ecdf(df, outdir)
    plot_wait_nodes_heatmap(df, outdir, log_counts=True)
    plot_cov_corr_heatmap(df, outdir)
    save_summary(df, outdir)

    return outdir


def plot_from_action_traces_json(json_path: str | Path, outdir: str | Path = "figs") -> Path:
    """
    Use this if you export JUST the action_traces as JSON (list or list of lists of action dicts).
    """
    with open(json_path, "r") as f:
        action_traces = json.load(f)
    return plot_from_result({"action_traces": action_traces}, outdir=outdir)

