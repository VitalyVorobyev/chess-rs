"""Plotting helpers for ML/OpenCV comparison."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return None
    return plt


def plot_cdf(errors_by_method: Dict[str, np.ndarray], out_path: str) -> None:
    plt = _import_matplotlib()
    if plt is None:
        print("matplotlib not installed; skipping CDF plot")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, errors in errors_by_method.items():
        values = np.asarray(errors, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        values = np.sort(values)
        y = np.linspace(0.0, 1.0, values.size, endpoint=True)
        ax.plot(values, y, label=name)

    ax.set_xlabel("error (px)")
    ax.set_ylabel("CDF")
    ax.set_ylim(bottom=0)
    ax.set_title("Refinement error CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_binned_metric(
    bin_edges: Iterable[float],
    metrics_by_method: Dict[str, List[dict]],
    metric_key: str,
    out_path: str,
    xlabel: str,
    title: str,
) -> None:
    plt = _import_matplotlib()
    if plt is None:
        print("matplotlib not installed; skipping binned plot")
        return

    edges = np.asarray(list(bin_edges), dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, metrics in metrics_by_method.items():
        ys = []
        for entry in metrics:
            val = entry.get(metric_key)
            ys.append(np.nan if val is None else float(val))
        ax.plot(centers, ys, marker="o", label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_key)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_conf_hist(
    conf_pos: np.ndarray, conf_neg: np.ndarray, out_path: str
) -> None:
    plt = _import_matplotlib()
    if plt is None:
        print("matplotlib not installed; skipping conf histogram")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(conf_pos, bins=20, alpha=0.6, label="pos")
    ax.hist(conf_neg, bins=20, alpha=0.6, label="neg")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("conf_pred")
    ax.set_ylabel("count")
    ax.set_title("Confidence histogram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
