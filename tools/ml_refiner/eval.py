#!/usr/bin/env python3
"""Evaluate a trained ML refiner model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from dataset import ShardDataset  # noqa: E402
from model import CornerRefinerNet  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def select_device(device_cfg: str) -> torch.device:
    device_cfg = device_cfg.lower()
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_cfg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("mps requested but not available; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_cfg)


def build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    dataset = ShardDataset(
        data_dir=cfg["data_dir"],
        split=split,
        val_split=float(cfg.get("val_split", 0.1)),
        patch_size=int(cfg.get("patch_size", 0)) or None,
        max_cache=int(cfg.get("max_cache", 2)),
    )

    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 0))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    all_dx_err: list[np.ndarray] = []
    all_dy_err: list[np.ndarray] = []
    all_conf_pred: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            dx_hat = outputs[:, 0].detach().cpu().numpy()
            dy_hat = outputs[:, 1].detach().cpu().numpy()
            conf_logit = outputs[:, 2].detach().cpu().numpy()

            dx = y[:, 0].detach().cpu().numpy()
            dy = y[:, 1].detach().cpu().numpy()
            conf = y[:, 2].detach().cpu().numpy()

            conf_pred = 1.0 / (1.0 + np.exp(-conf_logit))

            all_dx_err.append(dx_hat - dx)
            all_dy_err.append(dy_hat - dy)
            all_conf_pred.append(conf_pred)
            all_conf.append(conf)

    dx_err = np.concatenate(all_dx_err, axis=0)
    dy_err = np.concatenate(all_dy_err, axis=0)
    conf_pred = np.concatenate(all_conf_pred, axis=0)
    conf = np.concatenate(all_conf, axis=0)

    radial = np.sqrt(dx_err * dx_err + dy_err * dy_err)

    metrics: Dict[str, Any] = {
        "mae_dx": float(np.mean(np.abs(dx_err))),
        "mae_dy": float(np.mean(np.abs(dy_err))),
        "p50": float(np.percentile(radial, 50)),
        "p90": float(np.percentile(radial, 90)),
        "p95": float(np.percentile(radial, 95)),
        "conf_mse": float(np.mean((conf_pred - conf) ** 2)),
    }

    if np.std(conf_pred) > 1e-6 and np.std(conf) > 1e-6:
        metrics["conf_corr"] = float(np.corrcoef(conf_pred, conf)[0, 1])
    else:
        metrics["conf_corr"] = None

    try:
        from scipy.stats import spearmanr  # type: ignore
    except ModuleNotFoundError:
        metrics["conf_spearman"] = None
    else:
        metrics["conf_spearman"] = float(spearmanr(conf_pred, conf).correlation)

    bins = 10
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_pred = []
    bin_label = []
    for i in range(bins):
        if i == bins - 1:
            mask = (conf_pred >= edges[i]) & (conf_pred <= edges[i + 1])
        else:
            mask = (conf_pred >= edges[i]) & (conf_pred < edges[i + 1])
        if np.any(mask):
            bin_pred.append(float(np.mean(conf_pred[mask])))
            bin_label.append(float(np.mean(conf[mask])))
        else:
            bin_pred.append(float("nan"))
            bin_label.append(float("nan"))
    metrics["calibration_pred"] = bin_pred
    metrics["calibration_label"] = bin_label
    metrics["calibration_edges"] = edges.tolist()

    return metrics


def save_calibration_plot(metrics: Dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping calibration plot")
        return

    pred = np.array(metrics["calibration_pred"], dtype=float)
    label = np.array(metrics["calibration_label"], dtype=float)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(pred, label, marker="o", color="tab:blue")
    ax.set_xlabel("conf_pred")
    ax.set_ylabel("conf_label")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to train config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", help="Override device in config")
    parser.add_argument("--plot", help="Optional path to save calibration plot")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.device:
        cfg["device"] = args.device

    device = select_device(str(cfg.get("device", "auto")))
    loader = build_loader(cfg, "val")

    model = CornerRefinerNet()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)

    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))

    if args.plot:
        save_calibration_plot(metrics, Path(args.plot))


if __name__ == "__main__":
    main()
