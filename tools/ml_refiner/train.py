#!/usr/bin/env python3
"""Train a baseline CNN for subpixel corner refinement."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
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


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


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


def make_run_dir(base_dir: Path, name: str | None) -> Path:
    if name:
        run_dir = base_dir / name
    else:
        run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    suffix = 1
    while True:
        candidate = Path(f"{run_dir}_{suffix:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    dx_hat = outputs[:, 0]
    dy_hat = outputs[:, 1]
    conf_logit = outputs[:, 2]

    dx = targets[:, 0]
    dy = targets[:, 1]
    conf = targets[:, 2]

    reg_weight_bias = float(loss_cfg.get("reg_weight_bias", 0.2))
    reg_weight_scale = float(loss_cfg.get("reg_weight_scale", 0.8))
    lambda_conf = float(loss_cfg.get("lambda_conf", 0.2))

    reg_err = nn.functional.smooth_l1_loss(dx_hat, dx, reduction="none")
    reg_err = reg_err + nn.functional.smooth_l1_loss(dy_hat, dy, reduction="none")
    weights = reg_weight_bias + reg_weight_scale * conf.detach()
    reg_loss = torch.mean(weights * reg_err)

    conf_loss = nn.functional.binary_cross_entropy_with_logits(conf_logit, conf)

    total = reg_loss + lambda_conf * conf_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_reg": float(reg_loss.detach().cpu()),
        "loss_conf": float(conf_loss.detach().cpu()),
    }


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    sched_cfg: Dict[str, Any],
) -> float:
    kind = str(sched_cfg.get("kind", "none")).lower()
    warmup = int(sched_cfg.get("warmup_epochs", 0))

    if warmup > 0 and epoch < warmup:
        lr = base_lr * float(epoch + 1) / float(warmup)
    elif kind == "cosine":
        denom = max(1, total_epochs - warmup)
        progress = float(epoch - warmup) / float(denom)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif kind == "step":
        step_size = int(sched_cfg.get("step_size", max(1, total_epochs // 3)))
        gamma = float(sched_cfg.get("gamma", 0.1))
        steps = max(0, (epoch - warmup) // step_size)
        lr = base_lr * (gamma**steps)
    else:
        lr = base_lr

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def collect_metrics(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, np.ndarray]:
    dx_hat = outputs[:, 0].detach().cpu().numpy()
    dy_hat = outputs[:, 1].detach().cpu().numpy()
    conf_logit = outputs[:, 2].detach().cpu().numpy()

    dx = targets[:, 0].detach().cpu().numpy()
    dy = targets[:, 1].detach().cpu().numpy()
    conf = targets[:, 2].detach().cpu().numpy()

    conf_pred = 1.0 / (1.0 + np.exp(-conf_logit))

    return {
        "dx_err": dx_hat - dx,
        "dy_err": dy_hat - dy,
        "conf_pred": conf_pred,
        "conf": conf,
    }


def summarize_metrics(records: Dict[str, np.ndarray]) -> Dict[str, Any]:
    dx_err = records["dx_err"]
    dy_err = records["dy_err"]
    radial = np.sqrt(dx_err * dx_err + dy_err * dy_err)
    conf_pred = records["conf_pred"]
    conf = records["conf"]

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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    all_dx_err: list[np.ndarray] = []
    all_dy_err: list[np.ndarray] = []
    all_conf_pred: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        loss, loss_stats = compute_loss(outputs, y, loss_cfg)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss_stats["loss"] * batch_size
        total_count += batch_size

        with torch.no_grad():
            batch_metrics = collect_metrics(outputs, y)
        all_dx_err.append(batch_metrics["dx_err"])
        all_dy_err.append(batch_metrics["dy_err"])
        all_conf_pred.append(batch_metrics["conf_pred"])
        all_conf.append(batch_metrics["conf"])

    records = {
        "dx_err": np.concatenate(all_dx_err, axis=0),
        "dy_err": np.concatenate(all_dy_err, axis=0),
        "conf_pred": np.concatenate(all_conf_pred, axis=0),
        "conf": np.concatenate(all_conf, axis=0),
    }
    summary = summarize_metrics(records)
    summary["loss"] = total_loss / max(1, total_count)
    return summary


def build_loader(
    cfg: Dict[str, Any],
    split: str,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = ShardDataset(
        data_dir=cfg["data_dir"],
        split=split,
        val_split=float(cfg.get("val_split", 0.1)),
        patch_size=int(cfg.get("patch_size", 0)) or None,
        max_cache=int(cfg.get("max_cache", 2)),
    )

    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 0))

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    use_pin = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to train config")
    parser.add_argument("--data-dir", help="Override data_dir in config")
    parser.add_argument("--device", help="Override device in config")
    parser.add_argument("--epochs", type=int, help="Override epochs in config")
    parser.add_argument("--batch-size", type=int, help="Override batch_size in config")
    parser.add_argument("--run-name", help="Run directory name override")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.device:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = select_device(str(cfg.get("device", "auto")))

    model = CornerRefinerNet()
    model.to(device)

    train_loader = build_loader(cfg, "train", seed, device)
    val_loader = build_loader(cfg, "val", seed + 1, device)

    if len(train_loader.dataset) == 0:
        raise RuntimeError("training dataset is empty")

    epochs = int(cfg.get("epochs", 1))
    lr = float(cfg.get("lr", 1.0e-3))
    weight_decay = float(cfg.get("weight_decay", 1.0e-4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_cfg = cfg.get("save", {}) if isinstance(cfg.get("save"), dict) else {}
    run_root = Path(run_cfg.get("run_dir", "tools/ml_refiner/runs"))
    run_dir = make_run_dir(run_root, args.run_name)

    save_config(cfg, run_dir / "config.yaml")

    metrics_path = run_dir / "metrics.jsonl"
    best_p95 = None

    for epoch in range(epochs):
        lr_epoch = adjust_lr(
            optimizer,
            epoch,
            epochs,
            lr,
            cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler"), dict) else {},
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {},
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            cfg.get("loss", {}) if isinstance(cfg.get("loss"), dict) else {},
        )

        log = {
            "epoch": epoch,
            "lr": lr_epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

        print(
            f"epoch {epoch + 1:03d}/{epochs} "
            f"lr={lr_epoch:.3g} "
            f"train_loss={train_metrics['loss']:.4g} "
            f"val_p95={val_metrics['p95']:.4g} "
            f"val_conf_mse={val_metrics['conf_mse']:.4g}"
        )

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "metrics": val_metrics,
        }

        if run_cfg.get("save_last", True):
            torch.save(state, run_dir / "model_last.pt")

        if run_cfg.get("save_best", True):
            p95 = float(val_metrics["p95"])
            if best_p95 is None or p95 < best_p95:
                best_p95 = p95
                torch.save(state, run_dir / "model_best.pt")

    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
