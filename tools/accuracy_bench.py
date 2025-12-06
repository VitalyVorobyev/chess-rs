import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from detection.harris import run_harris_corners
from detection.chessboard import run_chessboard_corners
from trace.parser import parse_trace
from trace.runner import run_once, write_config

ROOT = Path(__file__).resolve().parents[1]
IMG_ROOT = ROOT / "testdata" / "imgs"
BASE_CFG = ROOT / "config" / "config_single.json"
BIN_PATH = ROOT / "target" / "release" / "chess-corners"
DEFAULT_OUTDIR = ROOT / "testdata" / "out" / "feature_refs"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_chess_detector(img_path: Path, base_cfg: dict) -> Tuple[np.ndarray, Dict[str, float]]:
    cfg = dict(base_cfg)
    cfg["image"] = str(img_path)
    cfg["output_json"] = None
    cfg["output_png"] = None

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cfg_path = tmpdir / "config.json"
        write_config(cfg, cfg_path)
        stdout = run_once(ROOT, BIN_PATH, cfg_path, tmpdir)

        out_path = tmpdir / "out.json"
        with out_path.open("r", encoding="utf-8") as f:
            result = json.load(f)

    corners_raw = result.get("corners", [])
    corners = (
        np.array([[c["x"], c["y"]] for c in corners_raw], dtype=float)
        if corners_raw
        else np.empty((0, 2), dtype=float)
    )
    metrics = parse_trace(stdout)
    return corners, metrics


def nearest_deltas(
    chess_pts: np.ndarray,
    ref_pts: np.ndarray,
    *,
    ref_label: str,
    warn_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if chess_pts.size == 0 or ref_pts.size == 0:
        return (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )
    diff = ref_pts[None, :, :] - chess_pts[:, None, :]
    dists = np.linalg.norm(diff, axis=2)
    nn_idx = np.argmin(dists, axis=1)
    deltas = ref_pts[nn_idx] - chess_pts
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    far_mask = dist > warn_threshold
    if np.any(far_mask):
        for idx in np.where(far_mask)[0]:
            print(
                f"Warning: nearest {ref_label} for ChESS corner {idx} is {dist[idx]:.2f}px away "
                f"(dx={dx[idx]:.2f}, dy={dy[idx]:.2f})"
            )

    return dx, dy, dist


def plot_hist(
    data: np.ndarray,
    title: str,
    xlabel: str,
    path: Path,
    *,
    non_negative: bool = False,
) -> None:
    if data.size == 0:
        return
    mean = float(np.mean(data))
    std = float(np.std(data))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=40, color="#1d3557", edgecolor="black", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    if not non_negative:
        ax.axvline(mean, color="#e63946", linestyle="--", linewidth=1.5, label=f"mean={mean:.3f}")
        ax.axvline(mean + std, color="#ffb703", linestyle=":", linewidth=1.2, label=f"σ={std:.3f}")
        ax.axvline(mean - std, color="#ffb703", linestyle=":", linewidth=1.2)
        ax.set_xlim(left=-1, right=1)
    else:
        ax.axvline(mean, color="#e63946", linestyle="--", linewidth=1.5, label=f"mean={mean:.3f}")
        ax.set_xlim(left=0, right=1)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)

def plot_compare_hist(
    data_a: np.ndarray,
    label_a: str,
    data_b: np.ndarray,
    label_b: str,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    if data_a.size == 0 or data_b.size == 0:
        return
    bins = 40
    low = float(min(data_a.min(), data_b.min()))
    high = float(max(data_a.max(), data_b.max()))
    edges = np.linspace(low, high, bins + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data_a, bins=edges, color="#1d3557", alpha=0.65, label=label_a, edgecolor="black")
    ax.hist(data_b, bins=edges, color="#2a9d8f", alpha=0.55, label=label_b, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)

def summarize(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: no matches")
        return
    stats = {
        "count": arr.size,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }
    print(f"{name}: {stats}")


def collect_images(root: Path) -> list[Path]:
    imgs: list[Path] = []
    for i in range(1, 21):
        for cam, prefix in (("leftcamera", "Im_L_"), ("rightcamera", "Im_R_")):
            path = root / cam / f"{prefix}{i}.png"
            if path.exists():
                imgs.append(path)
            else:
                print(f"Skipping missing image: {path}")
    return imgs


def main():
    if not BIN_PATH.exists():
        raise SystemExit(
            f"Missing binary at {BIN_PATH}; run `cargo build -p chess-corners --release` first."
        )

    parser = argparse.ArgumentParser(description="Compare ChESS corners against Harris and findChessboardCornersSB.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    base_cfg = load_config(BASE_CFG)

    images = collect_images(IMG_ROOT)
    if not images:
        raise SystemExit(f"No images found under {IMG_ROOT}")

    dx_harris: list[np.ndarray] = []
    dy_harris: list[np.ndarray] = []
    dist_harris: list[np.ndarray] = []

    dx_cb: list[np.ndarray] = []
    dy_cb: list[np.ndarray] = []
    dist_cb: list[np.ndarray] = []

    dx_harris_cb: list[np.ndarray] = []
    dy_harris_cb: list[np.ndarray] = []
    dist_harris_cb: list[np.ndarray] = []

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        chess_pts, _ = run_chess_detector(img_path, base_cfg)
        harris_pts, _ = run_harris_corners(img)
        chessboard_pts, _ = run_chessboard_corners(img)

        dx, dy, dist = nearest_deltas(chess_pts, harris_pts, ref_label=f"Harris in {img_path.name}")
        dx_harris.append(dx)
        dy_harris.append(dy)
        dist_harris.append(dist)

        dx, dy, dist = nearest_deltas(chess_pts, chessboard_pts, ref_label=f"findChessboardCornersSB in {img_path.name}")
        dx_cb.append(dx)
        dy_cb.append(dy)
        dist_cb.append(dist)

        dx, dy, dist = nearest_deltas(chessboard_pts, harris_pts, ref_label=f"Harris vs findChessboardCornersSB in {img_path.name}")
        dx_harris_cb.append(dx)
        dy_harris_cb.append(dy)
        dist_harris_cb.append(dist)

    dx_harris_all = np.concatenate([d for d in dx_harris if d.size]) if dx_harris else np.array([])
    dy_harris_all = np.concatenate([d for d in dy_harris if d.size]) if dy_harris else np.array([])
    dist_harris_all = np.concatenate([d for d in dist_harris if d.size]) if dist_harris else np.array([])

    dx_cb_all = np.concatenate([d for d in dx_cb if d.size]) if dx_cb else np.array([])
    dy_cb_all = np.concatenate([d for d in dy_cb if d.size]) if dy_cb else np.array([])
    dist_cb_all = np.concatenate([d for d in dist_cb if d.size]) if dist_cb else np.array([])

    dx_harris_cb_all = np.concatenate([d for d in dx_harris_cb if d.size]) if dx_harris_cb else np.array([])
    dy_harris_cb_all = np.concatenate([d for d in dy_harris_cb if d.size]) if dy_harris_cb else np.array([])
    dist_harris_cb_all = np.concatenate([d for d in dist_harris_cb if d.size]) if dist_harris_cb else np.array([])

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_hist(dx_harris_all, "Harris Δx vs ChESS", "Δx (px)", args.outdir / "harris_dx.png")
    plot_hist(dy_harris_all, "Harris Δy vs ChESS", "Δy (px)", args.outdir / "harris_dy.png")
    plot_hist(dist_harris_all, "Harris distance vs ChESS", "Distance (px)", args.outdir / "harris_dist.png", non_negative=True)

    plot_hist(dx_cb_all, "Chessboard Δx vs ChESS", "Δx (px)", args.outdir / "chessboard_dx.png")
    plot_hist(dy_cb_all, "Chessboard Δy vs ChESS", "Δy (px)", args.outdir / "chessboard_dy.png")
    plot_hist(dist_cb_all, "Chessboard distance vs ChESS", "Distance (px)", args.outdir / "chessboard_dist.png", non_negative=True)

    plot_hist(dx_harris_cb_all, "Chessboard Δx vs Harris", "Δx (px)", args.outdir / "harris_vs_chessboard_dx.png")
    plot_hist(dy_harris_cb_all, "Chessboard Δy vs Harris", "Δy (px)", args.outdir / "harris_vs_chessboard_dy.png")
    plot_hist(dist_harris_cb_all, "Chessboard distance vs Harris", "Distance (px)", args.outdir / "harris_vs_chessboard_dist.png", non_negative=True)

    plot_compare_hist(
        dist_harris_all,
        "Harris vs ChESS",
        dist_cb_all,
        "Chessboard vs ChESS",
        "Distance vs ChESS: Harris vs Chessboard",
        "Distance (px)",
        args.outdir / "compare_distance.png",
    )
    plot_compare_hist(
        dx_harris_all,
        "Harris vs ChESS",
        dx_cb_all,
        "Chessboard vs ChESS",
        "Δx vs ChESS: Harris vs Chessboard",
        "Δx (px)",
        args.outdir / "compare_dx.png",
    )
    plot_compare_hist(
        dy_harris_all,
        "Harris vs ChESS",
        dy_cb_all,
        "Chessboard vs ChESS",
        "Δy vs ChESS: Harris vs Chessboard",
        "Δy (px)",
        args.outdir / "compare_dy.png",
    )
    print("\n=== Harris vs ChESS ===")
    summarize("Δx", dx_harris_all)
    summarize("Δy", dy_harris_all)
    summarize("Distance", dist_harris_all)

    print("\n=== findChessboardCornersSB vs ChESS ===")
    summarize("Δx", dx_cb_all)
    summarize("Δy", dy_cb_all)
    summarize("Distance", dist_cb_all)

    print("\n=== findChessboardCornersSB vs Harris ===")
    summarize("Δx", dx_harris_cb_all)
    summarize("Δy", dy_harris_cb_all)
    summarize("Distance", dist_harris_cb_all)
    plt.show()


if __name__ == "__main__":
    main()
