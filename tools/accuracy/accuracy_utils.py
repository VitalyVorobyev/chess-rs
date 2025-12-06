from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import cv2

from chess_plot import plot_overlay, plot_offset_hist

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "testdata" / "out" / "parameters.npz"
DEFAULT_OUTDIR = ROOT / "testdata" / "out" / "accuracy"

DetectorFn = Callable[[Path, np.ndarray | None], Tuple[np.ndarray, Dict[str, float]]]


def image_path(camera: str, index: int) -> Path:
    camdir = "leftcamera" if camera == "l" else "rightcamera"
    suff = "L" if camera == "l" else "R"
    return ROOT / "testdata" / "imgs" / camdir / f"Im_{suff}_{index + 1}.png"


def load_ground_truth(params_path: Path = PARAMS_PATH) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(params_path)
    return data["L_Imgpoints"], data["R_Imgpoints"]


def select_gt(camera: str, idx: int, l_pts: np.ndarray, r_pts: np.ndarray) -> np.ndarray:
    points = l_pts if camera == "l" else r_pts
    if idx < 0 or idx >= len(points):
        raise IndexError(
            f"imgindex {idx} out of range for camera '{camera}' with {len(points)} frames"
        )
    return points[idx]


def _detect_with_optional_gt(
    detect_fn: DetectorFn, img_path: Path, gt: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    params = list(inspect.signature(detect_fn).parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in params) or len(params) >= 2:
        return detect_fn(img_path, gt)
    return detect_fn(img_path)  # type: ignore[arg-type]

def nearest_offsets(detected: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if detected.size == 0 or gt.size == 0:
        return np.empty((0,), dtype=float)
    diff = detected[:, None, :] - gt[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists.min(axis=1)


def summarize_offsets(offsets: np.ndarray) -> dict:
    if offsets.size == 0:
        return {}
    return {
        "mean_px": float(np.mean(offsets)),
        "median_px": float(np.median(offsets)),
        "p95_px": float(np.percentile(offsets, 95)),
        "max_px": float(np.max(offsets)),
    }

def evaluate_image(
    camera: str,
    idx: int,
    gt: np.ndarray,
    detect_fn: DetectorFn,
    detector_label: str | None = None,
) -> Tuple[dict, np.ndarray]:
    img_path = image_path(camera, idx)
    detected, metrics = _detect_with_optional_gt(detect_fn, img_path, gt)
    errors = nearest_offsets(detected, gt)
    acc = summarize_offsets(errors)

    if overlay:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        plot_overlay(img, chess_pts=)

    report = {
        "camera": camera,
        "index": idx,
        "image": str(img_path.relative_to(ROOT)),
        "ground_truth_points": int(len(gt)),
        "detected_points": int(len(detected)),
        "accuracy_px": acc,
        "trace_ms": metrics,
    }
    if detector_label:
        report["detector"] = detector_label
    return report, errors


def aggregate_errors(errors: Iterable[np.ndarray]) -> dict:
    collected: List[np.ndarray] = [e for e in errors if e.size]
    if not collected:
        return {}
    combined = np.concatenate(collected)
    return summarize_errors(combined)


def _collect_error_arrays(agg: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_errors = np.concatenate([e for e in agg["l"] if e.size]) if agg["l"] else np.array([])
    right_errors = np.concatenate([e for e in agg["r"] if e.size]) if agg["r"] else np.array([])
    overall_errors = (
        np.concatenate([left_errors, right_errors])
        if left_errors.size or right_errors.size
        else np.array([])
    )
    return overall_errors


def _plot_all_errors(agg: Dict[str, List[np.ndarray]], outdir: Path, label: str | None) -> Dict[str, str]:
    label_safe = label.replace(" ", "_").replace("/", "_") if label else ""
    suffix = f"_{label_safe}" if label_safe else ""
    paths = {
        "error_hist_left": outdir / f"error_hist_left{suffix}.png",
        "error_hist_right": outdir / f"error_hist_right{suffix}.png",
        "error_hist_overall": outdir / f"error_hist_overall{suffix}.png",
        "error_scatter_left": outdir / f"error_scatter_left{suffix}.png",
        "error_scatter_right": outdir / f"error_scatter_right{suffix}.png",
        "error_scatter_overall": outdir / f"error_scatter_overall{suffix}.png",
    }

    overall_errors = _collect_error_arrays(agg)
    plot_offset_hist(overall_errors, "Overall error histogram", paths["error_hist_overall"])

    return {k: str(v) for k, v in paths.items()}

def run_batch(
    detect_fn: DetectorFn,
    l_pts: np.ndarray,
    r_pts: np.ndarray,
    outdir: Path,
    *,
    detector_label: str | None = None,
    report_filename: str = "accuracy_report.json",
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    reports: List[dict] = []
    agg: Dict[str, List[np.ndarray]] = {"l": [], "r": []}

    for camera, pts in (("l", l_pts), ("r", r_pts)):
        for idx, gt in enumerate(pts):
            report, errors = evaluate_image(
                camera,
                idx,
                gt,
                detect_fn,
                detector_label=detector_label,
                overlay=False,
                show_overlay=False,
            )
            reports.append(report)
            agg[camera].append(errors)
            print(
                f"{camera}{idx+1:02d}: gt={report['ground_truth_points']} "
                f"det={report['detected_points']} acc={report['accuracy_px']}"
            )

    summary = {
        "left": aggregate_errors(agg["l"]),
        "right": aggregate_errors(agg["r"]),
    }
    summary["overall"] = aggregate_errors(agg["l"] + agg["r"])

    plots = _plot_all_errors(agg, outdir, detector_label)

    out = {"summary": summary, "images": reports, "plots": plots}
    if detector_label:
        out["detector"] = detector_label

    report_path = outdir / report_filename
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote report: {report_path}")
    return report_path
