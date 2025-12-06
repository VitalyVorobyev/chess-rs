import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from accuracy.accuracy_utils import (
    DEFAULT_OUTDIR,
    PARAMS_PATH,
    ROOT,
    evaluate_image,
    load_ground_truth,
    run_batch,
    select_gt,
)
from trace.parser import parse_trace
from trace.runner import run_once, write_config

BASE_CFG = ROOT / "config" / "config_single.json"
BIN_PATH = ROOT / "target" / "release" / "chess-corners"

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_chess_detector(base_cfg: dict):
    def detect(img_path: Path):
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

    return detect


def main():
    if not BIN_PATH.exists():
        raise SystemExit(
            f"Missing binary at {BIN_PATH}; run `cargo build -p chess-corners --release` first."
        )

    parser = argparse.ArgumentParser(description="ChESS accuracy benchmark")
    parser.add_argument(
        "camera",
        choices=("l", "r"),
        nargs="?",
        default="l",
        help="Camera side: l or r (ignored in --batch mode).",
    )
    parser.add_argument(
        "-i", "--imgindex", type=int, default=0, help="Image index to load"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all left/right images, save plots and JSON report.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for batch plots/report.",
    )
    args = parser.parse_args()

    l_pts, r_pts = load_ground_truth(PARAMS_PATH)
    detector = make_chess_detector(load_config(BASE_CFG))

    if args.batch:
        run_batch(
            detector,
            l_pts,
            r_pts,
            args.outdir,
            detector_label="chess-corners",
            report_filename="accuracy_report.json",
        )
        return

    gt = select_gt(args.camera, args.imgindex, l_pts, r_pts)
    report, errors = evaluate_image(
        args.camera,
        args.imgindex,
        gt,
        detector,
        detector_label="chess-corners",
        overlay=True,
        show_overlay=True,
    )

    print(f"Image: {report['image']}")
    print(
        f"Ground truth points: {report['ground_truth_points']} | "
        f"Detected corners: {report['detected_points']}"
    )
    if report["trace_ms"]:
        print("Trace (ms):", report["trace_ms"])
    if errors.size:
        print("Accuracy (px):", report["accuracy_px"])
    else:
        print("Accuracy (px): no matches to compare.")


if __name__ == "__main__":
    main()
