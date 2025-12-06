#!/usr/bin/env python3
"""
Run the Rust chess-corners detector on a given image/config and compare its
corners with reference detectors implemented in OpenCV (Harris and
findChessboardCornersSB).

By default this script:
- runs `chess-corners` with the provided config (using the given image),
- overlays the ChESS corners on top of the image, and
- when enabled, overlays reference detector corners from OpenCV on either the same axes or a separate subplot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import cv2

from detection.chess import run_chess_corners, DEFAULT_BIN
from detection.chessboard import run_chessboard_corners
from detection.harris import run_harris_corners
from plotting.chess_plot import (
    plot_chess_corners,
    plot_chessboard_corners,
    plot_harris_corners,
    plot_overlay,
)

def _save_overlay_image(
    img,
    pts,
    plot_fn: Callable,
    title: str,
    path: Path,
) -> None:
    """Render a single overlay and save it to disk."""
    if pts is None:
        return
    if hasattr(pts, "__len__") and len(pts) == 0:
        return
    if hasattr(pts, "size") and getattr(pts, "size", 0) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray")
    try:
        plot_fn(ax, pts, show_labels=False)
    except TypeError:
        plot_fn(ax, pts)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    ax.margins(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {title} overlay to {path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chess-corners and optional OpenCV reference detectors on an image and plot corners."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config JSON for chess-corners (image field will be overridden).",
    )
    parser.add_argument(
        "--bin",
        type=Path,
        default=DEFAULT_BIN,
        help="Path to the chess-corners binary.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to save the figure (PNG).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Show ChESS and reference detector corners in side-by-side plots instead of overlay.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure, only save it if --out is given.",
    )
    parser.add_argument(
        "--harris",
        action="store_true",
        help="Include Harris detector",
    )
    parser.add_argument(
        "--chessboard",
        action="store_true",
        help="Include OpenCV findChessboardCornersSB detector",
    )
    parser.add_argument(
        "--pattern-size",
        type=int,
        nargs=2,
        metavar=("COLS", "ROWS"),
        default=(11, 7),
        help="Inner corner pattern size (cols rows) for findChessboardCornersSB.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save per-detector overlays as separate images with automatic filenames.",
    )
    parser.add_argument(
        "--overlays-dir",
        type=Path,
        help="Optional directory for per-detector overlays (defaults to --out directory if set, else the input image directory).",
    )
    args = parser.parse_args()

    img_path = args.image
    overlay_dir: Path | None = None
    if args.save_overlays:
        overlay_dir = args.out.parent if args.out else img_path.parent
        if overlay_dir is not None:
            overlay_dir = overlay_dir / "overlays"

    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Run Rust detector
    chess_pts = run_chess_corners(img_path, args.config, args.bin)

    harris_pts = None
    if args.harris:
        harris_pts, trace = run_harris_corners(img)
        print("Harris detector trace:")
        for k, v in trace.items():
            print(f"  {k}: {v:.2f} ms")

    chessboard_pts = None
    if args.chessboard:
        pattern_size = tuple(args.pattern_size)
        chessboard_pts, cb_trace = run_chessboard_corners(img, pattern_size)
        print("findChessboardCornersSB trace:")
        for k, v in cb_trace.items():
            print(f"  {k}: {v:.2f} ms")

    if overlay_dir:
        base = img_path.stem
        _save_overlay_image(
            img,
            chess_pts,
            plot_chess_corners,
            "ChESS corners",
            overlay_dir / f"{base}_chess.png",
        )
        if harris_pts is not None:
            _save_overlay_image(
                img,
                harris_pts,
                plot_harris_corners,
                "Harris corners",
                overlay_dir / f"{base}_harris.png",
            )
        if chessboard_pts is not None:
            _save_overlay_image(
                img,
                chessboard_pts,
                plot_chessboard_corners,
                "findChessboardCornersSB",
                overlay_dir / f"{base}_chessboard.png",
            )

    fig = plot_overlay(
        img,
        chess_pts,
        harris_pts=harris_pts,
        chessboard_pts=chessboard_pts,
        split=args.split,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    import os
    main()
