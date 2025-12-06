""" OpenCV chessboard detector using findChessboardCornersSB. """

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

PatternSize = Tuple[int, int]


def run_chessboard_corners(
    gray: np.ndarray,
    pattern_size: PatternSize = (11, 7),
    flags: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Detect chessboard corners using OpenCV's findChessboardCornersSB.

    Args:
        gray: Grayscale image (H x W) or BGR image.
        pattern_size: (columns, rows) of inner chessboard corners.
        flags: Optional OpenCV flags to pass to the detector.
    Returns:
        (corners array shaped [N, 2], timing metrics in ms)
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    detector_flags = flags if flags is not None else cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    detect_start = time.perf_counter()
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=detector_flags)
    detect_ms = (time.perf_counter() - detect_start) * 1000.0

    pts = np.empty((0, 2), dtype=float)
    if bool(found) and corners is not None:
        pts = corners.reshape(-1, 2).astype(float)

    return pts, {"detect_ms": detect_ms}
