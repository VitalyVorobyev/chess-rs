""" OpenCV Harris corner detector tool. """

from pathlib import Path

import time
import numpy as np
import cv2

def merge_close_points(pts: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """
    Merge points that fall within `radius` of each other into a single averaged point.

    Prefers a fast cKDTree-based merge when SciPy is available; otherwise uses a
    grid hashing approach so runtime scales roughly linearly with point count.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= 1 or radius <= 0:
        return pts

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        cKDTree = None

    if cKDTree:
        tree = cKDTree(pts)
        neighbors = tree.query_ball_tree(tree, r=radius)

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i, neigh in enumerate(neighbors):
            for j in neigh:
                if j <= i:
                    continue
                union(i, j)

        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        merged = [pts[idxs].mean(axis=0) for idxs in clusters.values()]
        return np.vstack(merged)

    # Fallback: bucket points into a grid and union overlaps locally.
    cell_size = radius
    cell_coords = np.floor(pts / cell_size).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cell_coords):
        key = (int(cell[0]), int(cell[1]))
        buckets.setdefault(key, []).append(idx)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    neighbor_offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for (cx, cy), indices in buckets.items():
        for dx, dy in neighbor_offsets:
            nkey = (cx + dx, cy + dy)
            if nkey not in buckets:
                continue
            for i in indices:
                for j in buckets[nkey]:
                    if j <= i:
                        continue
                    if np.linalg.norm(pts[i] - pts[j]) <= radius:
                        union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    merged = [pts[idxs].mean(axis=0) for idxs in clusters.values()]
    return np.vstack(merged)

def run_harris_corners(gray: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    detect_start = time.perf_counter()
    # Harris corner response tuned for chessboard-like corners.
    harris = cv2.cornerHarris(
        np.float32(gray),
        blockSize=3,
        ksize=3,
        k=0.04,
    )
    harris = cv2.dilate(harris, None)
    thresh = 0.01 * harris.max()
    ys, xs = np.where(harris > thresh)
    pts = np.stack([xs, ys], axis=1).astype(np.float32) if xs.size else np.empty((0, 2), dtype=float)
    detect_ms = (time.perf_counter() - detect_start) * 1000.0

    # Refine to sub-pixel accuracy.
    refine_start = time.perf_counter()
    if pts.size:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        refined = cv2.cornerSubPix(
            gray,
            pts.astype(np.float32).reshape(-1, 1, 2),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        pts = refined.reshape(-1, 2)
    refine_ms = (time.perf_counter() - refine_start) * 1000.0

    merge_start = time.perf_counter()
    pts = merge_close_points(pts, radius=3.0)
    merge_ms = (time.perf_counter() - merge_start) * 1000.0

    return pts, {
        'detect_ms': detect_ms,
        'refine_ms': refine_ms,
        'merge_ms': merge_ms,
    }
