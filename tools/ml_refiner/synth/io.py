"""I/O helpers for synthetic dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_shard(
    path: Path,
    patches: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    conf: np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "patches": patches,
        "dx": dx,
        "dy": dy,
        "conf": conf,
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
