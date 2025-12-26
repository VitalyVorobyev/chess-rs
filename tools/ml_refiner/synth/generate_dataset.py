#!/usr/bin/env python3
"""Generate synthetic subpixel corner refinement datasets."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from synth import augment, io as synth_io, render_corner  # noqa: E402


REQUIRED_KEYS = (
    "seed",
    "patch_size",
    "num_samples",
    "shard_size",
    "dx_range",
    "dy_range",
    "rotation",
    "noise_sigma",
    "blur_sigma",
    "contrast",
    "brightness",
    "gamma",
    "conf_params",
)


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")
    return cfg


def _as_range(cfg: Dict[str, Any], key: str) -> Tuple[float, float]:
    value = cfg[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a 2-element list")
    return float(value[0]), float(value[1])


def _get_git_commit(repo_root: Path) -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return commit.decode("utf-8").strip()


def _compute_extent(cfg: Dict[str, Any]) -> float:
    patch_size = int(cfg["patch_size"])
    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")
    max_offset = max(
        abs(dx_range[0]),
        abs(dx_range[1]),
        abs(dy_range[0]),
        abs(dy_range[1]),
    )
    margin = float(cfg.get("render_margin", 2.0))
    return (patch_size - 1) / 2.0 + max_offset + margin


def _compute_confidence(
    blur_sigma: np.ndarray,
    noise_sigma: np.ndarray,
    conf_params: Dict[str, Any],
) -> np.ndarray:
    a = float(conf_params.get("a", 0.0))
    b = float(conf_params.get("b", 0.0))
    conf = np.exp(-a * (blur_sigma**2) - b * (noise_sigma**2))
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _sample_params(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
) -> Dict[str, np.ndarray]:
    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")
    rot_range = _as_range(cfg, "rotation")
    noise_range = _as_range(cfg, "noise_sigma")
    blur_range = _as_range(cfg, "blur_sigma")
    scale_range = _as_range(cfg, "scale") if "scale" in cfg else (1.0, 1.0)

    return {
        "dx": rng.uniform(*dx_range, size=count).astype(np.float32),
        "dy": rng.uniform(*dy_range, size=count).astype(np.float32),
        "theta": rng.uniform(*rot_range, size=count).astype(np.float32),
        "scale": rng.uniform(*scale_range, size=count).astype(np.float32),
        "noise_sigma": rng.uniform(*noise_range, size=count).astype(np.float32),
        "blur_sigma": rng.uniform(*blur_range, size=count).astype(np.float32),
    }


def generate_samples(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    count: int,
    render_x: np.ndarray,
    render_y: np.ndarray,
    patch_x: np.ndarray,
    patch_y: np.ndarray,
    extent: float,
    super_res: int,
) -> Dict[str, np.ndarray]:
    patch_size = int(cfg["patch_size"])
    edge_softness = float(cfg.get("edge_softness", 0.15))

    contrast_range = _as_range(cfg, "contrast")
    brightness_range = _as_range(cfg, "brightness")
    gamma_range = _as_range(cfg, "gamma")

    params = _sample_params(cfg, rng, count)
    conf = _compute_confidence(
        params["blur_sigma"], params["noise_sigma"], cfg["conf_params"]
    )

    patches = np.empty((count, patch_size, patch_size), dtype=np.uint8)

    for idx in range(count):
        ideal = render_corner.render_ideal_corner_from_grid(
            render_x,
            render_y,
            float(params["theta"][idx]),
            float(params["scale"][idx]),
            edge_softness,
        )
        patch = render_corner.sample_patch_from_image(
            ideal,
            extent,
            super_res,
            patch_x,
            patch_y,
            float(params["dx"][idx]),
            float(params["dy"][idx]),
        )
        patch = augment.apply_blur(patch, float(params["blur_sigma"][idx]))
        patch = augment.apply_photometric(
            patch,
            rng,
            contrast_range,
            brightness_range,
            gamma_range,
        )
        patch = augment.apply_noise(patch, rng, float(params["noise_sigma"][idx]))
        patches[idx] = np.clip(patch * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    return {
        "patches": patches,
        "dx": params["dx"],
        "dy": params["dy"],
        "conf": conf,
        "noise_sigma": params["noise_sigma"],
        "blur_sigma": params["blur_sigma"],
    }


def run_self_test(cfg: Dict[str, Any]) -> None:
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required: pip install numpy")
    cfg = dict(cfg)
    cfg["seed"] = 0
    count = 128
    rng = np.random.default_rng(cfg["seed"])

    patch_size = int(cfg["patch_size"])
    extent = _compute_extent(cfg)
    super_res = int(cfg.get("super_res", 4))
    render_x, render_y = render_corner.make_render_grid(super_res, extent)
    patch_x, patch_y = render_corner.make_patch_grid(patch_size)

    data = generate_samples(
        cfg, rng, count, render_x, render_y, patch_x, patch_y, extent, super_res
    )

    patches = data["patches"]
    dx = data["dx"]
    dy = data["dy"]
    conf = data["conf"]

    dx_range = _as_range(cfg, "dx_range")
    dy_range = _as_range(cfg, "dy_range")

    assert patches.dtype == np.uint8
    assert patches.shape == (count, patch_size, patch_size)
    assert np.all(patches >= 0) and np.all(patches <= 255)
    assert np.all(dx >= dx_range[0]) and np.all(dx <= dx_range[1])
    assert np.all(dy >= dy_range[0]) and np.all(dy <= dy_range[1])
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)

    print("self-test OK")


def save_preview(
    out_dir: Path, patches: np.ndarray, dx: np.ndarray, dy: np.ndarray
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping preview")
        return

    count = patches.shape[0]
    grid = int(math.ceil(math.sqrt(count)))
    patch_size = patches.shape[1]
    center = (patch_size - 1) / 2.0

    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2.0, grid * 2.0))
    axes = np.array(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= count:
            continue
        patch = patches[idx]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.arrow(
            center,
            center,
            float(dx[idx]),
            float(dy[idx]),
            color="red",
            width=0.3,
            head_width=2.0,
            length_includes_head=True,
        )

    out_path = out_dir / "preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", help="Output directory for dataset shards")
    parser.add_argument("--preview", type=int, default=0, help="Save preview grid")
    parser.add_argument("--self-test", action="store_true", help="Run a quick self-test")
    args = parser.parse_args()

    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required: pip install numpy")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.self_test:
        run_self_test(cfg)
        return

    if not args.out:
        parser.error("--out is required unless --self-test is set")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(cfg["patch_size"])
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    extent = _compute_extent(cfg)
    super_res = int(cfg.get("super_res", 4))
    if patch_size <= 0 or super_res <= 0:
        raise ValueError("patch_size and super_res must be positive")

    render_x, render_y = render_corner.make_render_grid(super_res, extent)
    patch_x, patch_y = render_corner.make_patch_grid(patch_size)

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    num_samples = int(cfg["num_samples"])
    shard_size = int(cfg["shard_size"])
    if num_samples <= 0 or shard_size <= 0:
        raise ValueError("num_samples and shard_size must be positive")
    num_shards = int(math.ceil(num_samples / shard_size))

    preview_count = max(0, int(args.preview))
    preview_data = None
    if preview_count > 0:
        preview_rng = np.random.default_rng(seed + 1)
        preview_data = generate_samples(
            cfg,
            preview_rng,
            preview_count,
            render_x,
            render_y,
            patch_x,
            patch_y,
            extent,
            super_res,
        )

    meta = {
        "generator": "ml_refiner_synth_v1",
        "seed": seed,
        "patch_dtype": "uint8",
        "config": cfg,
        "git_commit": _get_git_commit(ML_ROOT.parents[1]),
    }
    synth_io.write_json(out_dir / "meta.json", meta)

    written = 0
    for shard_idx in range(num_shards):
        count = min(shard_size, num_samples - written)
        data = generate_samples(
            cfg,
            rng,
            count,
            render_x,
            render_y,
            patch_x,
            patch_y,
            extent,
            super_res,
        )

        shard_path = out_dir / f"shard_{shard_idx:05d}.npz"
        synth_io.save_shard(
            shard_path,
            data["patches"],
            data["dx"],
            data["dy"],
            data["conf"],
            extra={
                "noise_sigma": data["noise_sigma"],
                "blur_sigma": data["blur_sigma"],
            },
        )
        written += count
        print(f"wrote {shard_path} ({count} samples)")

    if preview_data is not None:
        save_preview(out_dir, preview_data["patches"], preview_data["dx"], preview_data["dy"])


if __name__ == "__main__":
    main()
