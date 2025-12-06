""" Run ChESS detector """

import json
import os
from pathlib import Path
import subprocess
import tempfile

from .chesscorner import ChESSCorner, load_corners

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "target" / "release" / "chess-corners"

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def run_chess_corners(
    image_path: Path,
    cfg_path: Path,
    bin_path: Path | None = None,
) -> list[ChESSCorner]:
    bin_path = bin_path or DEFAULT_BIN
    if not bin_path.exists():
        raise SystemExit(
            f"Missing chess-corners binary at {bin_path}. "
            "Build it with `cargo build -p chess-corners --release`."
        )

    base_cfg = load_config(cfg_path)
    cfg = dict(base_cfg)
    cfg["image"] = str(image_path)
    cfg["output_json"] = None
    cfg["output_png"] = None

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        tmp_cfg = tmpdir / "config.json"
        with tmp_cfg.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        out_json = tmpdir / "out.json"

        cmd = [
            str(bin_path),
            "run",
            str(tmp_cfg),
            "--output-json",
            str(out_json),
        ]
        env = dict(**os.environ)
        env.setdefault("RUST_LOG", "info")
        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
        )

        if not out_json.exists():
            raise SystemExit(f"chess-corners did not produce {out_json}")

        return load_corners(out_json)
