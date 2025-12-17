# Tools

This directory contains helper scripts used to benchmark and visualize the
`chess-corners` CLI and compare it to reference detectors. They are **not**
required to use the Rust crates, but they are useful for:

- generating synthetic chessboard datasets with ground-truth corners,
- running accuracy benchmarks (GT matching + plots),
- visualizing detector outputs on individual images, and
- profiling performance across feature combinations (`rayon`, `simd`, etc.).

Most scripts assume:

- Python 3.10+ (recommended: Python 3.12 for best `opencv-python` wheel availability),
- NumPy,
- Matplotlib,
- OpenCV (`opencv-python`).

A minimal setup with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python matplotlib
```

Some scripts optionally use SciPy (for faster point merging in Harris):

```bash
pip install scipy
```

Scripts expect the `chess-corners` CLI binary to be built first:

```bash
cargo build -p chess-corners --release
```

---

## Synthetic dataset generation

`synthimages.py` generates a folder containing:

- `dataset.json` — metadata + per-image ground truth (`gt_corners_uv`)
- `images/000000.png`, `images/000001.png`, … — rendered grayscale images

Example:

```bash
python3 tools/synthimages.py \
  --out testdata/synthetic_sigma_3 \
  --num 200 --seed 1 \
  --inner 9 6 --square_size 0.03 --pps 90 \
  --img_size 1280 960 --fx 1100 --fy 1100 \
  --yaw 35 --pitch 30 --roll 15 \
  --z_range 0.7 1.6 \
  --noise_sigma 3.0 --blur_sigma 0.8 \
  --gamma_range 0.9 1.2 --contrast_range 0.9 1.1 --brightness_range -10 10 \
  --vignetting 0.15
```

This writes `testdata/synthetic_sigma_3/dataset.json` and an `images/` subfolder.

---

## Synthetic accuracy benchmark (GT matching)

`accuracy_bench.py` runs three corner detectors on the synthetic dataset and
computes match statistics against ground truth:

- `chess-corners` (Rust CLI)
- OpenCV Harris (+ `cornerSubPix`)
- OpenCV `findChessboardCornersSB`

Run it like this:

```bash
python3 tools/accuracy_bench.py \
  --dataset testdata/synthetic_sigma_3/dataset.json \
  --config config/config_single.json \
  --bin target/release/chess-corners \
  --outdir testdata/out/synthetic_accuracy_sigma_3 \
  --match-threshold-px 2.5 \
  --worst-k 8
```

Outputs:

- `report.json` — summary + per-image metrics
- `plots/` — error hist/CDF, residual scatter, precision/recall/F1, counts
- `overlays/` — per-detector “worst case” image grids (FP/FN visualization)

Tip: for a quick smoke test, add `--max-images 10`.

---

## Single-image visualization / debugging

`detect.py` runs the Rust detector (and optionally OpenCV reference detectors)
on one image and produces an overlay:

```bash
python3 tools/detect.py testdata/synthetic_sigma_3/images/000000.png \
  --config config/config_single.json \
  --harris --chessboard --pattern-size 9 6 \
  --out /tmp/overlay.png
```

`plot_detectors.py` is a thin wrapper around `detect.py` for compatibility.

---

## Performance benchmarking and tracing

- `perf_bench.py` – builds and runs the CLI across feature combinations
  (`simd`, `rayon`, `par_pyramid`), repeats runs, and writes a JSON report under
  `testdata/out/`.
- `trace/` – helpers used by `perf_bench.py` (run the CLI with `--json-trace`,
  parse spans, aggregate metrics).
- `analyze_trace.py` – analyze a raw line-delimited tracing log (from
  `chess-corners ... --json-trace`) and generate a simple perf plot.

---

## Misc scripts

- `crop.py` – small helper for ad-hoc image cropping during experiments.
- `detection/` and `plotting/` – internal Python modules used by the scripts
  above (OpenCV detectors + plotting helpers).

For details, check each script’s docstring and CLI `--help`.
