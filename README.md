# chess-rs

A dedicated, high-performance Rust implementation of the **ChESS** (Chess-board Extraction by Subtraction and Summation) corner response.

ChESS is a *classical*, ID-free detector for chessboard **X-junction corners**. It produces a dense response map and (later in this repo) subpixel corner candidates.  
This crate focuses on the low-level primitive: **robust chessboard vertex detection**. Board topology recovery / grid ordering lives in higher-level crates.

## Status

🚧 Work in progress (Phase A+B).
Implemented:
- Workspace skeleton
- Canonical 16-sample ring (r=5, r=10)
- Scalar response kernel following the paper:
  - Sum Response (SR)
  - Diff Response (DR)
  - Mean Response (MR)
  - Final score `R = SR − DR − 16·MR`

Planned next:
- Golden tests vs reference maps
- NMS + candidate extraction
- Subpixel refinement
- SIMD + rayon acceleration
- Multi-scale pyramid
- Optional orientation labeling

## Crates

- `chess-core`  
  Minimal core implementation (response computation). Intended to stay lean, SIMD/parallel-friendly, and potentially `no_std` in the future.

- `chess`  
  Ergonomic facade with `image` crate interop.

Workspace layout:

```

chess-rs/
crates/
chess-core/
chess/
testdata/
benches/

```

## Algorithm (short)

For each pixel where the ring fits:
1. Sample 16 intensities on a circular ring (default radius 5 px).
2. Compute:
   - `SR`: strong for chessboard crossings
   - `DR`: strong for single edges (suppressed)
   - `MR`: rejects stripe-like false positives
3. Final response:
```

R = SR − DR − 16 * MR

````
4. Higher response means more likely an X-corner.

## Quick start (will work once Phase C lands)

```rust
use chess::{ChessParams, chess_response_image};
use image::io::Reader as ImageReader;

let img = ImageReader::open("board.png")?.decode()?.to_luma8();
let params = ChessParams::default();
let resp = chess_response_image(&img, &params);

// Later: find_corners_image(&img, &params)
````

## License

Dual-licensed under MIT or Apache-2.0.

## References

* Bennett, Lasenby, *ChESS: A Fast and Accurate Chessboard Corner Detector*, CVIU 2014.
