# chess-corners-core

Core primitives for computing ChESS responses and extracting subpixel chessboard corners.

This crate implements:

- 16-sample ChESS rings (`ring` module) at radii 5 and 10.
- Dense response computation on 8-bit grayscale images (`response` module).
- Thresholding, non-maximum suppression, and pluggable subpixel refinement (`detect` + `refine` modules).
- Conversion from raw peaks to rich corner descriptors (`descriptor` module).

Feature flags:

- `std` *(default)* – use the Rust standard library; disabling this yields `no_std` + `alloc`.
- `rayon` – parallelize response computation over image rows.
- `simd` – enable portable-SIMD acceleration of the response kernel (nightly only).
- `tracing` – emit structured spans around response and detector code for profiling.

Basic usage:

```rust
use chess_corners_core::{detect::find_corners_u8, ChessParams, RefinerKind};

fn detect(img: &[u8], w: usize, h: usize) {
    let mut params = ChessParams::default();
    // Default = center-of-mass refinement on the response map.
    let corners = find_corners_u8(img, w, h, &params);
    println!("found {} corners", corners.len());

    // Opt into Förstner or saddle-point refiners on the image intensities:
    params.refiner = RefinerKind::Forstner(Default::default());
    let refined = find_corners_u8(img, w, h, &params);
    println!("found {} corners with Förstner", refined.len());
}
```

For a higher-level, image-friendly API (including multiscale detection and an optional CLI),
see the `chess-corners` crate in this workspace.
