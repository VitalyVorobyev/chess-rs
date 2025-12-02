# Part IV: Multiscale and Pyramids

> Placeholder for future content. This part will explain the image pyramid implementation and the coarse-to-fine detector built on top.

Planned sections:

- **4.1 Image pyramids**
  - `ImageView`, `ImageBuffer`.
  - `PyramidParams`, `PyramidBuffers`, `build_pyramid`.
  - Downsampling and feature combinations (`rayon`, `simd`).
- **4.2 Coarse-to-fine detection**
  - `CoarseToFineParams` design.
  - `find_chess_corners_buff` and `find_chess_corners`.
  - ROI selection, border logic, merging duplicates.
- **4.3 Choosing multiscale configs**
  - Recommended presets.
  - Trade-offs between speed, robustness, and corner density.

