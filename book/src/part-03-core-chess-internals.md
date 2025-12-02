# Part III: Core ChESS Internals

> Placeholder for future content. This part will cover the ChESS response math, the detector pipeline, and how the core crate is structured.

Planned sections:

- **3.1 Rings and sampling geometry**
  - `RingOffsets`, r=5 and r=10 rings.
  - Coordinate conventions.
- **3.2 Dense response computation**
  - `chess_response_u8` and `ResponseMap`.
  - ROI support via `Roi`.
  - Border handling.
- **3.3 Detection pipeline**
  - Thresholding, NMS, cluster filtering.
  - 5×5 subpixel refinement.
- **3.4 Corner descriptors**
  - `Corner` vs `CornerDescriptor`.
  - Orientation, phase, anisotropy estimation.

