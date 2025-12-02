# Part II: Using the Detector

> Placeholder for future content. This part will walk through the high-level APIs, from simple single-scale use with `image` to multiscale detection and CLI workflows.

Planned sections:

- **2.1 Single-scale detection with `image`**
  - Using `ChessConfig::single_scale`.
  - Calling `find_chess_corners_image`.
  - Interpreting `CornerDescriptor`.
- **2.2 Raw buffer API**
  - `find_chess_corners_u8` and `ImageView`.
  - Handling strides / custom buffers.
- **2.3 CLI workflows**
  - `DetectionConfig` JSON.
  - Running the `chess-corners` binary.
  - Inspecting JSON and PNG outputs.

