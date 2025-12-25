# chess_corners (Python)

Python bindings for the `chess-corners` detector.

## Development

```bash
maturin develop -m crates/chess-corners-py/pyproject.toml
```

## Quick start

```python
import numpy as np
import chess_corners

img = np.zeros((128, 128), dtype=np.uint8)

cfg = chess_corners.ChessConfig()
cfg.threshold_rel = 0.2

corners = chess_corners.find_chess_corners(img, cfg)
print(corners.shape, corners.dtype)
```
