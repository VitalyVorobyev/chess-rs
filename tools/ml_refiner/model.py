"""Baseline CNN for subpixel corner refinement."""

from __future__ import annotations

import torch
from torch import nn


class CornerRefinerNet(nn.Module):
    def __init__(self, base_channels: int = 16, head_dim: int = 64) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.backbone = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)
