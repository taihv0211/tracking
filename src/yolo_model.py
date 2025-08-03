"""Simple YOLO-style model built with PyTorch only.

This module defines a minimal YOLOv8-like architecture without relying on the
Ultralytics package.  The model is intentionally small so that you can easily
modify or extend its layers.  It outputs raw detection tensors that can be
further processed into bounding boxes.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


class Conv(nn.Module):
    """Convolution followed by BatchNorm and SiLU activation."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    """A light-weight residual block used in YOLOv8."""

    def __init__(self, c1: int, c2: int, n: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = c1
        for i in range(n):
            layers += [Conv(ch, c2, 3), Conv(c2, c2, 3)]
            ch = c2
        self.blocks = nn.Sequential(*layers)
        self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SPPF(nn.Module):
    """Spatial pyramid pooling layer."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c2 * 4, c2, 1, 1, 0)
        self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Detect(nn.Module):
    """Final detection layer producing raw outputs."""

    def __init__(self, c1: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, num_classes + 4 + 1, 1)
        self.out_channels = num_classes + 5
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch, num_classes + 5, h, w)
        return self.conv(x)


class CustomYOLOv8(nn.Module):
    """Tiny YOLOv8-style network for easy modification."""

    def __init__(self, num_classes: int = 80) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList(
            [
                Conv(3, 32, 3, 2),  # 0
                Conv(32, 64, 3, 2),  # 1
                C2f(64, 64, 2),  # 2
                Conv(64, 128, 3, 2),  # 3
                C2f(128, 128, 2),  # 4
                Conv(128, 256, 3, 2),  # 5
                C2f(256, 256, 2),  # 6
                SPPF(256, 256, 5),  # 7
            ]
        )
        self.detect = Detect(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.detect(x)
