"""Compact U-Net baseline / fallback.

Lets the whole CrackFM pipeline train and evaluate end-to-end on a free GPU
without downloading any foundation-model weights. Outputs single-channel logits
at input resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        chans = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        cin = in_channels
        for c in chans[:-1]:
            self.downs.append(_DoubleConv(cin, c))
            self.pools.append(nn.MaxPool2d(2))
            cin = c
        self.bottleneck = _DoubleConv(chans[-2], chans[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for c in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.up_convs.append(_DoubleConv(c * 2, c))
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                              align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))
        return self.head(x)
