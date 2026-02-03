from __future__ import annotations
import torch
import torch.nn as nn
from .moving_avg import MovingAvg


class SeriesDecomp(nn.Module):
    """
    Series decomposition:
      x = seasonal + trend
    where trend is a moving average and seasonal is residual.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor [B, L, C]
        Returns:
            seasonal: [B, L, C]
            trend:    [B, L, C]
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend
