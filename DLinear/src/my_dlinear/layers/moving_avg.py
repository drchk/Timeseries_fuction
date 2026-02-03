from __future__ import annotations
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """
    Moving average over the time dimension (L) for inputs shaped [B, L, C].
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        # AvgPool1d expects [B, C, L]
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, L, C]
        Returns:
            trend: Tensor [B, L, C]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B,L,C], got {tuple(x.shape)}")

        # replicate padding
        pad = (self.kernel_size - 1) // 2
        if pad <= 0:
            return x

        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)  # [B, L+2pad, C]

        # pool over time
        x_pad = x_pad.permute(0, 2, 1)            # [B, C, L+2pad]
        trend = self.avg(x_pad)                   # [B, C, L]
        trend = trend.permute(0, 2, 1)            # [B, L, C]
        return trend
