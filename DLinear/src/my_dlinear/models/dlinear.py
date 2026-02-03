from __future__ import annotations
import torch
import torch.nn as nn
from ..layers.series_decomp import SeriesDecomp


class DLinear(nn.Module):
    """
    DLinear :
      - Decompose x into seasonal + trend via moving average
      - Forecast each via Linear(seq_len -> pred_len)
      - Sum them to get output

    Input:  [B, seq_len, enc_in]
    Output: [B, pred_len, enc_in]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_size: int = 25,
        individual: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.individual = bool(individual)

        self.decomp = SeriesDecomp(kernel_size=int(kernel_size))

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, enc_in]
        Returns:
            y: [B, pred_len, enc_in]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, L, C], got {tuple(x.shape)}")

        B, L, C = x.shape
        if L != self.seq_len or C != self.enc_in:
            raise ValueError(
                f"Expected x shape [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}"
            )

        seasonal, trend = self.decomp(x)

        if self.individual:
            seasonal_out = torch.zeros((B, self.pred_len, C), device=x.device, dtype=x.dtype)
            trend_out = torch.zeros((B, self.pred_len, C), device=x.device, dtype=x.dtype)

            # per-channel linear
            for i in range(C):
                seasonal_out[:, :, i] = self.linear_seasonal[i](seasonal[:, :, i])
                trend_out[:, :, i] = self.linear_trend[i](trend[:, :, i])
        else:
            # shared linear: need [B,C,L] -> Linear(L->H) -> [B,C,H] -> [B,H,C]
            seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        return seasonal_out + trend_out
