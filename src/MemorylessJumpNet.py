import torch
import torch.nn as nn

class MemorylessJumpNet(nn.Module):
    """
    Memory-free block-to-block model.
    Maps one input block [B, L, C] -> predicted future block [B, L, C] in a single pass.
    No recurrent state, no attention cache, no dataset running stats.
    """
    def __init__(self, C, L, d_model=256, depth=2, use_revin=True):
        super().__init__()
        self.C, self.L = C, L
        self.use_revin = use_revin

        hidden_c = max(32, d_model // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(C, hidden_c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_c, hidden_c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        in_feat = hidden_c * L
        mlp = [nn.LayerNorm(in_feat), nn.Linear(in_feat, d_model), nn.GELU()]
        for _ in range(depth - 1):
            mlp += [nn.Linear(d_model, d_model), nn.GELU()]
        mlp += [nn.Linear(d_model, L * C)]
        self.mlp = nn.Sequential(*mlp)

        if self.use_revin:
            self.gamma = nn.Parameter(torch.ones(C))
            self.beta  = nn.Parameter(torch.zeros(C))

    def revin_in(self, xLC):
        mu = xLC.mean(dim=1, keepdim=True)              # [B,1,C]
        x_ = xLC - mu
        sig = x_.pow(2).mean(dim=1, keepdim=True).sqrt()
        xN = (x_ / (sig + 1e-6)) * self.gamma + self.beta
        return xN, (mu, sig)

    def revin_out(self, yLC, stats):
        mu, sig = stats
        y = (yLC - self.beta) / (self.gamma + 1e-12)
        return y * (sig + 1e-6) + mu

    def forward(self, x):  # x: [B,L,C]
        if self.use_revin:
            x, stats = self.revin_in(x)
        else:
            stats = None

        z = x.transpose(1, 2)             # [B,C,L]
        z = self.conv(z)                  # [B,hidden_c,L]
        z = z.reshape(z.size(0), -1)      # [B, hidden_c*L]
        yhat = self.mlp(z).view(-1, self.L, self.C)

        if self.use_revin:
            yhat = self.revin_out(yhat, stats)

        return yhat
