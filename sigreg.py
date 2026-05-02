"""SIGReg from LeJEPA (arXiv:2511.08544); input shape (V, N, D)."""

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        proj = proj.contiguous()
        A = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype)
        norm = A.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12)
        A = A / norm
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        w = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + (x_t.sin().mean(-3)).square()
        statistic = err.contiguous() @ w
        statistic = statistic * proj.size(-2)
        return statistic.mean()
