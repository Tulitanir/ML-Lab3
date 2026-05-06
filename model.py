import torch
import torch.nn as nn


class HopfieldDenoiser(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        latent_dim: int = 64,
        num_steps: int = 4,
        beta: float = 1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.n = img_size * img_size
        self.num_steps = num_steps
        self.beta = beta

        scale = (2.0 / (self.n + latent_dim)) ** 0.5
        self.A = nn.Parameter(torch.randn(self.n, latent_dim) * scale)
        self.bias = nn.Parameter(torch.zeros(self.n))

    def _step(self, x: torch.Tensor) -> torch.Tensor:

        h = x @ self.A
        Wx = h @ self.A.t()
        return torch.tanh(self.beta * (Wx + self.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = 2.0 * x.view(b, self.n) - 1.0

        for _ in range(self.num_steps):
            x_flat = self._step(x_flat)

        return ((x_flat + 1.0) / 2.0).view(b, c, h, w)
