import torch
from torch import nn
from ..codebook import vsa


class Binder(nn.Module):
    n_features: int
    latent_dim: int
    hd_placeholders: nn.Parameter

    def __init__(self):
        super().__init__()


class FourierBinder(Binder):
    def __init__(self, placeholders: torch.tensor):
        super().__init__()
        self.hd_placeholders = nn.Parameter(data=placeholders.unsqueeze(0))

    def forward(self, z):
        out = vsa.bind(self.hd_placeholders.data, z)
        return out
