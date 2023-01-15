from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch import nn


class AttentionModule(pl.LightningModule):
    def __init__(self,
                 vsa_features: List[torch.Tensor],
                 n_features: int = 5,
                 latent_dim: int = 1024,
                 scale: Optional[float] = None,
                 ):
        super().__init__()
        self.vsa_features = vsa_features
        self.n_features = n_features
        self.latent_dim = latent_dim

        if scale is None:
            self.scale = 1 / (latent_dim ** 0.5)
        else:
            self.scale = scale

        self.softmax = nn.Softmax(dim=1)
        self.q_proj = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in
                                     range(n_features)])
        self.k_proj = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in
                                     range(n_features)])

    def forward(self, x):
        query: List[torch.tensor] = [self.q_proj[i](x)
                                     for i in range(self.n_features)]
        key: List[torch.tensor] = [self.k_proj[i](feature.to(self.device))
                                   for i, feature in enumerate(self.vsa_features)]

        k: torch.tensor
        attn_logits = [torch.matmul(query[i], key[i].transpose(-2, -1))
                       for i in range(self.n_features)]
        attn_logits = [logit * self.scale for logit in attn_logits]
        attention = [self.softmax(logit) for logit in attn_logits]
        max_values = [torch.mean(torch.max(attn, dim=1).values)
                      for attn in attention]

        values = [torch.matmul(attention[i], self.vsa_features[i].to(self.device))
                  for i in range(self.n_features)]
        values = torch.stack(values, dim=1)

        return values, max_values
