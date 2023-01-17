import operator
from functools import reduce
from typing import Tuple

import torch
from torch import nn

from ..utils import product


class Decoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),

                 latent_dim: int = 1024,
                 hidden_channels: int = 64,
                 in_channels: int = 64):
        super(Decoder, self).__init__()
        # Dataset parameters
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.out_channels = self.image_size[0]

        # Layer parameters
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.reshape = (self.in_channels, 4, 4)
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)

        # Latent layers
        self.latent_layers = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU(),
            nn.Linear(self.latent_dim, product(self.reshape)), nn.GELU()
        )

        if image_size == (1, 64, 64):
            self.cnn_layers = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, self.hidden_channels,
                                   **cnn_kwargs),
                nn.BatchNorm2d(self.hidden_channels),
                nn.GELU(),

                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels,
                                   **cnn_kwargs),
                nn.BatchNorm2d(self.hidden_channels),
                nn.GELU(),

                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels,
                                   **cnn_kwargs),
                nn.BatchNorm2d(self.hidden_channels),
                nn.GELU(),

                nn.ConvTranspose2d(self.hidden_channels, self.out_channels,
                                   **cnn_kwargs)
            )
        elif image_size == (3, 128, 128):
            self.cnn_layers = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, self.hidden_channels, **cnn_kwargs),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs),
                nn.GELU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ConvTranspose2d(self.hidden_channels, self.out_channels, **cnn_kwargs)
            )
        else:
            raise NotImplemented(f"Wrong Image size {self.image_size}")

        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        # Linear layers
        x = self.latent_layers(x)
        x = x.view(-1, *self.reshape)
        # Conv layers
        x = self.final_activation(self.cnn_layers(x))
        return x


if __name__ == '__main__':
    # Clevr
    image_size = (3, 128, 128)
    n_features = 6
    latent_dim = 1024
    decoder = Decoder(image_size=image_size, latent_dim=latent_dim)

    x = torch.randn(2, 1024)
    out = decoder(x)

    print("Done")

    # Dsprites
    image_size = (1, 64, 64)
    n_features = 5
    latent_dim = 1024
    decoder = Decoder(image_size=image_size, latent_dim=latent_dim)

    x = torch.randn(2, 1024)
    out = decoder(x)

    print("Done")
