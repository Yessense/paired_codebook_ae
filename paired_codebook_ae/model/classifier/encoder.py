import operator
from functools import reduce
from typing import Tuple

import torch
from torch import nn

from ...utils import product


class Encoder(nn.Module):
    def __init__(self,
                 image_size: Tuple[int, int, int] = (3, 128, 128),
                 latent_dim: int = 1024,
                 hidden_channels: int = 64):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # NN parameters
        self.hidden_channels = hidden_channels
        self.in_channels = self.image_size[0]

        self.out_channels = self.latent_dim

        self.activation = torch.nn.ReLU()
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)

        if image_size == (3, 128, 128):
            # Convolutional layers
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
            )
            self.reshape = (self.hidden_channels, 4, 4)
        elif image_size == (1, 64, 64):
            # Convolutional layers
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.BatchNorm2d(self.hidden_channels),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.BatchNorm2d(self.hidden_channels),
            )
            self.reshape = (self.hidden_channels, 4, 4)
        else:
            raise ValueError("Wrong image size")

        self.final_layers = nn.Sequential(
            nn.Linear(product(self.reshape), self.out_channels), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.out_channels, self.out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # x -> (batch_size, in_channels, width, height)
        x = self.cnn_layers(x)
        x = x.reshape((batch_size, -1))
        # x -> (batch_size, self.reshape)

        x = self.final_layers(x)

        return x
