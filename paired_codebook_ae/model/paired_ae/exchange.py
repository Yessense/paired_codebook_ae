from typing import Tuple

import torch
from torch import nn


class ExchangeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                image_features: torch.tensor,
                donor_features: torch.tensor,
                exchange_labels: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        exchange_labels = exchange_labels.expand(image_features.size())

        image_with_same_donor_elements = torch.where(exchange_labels,
                                                     image_features,
                                                     donor_features)

        donor_with_same_image_elements = torch.where(exchange_labels,
                                                     donor_features,
                                                     image_features)

        return image_with_same_donor_elements, donor_with_same_image_elements
