from dataclasses import dataclass
from typing import List
import torch
from torch import nn
from torch.utils.data import Dataset

from ..dataset._dataset_info import DatasetInfo, DatasetWithInfo
from . import vsa


@dataclass
class Feature:
    name: str
    n_values: int
    contiguous: bool = False
    density: float = 1.


class Codebook(nn.Module):
    features: List[Feature]
    latent_dim: int
    codebook: List[torch.tensor]

    @staticmethod
    def make_features_from_dataset(dataset_info: DatasetInfo) -> List[Feature]:
        features: List[Feature] = []
        for feature_name, n_values, contiguous in zip(dataset_info.feature_names,
                                                      dataset_info.feature_counts,
                                                      dataset_info.is_contiguous):
            features.append(Feature(name=feature_name,
                                    n_values=n_values,
                                    contiguous=contiguous))
        return features

    def __init__(self, features: List[Feature], latent_dim: int, seed: int = 0,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        self.features = features
        self.n_features = len(features)
        # Add placeholders Feature class to automatic creation later
        placeholders = Feature(name='Placeholders', n_values=len(self.features))
        self.features.insert(0, placeholders)

        self.latent_dim = latent_dim
        codebook = []

        for feature in features:
            feature_vectors = torch.zeros((feature.n_values, latent_dim),
                                          dtype=torch.float32)

            if feature.contiguous:
                base_vector = vsa.generate(self.latent_dim)
                base_vector = vsa.make_unitary(base_vector)

                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.pow(base_vector,
                                                 1 + i * feature.density)
            else:
                for i in range(feature.n_values):
                    feature_vectors[i] = vsa.generate(self.latent_dim)
            codebook.append(feature_vectors)

        self.placeholders = nn.Parameter(codebook[0])
        self.vsa_features = nn.ParameterList(
            [nn.Parameter(feature_vectors) for feature_vectors in codebook[1:]])


if __name__ == '__main__':
    features = [Feature('shape', 3), Feature('scale', 6, contiguous=True)]
    latent_dim = 1024
    codebook = Codebook(features, latent_dim)

    pass
