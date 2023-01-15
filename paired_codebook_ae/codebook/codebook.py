from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import Dataset

from . import vsa


@dataclass
class Feature:
    name: str
    n_values: int
    contiguous: bool = False
    density: float = 1.


class Codebook:
    features: List[Feature]
    latent_dim: int
    codebook: List[torch.tensor]

    @staticmethod
    def make_features_from_dataset(dataset: Dataset) -> List[Feature]:
        features: List[Feature] = []
        for feature_name, n_values, contiguous in zip(dataset.feature_names,
                                                      dataset.feature_counts,
                                                      dataset.is_contiguous):
            features.append(Feature(name=feature_name,
                                    n_values=n_values,
                                    contiguous=contiguous))
        return features

    @property
    def placeholders(self) -> torch.tensor:
        return self.codebook[0]

    @property
    def vsa_features(self) -> List[torch.tensor]:
        return self.codebook[1:]

    def __init__(self, features: List[Feature],
                 latent_dim: int,
                 seed: int = 0,
                 device: torch.device = torch.device('cpu')):
        torch.manual_seed(seed)
        self.device = device
        self.features = features
        self.n_features = len(features)
        # Add placeholders Feature class to automatic creation later
        placeholders = Feature(name='Placeholders', n_values=len(self.features))
        self.features.insert(0, placeholders)

        self.latent_dim = latent_dim
        self.codebook = []
        self.features_count = []

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
            self.codebook.append(feature_vectors)


if __name__ == '__main__':
    features = [Feature('shape', 3), Feature('scale', 6, contiguous=True)]
    latent_dim = 1024
    codebook = Codebook(features, latent_dim)

    pass
