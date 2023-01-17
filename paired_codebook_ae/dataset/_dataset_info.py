from abc import ABC
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from torch.utils.data import Dataset


@dataclass
class DatasetInfo():
    # Number of features
    n_features: int
    # List of feature names
    feature_names: Tuple[str, ...]
    # Count each feature counts
    feature_counts: Tuple[int, ...]
    # Is feature contiguous
    is_contiguous: Tuple[bool, ...]
    # Feature numbers
    features_list: List[int]
    # Ranges for each feature possible values
    features_range: List[np.array]
    # Image size
    image_size: Tuple[int, int, int]

    def __post_init__(self):
        self.features_list = list(range(len(self.feature_names)))
        self.features_range = [np.array(list(range(i))) for i in self.feature_counts]


class DatasetWithInfo(Dataset, ABC):
    dataset_info: DatasetInfo

    def __init__(self, dataset_info: DatasetInfo):
        super().__init__()
        self.dataset_info = dataset_info
