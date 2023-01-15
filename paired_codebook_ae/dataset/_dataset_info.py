from abc import ABC
from typing import Tuple, List, Optional

from torch.utils.data import Dataset


class DatasetInfo(ABC):
    # Number of features
    n_features: int
    # List of feature names
    feature_names: Tuple[str, ...]
    # Count each feature counts
    feature_counts: Tuple[int]
    # Is feature contiguous
    is_contiguous: Tuple[bool]
    # Feature numbers
    features_list: List[int]
    # Ranges for each feature possible values
    features_range: List[Optional[int]]
    # Image size
    image_size: Tuple[int, int, int]


class DatasetWithInfo(Dataset, DatasetInfo, ABC):
    pass
