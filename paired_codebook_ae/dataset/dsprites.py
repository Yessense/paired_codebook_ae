import random
from abc import ABC
from pathlib import Path
from typing import Tuple, List, Set, Optional

import torch
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import itertools
import operator

from ._dataset_info import DatasetWithInfo


class Dsprites(DatasetWithInfo):
    """Store dsprites images"""

    # List of feature names
    feature_names: Tuple[str, ...] = (
        'shape', 'scale', 'orientation', 'posX', 'posY')
    # Count each feature counts
    feature_counts: Tuple[int] = [3, 6, 40, 32, 32]

    # Is feature contiguous
    is_contiguous: Tuple[bool] = [False, True, True, True, True]

    # Feature numbers
    features_list: List[int] = list(range(len(feature_names)))

    # Ranges for each feature possible values
    features_range = [np.array(list(range(i))) for i in
                      feature_counts]

    # Getting multipler for each feature position
    multiplier = list(itertools.accumulate(
        feature_counts[-1:0:-1], operator.mul))[::-1] + [1]

    n_features = 5

    image_size: Tuple[int, int, int] = (1, 64, 64)

    def __init__(self, path='data/dsprites/dsprites.npz'):
        # Load npz numpy archive
        dataset_zip = np.load(path, allow_pickle=True, encoding='latin1')

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dataset_zip['imgs']

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dataset_zip['latents_classes'][:, 1:]
        self.possible_values = dataset_zip['metadata'][()][
            'latents_possible_values']

        # Size of dataset (737280)
        self.size: int = self.imgs.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx]).unsqueeze(0).float()
        labels = torch.from_numpy(self.labels[idx])
        return img, labels

    @staticmethod
    def get_element_pos(labels: np.ndarray) -> int:
        """
        Get position of image with `labels` in dataset

        Parameters
        ----------
        labels: np.ndarray

        Returns
        -------
        pos: int
            Position in dataset
        """
        pos = 0
        for mult, label in zip(Dsprites.multiplier, labels):
            pos += mult * label
        return pos


class DspritesDatamodule(pl.LightningDataModule):
    dataset: Dataset
    image_size = (1, 64, 64)

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 train_size: int = 100_000,
                 val_size: int = 30_000):
        super().__init__()
        self.path_to_data_dir = Path(path_to_data_dir)
        self.path_to_dsprites_dataset = str(self.path_to_data_dir / 'dsprites.npz')
        self.batch_size = batch_size
        self.train_size, self.val_size = train_size, val_size

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = Dsprites(self.path_to_dsprites_dataset)

        last_chunk = len(dataset) - self.train_size - self.val_size
        self.dsprites_train, self.dsprites_val, _ = random_split(dataset,
                                                                 [self.train_size, self.val_size,
                                                                  last_chunk])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dsprites_train, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dsprites_val, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()
