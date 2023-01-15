import itertools
from pathlib import Path

import pytorch_lightning as pl
import random
from typing import List, Tuple, Optional

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Subset, DataLoader

from paired_codebook_ae.dataset._dataset_info import DatasetWithInfo
from .dsprites import Dsprites


def make_indices(train_size: int, test_size: int, seed: int = 0) -> Tuple[List[int], List[int]]:
    random.seed(seed)

    used_indices = set()
    train_indices: List[int] = []
    test_indices: List[int] = []

    labels = list(itertools.product(*Dsprites.features_range))

    # train
    for mode, indices_list, indices_len in zip(["train", "test"],
                                               [train_indices, test_indices],
                                               [train_size, test_size]):
        random.shuffle(labels)

        for label in labels:
            if len(indices_list) >= indices_len:
                break

            idx = Dsprites.get_element_pos(label)

            if mode == "train":
                # shape = square and posx > 0.5
                if label[0] == 0 and label[3] > 15:
                    continue

            if idx not in used_indices:
                used_indices.add(idx)
                indices_list.append(idx)
            else:
                continue

    return train_indices, test_indices


class GeneralizationDspritesDataModule(pl.LightningDataModule):
    dataset: DatasetWithInfo

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 train_size: int = 100_000,
                 val_size: int = 30_000):
        super().__init__()
        self.path_to_data_dir = Path(path_to_data_dir)
        self.path_to_dsprites_dataset = str(self.path_to_data_dir / 'dsprites_train.npz')
        self.batch_size = batch_size
        self.train_size, self.val_size = train_size, val_size

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = Dsprites(self.path_to_dsprites_dataset)
        train_indices, test_indices = make_indices(self.train_size, self.val_size)

        self.dsprites_train = Subset(dataset, train_indices)
        self.dsprites_val = Subset(dataset, test_indices)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dsprites_train, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.dsprites_val, batch_size=self.batch_size, drop_last=True)


if __name__ == '__main__':
    indices = make_indices(10, 5)
    pass
