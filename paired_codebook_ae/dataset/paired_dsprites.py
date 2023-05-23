import itertools
import operator
import os
from pathlib import Path
import random
from typing import Set, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from ._dataset_info import DatasetWithInfo
from .dsprites import Dsprites


def make_pair_dsprites_indices(train_size: int,
                               test_size: int,
                               seed: int = 42,
                               max_exchanges: int = 1,
                               dsprites_path: str = "/data/dsprites/dsprites.npz"
                               ):
    """ Create indices for Paired Dsprites dataset """
    dataset = Dsprites(path=dsprites_path)

    random.seed(seed)
    # set random seed
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    def get_pair(img_labels: np.ndarray, max_exchanges: int):
        pair_img_labels = np.copy(img_labels)

        n_exchanged = random.randint(1, max_exchanges)
        exchange_feature_numbers = random.sample(Dsprites.features_list,
                                                 k=n_exchanged)

        for feature_number in exchange_feature_numbers:
            while pair_img_labels[feature_number] == img_labels[feature_number]:
                pair_img_labels[feature_number] = random.randrange(0,
                                                                   Dsprites.feature_counts[
                                                                       feature_number])

        return torch.from_numpy(pair_img_labels)

    # Create list of allowed combination
    allowed_combinations = np.array(
        [np.array(x) for x in itertools.product(*Dsprites.features_range)])
    np.random.shuffle(allowed_combinations)

    # Already used pairs
    used_indices: Set = set()

    train_pairs: List = []
    train_exchanges: List = []
    test_pairs: List = []
    test_exchanges: List = []

    assert train_size <= len(allowed_combinations) // 2

    # Add train samples
    i = 0
    while len(train_pairs) < train_size:
        img_idx = Dsprites.get_element_pos(allowed_combinations[i])
        _, img_labels = dataset[img_idx]

        # Conditional generalization:
        # No Square shape on right
        if img_idx in used_indices:
            i += 1
            continue

        # get pair image idx
        # repeat while not under conditional generalization
        # and if already used
        pair_img_labels = get_pair(img_labels, max_exchanges)
        pair_img_idx = Dsprites.get_element_pos(pair_img_labels)

        while pair_img_idx in used_indices:
            pair_img_labels = get_pair(img_labels, max_exchanges)
            pair_img_idx = Dsprites.get_element_pos(pair_img_labels)

        used_indices.add(img_idx)
        used_indices.add(pair_img_idx)

        exchanges = (img_labels != pair_img_labels).long()
        train_pairs.append(torch.tensor([img_idx, pair_img_idx]))
        train_exchanges.append(exchanges)

        i += 1

    # Add test samples
    while len(test_pairs) < test_size:
        img_idx = random.randrange(len(dataset))

        _, img_labels = dataset[img_idx]

        if img_idx in used_indices:
            continue

        # get pair image idx
        # if not already used
        pair_img_labels = get_pair(img_labels, max_exchanges)
        pair_img_idx = Dsprites.get_element_pos(pair_img_labels)
        while pair_img_idx in used_indices:
            pair_img_labels = get_pair(img_labels, max_exchanges)
            pair_img_idx = Dsprites.get_element_pos(pair_img_labels)

        used_indices.add(img_idx)
        used_indices.add(pair_img_idx)
        exchanges = (img_labels != pair_img_labels).long()
        test_pairs.append(torch.tensor([img_idx, pair_img_idx]))
        test_exchanges.append(exchanges)

        i += 1

    return torch.stack(train_pairs), torch.stack(train_exchanges), torch.stack(
        test_pairs), torch.stack(test_exchanges)


class PairedDspritesDataset(DatasetWithInfo):
    # List of feature names

    dataset_info = Dsprites.dataset_info
    # Getting multipler for each feature position
    multiplier = list(itertools.accumulate(
        dataset_info.feature_counts[-1:0:-1], operator.mul))[::-1] + [1]

    def __init__(self, dsprites_path='../data/dsprites/dsprites.npz',
                 paired_dsprites_path='../data/paired_dsprites/train.npz'):
        # Load npz numpy archive
        super().__init__(self.dataset_info)
        dsprites = np.load(dsprites_path, allow_pickle=True)
        paired_dsprites = torch.load(paired_dsprites_path)

        self.data = paired_dsprites['pairs']
        self.exchanges = paired_dsprites['exchanges']

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dsprites['imgs']

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dsprites['latents_classes'][:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.imgs[self.data[idx][0]]
        image = torch.from_numpy(image).unsqueeze(0).float()

        donor = self.imgs[self.data[idx][1]]
        donor = torch.from_numpy(donor).unsqueeze(0).float()

        exchange_labels = self.exchanges[idx].bool().unsqueeze(-1)

        image_labels = self.labels[self.data[idx][0]]
        donor_labels = self.labels[self.data[idx][1]]
        return (image, donor), (image_labels, donor_labels), exchange_labels


class PairedDspritesDatamodule(pl.LightningDataModule):
    dataset_type: DatasetWithInfo = PairedDspritesDataset
    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 mode: str = "paired_dsprites"):
        super().__init__()
        path_to_data_dir = Path(path_to_data_dir)
        self.path_to_dsprites_dataset = str(path_to_data_dir / 'dsprites' / 'dsprites.npz')
        self.path_to_paired_dsprites_dir = path_to_data_dir / mode
        self.batch_size = batch_size
        self.image_size = (1, 64, 64)

    def setup(self, stage=None):
        self.train_dataset = PairedDspritesDataset(
            dsprites_path=self.path_to_dsprites_dataset,
            paired_dsprites_path=str(
                self.path_to_paired_dsprites_dir / 'paired_dsprites_train.npz'))

        self.val_dataset = PairedDspritesDataset(
            dsprites_path=self.path_to_dsprites_dataset,
            paired_dsprites_path=str(self.path_to_paired_dsprites_dir / 'paired_dsprites_test.npz'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()


def plot_dataset(
        dsprites_path: str,
        paired_dsprites_path: str,
        show_test: bool = True) -> None:
    """ Plot 5 exapmles in Paired Dsprites Dataset"""
    import matplotlib.pyplot as plt

    if show_test:
        path_to_split = os.path.join(paired_dsprites_path, 'paired_dsprites_test.npz')
    else:
        path_to_split = os.path.join(paired_dsprites_path, 'paired_dsprites_train.npz')

    pd = PairedDspritesDataset(
        dsprites_path=dsprites_path,
        paired_dsprites_path=path_to_split)
    # md = Dsprites(max_exchanges=1, block_orientation=True)
    #
    batch_size = 5
    loader = DataLoader(pd, batch_size=5, shuffle=True)
    #
    batch = next(iter(loader))

    fig, ax = plt.subplots(2, batch_size, figsize=(10, 5))
    for i in range(batch_size):
        img = batch[0][0][i]
        pair_img = batch[0][1][i]
        exchange_labels = batch[2][i].squeeze()

        ax[0, i].imshow(img.detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[0, i].set_axis_off()
        ax[1, i].imshow(pair_img.detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[1, i].set_axis_off()
        print(
            f'{i} pair has [{" ,".join([pd.dataset_info.feature_names[idx] for idx, label in enumerate(exchange_labels) if label])}] feature(s) exchanged')

    plt.show()


def make_dataset(max_exchanges: int,
                 path_to_dsprites_train: str,
                 save_path: str,
                 train_size: int,
                 test_size: int) -> None:
    pairs = make_pair_dsprites_indices(train_size, test_size,
                                       max_exchanges=max_exchanges,
                                       dsprites_path=path_to_dsprites_train)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_name = 'paired_dsprites_train.npz'
    test_name = 'paired_dsprites_test.npz'

    torch.save({'pairs': pairs[0], 'exchanges': pairs[1]},
               os.path.join(save_path, train_name), )

    torch.save({'pairs': pairs[2], 'exchanges': pairs[3]},
               os.path.join(save_path, test_name), )


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['make_dataset', 'show_dataset'],
                        default='show_dataset')
    parser.add_argument("--max_exchanges", type=int, choices=[1, 2, 3, 4, 5],
                        default=1)
    parser.add_argument("--path_to_dsprites_train", type=str,
                        default='../../data/dsprites/dsprites.npz')
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--test_size", type=int, default=30_000)
    parser.add_argument("--save_path", type=str,
                        default='../../data/paired_dsprites/')
    args = parser.parse_args()

    if args.mode == 'make_dataset':
        make_dataset(max_exchanges=args.max_exchanges,
                     path_to_dsprites_train=args.path_to_dsprites_train,
                     save_path=args.save_path,
                     train_size=args.train_size,
                     test_size=args.test_size)
    elif args.mode == 'show_dataset':
        plot_dataset(dsprites_path=args.path_to_dsprites_train,
                     paired_dsprites_path=args.save_path,
                     show_test=False)
    else:
        raise ValueError("Wrong mode")


if __name__ == '__main__':
    cli()
