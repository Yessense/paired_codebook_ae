import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.io.image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from ..dataset._dataset_info import DatasetWithInfo, DatasetInfo


class PairedClevr(DatasetWithInfo):
    dataset_info = DatasetInfo(
        feature_names=('shape', 'color', 'size', 'material', 'posX', 'posY'),
        feature_counts=(3, 8, 2, 2, 32, 32),
        is_contiguous=(False, False, False, False, True, True),
        n_features=6,
        image_size=(3, 128, 128),
        features_list=[],
        features_range=[],
    )

    def __init__(self, dataset_dir: Path,
                 indices, with_labels=False):
        super().__init__(self.dataset_info)
        self.with_labels = with_labels
        self.scenes_dir = dataset_dir / 'scenes'
        self.img_dir = dataset_dir / 'images'
        self.img_template = "{name}_{idx:06d}.png"
        self.json_template = "scene_{idx:06d}.json"
        self._size = len(os.listdir(self.scenes_dir))
        self.indices = indices

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        json_path = os.path.join(self.scenes_dir, self.json_template.format(idx=idx))
        with open(json_path) as json_file:
            annotations = json.load(json_file)

        images = []
        labels = []
        for name in ['img', 'pair']:
            img_filename = self.img_template.format(name=name, idx=idx)
            img_path = os.path.join(self.img_dir, img_filename)
            img = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB) / 255
            images.append(img)
            label = []
            for obj in annotations:
                if obj['image_filename'] == img_filename:
                    o = obj['objects'][0]
                    label.append(o['shape'])
                    label.append(o['color'])
                    label.append(o['size'])
                    label.append(o['material'])
                    label.append(int(((o['3d_coords'][0] + 3) * 32) // 6))
                    label.append(int(((o['3d_coords'][1] + 3) * 32) // 6))
                    labels.append(label)
                    break

        exchange_labels = torch.zeros((self.dataset_info.n_features), dtype=bool)

        for i in range(self.dataset_info.n_features):
            exchange_labels[i] = labels[0][i] != labels[1][i]

        exchange_labels = exchange_labels.unsqueeze(-1)

        return images, labels, exchange_labels


class PairedClevrDatamodule(pl.LightningDataModule):
    dataset_type: DatasetWithInfo = PairedClevr
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 mode: str = "paired_clevr",
                 num_workers: int = 4):
        super().__init__()
        self.num_workers = num_workers
        path_to_data_dir = Path(path_to_data_dir)
        self.path_to_paired_clevr_dir = path_to_data_dir / mode
        self.batch_size = batch_size
        self.image_size = (3, 128, 128)

    def setup(self, stage):
        self.train_dataset = PairedClevr(
            dataset_dir=self.path_to_paired_clevr_dir / 'train',
            indices=list(range(10000))
        )

        self.val_dataset = PairedClevr(
            dataset_dir=self.path_to_paired_clevr_dir / 'val',
            indices=list(range(1000))
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    dataset = PairedClevr(
        dataset_dir=Path('/home/yessense/data/paired_codebook_ae/data/paired_clevr/train/'),
        indices=list(range(10000)))
    start_idx = 300
    n_images = 2

    plt.figure(figsize=(30, 20))
    fig, ax = plt.subplots(n_images, 2)
    for i in range(n_images):
        (img, pair), (labels), exchange_label = dataset[i + start_idx]
        print(
            *[dataset.dataset_info.feature_names[i] for i, exchange in enumerate(exchange_label) if
              exchange])
        ax[i, 0].imshow(img.permute(1, 2, 0))
        ax[i, 1].imshow(pair.permute(1, 2, 0))
    plt.show()
