from typing import Any
from torch.utils.data import Dataset
from ._dataset_info import DatasetWithInfo, DatasetInfo
import torchvision
import pandas as pd
import numpy as np
import torchvision
from torchvision.io import read_image
from torchvision.transforms import Resize
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class PairedCeleba(DatasetWithInfo):
    dataset_info = DatasetInfo(
        feature_names=['Hair_type', 'Beard', 'Nose', 'Hair_Form', 'Male', 'Young', 'Pale_Skin'],
        feature_counts=(5, 2, 2, 2, 2, 2, 2),
        is_contiguous=(False, False, False, False, False, False, False),
        image_size=(1, 64, 64),
        n_features=7,
        features_list=[],
        features_range=[]
    )

    def __init__(self, dataset_dir="data/celeba/"):
        super().__init__(dataset_info=self.dataset_info)
        data = np.load(dataset_dir + 'celeba_indices.npy', allow_pickle=True)
        data = data.tolist()
        self.transform = Resize((128,128))
        self._path = dataset_dir
        self.imgs = data.get('imgs')
        self.pair_imgs = data.get('pair_imgs')
        self.labels = data.get('img_labels')
        self.pair_labels = data.get('pair_labels')
        self.exchange_indices = data.get('exchange_indices')
        self.img_template = "{idx:06d}.jpg"
        self._size = len(self.imgs)


    def __len__(self):
        return self._size

    def __getitem__(self, index) -> Any:
        img = self.img_template.format(idx=self.imgs[index])
        img =self.transform( read_image(self._path + '/img_align_celeba/' + img) / 255)

        pair_img = self.img_template.format(idx=self.pair_imgs[index])
        pair_img = self.transform(read_image(self._path + '/img_align_celeba/' + pair_img) / 255)

        img_labels = self.labels[index]
        pair_lables = self.pair_labels[index]
        exchange_indices = torch.from_numpy(self.exchange_indices[index]).unsqueeze(-1)


        return (img, pair_img), (img_labels, pair_lables), exchange_indices
    


class PairedCelebaDatamodule(pl.LightningDataModule):
    dataset_type: DatasetWithInfo = PairedCeleba
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 mode: str = "paired_celeba",
                 num_workers: int = 4):
        super().__init__()
        self.mode = mode
        self.path_to_data_dir = path_to_data_dir + 'celeba/'
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = (3, 128, 128)

    def setup(self, stage):
        dataset = PairedCeleba(
            dataset_dir=self.path_to_data_dir,
        )
        gen = torch.Generator()
        gen.manual_seed(0)
        self.train_dataset, self.val_dataset, _ = torch.utils.data.random_split(dataset, [10000, 1000, len(dataset) - 11000], generator=gen)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    dataset = PairedCeleba()

    import matplotlib.pyplot as plt

    # Get random indices for example images
    start_idx = 300
    n_images = 4

    plt.figure(figsize=(30, 20))
    fig, ax = plt.subplots(n_images, 2)
    for i in range(n_images):
        (img, pair), (labels), exchange_label = dataset[i + start_idx]
        print(
            *[dataset.dataset_info.feature_names[i] for i, exchange in enumerate(exchange_label) if
              exchange])
        ax[i, 0].imshow(img.permute(1, 2, 0))
        ax[i, 1].imshow(pair.permute(1, 2, 0))
    plt.savefig('plot.png')