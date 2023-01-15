import os.path
from typing import List, Optional

import pytorch_lightning as pl
import torch
import wandb

from paired_codebook_ae.dataset.dsprites import Dsprites


class GeneralizationVisualizationCallback(pl.Callback):
    def __init__(self, samples: Optional[List] = None,
                 path_to_data_dir: str = './data/'):
        if samples is None:
            # shape, scale, orientation, x, y
            samples = [
                [0, 0, 0, 31, 0],
                [0, 0, 0, 31, 31],
                [0, 0, 5, 31, 0],
                [0, 0, 5, 31, 31],
                [0, 0, 10, 31, 0],
                [0, 0, 10, 31, 31],
                [0, 0, 10, 31, 0],
                [0, 0, 10, 31, 31],
                [0, 4, 0, 31, 0],
                [0, 4, 0, 31, 31],
                [0, 4, 5, 31, 0],
                [0, 4, 5, 31, 31],
                [0, 4, 10, 31, 0],
                [0, 4, 10, 31, 31],
                [0, 4, 10, 31, 0],
                [0, 4, 10, 31, 31],
            ]

        dataset = Dsprites(path_to_data_dir)
        self.samples = []
        for label in samples:
            img, _ = dataset[dataset.get_element_pos(label)]
            self.samples.append(img)

        self.samples = torch.tensor(samples)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:

        z = pl_module.encoder(self.samples)
        z, _ = pl_module.attention(z)
        z = pl_module.binder(z)
        z = torch.sum(z, dim=1)
        recons = pl_module.decode(torch.sum(z, dim=1))

        trainer.logger.experiment.log({
            'Validation/reconstructions': [wandb.Image(img) for img in recons]
        })
