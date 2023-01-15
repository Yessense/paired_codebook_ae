from typing import List, Optional

import pytorch_lightning as pl
import torch
import wandb


class GeneralizationVisualizationCallback(pl.Callback):
    def __init__(self, samples: Optional[List] = None):
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
        self.samples = torch.LongTensor(samples)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        recons = pl_module(self.samples)
        trainer.logger.experiment.log({
            'reconstructions': [wandb.Image(img) for img in recons]
        })

