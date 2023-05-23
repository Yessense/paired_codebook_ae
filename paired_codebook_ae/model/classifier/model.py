from typing import Any, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler
import torch

from paired_codebook_ae.dataset.paired_clevr import PairedClevrDatamodule
from paired_codebook_ae.dataset.paired_dsprites import PairedDspritesDatamodule
from paired_codebook_ae.model.classifier.encoder import Encoder
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, cfg, datamodule: Union[PairedClevrDatamodule,
    PairedDspritesDatamodule]) -> None:
        super().__init__()

        self.cfg = cfg
        self.cfg.experiment.steps_per_epoch = cfg.dataset.train_size // cfg.experiment.batch_size

        self.encoder = Encoder(image_size=cfg.dataset.image_size,
                               latent_dim=cfg.model.latent_dim,
                               hidden_channels=64)

        self.dataset_info = datamodule.dataset_type.dataset_info
        self.feature_index = cfg.experiment.feature_index
        self.num_classes = self.dataset_info.feature_counts[self.feature_index]
        # if self.conti
        self.classifier = nn.Sequential(nn.Linear(cfg.model.latent_dim, cfg.model.latent_dim),
                                        nn.ReLU(),
                                        nn.Linear(cfg.model.latent_dim, self.num_classes))
        self.activation = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)
        self.save_hyperparameters()

    def step(self, batch, batch_idx, mode: str = 'Train') -> Tensor:
        image: Tensor
        image_labels: Tensor
        donor: Tensor
        donor_labels: Tensor
        exchange_labels: Tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        images = torch.cat((image, donor), 0)
        labels = torch.cat((image_labels, donor_labels), 0)

        z = self.activation(self.encoder(images))

        pred = self.softmax(self.classifier(z))
        loss = self.loss_f(pred, labels[:, self.cfg.experiment.feature_index])
        metrics_cls = MulticlassAccuracy(num_classes=self.num_classes, average='micro').to(self.device)
        metrics = metrics_cls(pred, labels[:, self.feature_index])

        self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_loss": loss})
        self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_metric": metrics})

        self.log(f"{mode}/Total", loss)

        return loss

    def loss_f(self, pred: Tensor, target: Tensor):
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='Train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='Validation')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.experiment.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.experiment.lr,
                                            epochs=self.cfg.experiment.trainer.max_epochs,
                                            steps_per_epoch=self.cfg.experiment.steps_per_epoch,
                                            pct_start=self.cfg.experiment.pct_start)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }
