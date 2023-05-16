from typing import Any, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler
import torch
from torchvision import transforms
import torchvision

from paired_codebook_ae.dataset.paired_clevr import PairedClevrDatamodule
from paired_codebook_ae.dataset.paired_dsprites import PairedDspritesDatamodule
from paired_codebook_ae.model.classifier.encoder import Encoder
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, cfg, datamodule: Union[PairedClevrDatamodule,
                                              PairedDspritesDatamodule]) -> None:
        super().__init__()

        self.IMAGE_SIZE = 224
        # specify ImageNet mean and standard deviation
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        self.cfg = cfg
        self.cfg.experiment.steps_per_epoch = cfg.dataset.train_size // cfg.experiment.batch_size

        self.dataset_info = datamodule.dataset_type.dataset_info
        self.feature_index = cfg.experiment.feature_index
        # TODO
        self.contigious = self.dataset_info.is_contiguous[self.feature_index]
        self.num_classes = self.dataset_info.feature_counts[self.feature_index]

        for name, param in self.model.named_parameters():
            if 'fc' not in str(name):
                param.requires_grad = False
        self.model = torchvision.models.resnet34(pretrained=True)

        if self.contigious:
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        else:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


        nn.init.xavier_uniform_(self.model.fc.weight)
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            transforms.Resize((224, 224)),
        ])

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
        labels = labels[:, self.cfg.experiment.feature_index]
        images = self.transforms(images)
        pred = self.model(images)

        if self.contigious:
            # -> [0, 1]
            labels = labels.float() / (self.num_classes - 1)
            # huber loss
            loss = torch.nn.functional.huber_loss(pred, labels)
            # log loss
            self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_mse": loss})

            # calculate accuracy
            num_classes = self.dataset_info.feature_counts[self.feature_index]
            # metrics_cls = MulticlassAccuracy(
            #     num_classes=num_classes, average='micro').to(self.device)
            # metrics = metrics_cls(torch.round(torch.clip(pred,
            #                                  min=0,
            #                                  max=self.dataset_info.feature_counts[self.feature_index] - 1,
            #                                  ), 0).long(), labels.long())
            self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@0.1": self.calculate_map(pred, labels, 0.1)})
            self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@0.25":self.calculate_map(pred, labels, 0.25)})
            self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@0.5": self.calculate_map(pred, labels, 0.5)})
            self.log_dict({f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@-1": self.calculate_map(pred, labels, -1)})

        else:
            # cross entropy
            loss = self.loss_f(pred, labels)
            self.log_dict(
                {f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_cross_entr": loss})

            # metrics
            metrics_cls = MulticlassAccuracy(
                num_classes=self.num_classes, average='micro').to(self.device)
            metrics = metrics_cls(pred, labels)
            self.log_dict(
                {f"{mode}/{self.dataset_info.feature_names[self.feature_index]}_class_acc": metrics})

        self.log(f"{mode}/Total", loss)

        return loss

    def calculate_map(self, pred: Tensor, target: Tensor, distance_threshold: float) -> float:
        assert distance_threshold <= 1 and distance_threshold >= 0. or distance_threshold == -1
        assert len(pred) == len(target)
        assert torch.all(target <= 1.) and torch.all(target >= 0.)

        batch_size = pred.shape[0]
        out = 0.
        if distance_threshold == -1:
            return 1

        for p, t in zip(pred, target):
            if p - distance_threshold <= t <= p + distance_threshold:
                out += 1
        return out / batch_size


    def loss_f(self, pred: Tensor, target: Tensor):
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='Train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode='Validation')

    def configure_optimizers(self):
        lr, weight_decay = 1e-5, 5e-4
        params = [param for name, param in self.model.named_parameters()
                  if 'fc' not in str(name)]
        optimizer = torch.optim.Adam([{'params': params},
                                      {'params': self.model.fc.parameters(),
                                       'lr': lr*10},],
                                     lr=lr,
                                     weight_decay=weight_decay)

        return {"optimizer": optimizer}
        # "lr_scheduler": {'scheduler': scheduler,
        #                  'interval': 'step',
        #                  'frequency': 1}, }
