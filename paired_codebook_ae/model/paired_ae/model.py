import pathlib
import random
from typing import Any, Optional, List, Union
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from torchvision import transforms
from torch import Tensor

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler
from paired_codebook_ae.config import PairedAEClevrSetupConfig, PairedAEDspritesSetupConfig

from paired_codebook_ae.dataset.paired_clevr import PairedClevr, PairedClevrDatamodule
from paired_codebook_ae.dataset._dataset_info import DatasetInfo
from paired_codebook_ae.metrics.test_visualization import reconstruction_from_one_feature, \
    exchange_between_two_random_objects, exchange_between_two_dataset_objects, true_unbinding
from .attention import AttentionModule
from .exchange import ExchangeModule
from paired_codebook_ae.metrics.vsa import vsa_decoding_accuracy
from paired_codebook_ae.utils import iou_pytorch
from paired_codebook_ae.dataset.paired_dsprites import Dsprites, PairedDspritesDatamodule
from paired_codebook_ae.codebook.codebook import Codebook
from .binder import Binder, FourierBinder
from .decoder import Decoder
from .encoder import Encoder
from torchmetrics.classification import MulticlassAccuracy


class VSADecoder(pl.LightningModule):
    binder: Binder
    cfg: Union[PairedAEClevrSetupConfig, PairedAEDspritesSetupConfig]
    dataset_info: DatasetInfo

    def __init__(self, cfg: Union[PairedAEClevrSetupConfig, PairedAEDspritesSetupConfig],
                 datamodule: Union[PairedClevrDatamodule, PairedDspritesDatamodule],
                 **kwargs):
        super().__init__()
        self.cfg = cfg

        self.dataset_info = datamodule.dataset_type.dataset_info
        self.cfg.experiment.steps_per_epoch = cfg.dataset.train_size // cfg.experiment.batch_size

        features = Codebook.make_features_from_dataset(self.dataset_info)
        if cfg.dataset.requires_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)

        self.encoder = Encoder(image_size=cfg.dataset.image_size,
                               latent_dim=cfg.model.latent_dim,
                               hidden_channels=cfg.model.encoder_config.hidden_channels)

        self.decoder = Decoder(image_size=cfg.dataset.image_size,
                               latent_dim=cfg.model.latent_dim,
                               in_channels=cfg.model.decoder_config.in_channels,
                               hidden_channels=cfg.model.decoder_config.hidden_channels)
        self.codebook = Codebook(features=features,
                                 latent_dim=cfg.model.latent_dim,
                                 seed=cfg.experiment.seed)

        self.attention = AttentionModule(vsa_features=self.codebook.vsa_features,
                                         n_features=cfg.dataset.n_features,
                                         latent_dim=cfg.model.latent_dim,
                                         scale=None)
        self.exchange_module = ExchangeModule()
        self.loss_f = F.mse_loss

        if cfg.model.binder == 'fourier':
            self.binder = FourierBinder(
                placeholders=self.codebook.placeholders)
        else:
            raise NotImplemented(f"Wrong binder type {cfg.model.binder}")

        self.save_hyperparameters()

    def step(self, batch, batch_idx, mode: str = 'Train') -> torch.tensor:
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            def log_images(x): return x == 0
        elif mode == 'Validation':
            def log_images(x): return x % 10 == 0
        else:
            raise ValueError

        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        image_latent = self.encoder(image)
        donor_latent = self.encoder(donor)

        image_features, image_max_values, _ = self.attention(image_latent)
        donor_features, donor_max_values, _ = self.attention(donor_latent)

        image_with_same_donor_elements, donor_with_same_image_elements = self.exchange_module(
            image_features, donor_features, exchange_labels)

        image_like_binded = self.binder(image_with_same_donor_elements)
        donor_like_binded = self.binder(donor_with_same_image_elements)

        recon_image_like = self.decoder(torch.sum(image_like_binded, dim=1))
        recon_donor_like = self.decoder(torch.sum(donor_like_binded, dim=1))

        image_loss = self.loss_f(
            recon_image_like, image, reduction=self.cfg.experiment.reduction)
        donor_loss = self.loss_f(
            recon_donor_like, donor, reduction=self.cfg.experiment.reduction)
        total_loss = (image_loss + donor_loss) * \
            0.5  # + self.kld_coef * kld_loss

        iou_image = iou_pytorch(recon_image_like, image)
        iou_donor = iou_pytorch(recon_image_like, image)
        total_iou = (iou_image + iou_donor) / 2

        # ----------------------------------------
        # Logs
        # ----------------------------------------

        self.log(f"{mode}/Total", total_loss)
        self.log(f"{mode}/Reconstruct Image", image_loss)
        self.log(f"{mode}/Reconstruct Donor", donor_loss)
        self.log(f"{mode}/Mean Reconstruction", (image_loss + donor_loss) / 2)
        # self.log(f"{mode}/KLD", kld_loss * self.kld_coef)
        self.log(f"{mode}/iou total", total_iou)
        self.log(f"{mode}/iou image", iou_image)
        self.log(f"{mode}/iou donor", iou_donor)

        self.logger.experiment.log(
            {f"{mode}/image max " + self.dataset_info.feature_names[i]: image_max_values[i] for i in
             range(self.cfg.dataset.n_features)}, commit=False)
        self.logger.experiment.log(
            {f"{mode}/donor max " + self.dataset_info.feature_names[i]: donor_max_values[i] for i in
             range(self.cfg.dataset.n_features)})

        if mode == 'Validation' and self.cfg.dataset.requires_fid:
            self.fid.update(image, real=True)
            self.fid.update(recon_image_like, real=False)
            fid = self.fid.compute()
            self.log(f"{mode}/FID", fid)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(donor[0], caption='Donor'),
                    wandb.Image(recon_image_like[0],
                                caption='Recon like Image'),
                    wandb.Image(recon_donor_like[0],
                                caption='Recon like Donor'),
                ]})

        return {'loss': total_loss}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        self.step(batch, batch_idx, mode='Validation')
        return None

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

    def on_test_start(self) -> None:
        # self.latent_vectors = []
        # self.latent_features = []
        # self.labels = []
        classifiers_dir = '/home/akorchemnyi/paired_codebook_ae/paired_codebook_ae/classifier_weights'
        classifier_ext = '.ckpt'
        self.classifiers = []

        self.IMAGE_SIZE = 224
        # specify ImageNet mean and standard deviation
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        self.transforms = transforms.Compose([
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
        ])

        for contigious, num_classes, name in zip(self.dataset_info.is_contiguous,
                                                 self.dataset_info.feature_counts,
                                                 self.dataset_info.feature_names):
            classifier = torchvision.models.resnet34()

            if contigious:
                classifier.fc = nn.Linear(classifier.fc.in_features, 1)
            else:
                classifier.fc = nn.Linear(
                    classifier.fc.in_features, num_classes)
            path = f'{classifiers_dir}/{name}{classifier_ext}'
            state_dict = torch.load(path)['state_dict']
            state_dict = {key[len("model."):]: value for key, value in state_dict.items(
            ) if key.startswith("model")}
            classifier.load_state_dict(state_dict)
            classifier.to(self.device)
            self.classifiers.append(classifier)

    def test_step(self, batch, batch_idx):
        total_single = 0.
        total_entropy = 0.
        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        images = torch.cat((image, donor), 0)
        image_labels = torch.cat((image_labels, donor_labels), 0)
        batch_size = image_labels.shape[0]
        image_latent = self.encoder(images)

        # 64, 6, 1024
        image_features, image_max_values, _ = self.attention(image_latent)

        image_like_binded = self.binder(image_features)
        recon_image_like = self.decoder(torch.sum(image_like_binded, dim=1))

        default_labels = torch.zeros_like(image_labels, dtype=float)
        for i, (classifier, contiguous) in enumerate(zip(self.classifiers, self.dataset_info.is_contiguous)):
            with torch.no_grad():
                pred = classifier(recon_image_like)
            if contiguous:

                # torch.clip(pred.squeeze(), min=0., max=1.)
                default_labels[:, i] = pred.squeeze()
            else:
                default_labels[:, i] = torch.argmax(pred, dim=1)

        print(
            f'Total accuracy: {torch.mean((default_labels[:, :4] == image_labels[:, :4]).float())}')
        # classify

        # for i, (classifier, contiguous) in enumerate(zip(self.classifiers, self.dataset_info.is_contiguous)):

        #     pred = classifier(recon_image_like)
        #     labels = image_labels[:, i]

        #     if contiguous:
        #         labels = labels.float() / (self.dataset_info.feature_counts[i] - 1)

        #         # self.log_dict({f"Test/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@0.1": self.calculate_map(pred, labels, 0.1)})
        #         # self.log_dict({f"Test/{self.dataset_info.feature_names[i]}_reg_acc@0.25":self.calculate_map(pred, labels, 0.25)})
        #         self.log_dict({f"Test/{self.dataset_info.feature_names[i]}_reg_acc@0.5": self.calculate_map(pred, labels, 0.5)})
        #         # self.log_dict({f"Test/{self.dataset_info.feature_names[i]}_reg_acc@-1": self.calculate_map(pred, labels, -1)})

        #     else:

        #         metrics_cls = MulticlassAccuracy(
        #             num_classes=self.dataset_info.feature_counts[i], average='micro').to(self.device)
        #         metrics = metrics_cls(pred, labels)

        #         self.log_dict(
        #             {f"Test/{self.dataset_info.feature_names[i]}_class_acc": metrics})

        for j, vsa_vectors in enumerate(self.codebook.vsa_features):
            # -> (64, 32, 6)

            metric_arr = torch.zeros(batch_size, vsa_vectors.shape[0], len(
                self.classifiers)).to(self.device)
            space_entropy_vec = torch.zeros(
                len(self.classifiers)).to(self.device)

            for k, vector in enumerate(vsa_vectors):

                vector = vector.unsqueeze(0).repeat(batch_size, 1)
                changed_image_features = image_features.clone()
                changed_image_features[:, j] = vector

                changed_binded = self.binder(changed_image_features)
                changed_recon_image = self.decoder(
                    torch.sum(changed_binded, dim=1))

                entropy_vec = torch.zeros(
                    len(self.classifiers)).to(self.device)

                for i, (classifier, contiguous) in enumerate(zip(self.classifiers, self.dataset_info.is_contiguous)):
                    with torch.no_grad():
                        pred = classifier(changed_recon_image)

                    labels = default_labels[:, i]

                    if contiguous:
                        pred_labels = self.classify_average_precision(
                            pred.squeeze(), labels, 0.5)
                        metric_arr[:, k, i] = pred_labels

                        # labels = labels.float() / (self.dataset_info.feature_counts[i] - 1)

                        # self.log_dict({f"Test/{self.dataset_info.feature_names[self.feature_index]}_reg_acc@0.1": self.calculate_map(pred, labels, 0.1)})
                        # self.log_dict({f"Test/{self.dataset_info.feature_names[i]}_reg_acc@0.25":self.calculate_map(pred, labels, 0.25)})
                        metrics = self.calculate_map(pred, labels, 0.5)
                        self.log_dict(
                            {f"Test/{self.dataset_info.feature_names[i]}_reg_acc@0.5": metrics})
                        # self.log_dict({f"Test/{self.dataset_info.feature_names[i]}_reg_acc@-1": self.calculate_map(pred, labels, -1)})
                        entropy_vec[i] += metrics
                    else:
                        pred_labels = torch.argmax(pred, dim=1)
                        metric_arr[:, k, i] = (pred_labels == labels).long()

                        metrics_cls = MulticlassAccuracy(
                            num_classes=self.dataset_info.feature_counts[i], average='micro').to(self.device)
                        metrics = metrics_cls(pred, labels)

                        self.log_dict(
                            {f"Test/{self.dataset_info.feature_names[i]}_class_acc": metrics})
                        entropy_vec[i] += metrics

                space_entropy_vec += (1 - entropy_vec)

            single_metric = metric_arr.sum(dim=(1, 2)) / metric_arr.shape[1]

            single_metric = torch.abs(single_metric - 1).mean()
            total_single += single_metric

            # print(single_metric)

            # print(1 - entropy_vec)
            # print("")

            space_entropy_vec /= len(vsa_vectors)
            t = 10
            probs = F.softmax(space_entropy_vec * t, dim=0)
            entropy_value = -torch.sum(probs * torch.log(probs))
            total_entropy += entropy_value
            print(
                f'{j}. Entropy vec: {space_entropy_vec}, probs: {probs}\n\t, entropy value: {entropy_value: 0.2f}, single: {single_metric.item(): 0.2f}')
            

            self.log_dict(
                {f"Test/{self.dataset_info.feature_names[j]}_entropy_val": entropy_value})
            self.log_dict(
                {f"Test/{self.dataset_info.feature_names[j]}_single_val": single_metric})
        self.log('single', total_single / len(self.classifiers))
        self.log('entropy', total_entropy / len(self.classifiers))

        # for name, num_classes, contiguous, classifier in zip(self.dataset_info.feature_names,
        #                              self.dataset_info.feature_counts,
        #                              self.dataset_info.is_contiguous,
        #                              self.classifiers):
        #     # -> (64, 32, 6)
        #     metric_arr = torch.zeros(batch_size, num_classes, len(self.classifiers))

        # features =

        # true_unbinding(self, batch)
        # if batch_idx == 0:
        # pass
        # reconstruction_from_one_feature(self)
        # if batch_idx < 20:
        # pass
        # exchange_between_two_dataset_objects(self, batch, batch_idx)

        # for i in range(n_samples):
        #     self.logger.experiment.log({
        #         "image": [wandb.Image(image[i], caption='Image')]
        #     }, commit=False)
        #     for feature_idx, feature_name in enumerate(self.dataset_info.feature_names):
        #         img_batch = torch.zeros(len(self.codebook.vsa_features[feature_idx]),
        #                                 self.cfg.dataset.n_features,
        #                                 self.cfg.model.latent_dim).to(self.device)
        #
        #         for feature_number, feature_value in enumerate(
        #                 self.codebook.vsa_features[feature_idx]):
        #             img_batch[feature_number] = image_features[i]
        #             img_batch[feature_number, feature_idx] = feature_value
        #
        #         img_batch = self.binder(img_batch)
        #         img_batch = torch.sum(img_batch, dim=1)
        #         img_batch = self.decoder(img_batch)
        #         commit = feature_idx == (len(self.dataset_info.feature_names) - 1)
        #         self.logger.experiment.log({
        #             feature_name: [wandb.Image(im) for im in img_batch]
        #         }, commit=commit)
        # return

    def calculate_map(self, pred: Tensor, target: Tensor, distance_threshold: float) -> float:
        assert distance_threshold <= 1 and distance_threshold >= 0. or distance_threshold == -1
        assert len(pred) == len(target)
        # assert torch.all(target <= 1.) and torch.all(target >= 0.)

        batch_size = pred.shape[0]
        out = 0.
        if distance_threshold == -1:
            return 1.

        for p, t in zip(pred, target):
            if p - distance_threshold <= t <= p + distance_threshold:
                out += 1
        return out / batch_size

    def classify_average_precision(self, pred: Tensor, target: Tensor, distance_threshold: float) -> Tensor:
        assert distance_threshold <= 1 and distance_threshold >= 0. or distance_threshold == -1
        assert len(pred) == len(target)

        if distance_threshold == -1:
            return torch.ones_like(pred).long()

        more = pred - distance_threshold <= target
        less = target <= pred + distance_threshold
        return  (less & more).long() 
