from typing import List, Callable, Union, Any, TypeVar, Tuple

import torch
import torchvision
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torch import tensor as Tensor
from torch import nn
from abc import abstractmethod

import os
import math
import torch
from torch import optim
import wandb
from paired_codebook_ae.config import MainConfig, BetaVAESetupConfig
from paired_codebook_ae.dataset.paired_clevr import PairedClevrDatamodule
from paired_codebook_ae.dataset.paired_dsprites import PairedDspritesDatamodule
from paired_codebook_ae.model.beta_vae.decoder import Decoder
from paired_codebook_ae.model.beta_vae.encoder import Encoder
from ...dataset._dataset_info import DatasetInfo
from ...utils import iou_pytorch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 dataset_info: DatasetInfo,
                 latent_dim: int,
                 beta: int = 1,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs,
                 ) -> None:
        super().__init__()

        self.dataset_info = dataset_info
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.encoder = Encoder(self.dataset_info.image_size, self.latent_dim)

        self.decoder = Decoder(self.dataset_info.image_size, self.latent_dim)
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # print(result.shape)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        # print(z.shape)
        # result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      kld_weight,
                      ) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 cfg: BetaVAESetupConfig,
                 datamodule: Union[PairedClevrDatamodule, PairedDspritesDatamodule]
                 ) -> None:
        super(VAEXperiment, self).__init__()

        if cfg.dataset.requires_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            self.requires_fid = True
        else:
            self.requires_fid = False

        self.dataset_info = datamodule.dataset_type.dataset_info
        self.model = BetaVAE(
            dataset_info=self.dataset_info,
            latent_dim=cfg.model.latent_dim,
            loss_type=cfg.model.loss_type,
            gamma=cfg.model.gamma,
            max_capacity=cfg.model.max_capacity,
            Capacity_max_iter=cfg.model.Capacity_max_iter)
        self.cfg = cfg
        self.save_hyperparameters()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def step(self, batch, batch_idx, mode: str = 'Train') -> torch.tensor:
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        else:
            raise ValueError

        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        new_batch = torch.cat((image, donor), 0)

        decoded, original, mu, log_var = self.forward(new_batch)

        loss = self.model.loss_function(decoded, original, mu, log_var, kld_weight=self.cfg.experiment.kld_weight)

        iou_image = iou_pytorch(decoded, original)

        self.log(f"{mode}/Total", loss['loss'])
        self.log(f"{mode}/Reconstruct", loss['Reconstruction_Loss'])
        self.log(f"{mode}/KLD", loss['KLD'])
        self.log(f"{mode}/IOU", iou_image)

        if mode == 'Validation' and self.requires_fid:
            self.fid.update(original, real=True)
            self.fid.update(decoded, real=False)
            fid = self.fid.compute()
            self.log(f"{mode}/FID", fid)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(original[0], caption='Image'),
                    wandb.Image(decoded[0],
                                caption='Reconstructed'),
                ]})

        return loss['loss']

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, mode='Validation')
        return loss

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.experiment.lr,
                               weight_decay=self.cfg.experiment.weight_decay)
        optims.append(optimizer)

        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                     gamma=self.cfg.experiment.scheduler_gamma)
        scheds.append(scheduler)
        return {"optimizer": optims[0], "lr_scheduler": scheds[0]}


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
        self.means = torch.tensor([1.4093, -0.6671, -1.5208,  2.7608,  5.0833, -1.3743, -0.0607, -3.2667,
        -0.1273, -9.8282])
        self.stds = torch.tensor([1.4391, 2.1054, 1.4354, 2.2973, 3.5389, 3.8463, 2.3919, 1.9964, 4.6758,
        4.2449])
        self.lin_values = []

        for mean, std, i in zip(self.means, self.stds, range(10)):
            lin_values = torch.linspace(mean - 3* std, mean + 3* std, 32)
            self.lin_values.append(lin_values)

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
        # self.means = []
    
    def test_step(self, batch, batch_idx):
        total_single = 0.
        total_entropy = 0.
        image: torch.tensor
        image_labels: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        images = torch.cat((image, donor), 0)
        image_labels = torch.cat((image_labels, donor_labels), 0)
        batch_size = image_labels.shape[0]
        image_latent = self.model.encoder(images)
        image_latent = self.model.fc_mu(image_latent)
        # self.means.append(image_latent)

        recon_image_like = self.model.decode(image_latent)

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
        # print("model")

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


        for j, lin_values in enumerate(self.lin_values):
            # -> (64, 32, 6)

            metric_arr = torch.zeros(batch_size, lin_values.shape[0], len(
                self.classifiers)).to(self.device)
            space_entropy_vec = torch.zeros(
                len(self.classifiers)).to(self.device)

            for k, vector in enumerate(lin_values):

                vector = vector.repeat(batch_size)
                changed_image_features = image_latent.clone()
                changed_image_features[:, j] = vector

                changed_recon_image = self.model.decoder(changed_image_features)

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

            space_entropy_vec /= 32
            t = 10
            probs = F.softmax(space_entropy_vec * t, dim=0)
            entropy_value = -torch.sum(probs * torch.log(probs))
            total_entropy += entropy_value
            print(
                f'{j}. Entropy vec: {space_entropy_vec}, probs: {probs}\n\t, entropy value: {entropy_value: 0.2f}, single: {single_metric.item(): 0.2f}')
            

            self.log_dict(
                {f"Test/{j}_entropy_val": entropy_value})
            self.log_dict(
                {f"Test/{j}_single_val": single_metric})
        self.log('single', total_single / len(self.lin_values))
        self.log('entropy', total_entropy / len(self.lin_values))

    # def on_test_end(self) -> None:
    #     vals = torch.cat(self.means, dim=0)
    #     vector_means = torch.mean(vals, dim=0)
    #     vector_std = torch.std(vals, dim=0)
    #     vector_max = torch.max(vals, dim=0).values
    #     vector_min = torch.min(vals, dim=0).values

    #     print("means: ", vector_means)
    #     print("std: ", vector_std)
    #     print("max: ", vector_max)
    #     print("min: ", vector_min)

    #     return super().on_test_end()


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
