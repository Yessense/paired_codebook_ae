from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torch import tensor as Tensor
from torch import nn
from abc import abstractmethod

import os
import math
import torch
from torch import optim
import wandb
from paired_codebook_ae.config import MainConfig
from paired_codebook_ae.config import MainConfig
from paired_codebook_ae.model.factor_vae.utils import kl_divergence, permute_dims, recon_loss
from paired_codebook_ae.utils import iou_pytorch

import pytorch_lightning as pl
from torch.nn import functional as F
from .clevr_vae import CLEVRFactorVAE
from .dsprites_vae import DspritesFactorVAE
from .discriminator import Discriminator
# from paired_codebook_ae.config import FactorVAECompareConfig

torch.autograd.set_detect_anomaly(True)

class FactorVAEXperiment(pl.LightningModule):
    def __init__(self,
                 cfg: MainConfig, **kwargs) -> None:
        super(FactorVAEXperiment, self).__init__()

        if cfg.dataset.requires_fid:
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
            self.requires_fid = True
        else:
            self.requires_fid = False

        if cfg.dataset.datamodule.mode == 'paired_clevr':
            self.cfg = cfg.model.factor_vae.clevr_experiment
            self.model = CLEVRFactorVAE(self.cfg.z_dim)
        elif cfg.dataset.datamodule.mode == 'paired_dsprites':
            self.cfg = cfg.model.factor_vae.dsprites_experiment
            self.model = DspritesFactorVAE(self.cfg.z_dim)
        else:
            raise ValueError("Wrong dataset mode")

        self.discriminator = Discriminator(self.cfg.z_dim)

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

        self.automatic_optimization = False

        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        ones = torch.ones(image.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros(
            image.shape[0], dtype=torch.long, device=self.device)

        # random permute donor images for discriminator
        dim = 0

        idx = torch.randperm(donor.shape[dim])
        donor = donor[idx]

        # new_batch = torch.cat((image, donor), 0)

        optimizer_g, optimizer_d = self.optimizers()

        # self.toggle_optimizer(optimizer_g)
        x_recon, mu, logvar, z = self.model(image)
        d_z = self.discriminator(z)

        vae_recon_loss = recon_loss(image, x_recon)
        vae_kld = kl_divergence(mu, logvar)
        vae_tc_loss = (d_z[:, :1] - d_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.cfg.gamma * vae_tc_loss

        vae_loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_g.zero_grad()

        # self.untoggle_optimizer(optimizer_g)

        # train discriminator

        # self.toggle_optimizer(optimizer_d)
        z_prime = self.model(donor, no_dec=True)
        z_pperm = permute_dims(z_prime).detach()

        d_z_perm = self.discriminator(z_pperm)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_perm, ones))

        d_tc_loss.backward()
        optimizer_d.step()
        optimizer_d.zero_grad()

        # self.untoggle_optimizer(optimizer_d)

        iou_image = iou_pytorch(image, x_recon)

        self.log(f"{mode}/Total", vae_loss)
        self.log(f"{mode}/Reconstruct", vae_recon_loss)
        self.log(f"{mode}/KLD", vae_kld)
        self.log(f"{mode}/IOU", iou_image)

        if mode == 'Validation' and self.requires_fid:
            self.fid.update(image, real=True)
            self.fid.update(x_recon, real=False)
            fid = self.fid.compute()
            self.log(f"{mode}/FID", fid)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(x_recon[0],
                                caption='Reconstructed'),
                ]})
        return None

    def training_step(self, batch, batch_idx):
        mode = 'Train'
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            def log_images(x): return x == 0
        elif mode == 'Validation':
            def log_images(x): return x % 10 == 0
        else:
            raise ValueError

        self.automatic_optimization = False

        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        ones = torch.ones(image.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros(
            image.shape[0], dtype=torch.long, device=self.device)

        # random permute donor images for discriminator
        dim = 0

        idx = torch.randperm(donor.shape[dim])
        donor = donor[idx]

        # new_batch = torch.cat((image, donor), 0)

        optimizer_g, optimizer_d = self.optimizers()

        # self.toggle_optimizer(optimizer_g)
        x_recon, mu, logvar, z = self.model(image)
        d_z = self.discriminator(z)

        vae_recon_loss = recon_loss(image, x_recon)
        vae_kld = kl_divergence(mu, logvar)
        vae_tc_loss = (d_z[:, :1] - d_z[:, 1:]).abs().mean()

        vae_loss = vae_recon_loss + vae_kld + self.cfg.gamma * vae_tc_loss

        vae_loss.backward(retain_graph=True)

        # self.untoggle_optimizer(optimizer_g)

        # train discriminator

        # self.toggle_optimizer(optimizer_d)
        z_prime = self.model(donor, no_dec=True)
        z_pperm = permute_dims(z_prime).detach()

        d_z_perm = self.discriminator(z_pperm)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
                           F.cross_entropy(d_z_perm, ones))

        d_tc_loss.backward()
        optimizer_g.step()
        optimizer_g.zero_grad()

        optimizer_d.step()
        optimizer_d.zero_grad()

        # self.untoggle_optimizer(optimizer_d)

        iou_image = iou_pytorch(x_recon, image)

        self.log(f"{mode}/Total", vae_loss)
        self.log(f"{mode}/Reconstruct", vae_recon_loss)
        self.log(f"{mode}/KLD", vae_kld)
        self.log(f"{mode}/iou total", iou_image)
        self.log(f"{mode}/VAE TC", vae_tc_loss)
        self.log(f"{mode}/Discriminator TC", d_tc_loss)

        if mode == 'Validation' and self.requires_fid:
            self.fid.update(image, real=True)
            self.fid.update(x_recon, real=False)
            fid = self.fid.compute()
            self.log(f"{mode}/FID", fid)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(x_recon[0],
                                caption='Reconstructed'),
                ]})
        return None

    def validation_step(self, batch, batch_idx):
        mode = 'Validation'
        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            def log_images(x): return x == 0
        elif mode == 'Validation':
            def log_images(x): return x % 10 == 0
        else:
            raise ValueError

        self.automatic_optimization = False

        image: torch.tensor
        image_labels: torch.tensor
        donor: torch.tensor
        donor_labels: torch.tensor
        exchange_labels: torch.tensor

        (image, donor), (image_labels, donor_labels), exchange_labels = batch

        ones = torch.ones(image.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros(
            image.shape[0], dtype=torch.long, device=self.device)

        # random permute donor images for discriminator
        dim = 0

        idx = torch.randperm(donor.shape[dim])
        donor = donor[idx]

        # new_batch = torch.cat((image, donor), 0)


        # self.toggle_optimizer(optimizer_g)
        x_recon, mu, logvar, z = self.model(image)
        d_z = self.discriminator(z)

        vae_recon_loss = recon_loss(image, x_recon)
        vae_kld = kl_divergence(mu, logvar)
        vae_tc_loss = (d_z[:, :1] - d_z[:, 1:]).mean()

        vae_loss = vae_recon_loss + vae_kld + self.cfg.gamma * vae_tc_loss

        # self.untoggle_optimizer(optimizer_g)

        # train discriminator

        # self.toggle_optimizer(optimizer_d)
        # z_prime = self.VAE(donor, no_dec=True)
        # z_pperm = permute_dims(z_prime).detach()

        # d_z_perm = self.discriminator(z_pperm)
        # d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) +
        #                    F.cross_entropy(d_z_perm, ones))

        # d_tc_loss.backward()
        # optimizer_d.step()
        # optimizer_d.zero_grad()

        # self.untoggle_optimizer(optimizer_d)

        iou_image = iou_pytorch(x_recon, image)

        self.log(f"{mode}/Total", vae_loss)
        self.log(f"{mode}/Reconstruct", vae_recon_loss)
        self.log(f"{mode}/KLD", vae_kld)
        self.log(f"{mode}/VAE TC", vae_tc_loss)
        self.log(f"{mode}/iou total", iou_image)

        if mode == 'Validation' and self.requires_fid:
            self.fid.update(image, real=True)
            self.fid.update(x_recon, real=False)
            fid = self.fid.compute()
            self.log(f"{mode}/FID", fid)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(x_recon[0],
                                caption='Reconstructed'),
                ]})
        return None

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(),
                                 lr=self.cfg.opt_g_lr,
                                 betas=(self.cfg.beta1_vae, self.cfg.beta2_vae))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.opt_d_lr,
            betas=(self.cfg.beta1_d, self.cfg.beta2_d)
        )
        return [opt_g, opt_d], []
