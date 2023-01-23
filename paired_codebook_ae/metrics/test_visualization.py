import random

import torch
import wandb

from ..codebook import vsa
# from ..model.paired_ae import VSADecoder


def reconstruction_from_one_feature(paired_ae):
    log_images = []
    for feature_number in range(paired_ae.cfg.dataset.n_features):
        for i, feature_value in enumerate(paired_ae.codebook.vsa_features[feature_number]):
            feature_value = feature_value.unsqueeze(0)
            placeholder = paired_ae.codebook.placeholders[feature_number]
            latent_vector = vsa.bind(feature_value, placeholder)
            recon = paired_ae.decoder(latent_vector)
            log_images.append(wandb.Image(recon[0],
                                          caption=f"{paired_ae.dataset_info.feature_names[feature_number]} {i}"))
    paired_ae.logger.experiment.log({
        "Reconstruction_from_one_feature": log_images
    })


def exchange_between_two_random_objects(paired_ae, batch):
    image: torch.tensor
    image_labels: torch.tensor
    donor: torch.tensor
    donor_labels: torch.tensor
    exchange_labels: torch.tensor

    (image, donor), (image_labels, donor_labels), exchange_labels = batch

    image_latent = paired_ae.encoder(image)
    image_features, _ = paired_ae.attention(image_latent)

    i = random.randint(0, paired_ae.cfg.experiment.batch_size - 1)
    j = random.randint(0, paired_ae.cfg.experiment.batch_size - 1)
    paired_ae.logger.experiment.log({
        "image": [wandb.Image(image[i], caption='Image'),
                  wandb.Image(image[j], caption='Donor')]
    }, commit=False)
    c, h, w = paired_ae.cfg.dataset.image_size
    recons = torch.zeros(paired_ae.cfg.dataset.n_features, c, h, w).to(paired_ae.device)
    latents = paired_ae.encoder(image)
    image_latent = latents[i].unsqueeze(0)
    donor_latent = latents[j].unsqueeze(0)

    image_features, image_max_values = paired_ae.attention(image_latent)
    donor_features, donor_max_values = paired_ae.attention(donor_latent)

    for feature_number in range(paired_ae.cfg.dataset.n_features):
        exchange_labels = torch.zeros(1, paired_ae.cfg.dataset.n_features).bool().to(
            paired_ae.device).unsqueeze(-1)
        print(exchange_labels.shape)
        exchange_labels[:, feature_number, :] = True

        image_with_same_donor_elements, donor_with_same_image_elements = paired_ae.exchange_module(
            image_features, donor_features, exchange_labels)

        donor_like_binded = paired_ae.binder(donor_with_same_image_elements)

        recon_donor_like = paired_ae.decoder(torch.sum(donor_like_binded, dim=1))
        recons[feature_number] = recon_donor_like[0]
    paired_ae.logger.experiment.log({
        paired_ae.dataset_info.feature_names[feature_number]: wandb.Image(
            recons[feature_number]) for feature_number in
        range(paired_ae.cfg.dataset.n_features)}
        , commit=True)
