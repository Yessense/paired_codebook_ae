import torch
from paired_codebook_ae.model.paired_ae.model import VSADecoder
from .config import MainConfig, PairedAEClevrSetupConfig, PairedAEDspritesSetupConfig, PairedAEModelConfig, PairedClevrDatasetConfig, PairedDspritesDatasetConfig, \
    PairedClevrDatamoduleConfig
from .dataset.dsprites import Dsprites
from .dataset import PairedDspritesDatamodule, PairedClevrDatamodule
from .callbacks.logger import GeneralizationVisualizationCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
import wandb
import hydra
import os
import pathlib
import sys
from typing import Union

from hydra.utils import instantiate

# from paired_codebook_ae.model.beta_vae_model import VAEXperiment

sys.path.append("..")


cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)

cs.store(group="config.setup", name="paired_dsprites_setup",
         node=PairedAEDspritesSetupConfig)
cs.store(group="config.setup", name="paired_clevr_setup",
         node=PairedAEClevrSetupConfig)

cs.store(group="model", name="paired_model", node=PairedAEModelConfig)

cs.store(group="dataset.datamodule", name="paired_clevr",
         node=PairedClevrDatamoduleConfig)
cs.store(group="dataset", name="paired_clevr", node=PairedClevrDatasetConfig)
cs.store(group="dataset", name="paired_dsprites",
         node=PairedDspritesDatasetConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: MainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.setup.experiment.seed)

    datamodule: pl.LightningDataModule = instantiate(
        cfg.setup.dataset.datamodule, batch_size=cfg.setup.experiment.batch_size)

    ckpt = '/home/akorchemnyi/paired_codebook_ae/paired_codebook_ae/clevr_our_weights/best-epoch=599.ckpt'
    # ckpt = torch.load(cfg.setup.checkpoint.ckpt_path)
    ckpt = torch.load(ckpt)

    # Model
    model = instantiate(cfg.setup.model.model_class, cfg.setup, datamodule)
    model.load_state_dict(ckpt['state_dict'])

    # Logger
    wandb_logger = WandbLogger(
        project=cfg.setup.dataset.datamodule.mode + '_our_metric',
        name=f'{cfg.setup.dataset.datamodule.mode} -l {cfg.setup.model.latent_dim} '
             f'-s {cfg.setup.experiment.seed} '
             f'-bs {cfg.setup.experiment.batch_size} '
             f'vsa',
        save_dir=cfg.setup.experiment.logging_dir)

    # Move wandb cache dir to experiment dir
    os.environ['WANDB_CACHE_DIR'] = os.path.join(
        cfg.setup.experiment.logging_dir, 'cache')
    # wandb_logger.watch(
    #     model, log_graph=cfg.setup.experiment.log_training_graph)


    # top_metric_callback = ModelCheckpoint(monitor=cfg.setup.model.monitor,
    #                                       filename='best-{epoch}',
    #                                       save_top_k=cfg.setup.checkpoint.save_top_k)
    # every_epoch_callback = ModelCheckpoint(every_n_epochs=cfg.setup.checkpoint.every_k_epochs,
    #                                        filename='last-{epoch}')

    # # Learning rate monitor
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # # gen_viz_callback = GeneralizationVisualizationCallback(
    # #     path_to_data_dir=cfg.dataset.path_to_dataset + '/dsprites/dsprites.npz')

    # callbacks = [

    #     # gen_viz_callback,
    #     top_metric_callback,
    #     every_epoch_callback,
    #     lr_monitor,
    # ]
    # Train process
    trainer = pl.Trainer(**cfg.setup.experiment.trainer,
                         logger=wandb_logger,)
                        #  callbacks=callbacks,
                        #  gradient_clip_val=cfg.setup.experiment.gradient_clip)

    trainer.test(model,
                 datamodule=datamodule)

    print(f"Trained. Logging dir: {cfg.setup.experiment.logging_dir}")

if __name__ == '__main__':
    run()
