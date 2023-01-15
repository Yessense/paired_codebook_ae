import os
import pathlib
import sys

sys.path.append("..")

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from .utils import find_best_model
from .dataset.generalization_dsprites import GeneralizationDspritesDataModule
from .dataset.paired_dsprites import PairedDspritesDatamodule
from .model.paired_ae import VSADecoder
from .config import VSADecoderConfig

cs = ConfigStore.instance()
cs.store(name="config", node=VSADecoderConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: VSADecoderConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)

    if cfg.dataset.mode == 'paired_dsprites':
        datamodule = PairedDspritesDatamodule(
            path_to_data_dir=cfg.dataset.path_to_dataset,
            batch_size=cfg.experiment.batch_size)
    else:
        raise NotImplemented(f"Wrong dataset mode {cfg.dataset.path_to_dataset!r}")

    cfg.experiment.steps_per_epoch = cfg.dataset.train_size // cfg.experiment.batch_size

    model = VSADecoder(cfg=cfg)

    checkpoints_path = os.path.join(cfg.experiment.logging_dir, "checkpoints")

    top_metric_callback = ModelCheckpoint(monitor=cfg.model.monitor,
                                          dirpath=checkpoints_path,
                                          filename='best-{epoch}',
                                          save_top_k=cfg.checkpoint.save_top_k)
    every_epoch_callback = ModelCheckpoint(every_n_epochs=cfg.checkpoint.every_k_epochs,
                                           filename='last-{epoch}',
                                           dirpath=checkpoints_path)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        top_metric_callback,
        every_epoch_callback,
        lr_monitor,
    ]

    wandb_logger = WandbLogger(
        project=cfg.dataset.mode + '_vsa',
        name=f'{cfg.dataset.mode} -l {cfg.model.latent_dim} '
             f'-s {cfg.experiment.seed} '
             f'-bs {cfg.experiment.batch_size} '
             f'vsa',
        save_dir=cfg.experiment.logging_dir)

    os.environ['WANDB_CACHE_DIR'] = os.path.join(cfg.experiment.logging_dir, 'cache')
    wandb_logger.watch(model)

    # trainer
    trainer = pl.Trainer(accelerator=cfg.experiment.accelerator,
                         devices=cfg.experiment.devices,
                         max_epochs=cfg.experiment.max_epochs,
                         profiler=cfg.experiment.profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         check_val_every_n_epoch=cfg.checkpoint.check_val_every_n_epochs,
                         gradient_clip_val=cfg.experiment.gradient_clip)
    # Train
    trainer.fit(model,
                datamodule=datamodule,
                ckpt_path=cfg.checkpoint.ckpt_path)

    print(f"Trained. Logging dir: {cfg.experiment.logging_dir}")


if __name__ == '__main__':
    main()
