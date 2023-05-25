from dataclasses import dataclass, field
from typing import Any, Tuple, Optional, List, Union

from omegaconf import MISSING

import paired_codebook_ae.dataset
from paired_codebook_ae.dataset.paired_dsprites import Dsprites


# ----------------------------------------------------------------
# Base Dataset configs
# ----------------------------------------------------------------

@dataclass
class BaseDatamoduleConfig:
    _target_: str = MISSING
    batch_size: int = field(
        default_factory=lambda: BaseExperimentConfig.batch_size)
    mode: str = MISSING


@dataclass
class BaseDatasetConfig:
    n_features: int = MISSING
    datamodule: BaseDatamoduleConfig = MISSING
    train_size: int = MISSING
    val_size: int = MISSING
    image_size: Tuple[int, int, int] = MISSING
    requires_fid: bool = False

# ----------------------------------------------------------------
# Base Model configuration
# ----------------------------------------------------------------


@dataclass
class BaseModelClassConfig:
    _target_: str = MISSING


@dataclass
class BaseModelConfig:
    model_class: Any = MISSING

# ----------------------------------------------------------------
# Base Experiment Config
# ----------------------------------------------------------------


@dataclass
class BaseTrainerConfig:
    accelerator: str = 'gpu'
    max_epochs: int = 600
    devices: List[int] = field(default_factory=lambda: [2])
    profiler: Optional[str] = None
    check_val_every_n_epoch: int = 100


@dataclass
class BaseExperimentConfig:
    seed: int = 0
    batch_size: int = 64
    gradient_clip: float = 0.0
    logging_dir: str = "${hydra:run.dir}/"
    log_training_graph: bool = True
    trainer: BaseTrainerConfig = field(default_factory=BaseTrainerConfig)


# ----------------------------------------------------------------
# Base Checkpoints Config
# ----------------------------------------------------------------


@dataclass
class BaseCheckpointsConfig:
    save_top_k: int = 1
    every_k_epochs: int = 5
    check_val_every_n_epochs: int = 50
    ckpt_path: Optional[str] = None

# ----------------------------------------------------------------
# Base Setup Config
# ----------------------------------------------------------------


@dataclass
class BaseSetupConfig:
    model: BaseModelConfig = MISSING
    dataset: BaseDatasetConfig = MISSING
    experiment: BaseExperimentConfig = MISSING
    checkpoint: BaseCheckpointsConfig = MISSING

# ----------------------------------------------------------------
# Paired AE Datamodule
# ----------------------------------------------------------------

# Paired Clevr


@dataclass
class PairedClevrDatamoduleConfig(BaseDatamoduleConfig):
    _target_: str = 'paired_codebook_ae.dataset.PairedClevrDatamodule'
    path_to_data_dir: str = "${hydra:runtime.cwd}/data/"
    mode: str = 'paired_clevr'
    num_workers: int = 4


@dataclass
class PairedClevrDatasetConfig(BaseDatasetConfig):
    datamodule: BaseDatamoduleConfig = field(
        default_factory=PairedClevrDatamoduleConfig)
    train_size: int = 10_000
    val_size: int = 1_000
    n_features: int = 6
    image_size: Tuple[int, int, int] = (3, 128, 128)
    requires_fid: bool = True

# Paired Dsprites


@dataclass
class PairedDspritesDatamoduleConfig(BaseDatamoduleConfig):
    _target_: str = 'paired_codebook_ae.dataset.PairedDspritesDatamodule'
    path_to_data_dir: str = "${hydra:runtime.cwd}/data/"
    mode: str = 'paired_dsprites'


@dataclass
class PairedDspritesDatasetConfig(BaseDatasetConfig):
    datamodule: BaseDatamoduleConfig = field(
        default_factory=PairedDspritesDatamoduleConfig)
    train_size: int = 100_000
    val_size: int = 30_000
    n_features: int = 5
    image_size: Tuple[int, int, int] = (1, 64, 64)
    requires_fid: bool = False


# ----------------------------------------------------------------
# Paired AE config
# ----------------------------------------------------------------

@dataclass
class PairedAETrainerConfig(BaseTrainerConfig):
    pass


@dataclass
class PairedAECheckpointsConfig(BaseCheckpointsConfig):
    pass


@dataclass
class PairedAEClassConfig(BaseModelClassConfig):
    _target_: str = 'paired_codebook_ae.model.paired_ae.model.VSADecoder'


@dataclass
class DecoderConfig:
    in_channels: int = 64
    hidden_channels: int = 64


@dataclass
class EncoderConfig:
    hidden_channels: int = 64


@dataclass
class PairedAEModelConfig(BaseModelConfig):
    model_class: PairedAEClassConfig = field(
        default_factory=PairedAEClassConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    latent_dim: int = 1024
    binder: str = "fourier"
    monitor: str = "Validation/Total"


@dataclass
class PairedAEExperimentConfig(BaseExperimentConfig):
    seed: int = 0
    batch_size: int = 64
    gradient_clip: float = 0.0
    logging_dir: str = "${hydra:run.dir}/"
    log_training_graph: bool = True
    trainer: BaseTrainerConfig = field(default_factory=BaseTrainerConfig)
    pct_start: float = 0.2
    lr: float = 0.0001
    steps_per_epoch: int = 0
    scale: str = 'sqrt'
    reduction: str = 'sum'
    trainer: PairedAETrainerConfig = field(
        default_factory=PairedAETrainerConfig)


@dataclass
class PairedAEDspritesSetupConfig(BaseSetupConfig):
    """Paired AE Setup"""
    model: PairedAEModelConfig = field(
        default_factory=PairedAEModelConfig)
    dataset: PairedDspritesDatasetConfig = field(
        default_factory=PairedDspritesDatasetConfig)
    experiment: PairedAEExperimentConfig = field(
        default_factory=PairedAEExperimentConfig)
    checkpoint: PairedAECheckpointsConfig = field(
        default_factory=PairedAECheckpointsConfig)


@dataclass
class PairedAEClevrSetupConfig(BaseSetupConfig):
    """Paired AE Setup"""
    model: PairedAEModelConfig = field(
        default_factory=PairedAEModelConfig)
    dataset: BaseDatasetConfig = field(
        default_factory=PairedClevrDatasetConfig)
    experiment: PairedAEExperimentConfig = field(
        default_factory=PairedAEExperimentConfig)
    checkpoint: PairedAECheckpointsConfig = field(
        default_factory=PairedAECheckpointsConfig)

# ----------------------------------------------------------------
# Classifier 
# ----------------------------------------------------------------
@dataclass
class ClassifierModelClassConfig(BaseModelClassConfig):
    _target_: str = "paired_codebook_ae.model.classifier.model.Classifier"

@dataclass
class ClassifierResnetModelClassConfig(BaseModelClassConfig):
    _target_: str = "paired_codebook_ae.model.classifier.resnet.Classifier"

@dataclass
class ClassifierModelConfig(BaseModelConfig):
    model_class: ClassifierModelClassConfig = field(
        default_factory=ClassifierModelClassConfig)
    latent_dim: int = 1024
    monitor: str = "Validation/Total"

@dataclass
class ClassifierResnetModelConfig(BaseModelConfig):
    model_class: ClassifierResnetModelClassConfig = field(
        default_factory=ClassifierResnetModelClassConfig)
    latent_dim: int = 1024
    monitor: str = "Validation/Total"

@dataclass
class ClassifierExperimentConfig(BaseExperimentConfig):
    seed: int = 0
    batch_size: int = 128
    gradient_clip: float = 0.0
    logging_dir: str = "${hydra:run.dir}/"
    log_training_graph: bool = True
    trainer: BaseTrainerConfig = field(default_factory=BaseTrainerConfig)
    pct_start: float = 0.2
    lr: float = 0.0001
    steps_per_epoch: int = 1
    scale: str = 'sqrt'
    reduction: str = 'sum'
    feature_index: int = 5


@dataclass
class DspritesClassifierSetupConfig(BaseSetupConfig):
    """Paired AE Setup"""
    model: ClassifierModelConfig = field(
        default_factory=ClassifierModelConfig)
    dataset: PairedDspritesDatasetConfig = field(
        default_factory=PairedDspritesDatasetConfig)
    experiment: ClassifierExperimentConfig = field(
        default_factory=ClassifierExperimentConfig)
    checkpoint: PairedAECheckpointsConfig = field(
        default_factory=PairedAECheckpointsConfig)

@dataclass
class ClevrClassifierSetupConfig(BaseSetupConfig):
    """Paired AE Setup"""
    model: ClassifierModelConfig = field(
        default_factory=ClassifierModelConfig)
    dataset: PairedClevrDatasetConfig = field(
        default_factory=PairedClevrDatasetConfig)
    experiment: ClassifierExperimentConfig = field(
        default_factory=ClassifierExperimentConfig)
    checkpoint: PairedAECheckpointsConfig = field(
        default_factory=PairedAECheckpointsConfig)

@dataclass
class ClevrClassifierResnetSetupConfig(BaseSetupConfig):
    """Paired AE Setup"""
    model: ClassifierResnetModelConfig = field(
        default_factory=ClassifierResnetModelConfig)
    dataset: PairedClevrDatasetConfig = field(
        default_factory=PairedClevrDatasetConfig)
    experiment: ClassifierExperimentConfig = field(
        default_factory=ClassifierExperimentConfig)
    checkpoint: PairedAECheckpointsConfig = field(
        default_factory=PairedAECheckpointsConfig)


# ----------------------------------------------------------------
# Beta VAE Config
# ----------------------------------------------------------------

@dataclass
class BetaVAEModelClassConfig(BaseModelClassConfig):
    _target_: str = "paired_codebook_ae.model.beta_vae.beta_vae_model.VAEXperiment"


@dataclass
class BetaVAEModelConfig(BaseModelConfig):
    model_class: BetaVAEModelClassConfig = field(
        default_factory=BetaVAEModelClassConfig)
    in_channels: int = 3
    latent_dim: int = 10
    name: str = 'BetaVAE'
    loss_type: str = 'B'
    gamma: float = 10.0
    max_capacity: int = 25
    Capacity_max_iter: int = 100_000
    monitor: str = "Validation/Total"


@dataclass
class BetaVAETrainerConfig(BaseTrainerConfig):
    accelerator: str = 'gpu'
    max_epochs: int = 600
    devices: List[int] = field(default_factory=lambda: [2])
    profiler: Optional[str] = None
    check_val_every_n_epoch: int = 100


@dataclass
class BetaVAEExperimentConfig(BaseExperimentConfig):
    batch_size: int = 32
    patch_size: int = 64
    num_workers: int = 4
    lr: float = 0.0005
    weight_decay: float = 0.0
    scheduler_gamma: float = 0.995
    kld_weight: float = 0.00025
    trainer: BaseTrainerConfig = field(default_factory=BetaVAETrainerConfig)


@dataclass
class BetaVAECheckpointsConfig(BaseCheckpointsConfig):
    save_top_k: int = 1
    every_k_epochs: int = 5
    check_val_every_n_epochs: int = 100
    ckpt_path: Optional[str] = None


@dataclass
class BetaVAESetupConfig(BaseSetupConfig):
    """Beta VAE """
    model: BetaVAEModelConfig = field(
        default_factory=BetaVAEModelConfig)
    dataset: BaseDatasetConfig = field(
        default_factory=PairedClevrDatasetConfig)
    experiment: BetaVAEExperimentConfig = field(
        default_factory=BetaVAEExperimentConfig)
    checkpoint: BetaVAECheckpointsConfig = field(
        default_factory=BetaVAECheckpointsConfig)

# @dataclass
# class BetaVAEExpParamsConfig:
#     lr: float = 0.005
#     weight_decay: float = 0.0
#     scheduler_gamma: float = 0.995
#     kld_weight: float = 0.00025
#     manual_seed: int = 1265
#     devices: List[int] = field(default_factory=lambda: [0])

@dataclass
class FactorVAEModelClassConfig(BaseModelClassConfig):
    _target_: str = "paired_codebook_ae.model.factor_vae.factor_vae_model.FactorVAEXperiment"


@dataclass
class FactorVAEModelConfig(BaseModelConfig):
    model_class: FactorVAEModelClassConfig = field(
        default_factory=FactorVAEModelClassConfig)
    latent_dim: int = 10
    name: str = 'Factor_VAE'
    gamma: float = 10.0
    monitor: str = "Validation/Total"

@dataclass
class FactorVAEClevrConfig(BaseExperimentConfig):
    opt_g_lr: float = 0.0004
    opt_d_lr: float = 0.0004
    beta1_vae: float = 0.9
    beta2_vae: float = 0.999
    beta1_d: float = 0.5
    beta2_d: float = 0.9

@dataclass
class FactorVAESetupConfig(BaseSetupConfig):
    """Beta VAE """
    model: FactorVAEModelConfig = field(
        default_factory=FactorVAEModelConfig)
    dataset: BaseDatasetConfig = field(
        default_factory=PairedClevrDatasetConfig)
    experiment: BaseExperimentConfig = field(
        default_factory=FactorVAEClevrConfig)
    checkpoint: BetaVAECheckpointsConfig = field(
        default_factory=BetaVAECheckpointsConfig)

# @dataclass
# class FactorVAEDspritesConfig(BaseExperimentConfig):
#     z_dim: int = 10
#     gamma: float = 10
#     opt_g_lr: float = 0.0004
#     opt_d_lr: float = 0.0004
#     beta1_vae: float = 0.9
#     beta2_vae: float = 0.999
#     beta1_d: float = 0.5
#     beta2_d: float = 0.9


@dataclass
class MainConfig:
    setup: BaseSetupConfig = field(default_factory=BetaVAESetupConfig)
    PairedAEDspritesSetupConfig
    PairedAEClevrSetupConfig
    ClevrClassifierSetupConfig
    FactorVAESetupConfig
