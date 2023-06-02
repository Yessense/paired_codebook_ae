# Symbolic Disentangled Representations in Hyperdimensional Latent Space

## Dataset

dSprites datasets are stored in the directory `/data/`

[link](https://mega.nz/file/LlQQnZgA#c34zSV1UXM4NeX31T5Z22F3HPP0TdQeDXGUq5d0BC8c) to download CLEVR Paired dataset

## Models

All model architectures are located in the directory `./paired_codebook_ae/models/`

## Training
To run experiments, run the following command:

```bash
python3 -m paired_codebook_ae.train
```

The default hyperparameters for experiments are stored in ./paired_book_ae/config.py or ./conf/config.yaml. You can also pass these hyperparameters via the command line

To run a set of experiments, you can use the wandb sweep function for example:

```bash
wandb sweep sweep.yaml

# or

wandb sweep sweeps/latent_dim_sweep.yaml
wandb sweep sweeps/seed_sweep.yaml
```

For convenience, the different model configurations are combined into setup configurations. If you change parameters in the config.py file these can be :

    - PairedAEDspritesSetupConfig
    - PairedAEClevrSetupConfig
    - BetaVAESetupConfig
    - FactorVAESetupConfig

## Inference

To calculate metrics, run the following command:

```bash
python3 -m paired_codebook_ae.calculate_metrics
python3 -m paired_codebook_ae.calculate_our_metric
```
