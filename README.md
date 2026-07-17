# MJEPA CIFAR-10

Scripts for training a ViT model on CIFAR-10 using [MJEPA](https://github.com/TidalPaladin/mjepa).

## Requirements

- Python 3.11 through 3.14
- uv 0.11.28
- An NVIDIA Turing-generation or newer GPU for the default CUDA 13 PyTorch wheels
- NVIDIA driver 580.65.06 or newer on Linux
- A Weights & Biases account for training logs

## Setup

1. Run `make init` to create the virtual environment from `uv.lock`.
2. Run `wandb login`.

Use `make update` after changing a dependency pin. This upgrades `uv.lock` and synchronizes every dependency group.

## Development checks

- `make check` runs Ruff checks, basedpyright, and the CI-safe test suite without rewriting files.
- `make style` applies Ruff lint and formatting fixes.
- `make test-ci` runs the CI-safe tests and writes `coverage.xml`.
- `uv run pytest -m ci_skip` runs the multi-GPU tests when at least two CUDA devices are available.

## Training

To run model training:

1. Create your training configuration:

   ```bash
   cp Makefile.config.template Makefile.config
   ```

2. Edit `Makefile.config` with your training parameters (data path, device, etc.)
3. Run pretraining:

   ```bash
   make train          # runs distributed or single GPU based on NUM_TRAINERS
   make train-single   # forces single GPU training
   ```

4. Run finetuning from a backbone checkpoint:

   ```bash
   make finetune        # requires CONFIG to point at config/finetune/* and CHECKPOINT to a backbone.safetensors file
   make finetune-single # forces single GPU finetuning
   ```
