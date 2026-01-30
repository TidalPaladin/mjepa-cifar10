# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MJEPA-CIFAR10 trains a Vision Transformer (ViT) on CIFAR-10 using Masked Joint-Embedding Predictive Architecture (MJEPA) for self-supervised learning with a linear probe for classification evaluation.

## Common Commands

```bash
# Setup
make init              # Initialize venv and install dependencies with uv
cp Makefile.config.template Makefile.config  # Create config (edit with your paths)

# Training
make train             # Run distributed training (NUM_TRAINERS > 1)
make train-single      # Run single GPU training

# Code Quality
make check             # Run all checks (style + quality + types)
make style             # Auto-format code (autoflake, isort, autopep8, black)
make quality           # Check formatting (black --check)
make types             # Run Pyright static type checking (requires Node.js)

# Dependency Management
make update            # Update dependencies to latest versions
make clean             # Remove pycache and compiled files
```

## Architecture

**Entry Point:** `scripts/pretrain.py` - handles DDP initialization, YAML config loading, and orchestrates training.

**Core Package (`mjepa_cifar10/`):**
- `data.py` - CIFAR-10 dataloaders with train/val transforms and distributed sampling support
- `pretrain.py` - `CIFAR10MJEPA` class (MJEPA subclass with linear probe head) and `train()` loop

**Model Flow:**
```
CIFAR-10 Images → ViT-Small/4 Backbone → MJEPA (student/teacher encoders + predictor)
                                       → Probe Head (10-class classification)
                                       → Loss = MJEPA SSL + CrossEntropy
```

**External Dependencies (git repos):**
- `mjepa` - MJEPA implementation from github.com/TidalPaladin/mjepa
- `vit` - Vision Transformer from github.com/TidalPaladin/vit

## Configuration

- **Hyperparameters:** `config/pretrain/vit-small.yaml` contains all model and training settings
- **Runtime Config:** `Makefile.config` (user-created, gitignored) for paths, device, and trainer count

## Key Technical Details

- Uses `uv` as package manager (not pip)
- Logs metrics to Weights & Biases (wandb)
- Saves checkpoints in SafeTensors format
- Supports quantization via torchao (Int8DynamicActivationInt4Weight)
- Black line length: 120, isort line length: 119
