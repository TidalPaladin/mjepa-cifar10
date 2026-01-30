# MJEPA CIFAR-10

Scripts for training a ViT model on CIFAR-10 using [MJEPA](https://github.com/TidalPaladin/mjepa) 

## Setup

1. Run `make init` to set up a virtualenv and install dependencies
2. Log into Weights and Biases (used for logging)

## Training

To run model training:

1. Create your training configuration:
   ```bash
   cp Makefile.config.template Makefile.config
   ```
2. Edit `Makefile.config` with your training parameters (data path, device, etc.)
3. Run training:
   ```bash
   make train          # runs distributed or single GPU based on NUM_TRAINERS
   make train-single   # forces single GPU training
   ```
