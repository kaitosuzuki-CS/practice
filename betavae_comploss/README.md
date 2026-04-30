# Beta-VAE & Composite VAE-GAN on MNIST

This project implements a **Beta-Variational Autoencoder (Beta-VAE)** and a **Composite VAE-GAN** (VAE with Adversarial and Perceptual losses) using **PyTorch**. It serves as an experimental framework to compare standard VAE reconstruction quality against a hybrid approach that leverages a PatchGAN discriminator and Perceptual Image Patch Similarity (PIPS) loss for sharper generations.

The model is trained on the **MNIST** dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Project Files](#project-files)

## Project Overview

The project supports two primary training schemes defined in `training_schemes.py`:

1.  **Simple Scheme (Beta-VAE)**:
    - Standard VAE architecture with MLP Encoder/Decoder.
    - Optimizes a loss combining Reconstruction Loss (MSE) and KL Divergence (weighted by $\beta$).
    - Focuses on learning a disentangled latent representation.

2.  **Composite Scheme (VAE-GAN)**:
    - Uses the same VAE generator but adds a **PatchDiscriminator** as a critic.
    - Optimizes a composite objective: Reconstruction Loss + KL Divergence + Adversarial Loss + Perceptual Loss (PIPS).
    - Aims to produce sharper, more realistic image samples by combating the blurriness often associated with pure MSE losses.

## Project Structure

```
betavae/
├── checkpoints/          # Saved model weights
├── configs/              # JSON configuration files for models and training
├── data/                 # MNIST dataset storage
├── model/                # Neural network architectures
│   ├── components.py     # Building blocks (Linear, Conv blocks)
│   ├── decoder.py        # Decoder network
│   ├── discriminator.py  # PatchDiscriminator for composite scheme
│   ├── encoder.py        # Encoder network
│   ├── pips.py           # Perceptual Loss logic
│   └── vae.py            # Main VAE class
├── samples/              # Generated sample images
├── utils/                # Utility scripts
│   ├── dataset.py        # MNIST dataloader setup
│   ├── loss.py           # Loss function implementations
│   └── misc.py           # Miscellaneous helpers
├── infer.py              # Inference script for generating samples
├── train.py              # Main training entry point
└── training_schemes.py   # Training loops for simple/composite modes
```

## Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: PyTorch, Torchvision
- **Computation**: NumPy
- **Visualization**: Matplotlib
- **Data**: MNIST (Automatically downloaded via Torchvision)

## Getting Started

### Installation

1.  Clone the repository:

    ```bash
    git clone --filter=blob:none --sparse https://github.com/kaitosuzuki-CS/practice.git
    cd practice
    git sparse-checkout set betavae_comploss
    cd betavae_comploss
    ```

2.  Set up the environment:

    **Using Conda (recommended):**

    ```bash
    conda env create -f environment.yml
    conda activate betavae_comploss
    ```

    **Using Pip:**

    ```bash
    conda create -n betavae_comploss python=3.14
    conda activate betavae_comploss
    pip install -r requirements.txt
    ```

### Training

You can train the model in either `simple` or `composite` mode using the JSON configurations in `configs/`.

**1. Train Simple Beta-VAE:**

```bash
python train.py \
    --model_config configs/model_simple_config.json \
    --train_config configs/train_simple_config.json \
    --scheme simple
```

**2. Train Composite VAE-GAN:**

```bash
python train.py \
    --model_config configs/model_composite_config.json \
    --train_config configs/train_composite_config.json \
    --scheme composite
```

_Checkpoints will be saved to the `checkpoints/` directory._

### Inference

Generate sample images using a trained checkpoint. The script will output a grid of generated digits by traversing the latent space.

```bash
python infer.py \
    --model_config configs/model_simple_config.json \
    --ckpt_path checkpoints/final_model.pth
```

_(Replace `checkpoints/final_model.pth` with your actual checkpoint path)._

## Results

Comparison of generated samples from the two training schemes.

### Simple Scheme (Standard Beta-VAE)

_Standard VAEs often produce slightly blurry reconstructions due to the MSE loss._

![Simple Samples](samples/generated%20samples%20without%20composite%20loss.png)

### Composite Scheme (VAE-GAN)

_The composite loss aims to sharpen edges and improve perceptual quality._

![Composite Samples](samples/generated%20samples%20with%20composite%20loss.png)

## Project Files

- **`infer.py`**: Loads a checkpoint and generates a grid of images from the latent space.
- **`train.py`**: Entry point for training. Parses args and calls the appropriate routine from `training_schemes.py`.
- **`training_schemes.py`**: Contains the `train_with_composite` and `train_without_composite` functions, implementing the distinct training loops.
- **`configs/`**: Directory containing JSON configuration files for models and training.
  - **`configs/model_composite_config.json`**: Model configuration for the composite scheme.
  - **`configs/model_simple_config.json`**: Model configuration for the simple scheme.
  - **`configs/train_composite_config.json`**: Training configuration for the composite scheme.
  - **`configs/train_simple_config.json`**: Training configuration for the simple scheme.
- **`model/`**: Directory containing neural network architectures.
  - **`model/components.py`**: Building blocks (Linear, Conv blocks) for the neural networks.
  - **`model/decoder.py`**: Decoder network architecture.
  - **`model/discriminator.py`**: Defines `PatchDiscriminator`, used only in the composite scheme.
  - **`model/encoder.py`**: Encoder network architecture.
  - **`model/pips.py`**: Perceptual Loss (PIPS) logic implementation.
  - **`model/vae.py`**: Defines the `VAE` class, combining `Encoder` and `Decoder`.
- **`utils/`**: Directory for utility scripts.
  - **`utils/dataset.py`**: MNIST dataloader setup.
  - **`utils/loss.py`**: Implements `SimpleLoss` (MSE + KLD) and `CompositeLoss` (MSE + KLD + Adv + PIPS).
  - **`utils/misc.py`**: Miscellaneous helper functions.
- **`checkpoints/`**: Directory for saved model weights.
- **`data/`**: Directory for MNIST dataset storage.
- **`samples/`**: Directory for generated sample images.
