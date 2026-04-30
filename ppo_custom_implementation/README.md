# PPO Custom Implementation

## Introduction

This repository contains a custom implementation of the **Proximal Policy Optimization (PPO)** algorithm, introduced in the paper _"Proximal Policy Optimization Algorithms"_ (Schulman et al., 2017) [[paper](https://arxiv.org/pdf/1707.06347)]. This implementation is designed to train reinforcement learning agents in continuous control environments using the **DeepMind Control Suite (`dm_control`)**. The implementation is built from scratch using **PyTorch**.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application Info](#application-info)
- [Getting Started](#getting-started)
- [Project Files](#project-files)
- [Reference](#reference)

## Project Overview

The goal of this project is to provide a clean, understandable, and modifiable implementation of PPO. It supports continuous action spaces and utilizes the `dm_control` suite for simulation. The implementation features:

- **PPO Algorithm**: A robust on-policy reinforcement learning algorithm.
- **Actor-Critic Architecture**: Separate networks for policy (actor) and value function (critic).
- **Continuous Control**: tailored for environments with continuous action spaces (e.g., Mujoco physics).
- **Hyperparameter Configuration**: easy tuning via YAML configuration files.

## Project Structure

```
PPO_Custom_Implementation/
├── configs/
│   └── config.yml          # Hyperparameters for training and model
├── PPO/
│   ├── __init__.py
│   ├── network.py           # Neural network definitions (Actor, Critic)
│   └── ppo.py               # Main PPO algorithm logic
├── environment.yml          # Conda environment specification
├── test.py                  # Simple testing script
├── train.py                 # Main entry point for training
├── utils.py                 # Utility functions (config loading)
└── README.md                # Project documentation
```

## Tech Stack

- **Language**: Python 3.12
- **Deep Learning**: PyTorch
- **Simulation**: DeepMind Control Suite (`dm_control`), Mujoco
- **Configuration**: YAML

## Application Info

### PPO Implementation

The core logic resides in `PPO/ppo.py`. The `PPO` class handles:

- **Rollout**: Collecting trajectories (observations, actions, rewards) from the environment.
- **Advantage Estimation**: Using Generalized Advantage Estimation (GAE).
- **Optimization**: Updating the actor and critic networks using the clipped surrogate objective.

### Neural Networks

Defined in `PPO/network.py`:

- **Actor**: A feed-forward network that outputs a Normal distribution (mean and log-variance) for continuous actions.
- **Critic**: A feed-forward network that estimates the value function $V(s)$.

### Configuration

Hyperparameters are managed in `configs/config.yml`, allowing you to adjust learning rates, batch sizes, discount factors, and model dimensions without changing code.

## Getting Started

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation

1.  **Clone the repository**:

    ```bash
    git clone --filter=blob:none --sparse https://github.com/kaitosuzuki-CS/practice.git
    cd practice
    git sparse-checkout set ppo_custom_implementation
    cd ppo_custom_implementation
    ```

2.  **Install dependencies**:

    **Option 1: Using Conda**

    ```bash
    conda env create -f environment.yml
    conda activate ppo
    ```

    **Option 2: Using Pip**

    ```bash
    conda create -n ppo python=3.12
    conda activate ppo
    pip install -r requirements.txt
    ```

### Usage

To start training an agent, use the `train.py` script. You can specify the domain and task name from the `dm_control` suite.

**Basic Usage:**

```bash
python train.py --config_path configs/config.yml
```

**Custom Environment:**

```bash
python train.py --domain_name walker --task_name walk --config_path configs/config.yml
```

- `--domain_name`: The domain of the environment (default: `cheetah`).
- `--task_name`: The specific task within the domain (default: `run`).
- `--config_path`: Path to the YAML configuration file (required).

## Project Files

- **`train.py`**: The main execution script. It sets up the environment, loads the configuration, initializes the PPO agent, and starts the training loop.
- **`configs/config.yml`**: Contains all hyperparameters for the model (hidden dimensions) and training (learning rates, timesteps, batch sizes, etc.).
- **`PPO/ppo.py`**: The heart of the implementation. Contains the `PPO` class which implements the training loop, rollout data collection, advantage calculation (GAE), and network updates.
- **`PPO/network.py`**: Defines the PyTorch `Module` classes for the `Actor` and `Critic` networks. It uses orthogonal initialization for weights.
- **`utils.py`**: Helper functions, primarily for loading and parsing the YAML configuration into a Python object.
- **`test.py`**: A scratchpad script for testing basic NumPy operations (not part of the main training pipeline).
- **`environment.yml`**: Defines the dependencies required to run the project.

## Reference

- **Proximal Policy Optimization Algorithms** (Schulman et al., 2017): [https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)
