# CURL: Contrastive Unsupervised Representations for Reinforcement Learning

This repository contains a PyTorch implementation of **CURL** (Contrastive Unsupervised Representations for Reinforcement Learning) combined with **SAC** (Soft Actor-Critic). It is designed to learn control policies from raw pixel observations using data augmentation (random cropping) and contrastive learning to learn robust state representations.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application Info](#application-info)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Training](#running-training)
  - [Running Inference](#running-inference)
- [Project Files](#project-files)

## Project Overview

CURL extracts high-level features from raw images using a shared encoder that is trained simultaneously with the Reinforcement Learning (RL) objective and a contrastive auxiliary task. This approach significantly improves sample efficiency for pixel-based RL.

This implementation specifically targets continuous control tasks from the **DeepMind Control Suite**, with pre-configured settings for `Cheetah` and `Walker` domains.

## Project Structure

```
.
├── configs/                # Hyperparameter configurations (JSON)
│   ├── cheetah_config.json
│   └── walker_config.json
├── model/                  # Neural network architectures
│   ├── actor.py            # Actor network (Policy)
│   ├── components.py       # Encoder definitions
│   ├── critic.py           # Critic network (Q-function)
│   └── sac.py              # Main SAC + CURL agent logic
├── utils/                  # Utility scripts
│   ├── crop.py             # Image cropping/augmentation
│   ├── hps.py              # Hyperparameter loading
│   ├── misc.py             # Miscellaneous utils (seeds, video saving)
│   └── replay_buffer.py    # Experience Replay Buffer
├── env.py                  # Environment setup wrapper
├── infer.py                # Inference/Evaluation script
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
└── SAC_CURL_Notebook.ipynb # Jupyter Notebook version
```

## Tech Stack

- **Python**: Core programming language.
- **PyTorch**: Deep learning framework for building Actor, Critic, and Encoder networks.
- **MuJoCo**: Physics engine for the simulation environments.
- **DeepMind Control Suite (`dm_control`)**: Reinforcement learning environments.
- **NumPy**: Numerical operations.

## Application Info

The core algorithm combines:

1.  **Soft Actor-Critic (SAC)**: An off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.
2.  **CURL**: Uses contrastive learning to enforce consistency between augmented views (random crops) of the same observation.

**Key Components:**

- **Encoder**: A Convolutional Neural Network (CNN) that processes pixel inputs.
- **Data Augmentation**: Random cropping of input images (`100x100` -> `84x84`) is used to generate positive pairs for the contrastive loss.
- **Dual Objectives**: The encoder is updated by both the Critic's Q-learning loss and the CURL contrastive loss.

## Getting Started

### Prerequisites

- Python 3.8+
- **MuJoCo**: You must have MuJoCo installed and configured.

### Installation

You can set up the project using pip.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaitosuzuki-CS/practice.git
    cd Practice/CURL_Custom_Implementation
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt

### Running Training

To train the agent on a specific task (e.g., `cheetah` or `walker`):

```bash
python train.py --task cheetah
```

- **Arguments**:
  - `--task`: The name of the task (default: `cheetah`). Must have a corresponding `configs/<task>_config.json` file.
- **Output**: Checkpoints are saved to `checkpoints/<TaskName>/`.

### Running Inference

To evaluate a trained agent and generate a video of the performance:

```bash
python infer.py --task cheetah --ckpt checkpoints/Cheetah/final.pt
```

- **Arguments**:
  - `--task`: The name of the task (default: `cheetah`).
  - `--ckpt`: Path to the model checkpoint file (default: `final.pt`).
- **Output**: A video file `inference.mp4` will be saved in the `<task>` directory.

## Project Files

- **`train.py`**: The entry point for training. It initializes the environment, agent, and replay buffer, and runs the training loop.
- **`infer.py`**: Loads a pre-trained agent and runs it in the environment to record a video.
- **`model/sac.py`**: Contains the `SAC_CURL` class, which implements the core logic for the agent, including the actor, critic, and contrastive updates.
- **`utils/crop.py`**: Implements the random cropping logic essential for CURL's data augmentation.
- **`utils/replay_buffer.py`**: Efficient storage and sampling of experience tuples `(state, action, reward, next_state, done)`.
- **`configs/*.json`**: Configuration files defining model architecture (hidden dims, layers) and training hyperparameters (learning rate, batch size, etc.).
- **`SAC_CURL_Notebook.ipynb`**: A Jupyter Notebook providing an interactive environment for training and evaluating the SAC+CURL agent, optimized for Google Colab usage.
