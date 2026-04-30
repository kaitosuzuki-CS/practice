# Denoising Diffusion Probabilistic Model (DDPM)

## Project Overview

This project is an implementation of a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch. The model is trained to generate images by reversing a gradual noising process. This implementation provides scripts for training the model on different datasets (e.g., MNIST, CIFAR-100) and for generating images using a trained model.

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application](#application)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Using Custom Datasets](#using-custom-datasets)
- [Project Files](#project-files)

## Project Structure

```
/
├── checkpoints/
│   ├── CIFAR100/
│   └── MNIST/
├── config/
│   ├── cifar100_config.json
│   └── mnist_config.json
├── model/
│   ├── blocks/
│   ├── components/
│   ├── layers/
│   ├── models/
│   └── main.py
├── samples/
│   ├── CIFAR100/
│   │   └── README.md
│   └── MNIST/
│       └── README.md
├── utils/
│   ├── dataset.py
│   ├── hps.py
│   ├── loss.py
│   └── misc.py
├── .gitignore
├── infer.py
├── requirements_conda.txt
├── requirements_pip.txt
├── test.py
└── train.py
```

## Tech Stack

- Python 3
- PyTorch
- Transformers (for learning rate scheduler)

## Application

The application consists of two main functionalities:

1.  **Training**: The `train.py` script trains the DDPM model on a specified dataset. It handles the training loop, validation, saving checkpoints, and early stopping.
2.  **Inference**: The `infer.py` script uses a trained model to generate images. It loads a model checkpoint and performs the reverse diffusion process to create images from noise.

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (for CUDA setup) or pip

### Installation

You can set up the project using either Conda (recommended for CUDA) or pip.

#### Using Conda (with CUDA support)

1.  **Clone the repository:**

    ```bash
    git clone --filter=blob:none --sparse https://github.com/kaitosuzuki-CS/practice.git
    cd practice
    git sparse-checkout set ddpm
    cd ddpm
    ```

2.  **Create and activate a conda environment from `requirements_conda.txt`:**
    ```bash
    conda env create -f requirements_conda.txt
    conda activate ddpm
    ```

#### Using pip

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaitosuzuki-CS/Practice.git
    cd Practice/DDPM
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies from `requirements_pip.txt`:**
    ```bash
    pip install -r requirements_pip.txt
    ```

### Running the Application

After installation, you can train the model and run inference as follows:

#### Train the model:

To train the model on the MNIST dataset:

```bash
python train.py --data mnist
```

To train the model on the CIFAR-100 dataset:

```bash
python train.py --data cifar100
```

Checkpoints will be saved in the `checkpoints` directory.

#### Run inference:

Before running inference, make sure you have a trained model checkpoint. Update the `ckpt_path` in the corresponding config file (`config/mnist_config.json` or `config/cifar100_config.json`) to point to your trained model.

To generate images using the MNIST model:

```bash
python infer.py --data mnist
```

To generate images using the CIFAR-100 model:

```bash
python infer.py --data cifar100
```

Generated images will be saved in the `samples` directory.

## Using Custom Datasets

To train and infer with your own dataset, follow these steps:

### 1. Prepare Your Data

Organize your custom dataset in a directory structure that is easy to access. For image datasets, a common approach is to have all images in a single folder, or organized into subfolders if you need class labels (though for unconditional DDPMs, class labels might not be strictly necessary during training).

### 2. Create a Custom Configuration File

1.  **Duplicate an existing config file:** Start by copying either `config/mnist_config.json` or `config/cifar100_config.json` as a template. Name it appropriately, e.g., `config/my_dataset_config.json`.

    ```bash
    cp config/mnist_config.json config/my_dataset_config.json
    ```

2.  **Edit the new config file (`config/my_dataset_config.json`):**
    Open `config/my_dataset_config.json` and modify the following parameters to match your dataset:
    - `data_path`: Specify the path to your custom dataset directory.
    - `image_size`: Set the desired image dimensions (e.g., `64` for 64x64 pixels). All images will be resized to this.
    - `in_channels`: Number of color channels in your images (e.g., `1` for grayscale, `3` for RGB).
    - `num_classes`: If your dataset has classes and you plan to use a conditional DDPM, set this. For unconditional generation, you might keep it at `0` or adjust as needed.
    - Review other hyperparameters like `batch_size`, `learning_rate`, `epochs`, `noise_steps`, `beta_start`, `beta_end`, etc., and adjust them for your specific dataset and training requirements.
    - `name`: Change the name to reflect your custom dataset, e.g., `"my_dataset"`.

### 3. Update `utils/dataset.py` for Custom Data Loading

The `utils/dataset.py` file contains the logic for loading datasets. You will need to modify this file to correctly load your custom dataset.

1.  **Open `utils/dataset.py`**.
2.  **Add a new `Dataset` class**: Inherit from `torch.utils.data.Dataset` and implement `__init__`, `__len__`, and `__getitem__`.

- In `__init__`, load your images (e.g., using `torchvision.datasets.ImageFolder` if your data is organized by classes, or `glob` to find all image files in a directory). - In `__getitem__`, load an image, apply transformations (resizing, normalization, tensor conversion), and return it. - Example structure (conceptual):

      ```python
      import os
      from PIL import Image
      from torch.utils.data import Dataset
      from torchvision import transforms

      class MyCustomDataset(Dataset):
          def __init__(self, root_dir, image_size):
              self.root_dir = root_dir
              self.image_size = image_size
              self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] # Adjust extensions as needed
              self.transform = transforms.Compose([
                  transforms.Resize(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
              ])

          def __len__(self):
              return len(self.image_paths)

          def __getitem__(self, idx):
              img_path = self.image_paths[idx]
              image = Image.open(img_path).convert("L" if self.transform.transforms[2].mean[0] == 0.5 and len(self.transform.transforms[2].mean) == 1 else "RGB") # Convert to grayscale if 1 channel, else RGB
              return self.transform(image)

      ```

3.  **Add a `create_<custom dataset>_dataset` function**: This function should initialize the custom dataset and return a torch.utils.data.DataLoader object. It should return both a training set and a validation set.

4.  **Modify create_dataset function**: Add a new elif statement to call the `create_<custom dataset>_dataset` function with proper arguments.

### 4. Train the Model with Your Custom Dataset

Once your config file and `utils/dataset.py` are set up, you can train your model:

```bash
python train.py --data <name in load_dataset function in create_dataset function in dataset.py>
```

Replace `config/my_dataset_config.json` with the path to your custom configuration file. The training script will automatically load the hyperparameters and dataset settings from this file.

### 5. Run Inference with Your Custom Dataset

After training, you can use your trained model to generate images.

1.  **Update `ckpt_path` in your custom config file:**
    Open `config/my_dataset_config.json` and ensure that the `ckpt_path` parameter points to the `.pth` file of your trained model checkpoint (e.g., `checkpoints/my_dataset/best_model.pth`).

2.  **Run the inference script:**

    ```bash
    python infer.py --config_path config/my_dataset_config.json
    ```

    Replace `config/my_dataset_config.json` with the path to your custom configuration file. Generated images will be saved in the `samples/my_dataset/imgs` directory (or similar, based on your config's `name`).

## Project Files

- **`train.py`**: The main script for training the DDPM model.
- **`infer.py`**: The main script for generating images using a trained model.
- **`test.py`**: Script for testing the model (if applicable).
- **`config/*.json`**: Configuration files for different datasets, containing hyperparameters for the model, training, and inference.
- **`model/`**: This directory contains the source code for the DDPM model architecture.
  - **`main.py`**: The main model file that defines the DDPM.
  - **`blocks/`**, **`components/`**, **`layers/`**, **`models/`**: These subdirectories contain different modules of the model, such as encoder/decoder blocks, attention mechanisms, and positional embeddings.
- **`utils/`**: This directory contains utility scripts for various tasks.
  - **`dataset.py`**: Script for creating and loading datasets.
  - **`hps.py`**: Script for loading hyperparameters from config files.
  - **`loss.py`**: Defines the loss function used for training.
  - **`misc.py`**: Contains miscellaneous utility functions like early stopping and setting random seeds.
- **`checkpoints/`**: Directory where model checkpoints are saved during training.
- **`samples/`**: Directory where generated images are saved during inference.
- **`.gitignore`**: Specifies which files and directories to ignore in Git version control.
