import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST

parent_dir = Path(__file__).resolve().parent.parent


def create_mnist_dataset(save_dir, hps):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = MNIST(root=save_dir, transform=transform, train=True, download=True)
    val_dataset = MNIST(root=save_dir, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=hps.train_bs, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=hps.val_bs, shuffle=False)

    return train_loader, test_loader


def create_cifar100_dataset(save_dir, hps):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = CIFAR100(
        root=save_dir, transform=transform, train=True, download=True
    )
    val_dataset = CIFAR100(
        root=save_dir, transform=transform, train=False, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=hps.train_bs, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=hps.val_bs, shuffle=False)

    return train_loader, test_loader


def create_dataset(dataset, hps):
    save_dir = os.path.join(parent_dir, hps.data_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if dataset == "mnist":
        return create_mnist_dataset(save_dir, hps)
    elif dataset == "cifar100":
        return create_cifar100_dataset(save_dir, hps)

    raise ValueError("No available dataset")
