import os

import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class EarlyStopping:
    def __init__(self, patience, tol):
        self.patience = patience
        self.tol = tol
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None

    def step(self, model, loss):
        if loss < self.best_loss - self.tol:
            self.best_loss = loss
            self.counter = 0
            self.best_model = model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False


def set_seeds(hps):
    # Set seeds for reproducibility.
    seed = hps.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
