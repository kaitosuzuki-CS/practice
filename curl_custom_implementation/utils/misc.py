import os

import imageio
import numpy as np
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seeds(hps):
    seed = hps.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_video(frames, save_path):
    frames = np.stack(frames, axis=0)
    frames = frames.transpose(0, 2, 3, 1)

    imageio.mimsave(save_path, frames, fps=30)
