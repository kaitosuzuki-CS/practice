import argparse
import os
from pathlib import Path

import torch

from env import create_environment
from model.sac import SAC_CURL
from utils.crop import Crop
from utils.hps import load_hps
from utils.misc import save_video

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


def infer(env, agent, crop, action_spec, device):
    done = False
    state, reward, done, _ = env.reset()

    frames = []
    while not done:
        frames.append(state[-3:])
        with torch.no_grad():
            _state = torch.as_tensor(state, device=device).unsqueeze(0).float() / 255.0
            _state = crop.center_crop(_state)
            action = agent.select_action(_state, eval=True)
            action = action.detach().cpu().numpy().astype(action_spec.dtype)

        next_state, reward, done, _ = env.step(action)

        state = next_state

    frames.append(state[-3:])

    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="cheetah", help="Task name (Cheetah or Walker)"
    )
    parser.add_argument(
        "--ckpt", type=str, default="final.pt", help="Checkpoint path for evaluation"
    )

    args = parser.parse_args()
    task = args.task.lower()
    ckpt_path = args.ckpt

    config_file = os.path.join(parent_dir, "configs", f"{task}_config.json")
    model_hps, train_hps = load_hps(config_file)

    env = create_environment(
        train_hps.domain_name,  # type: ignore
        train_hps.task_name,  # type: ignore
        train_hps.action_repeat,  # type: ignore
        train_hps.frame_stack,  # type: ignore
        train_hps.image_size,  # type: ignore
    )
    action_spec = env.action_spec()
    agent = SAC_CURL(model_hps, train_hps, action_spec, device)
    crop = Crop(train_hps)

    ckpt_file = torch.load(ckpt_path, map_location=device)
    agent.init_weights(ckpt_file)

    frames = infer(env, agent, crop, action_spec, device)
    save_dir = os.path.join(parent_dir, task)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "inference.mp4")
    save_video(frames, save_path)
