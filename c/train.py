import argparse
import os
from pathlib import Path

import numpy as np
import torch

from env import create_environment
from model.sac import SAC_CURL
from utils.crop import Crop
from utils.hps import load_hps
from utils.misc import save_video, set_seeds
from utils.replay_buffer import ReplayBuffer

parent_dir = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(action_spec, env, agent, buffer, crop, train_hps, device, ckpt=None):
    save_dir = os.path.join(parent_dir, train_hps.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if ckpt is not None:
        ckpt = os.path.join(parent_dir, ckpt)
        assert os.path.exists(ckpt), f"{ckpt} does not exist"

        ckpt = torch.load(ckpt)
        agent.init_weights(ckpt)

        total_numsteps = ckpt["total_numsteps"] + 1
        num_episodes = ckpt["num_episodes"]
        updates = ckpt["updates"]
    else:
        agent.init_weights()
        total_numsteps = 0
        num_episodes = 0
        updates = 0

    agent.to(device)

    while total_numsteps < train_hps.total_steps:
        num_episodes += 1

        episode_reward = 0
        episode_steps = 0
        done = False
        state, reward, done, _ = env.reset()

        while not done:
            if total_numsteps < train_hps.warmup_steps:
                action = np.random.uniform(
                    action_spec.minimum, action_spec.maximum, size=action_spec.shape
                ).astype(action_spec.dtype)
            else:
                with torch.no_grad():
                    _state = (
                        torch.as_tensor(state, device=device).unsqueeze(0).float()
                        / 255.0
                    )
                    _state = crop.random_crop(_state)
                    action, _ = agent.select_action(_state)
                    action = action.detach().cpu().numpy().astype(action_spec.dtype)

            next_state, reward, done, _ = env.step(action)

            buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_numsteps += 1
            episode_steps += 1

            if buffer.size > train_hps.start_training_steps:
                updates += 1
                agent.update_parameters(crop, buffer, updates)

                if updates % 10000 == 0:
                    torch.save(
                        {
                            "agent_state_dict": agent.state_dict(),
                            "total_numsteps": total_numsteps,
                            "num_episodes": num_episodes,
                            "updates": updates,
                        },
                        f"{save_dir}/checkpoint_{updates}.pt",
                    )

                    # buffer.save(os.path.join(save_dir, 'buffer.pt'))
            if total_numsteps >= train_hps.total_steps:
                break

        print(
            f"Epsiode: {num_episodes}, Reward: {episode_reward}, Steps: {episode_steps}, Total Steps: {total_numsteps}"
        )

    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "total_numsteps": total_numsteps,
            "num_episodes": num_episodes,
            "updates": updates,
        },
        f"{save_dir}/final.pt",
    )

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for CURL training")
    parser.add_argument(
        "--task", type=str, default="cheetah", help="Task name (Cheetah or Walker)"
    )

    args = parser.parse_args()
    task = args.task.lower()

    config_file = os.path.join(parent_dir, "configs", f"{task}_config.json")
    model_hps, train_hps = load_hps(config_file)

    set_seeds(train_hps)

    env = create_environment(
        train_hps.domain_name,  # type: ignore
        train_hps.task_name,  # type: ignore
        train_hps.action_repeat,  # type: ignore
        train_hps.frame_stack,  # type: ignore
        train_hps.image_size,  # type: ignore
    )
    action_spec = env.action_spec()
    agent = SAC_CURL(model_hps, train_hps, action_spec, device)
    buffer = ReplayBuffer(
        train_hps.buffer_capacity,  # type: ignore
        model_hps.observation_shape,  # type: ignore
        action_spec.shape,  # type: ignore
        device,
    )
    crop = Crop(train_hps)

    train(action_spec, env, agent, buffer, crop, train_hps, device)
