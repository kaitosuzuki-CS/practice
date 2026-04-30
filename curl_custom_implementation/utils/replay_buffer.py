import os

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_shape, device):
        self._capacity = capacity
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self.device = device

        self.obs_buf = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.next_obs_buf = np.zeros((capacity, *observation_shape), dtype=np.uint8)

        self.action_buf = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def load(self, path):
        ckpt = torch.load(path)

        self._capacity = ckpt["capacity"]
        self._observation_shape = ckpt["observation_shape"]
        self._action_shape = ckpt["action_shape"]

        self.obs_buf = ckpt["obs_buf"]
        self.next_obs_buf = ckpt["next_obs_buf"]
        self.action_buf = ckpt["action_buf"]
        self.reward_buf = ckpt["reward_buf"]
        self.done_buf = ckpt["done_buf"]

        self.ptr = ckpt["ptr"]
        self.size = ckpt["size"]

    def save(self, save_path):
        torch.save(
            {
                "capacity": self._capacity,
                "observation_shape": self._observation_shape,
                "action_shape": self._action_shape,
                "ptr": self.ptr,
                "size": self.size,
                "obs_buf": self.obs_buf,
                "next_obs_buf": self.next_obs_buf,
                "action_buf": self.action_buf,
                "reward_buf": self.reward_buf,
                "done_buf": self.done_buf,
            },
            save_path,
            pickle_protocol=4,
        )

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self._capacity
        self.size = min(self.ptr + 1, self._capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device).float() / 255.0
        next_obs = (
            torch.as_tensor(self.next_obs_buf[idxs], device=self.device).float() / 255.0
        )

        action = torch.as_tensor(self.action_buf[idxs], device=self.device)
        reward = torch.as_tensor(self.reward_buf[idxs], device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device)

        return obs, action, reward, next_obs, done
