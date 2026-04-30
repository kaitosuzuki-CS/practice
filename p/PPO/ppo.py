import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
from tqdm import tqdm

from PPO.network import Actor, Critic

parent_dir = Path(__file__).parent.parent


class PPO:
    def __init__(self, env, hps, train_hps, device):
        self._env = env
        self._hps = hps
        self._train_hps = train_hps
        self._device = device

        self._init_hyperparameters()

        self.obs_shape, self.action_shape = self._get_env_info()

        _, _, _, ref_obs = self._env.reset()
        self.next_obs = self._flatten_obs(ref_obs)
        self.next_done = False

        self.actor = Actor(
            self.obs_shape[0], hps.hidden_dim, self.action_shape[0], device
        ).to(device)
        self.critic = Critic(self.obs_shape[0], hps.hidden_dim, 1, device).to(device)

        self.actor_optim = Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas, eps=1e-5  # type: ignore
        )
        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas,  # type: ignore
            eps=1e-5,
        )

    def _init_hyperparameters(self):
        self.num_timesteps_per_rollout = int(self._train_hps.num_timesteps_per_rollout)
        self.max_timesteps_per_episode = int(self._train_hps.max_timesteps_per_episode)
        self.total_timesteps = int(self._train_hps.total_timesteps)

        self.num_epochs_per_rollout = int(self._train_hps.num_epochs_per_rollout)
        self.minibatch_size = int(self._train_hps.minibatch_size)

        self.gamma = float(self._train_hps.gamma)
        self.lam = float(self._train_hps.lam)
        self.eps = float(self._train_hps.eps)
        self.alpha = float(self._train_hps.alpha)

        self.actor_lr = float(self._train_hps.actor_lr)
        self.critic_lr = float(self._train_hps.critic_lr)

        self.actor_betas = tuple(float(x) for x in self._train_hps.actor_betas)
        self.critic_betas = tuple(float(x) for x in self._train_hps.critic_betas)

        self.max_grad_norm = float(self._train_hps.max_grad_norm)

        self.checkpoint_path = os.path.join(parent_dir, self._train_hps.checkpoint_path)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint_interval = int(self._train_hps.checkpoint_interval)

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            keys = sorted(obs.keys())
            return np.concatenate([obs[k].ravel() for k in keys], axis=0)
        return obs

    def _get_env_info(self):
        _, _, _, ref_obs = self._env.reset()
        obs = self._flatten_obs(ref_obs)
        obs_shape = obs.shape

        action_shape = self._env.action_spec().shape

        return obs_shape, action_shape

    def _env_step(self, action):
        timestep = self._env.step(action)
        _, rew, _, obs = timestep
        obs = self._flatten_obs(obs)
        done = timestep.last()

        return rew, obs, done

    def _compute_advantages(self, rewards, values, dones):
        batch_advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        with torch.no_grad():
            next_value = self.critic(
                torch.tensor(self.next_obs, dtype=torch.float32, device=self._device)
            )

        for t in reversed(range(self.num_timesteps_per_rollout)):
            if t == self.num_timesteps_per_rollout - 1:
                nextnonterminal = 1.0 - int(self.next_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            gae = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

            batch_advantages[t] = gae
            lastgaelam = gae

        batch_returns = batch_advantages + values
        return batch_advantages, batch_returns

    def rollout(self):
        batch_obs = torch.zeros(
            (self.num_timesteps_per_rollout, *self.obs_shape),
            dtype=torch.float32,
            device=self._device,
        )
        batch_actions = torch.zeros(
            (self.num_timesteps_per_rollout, *self.action_shape),
            dtype=torch.float32,
            device=self._device,
        )
        batch_log_probs = torch.zeros(
            (self.num_timesteps_per_rollout,), dtype=torch.float32, device=self._device
        )
        batch_rewards = torch.zeros(
            (self.num_timesteps_per_rollout,), dtype=torch.float32, device=self._device
        )
        batch_dones = torch.zeros(
            (self.num_timesteps_per_rollout,), device=self._device
        )
        batch_values = torch.zeros(
            (self.num_timesteps_per_rollout,), dtype=torch.float32, device=self._device
        )

        for t in range(self.num_timesteps_per_rollout):
            batch_obs[t] = torch.tensor(self.next_obs, device=self._device)
            batch_dones[t] = self.next_done

            with torch.no_grad():
                action, log_prob, raw_action = self.actor(batch_obs[t].unsqueeze(0))
                value = self.critic(batch_obs[t].unsqueeze(0))

            batch_values[t] = value.flatten()
            batch_actions[t] = raw_action.flatten()
            batch_log_probs[t] = log_prob.flatten()

            env_action = action.detach().cpu().numpy().flatten()

            rew, obs, done = self._env_step(env_action)

            batch_rewards[t] = rew
            self.next_obs = self._flatten_obs(obs)
            self.next_done = done

            if done:
                _, _, _, next_obs = self._env.reset()
                self.next_obs = self._flatten_obs(next_obs)

        batch_advantages, batch_returns = self._compute_advantages(
            batch_rewards, batch_values, batch_dones
        )

        return (
            batch_obs,
            batch_actions,
            batch_log_probs,
            batch_advantages,
            batch_returns,
            batch_values,
        )

    def train(self):
        for t in tqdm(range(0, self.total_timesteps, self.num_timesteps_per_rollout)):
            (
                batch_obs,
                batch_actions,
                batch_log_probs,
                batch_advantages,
                batch_returns,
                batch_values,
            ) = self.rollout()

            b_inds = np.arange(self.num_timesteps_per_rollout)

            for epoch in range(self.num_epochs_per_rollout):
                np.random.shuffle(b_inds)

                for start in range(
                    0, self.num_timesteps_per_rollout, self.minibatch_size
                ):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = batch_obs[mb_inds]
                    mb_raw_actions = batch_actions[mb_inds]
                    mb_old_log_probs = batch_log_probs[mb_inds]
                    mb_advantages = batch_advantages[mb_inds]
                    mb_returns = batch_returns[mb_inds]

                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                    curr_log_probs, entropy = self.actor.evaluate(
                        mb_obs, mb_raw_actions
                    )
                    new_value = self.critic(mb_obs).squeeze()

                    ratio = (curr_log_probs - mb_old_log_probs).exp()

                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantages
                    )
                    actor_loss = (
                        -torch.min(surr1, surr2).mean() - self.alpha * entropy.mean()
                    )

                    v_loss_unclipped = (new_value - mb_returns) ** 2
                    v_clipped = batch_values[mb_inds] + torch.clamp(
                        new_value - batch_values[mb_inds], -self.eps, self.eps
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    critic_loss = (
                        0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    )

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            if (
                (t // self.num_timesteps_per_rollout) + 1
            ) % self.checkpoint_interval == 0:
                print(f"Global Step: {t}")
                print(
                    f"Policy Loss: {actor_loss.item():.4f}, Value Loss: {critic_loss.item():.4f}"
                )
                print(
                    f"Mean Return: {batch_returns.mean().item():.4f}, Mean Advantage: {batch_advantages.mean().item():.4f}"
                )

                torch.save(
                    {
                        "actor_state_dict": self.actor.state_dict(),
                        "critic_state_dict": self.critic.state_dict(),
                        "actor_optim_state_dict": self.actor_optim.state_dict(),
                        "critic_optim_state_dict": self.critic_optim.state_dict(),
                        "global_step": t,
                    },
                    f"{self.checkpoint_path}/checkpoint_{t}.pth",
                )

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            f"{self.checkpoint_path}/final_model.pth",
        )
