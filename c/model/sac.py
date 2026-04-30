from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from model.actor import Actor
from model.components import Encoder
from model.critic import SoftCritic


class SAC_CURL(nn.Module):
    def __init__(self, hps, train_hps, action_space, device):
        super(SAC_CURL, self).__init__()

        self.gamma = train_hps.gamma
        self.alpha = train_hps.alpha
        self.tau = train_hps.tau
        self.temp = train_hps.temp
        self.batch_size = train_hps.batch_size
        self.update_freq = train_hps.update_freq
        self.device = device

        self.encoder = Encoder(hps.observation_shape, hps.encoder)
        self.key_w = nn.Linear(hps.encoder.output_dim, hps.encoder.output_dim)
        self.encoder_optim = Adam(
            chain(self.encoder.parameters(), self.key_w.parameters()),
            lr=train_hps.lr,
            betas=tuple(train_hps.betas),
        )

        self.target_encoder = Encoder(hps.observation_shape, hps.encoder)

        self.actor = Actor(self.encoder, hps.actor)
        self.actor_optim = Adam(
            self.actor.mlp.parameters(), lr=train_hps.lr, betas=tuple(train_hps.betas)
        )

        self.critic1 = SoftCritic(self.encoder, hps.critic)
        self.critic2 = SoftCritic(self.encoder, hps.critic)
        self.critic1_optim = Adam(
            self.critic1.mlp.parameters(), lr=train_hps.lr, betas=tuple(train_hps.betas)
        )
        self.critic2_optim = Adam(
            self.critic2.mlp.parameters(), lr=train_hps.lr, betas=tuple(train_hps.betas)
        )

        self.target_critic1 = SoftCritic(self.target_encoder, hps.critic)
        self.target_critic2 = SoftCritic(self.target_encoder, hps.critic)

        self.target_entropy = -torch.prod(
            torch.Tensor(action_space.shape).to(device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = Adam(
            [self.log_alpha], lr=train_hps.alpha_lr, betas=tuple(train_hps.alpha_betas)
        )

    def init_weights(self, ckpt=None):
        if ckpt is None:
            self.encoder.init_weights()
            self.actor.init_weights()
            self.critic1.init_weights()
            self.critic2.init_weights()

            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        else:
            self.load_state_dict(ckpt["agent_state_dict"])

        self._freeze_parameters(self.target_encoder)
        self._freeze_parameters(self.target_critic1)
        self._freeze_parameters(self.target_critic2)

    def _freeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _soft_update(self, local_model, target_model):
        for param, target_param in zip(
            local_model.parameters(), target_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def q_forward(self, x, a):
        q1, q2 = self.target_critic1(x, a), self.target_critic2(x, a)

        return torch.min(q1, q2)

    def select_action(self, x, eval=False):
        if eval:
            _, _, mean = self.actor(x)
            return mean
        else:
            action, log_prob, _ = self.actor(x)
            return action, log_prob

    def update_parameters(self, crop, buffer, updates):
        batch = buffer.sample(self.batch_size)
        obs, action, reward, next_obs, done = batch

        obs_q, obs_k = crop.random_crop(obs), crop.random_crop(obs)
        next_obs = crop.random_crop(next_obs)

        mask = 1 - done.unsqueeze(1)
        reward = reward.unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor(next_obs)
            qf1_next_target, qf2_next_target = self.target_critic1(
                next_obs, next_state_action
            ), self.target_critic2(next_obs, next_state_action)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward + (mask * self.gamma * min_qf_next_target)

        qf1, qf2 = self.critic1(obs_q, action), self.critic2(obs_q, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        z_q = self.encoder(obs_q)
        with torch.no_grad():
            z_k = self.target_encoder(obs_k)

        z_k = self.key_w(z_k)
        z_k = F.normalize(z_k, dim=-1)

        logits = torch.matmul(z_q, z_k.T)
        logits = logits / self.temp

        labels = torch.arange(logits.shape[0]).long().to(self.device)
        curl_loss = F.cross_entropy(logits, labels)

        self.encoder_optim.zero_grad()
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        qf_loss.backward()
        curl_loss.backward()
        self.encoder_optim.step()
        self.critic1_optim.step()
        self.critic2_optim.step()

        pi, log_pi, _ = self.actor(obs_q)
        qf1_pi, qf2_pi = self.critic1(obs_q, pi), self.critic2(obs_q, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        if updates % self.update_freq == 0:
            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)
            self._soft_update(self.encoder, self.target_encoder)
