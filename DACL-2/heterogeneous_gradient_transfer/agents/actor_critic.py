from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .actor import GaussianActor
from .critic import ValueCritic


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    done: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    steps: int


class ActorCritic:
    def __init__(self, config, device="cpu"):
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        hidden = tuple(config.get("hidden_sizes", [64, 64]))
        self.actor = GaussianActor(hidden_sizes=hidden).to(self.device)
        self.critic = ValueCritic(hidden_sizes=hidden).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(), "config": self.config}

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def clone(self):
        result = ActorCritic(self.config, self.device)
        result.load_state_dict(copy.deepcopy(self.state_dict()))
        return result

    def collect_rollout(self, env, num_steps, observation=None):
        if observation is None:
            observation, _ = env.reset()
        obs, actions, rewards, terminated, done, values, next_values = [], [], [], [], [], [], []
        for _ in range(num_steps):
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_tensor, _ = self.actor.sample(obs_tensor.unsqueeze(0))
                value = self.critic(obs_tensor.unsqueeze(0)).item()
            action = action_tensor.squeeze(0).cpu().numpy()
            next_observation, reward, term, trunc, _ = env.step(action)
            with torch.no_grad():
                next_value = 0.0 if term else self.critic(torch.as_tensor(
                    next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
            obs.append(observation); actions.append(action); rewards.append(reward)
            terminated.append(float(term)); done.append(float(term or trunc))
            values.append(value); next_values.append(next_value)
            observation = next_observation
            if term or trunc:
                observation, _ = env.reset()
        tensors = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                   for x in (np.asarray(obs), np.asarray(actions), rewards, terminated, done,
                             values, next_values)]
        observations_t, actions_t, rewards_t, terminated_t, done_t, values_t, next_values_t = tensors
        advantages = torch.zeros_like(rewards_t)
        gae = torch.tensor(0.0, device=self.device)
        gamma, lam = self.config["gamma"], self.config["gae_lambda"]
        for index in reversed(range(num_steps)):
            delta = rewards_t[index] + gamma * (1.0 - terminated_t[index]) * next_values_t[index] - values_t[index]
            gae = delta + gamma * lam * (1.0 - done_t[index]) * gae
            advantages[index] = gae
        returns = advantages + values_t
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return Rollout(observations_t, actions_t, rewards_t, terminated_t, done_t, values_t,
                       next_values_t, advantages.detach(), returns.detach(), num_steps), observation

    def update(self, rollout, prior_actor=None, transfer_lambda=0.0):
        actor_loss = -(self.actor.log_prob(rollout.observations, rollout.actions) * rollout.advantages).mean()
        if prior_actor is not None and transfer_lambda:
            from diagnostics.kl import kl_loss
            actor_loss = actor_loss + transfer_lambda * kl_loss(prior_actor, self.actor, rollout.observations)
        critic_loss = 0.5 * (self.critic(rollout.observations) - rollout.returns).square().mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.config.get("max_grad_norm", 1.0))
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.config.get("max_grad_norm", 1.0))
        self.critic_optimizer.step()
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
