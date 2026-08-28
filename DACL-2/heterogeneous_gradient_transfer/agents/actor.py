from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim, output_dim, hidden_sizes):
    layers = []
    previous = input_dim
    for width in hidden_sizes:
        layers += [nn.Linear(previous, width), nn.Tanh()]
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """State-dependent mean and global diagonal std; actions are tanh-squashed."""

    def __init__(self, obs_dim=2, action_dim=2, hidden_sizes=(64, 64)):
        super().__init__()
        self.mean_net = mlp(obs_dim, action_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, observations):
        mean = self.mean_net(observations)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, observations):
        distribution = self.distribution(observations)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(-1)

    def log_prob(self, observations, actions):
        clipped = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(clipped)
        distribution = self.distribution(observations)
        value = distribution.log_prob(raw) - torch.log(1.0 - clipped.square() + 1e-6)
        return value.sum(-1)

    def deterministic_action(self, observations):
        return torch.tanh(self.distribution(observations).mean)

