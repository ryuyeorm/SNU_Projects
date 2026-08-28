"""Conventional on-policy actor–critic for a mathematically transparent test.

The critic learns a scalar state value with a one-step TD target. The actor uses the
log probability of actions that were collected from the current policy, weighted by a
detached TD(0) advantage. No replayed data, target networks, Q-gradient-through-action,
or learned SAC temperature enters an optimization update.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sac import Actor, mlp


class ValueCritic(nn.Module):
    """State-value baseline used to form TD advantages.

    Args:
        obs_dim: Number of observation coordinates.
        hidden_sizes: Width of each value-network hidden layer.
    """

    def __init__(self, obs_dim=2, hidden_sizes=(128, 128)):
        """Initialize a scalar-output multilayer perceptron."""
        super().__init__()
        self.network = mlp([obs_dim, *hidden_sizes, 1])

    def forward(self, obs):
        """Estimate the discounted return from each input state."""
        return self.network(obs)


@dataclass
class ActorCriticConfig:
    """Optimization settings for the conventional actor–critic.

    Attributes:
        gamma: Discount factor in the one-step TD target.
        learning_rate: Adam learning rate for actor and critic.
        entropy_coef: Fixed weight encouraging policy-distribution entropy.
        value_coef: Multiplier on critic mean-squared TD error.
        max_grad_norm: Joint per-network gradient clipping threshold.
        rollout_steps: On-policy transitions collected before each update.
        gae_lambda: Bias-variance interpolation used by generalized advantages.
        hidden_sizes: Actor and value-network hidden widths.
    """

    gamma: float = 0.99
    learning_rate: float = 3e-4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    rollout_steps: int = 32
    gae_lambda: float = 0.95
    hidden_sizes: tuple[int, ...] = (128, 128)


class ActorCriticAgent:
    """Gaussian actor and state-value critic with on-policy TD(0) updates.

    Args:
        config: Actor–critic hyperparameters.
        device: PyTorch execution device.
    """

    on_policy = True

    def __init__(self, config: ActorCriticConfig | None = None, device="cpu"):
        """Initialize actor/value networks and their independent optimizers."""
        self.config = config or ActorCriticConfig()
        self.device = torch.device(device)
        self.actor = Actor(hidden_sizes=self.config.hidden_sizes).to(self.device)
        self.critic = ValueCritic(hidden_sizes=self.config.hidden_sizes).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Produce one continuous NumPy action from one observation."""
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.sample(tensor, deterministic)[0].cpu().numpy()[0]

    def td_advantage(self, batch):
        """Calculate detached generalized advantages on a chronological rollout."""
        obs, _, rewards, next_obs, terminals = batch
        with torch.no_grad():
            values = self.critic(obs)
            next_values = self.critic(next_obs)
            deltas = rewards + self.config.gamma * (1 - terminals) * next_values - values
            advantage = torch.zeros_like(deltas)
            accumulator = torch.zeros_like(deltas[0])
            # Backward recursion implements generalized advantage estimation (GAE).
            for index in reversed(range(len(deltas))):
                accumulator = deltas[index] + (
                    self.config.gamma * self.config.gae_lambda
                    * (1 - terminals[index]) * accumulator
                )
                advantage[index] = accumulator
            target = values + advantage
        return target, advantage

    def diagnostic_actor_loss(self, batch, num_action_samples=1):
        """Return the ordinary policy-gradient loss on collected rollout actions.

        ``num_action_samples`` is accepted for a common diagnostic interface but is
        intentionally unused: on-policy actions are observed data, not resampled.
        """
        obs, actions = batch[0], batch[1]
        _, advantage = self.td_advantage(batch)
        # Centering supplies both positive and negative relative-action feedback;
        # scaling prevents reward magnitude from arbitrarily controlling actor steps.
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (
                advantage.std(unbiased=False) + 1e-8
            )
        policy_loss = -(self.actor.log_prob(obs, actions) * advantage).mean()
        entropy = self.actor.entropy_estimate(obs).mean()
        return policy_loss - self.config.entropy_coef * entropy

    def update(self, batch, prior_actor=None, transfer_lambda=0.0,
               transfer_measure="kl"):
        """Perform one value update and one on-policy actor update.

        Args:
            batch: Chronological rollout tensors collected by the current policy.
            prior_actor: Frozen source actor for the transfer condition.
            transfer_lambda: Weight on the same KL used by the diagnostic.
            transfer_measure: ``kl``, squared ``wasserstein``, or ``mean_mse``.
        """
        obs = batch[0]
        target, _ = self.td_advantage(batch)
        value_loss = self.config.value_coef * F.mse_loss(self.critic(obs), target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        actor_loss = self.diagnostic_actor_loss(batch)
        if prior_actor is not None and transfer_lambda:
            from gradient_transfer_test.transfer import policy_distance
            actor_loss = actor_loss + transfer_lambda * policy_distance(
                prior_actor, self.actor, obs, transfer_measure
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        return {"actor_loss": float(actor_loss.detach()),
                "critic_loss": float(value_loss.detach())}

    def state_dict(self):
        """Snapshot networks, optimizers, and all global RNG states."""
        return {
            "config": self.config,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }

    def load_state_dict(self, state, restore_rng=True):
        """Restore a complete branch snapshot."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])

    def clone(self):
        """Return an independent exact copy of this learner."""
        state = copy.deepcopy(self.state_dict())
        clone = ActorCriticAgent(self.config, self.device)
        clone.load_state_dict(state)
        return clone
