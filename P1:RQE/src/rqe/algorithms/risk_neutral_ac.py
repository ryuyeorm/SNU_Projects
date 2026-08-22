"""Implement the risk-neutral actor-critic baseline."""

import torch
from torch import Tensor, nn

from ..buffers.replay_buffer import ReplayBuffer
from ..models.actor import Actor
from ..models.critic import Critic
from ..training.updater import Updater


class RiskNeutralActorCritic(nn.Module):
    """Bundle action selection, rollout storage, and A2C updates."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coefficient: float = 0.01,
    ) -> None:
        super().__init__()

        self.actor = Actor(observation_dim, action_dim, hidden_dim)
        self.critic = Critic(observation_dim, hidden_dim)
        self.buffer = ReplayBuffer()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
        )
        self.updater = Updater(
            self.actor,
            self.critic,
            self.actor_optimizer,
            self.critic_optimizer,
            gamma=gamma,
            entropy_coefficient=entropy_coefficient,
        )

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        action, _, _ = self.actor.act(
            observation,
            deterministic=deterministic,
        )
        return action

    def observe(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
    ) -> None:
        self.buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

    def update(self) -> dict[str, float]:
        batch = self.buffer.get()
        metrics = self.updater.update(*batch)
        self.buffer.clear()
        return metrics
