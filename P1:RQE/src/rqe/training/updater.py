"""Apply optimizer steps, gradient processing, and target-network updates."""

import torch
from torch import Tensor

from ..models.actor import Actor
from ..models.critic import Critic
from .losses import actor_loss, advantages, critic_loss, td_target


class Updater:
    """Perform one actor-critic optimization step from a rollout batch."""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        entropy_coefficient: float = 0.01,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be nonnegative")

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient

    def update(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
    ) -> dict[str, float]:
        values = self.critic(observations)

        with torch.no_grad():
            next_values = self.critic(next_observations)
            targets = td_target(
                rewards,
                next_values,
                dones,
                self.gamma,
            )

        advantage_values = advantages(targets, values)
        log_probabilities, entropies = self.actor.evaluate_actions(
            observations,
            actions,
        )

        actor_loss_value = actor_loss(
            log_probabilities,
            advantage_values,
            entropies,
            self.entropy_coefficient,
        )
        critic_loss_value = critic_loss(values, targets)

        self.actor_optimizer.zero_grad()
        actor_loss_value.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss_value.backward()
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss_value.item(),
            "critic_loss": critic_loss_value.item(),
            "entropy": entropies.mean().item(),
        }
