"""Homework: implement a discrete-action policy network for A2C.

Complete every TODO in this file. You can run the file directly to execute
the small shape and gradient checks at the bottom:

    python src/models/actor.py
"""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network representing pi(action | observation).

    Args:
        observation_dim: Number of values in one observation.
        action_dim: Number of available discrete actions.
        hidden_dim: Width of both hidden layers.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )


    def forward(self, observations: Tensor) -> Tensor:
        """Return one action logit per observation and action.

        Expected shapes:
            [observation_dim]             -> [action_dim]
            [batch_size, observation_dim] -> [batch_size, action_dim]
        """

        return self.network(observations)

    def distribution(self, observations: Tensor) -> Categorical:
        """Construct the categorical policy for the given observations."""

        logits = self.forward(observations)
        return Categorical(logits = logits)
    
    def act(
        self,
        observations: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Choose actions and return actions, log probabilities, and entropy.

        During training, deterministic should normally be False. During
        evaluation, True chooses the action with the largest logit.
        """

        policy = self.distribution(observations)

        if deterministic:
            actions = policy.logits.argmax(dim=-1)
        else:
            actions = policy.sample()

        log_prob = policy.log_prob(actions)
        entropy = policy.entropy()

        return actions, log_prob, entropy


    def evaluate_actions(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate actions collected previously during an A2C rollout."""

        dist = self.distribution(observations)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, entropy

