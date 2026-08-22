"""Joint-action Q network for two-agent discrete Markov games."""

from torch import Tensor, nn


class JointActionCritic(nn.Module):
    """Map a state to one Q value for every joint action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim_1: int,
        action_dim_2: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim_1, action_dim_2, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")

        self.action_dim_1 = action_dim_1
        self.action_dim_2 = action_dim_2
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim_1 * action_dim_2),
        )

    def forward(self, observations: Tensor) -> Tensor:
        values = self.network(observations)
        return values.reshape(
            *values.shape[:-1],
            self.action_dim_1,
            self.action_dim_2,
        )
