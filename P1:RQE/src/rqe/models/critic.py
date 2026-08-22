import torch
from torch import Tensor, nn

class Critic(nn.Module):
    def __init__(
            self,
            observation_dim : int,
            hidden_dim : int = 128
    ):
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations).squeeze(-1)

    
