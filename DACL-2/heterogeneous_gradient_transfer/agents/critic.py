from torch import nn

from .actor import mlp


class ValueCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden_sizes=(64, 64)):
        super().__init__()
        self.value_net = mlp(obs_dim, 1, hidden_sizes)

    def forward(self, observations):
        return self.value_net(observations).squeeze(-1)

