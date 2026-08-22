import torch
from torch import Tensor, nn

class Adversary(nn.Module):
    def __init__(
            self, 
            observation_dim : int, 
            action_dim: int,
            hidden_dim : int = 128
            ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observations: Tensor) -> Tensor:
        return torch.softmax(self.network(observations), dim = -1) # transform the output(logits) into probability(normalizing them)

    # note : the output of this network has to be probability, which is p_i from the paper
    # it needs to be compared to the original true opponent policy pi_-i


