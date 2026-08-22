"""Build the game-gradient operator used by RQE optimization algorithms."""
import torch
from torch import Tensor

def gradient_operator(
    objective: Tensor,
    policy_probabilities: Tensor,
    adversary_probabilities: Tensor,
    create_graph: bool = False,
)-> tuple [Tensor, Tensor]:
    scalar_objective = objective.mean()
    
    policy_gradient, adversary_gradient = torch.autograd.grad(
        scalar_objective, 
        (policy_probabilities, adversary_probabilities),
        create_graph=create_graph
    )
    
    return policy_gradient, -adversary_gradient # adversary negated to represent minmax problem with just minimization
