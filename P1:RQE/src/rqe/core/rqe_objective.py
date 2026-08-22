"""Construct and evaluate the regularized quantal-response equilibrium objective."""
from .regularizers import kl_divergence, entropy
import torch
from torch import Tensor
# RQE Objective 

def rqe_objective(
    policy_probabilities: Tensor,
    adversary_probabilities: Tensor,
    q_values: Tensor,
    reference_opponent_probabilities: Tensor,
    tau: float,
    entropy_coefficient: float,
) -> Tensor:
    expected_value = torch.einsum(
        "...i,...ij,...j->...",
        policy_probabilities,
        q_values,
        adversary_probabilities,
    )

    kl_penalty = kl_divergence(
        adversary_probabilities,
        reference_opponent_probabilities,
    )

    policy_entropy = entropy(policy_probabilities)

    return (
        -expected_value
        - kl_penalty / tau
        - entropy_coefficient * policy_entropy
    )