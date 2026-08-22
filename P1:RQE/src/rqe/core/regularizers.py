"""Define RQE regularizers and their values or derivatives."""
from torch import Tensor

def entropy(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
    safe_probabilities = probabilities.clamp_min(eps)
    return -(probabilities * safe_probabilities.log()).sum(dim=-1) # elementwise operation


def log_barrier(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
    """Return the convex log-barrier regularizer on the simplex."""
    return -probabilities.clamp_min(eps).log().sum(dim=-1)

def kl_divergence(
        probabilities : Tensor,
        reference: Tensor,
        eps: float = 1e-8,
) -> Tensor:
    safe_probabilities = probabilities.clamp_min(eps)
    safe_reference = reference.clamp_min(eps)
    return (probabilities * (safe_probabilities.log() - safe_reference.log())).sum(dim=-1)
