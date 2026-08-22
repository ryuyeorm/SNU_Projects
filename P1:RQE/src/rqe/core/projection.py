"""Project vectors onto valid strategy spaces and other constraint sets."""
import torch


def project_simplex(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Euclidean projection onto the probability simplex:

        Delta = {p : p >= 0, sum(p) = 1}

    Supports batched tensors.

    Example:
        x.shape = [batch_size, num_actions]
        projection is performed along dim=-1.
    """

    # Sort values in descending order
    u, _ = torch.sort(x, dim=dim, descending=True)

    # Cumulative sum
    cssv = torch.cumsum(u, dim=dim)

    n = x.size(dim)

    # 1, 2, ..., n
    shape = [1] * x.ndim
    shape[dim] = n

    j = torch.arange(
        1,
        n + 1,
        device=x.device,
        dtype=x.dtype
    ).view(shape)

    # Condition:
    # u_j - (sum_{k<=j} u_k - 1) / j > 0
    condition = u - (cssv - 1.0) / j > 0

    # Number of entries satisfying condition
    rho = condition.sum(dim=dim, keepdim=True).clamp(min=1)

    # theta = (sum_{j<=rho} u_j - 1) / rho
    theta = (
        torch.gather(cssv, dim, rho - 1) - 1.0
    ) / rho.to(x.dtype)

    # Projection
    return torch.clamp(x - theta, min=0.0)
