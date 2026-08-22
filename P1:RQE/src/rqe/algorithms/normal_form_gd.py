"""Projected-gradient solvers for two-player normal-form RQE games."""

from collections.abc import Callable

import torch
from torch import Tensor

from ..core.projection import project_simplex
from ..core.regularizers import entropy, kl_divergence, log_barrier

Regularizer = Callable[[Tensor], Tensor]


def _policy_regularizer(name: str) -> Regularizer:
    if name == "log_barrier":
        return log_barrier
    if name == "negative_entropy":
        return lambda policy: -entropy(policy)
    raise ValueError("regularizer must be 'log_barrier' or 'negative_entropy'")


def solve_normal_form_game(
    player_1_payoffs: Tensor,
    player_2_payoffs: Tensor,
    iterations: int = 1_000,
    learning_rate: float = 0.01,
    tau: float = 1.0,
    epsilon: float = 0.2,
    regularizer: str = "log_barrier",
    initial_policy_1: Tensor | None = None,
    initial_policy_2: Tensor | None = None,
) -> dict[str, Tensor]:
    """Run the paper's simultaneous four-player projected-GD iteration.

    Payoff matrices use the common layout
    ``[player_1_action, player_2_action]``. Histories include iteration zero.
    """
    if player_1_payoffs.ndim != 2:
        raise ValueError("player_1_payoffs must be a matrix")
    if player_2_payoffs.shape != player_1_payoffs.shape:
        raise ValueError("both payoff matrices must have the same shape")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")

    rows, columns = player_1_payoffs.shape
    device, dtype = player_1_payoffs.device, player_1_payoffs.dtype
    player_2_payoffs = player_2_payoffs.to(device=device, dtype=dtype)
    nu = _policy_regularizer(regularizer)

    policy_1 = _initial_strategy(initial_policy_1, rows, device, dtype)
    policy_2 = _initial_strategy(initial_policy_2, columns, device, dtype)
    adversary_1 = policy_2.detach().clone().requires_grad_()
    adversary_2 = policy_1.detach().clone().requires_grad_()
    histories = {
        "policy_1": [policy_1.detach().clone()],
        "policy_2": [policy_2.detach().clone()],
        "adversary_1": [adversary_1.detach().clone()],
        "adversary_2": [adversary_2.detach().clone()],
    }

    for _ in range(iterations):
        player_1_cost = (
            -(policy_1 @ player_1_payoffs @ adversary_1)
            - kl_divergence(adversary_1, policy_2) / tau
            + epsilon * nu(policy_1)
        )
        player_2_cost = (
            -(policy_2 @ player_2_payoffs.T @ adversary_2)
            - kl_divergence(adversary_2, policy_1) / tau
            + epsilon * nu(policy_2)
        )

        policy_1_gradient = torch.autograd.grad(
            player_1_cost, policy_1, retain_graph=True
        )[0]
        policy_2_gradient = torch.autograd.grad(
            player_2_cost, policy_2, retain_graph=True
        )[0]
        adversary_1_gradient = torch.autograd.grad(
            -player_1_cost, adversary_1
        )[0]
        adversary_2_gradient = torch.autograd.grad(
            -player_2_cost, adversary_2
        )[0]

        with torch.no_grad():
            policy_1 = project_simplex(
                policy_1 - learning_rate * policy_1_gradient
            )
            policy_2 = project_simplex(
                policy_2 - learning_rate * policy_2_gradient
            )
            adversary_1 = project_simplex(
                adversary_1 - learning_rate * adversary_1_gradient
            )
            adversary_2 = project_simplex(
                adversary_2 - learning_rate * adversary_2_gradient
            )

        policy_1.requires_grad_()
        policy_2.requires_grad_()
        adversary_1.requires_grad_()
        adversary_2.requires_grad_()
        histories["policy_1"].append(policy_1.detach().clone())
        histories["policy_2"].append(policy_2.detach().clone())
        histories["adversary_1"].append(adversary_1.detach().clone())
        histories["adversary_2"].append(adversary_2.detach().clone())

    return {name: torch.stack(values) for name, values in histories.items()}


def solve_risk_neutral_game(
    player_1_payoffs: Tensor,
    player_2_payoffs: Tensor,
    iterations: int = 1_000,
    learning_rate: float = 0.01,
    epsilon: float = 0.2,
    regularizer: str = "log_barrier",
) -> dict[str, Tensor]:
    """Run projected GD for the two original players without adversaries."""
    if player_1_payoffs.ndim != 2 or player_2_payoffs.shape != player_1_payoffs.shape:
        raise ValueError("payoffs must be equally shaped matrices")
    rows, columns = player_1_payoffs.shape
    device, dtype = player_1_payoffs.device, player_1_payoffs.dtype
    player_2_payoffs = player_2_payoffs.to(device=device, dtype=dtype)
    policy_1 = _initial_strategy(None, rows, device, dtype)
    policy_2 = _initial_strategy(None, columns, device, dtype)
    nu = _policy_regularizer(regularizer)
    history_1 = [policy_1.detach().clone()]
    history_2 = [policy_2.detach().clone()]

    for _ in range(iterations):
        cost_1 = -(policy_1 @ player_1_payoffs @ policy_2) + epsilon * nu(policy_1)
        cost_2 = -(policy_1 @ player_2_payoffs @ policy_2) + epsilon * nu(policy_2)
        gradient_1 = torch.autograd.grad(cost_1, policy_1, retain_graph=True)[0]
        gradient_2 = torch.autograd.grad(cost_2, policy_2)[0]
        with torch.no_grad():
            policy_1 = project_simplex(policy_1 - learning_rate * gradient_1)
            policy_2 = project_simplex(policy_2 - learning_rate * gradient_2)
        policy_1.requires_grad_()
        policy_2.requires_grad_()
        history_1.append(policy_1.detach().clone())
        history_2.append(policy_2.detach().clone())

    return {
        "policy_1": torch.stack(history_1),
        "policy_2": torch.stack(history_2),
    }


def _initial_strategy(
    strategy: Tensor | None,
    actions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if strategy is None:
        strategy = torch.full(
            (actions,), 1.0 / actions, device=device, dtype=dtype
        )
    else:
        strategy = strategy.to(device=device, dtype=dtype)
        if strategy.shape != (actions,):
            raise ValueError(f"initial strategy must have shape ({actions},)")
        if (strategy < 0).any() or not torch.allclose(
            strategy.sum(), strategy.new_tensor(1.0)
        ):
            raise ValueError("initial strategy must lie on the simplex")
    return strategy.detach().clone().requires_grad_()
