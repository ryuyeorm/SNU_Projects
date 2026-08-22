import torch
from torch import Tensor

from rqe.core.gradient_operator import gradient_operator
from rqe.core.projection import project_simplex
from rqe.core.rqe_objective import rqe_objective


def solve_normal_form(
    q_values: Tensor,
    reference: Tensor,
    iterations: int = 1_000,
    learning_rate: float = 0.01,
    tau: float = 1.0,
    entropy_coefficient: float = 0.1,
) -> tuple[Tensor, Tensor]:
    own_actions, opponent_actions = q_values.shape

    policy = torch.full(
        (own_actions,),
        1.0 / own_actions,
        dtype=q_values.dtype,
        device=q_values.device,
        requires_grad=True,
    )

    adversary = torch.full(
        (opponent_actions,),
        1.0 / opponent_actions,
        dtype=q_values.dtype,
        device=q_values.device,
        requires_grad=True,
    )

    for _ in range(iterations):
        objective = rqe_objective(
            policy,
            adversary,
            q_values,
            reference,
            tau,
            entropy_coefficient,
        )

        policy_gradient, adversary_gradient = gradient_operator(
            objective,
            policy,
            adversary,
        )

        with torch.no_grad():
            policy = project_simplex(
                policy - learning_rate * policy_gradient
            )
            adversary = project_simplex(
                adversary - learning_rate * adversary_gradient
            )

        policy.requires_grad_()
        adversary.requires_grad_()

    return policy.detach(), adversary.detach()