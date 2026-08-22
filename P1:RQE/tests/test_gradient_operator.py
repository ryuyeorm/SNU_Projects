"""Test analytical and automatic game-gradient operator calculations."""
import torch

from rqe.core.gradient_operator import gradient_operator


def test_gradient_operator_signs():
    policy = torch.tensor(2.0, requires_grad=True)
    adversary = torch.tensor(3.0, requires_grad=True)

    # L(x, y) = x² - y²
    objective = policy.square() - adversary.square()

    policy_gradient, adversary_gradient = gradient_operator(
        objective,
        policy,
        adversary,
    )

    # ∇x L = 2x = 4
    torch.testing.assert_close(
        policy_gradient,
        torch.tensor(4.0),
    )

    # Operator uses -∇y L = -(-2y) = 6
    torch.testing.assert_close(
        adversary_gradient,
        torch.tensor(6.0),
    )
    
def test_gradient_operator_averages_batch():
    policy = torch.tensor([1.0, 2.0], requires_grad=True)
    adversary = torch.tensor([3.0, 4.0], requires_grad=True)

    objective = policy.square() - adversary.square()

    policy_gradient, adversary_gradient = gradient_operator(
        objective,
        policy,
        adversary,
    )

    torch.testing.assert_close(
        policy_gradient,
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        adversary_gradient,
        torch.tensor([3.0, 4.0]),
    )