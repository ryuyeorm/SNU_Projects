"""Test construction and evaluation of the RQE objective."""
import torch

from rqe.core.rqe_objective import rqe_objective


def test_rqe_objective_returns_scalar_for_single_game():
    policy = torch.tensor([0.4, 0.6])
    adversary = torch.tensor([0.7, 0.3])
    reference = torch.tensor([0.5, 0.5])
    q_values = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = rqe_objective(
        policy,
        adversary,
        q_values,
        reference,
        tau=1.0,
        entropy_coefficient=0.1,
    )

    assert result.shape == ()
    assert torch.isfinite(result)


def test_rqe_objective_supports_batches():
    batch_size = 4
    policy = torch.softmax(torch.randn(batch_size, 2), dim=-1)
    adversary = torch.softmax(torch.randn(batch_size, 3), dim=-1)
    reference = torch.full((batch_size, 3), 1 / 3)
    q_values = torch.randn(batch_size, 2, 3)

    result = rqe_objective(
        policy,
        adversary,
        q_values,
        reference,
        tau=1.0,
        entropy_coefficient=0.1,
    )

    assert result.shape == (batch_size,)


def test_rqe_objective_preserves_gradients():
    policy_logits = torch.randn(2, requires_grad=True)
    adversary_logits = torch.randn(3, requires_grad=True)

    policy = torch.softmax(policy_logits, dim=-1)
    adversary = torch.softmax(adversary_logits, dim=-1)
    reference = torch.full((3,), 1 / 3)
    q_values = torch.randn(2, 3)

    loss = rqe_objective(
        policy,
        adversary,
        q_values,
        reference,
        tau=1.0,
        entropy_coefficient=0.1,
    )

    loss.backward()

    assert policy_logits.grad is not None
    assert adversary_logits.grad is not None