"""Test feasibility and numerical behavior of strategy projections."""
import torch

from rqe.core.projection import project_simplex


def test_valid_distribution_is_unchanged():
    x = torch.tensor([0.2, 0.3, 0.5])
    torch.testing.assert_close(project_simplex(x), x)


def test_projection_is_on_simplex():
    x = torch.tensor([-1.0, 2.0, 3.0])
    result = project_simplex(x)

    assert (result >= 0).all()
    torch.testing.assert_close(result.sum(), torch.tensor(1.0))


def test_batched_projection():
    x = torch.randn(8, 4)
    result = project_simplex(x)

    assert result.shape == (8, 4)
    assert (result >= 0).all()
    torch.testing.assert_close(
        result.sum(dim=-1),
        torch.ones(8),
    )


def test_projection_supports_gradients():
    x = torch.randn(4, requires_grad=True)
    result = project_simplex(x)
    result.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()