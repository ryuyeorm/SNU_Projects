import torch

from rqe.core.regularizers import entropy, kl_divergence


def test_entropy_of_uniform_distribution():
    probabilities = torch.tensor([0.5, 0.5])
    torch.testing.assert_close(entropy(probabilities), torch.log(torch.tensor(2.0)))


def test_entropy_of_deterministic_distribution():
    probabilities = torch.tensor([1.0, 0.0])
    torch.testing.assert_close(entropy(probabilities), torch.tensor(0.0))


def test_kl_with_itself_is_zero():
    probabilities = torch.tensor([0.2, 0.8])
    torch.testing.assert_close(
        kl_divergence(probabilities, probabilities),
        torch.tensor(0.0),
    )


def test_regularizers_support_batches():
    probabilities = torch.tensor([
        [0.5, 0.5],
        [0.2, 0.8],
    ])

    assert entropy(probabilities).shape == (2,)