import copy

import torch

from agents import GaussianActor
from diagnostics.kl import kl_loss
from diagnostics.mean_mse import mse_loss
from diagnostics.wasserstein import wasserstein_loss


def test_identical_policy_distances_are_zero():
    actor = GaussianActor(hidden_sizes=(8,))
    duplicate = copy.deepcopy(actor)
    states = torch.randn(16, 2)
    for loss in (kl_loss, mse_loss, wasserstein_loss):
        assert abs(float(loss(actor, duplicate, states).detach())) < 1e-6
