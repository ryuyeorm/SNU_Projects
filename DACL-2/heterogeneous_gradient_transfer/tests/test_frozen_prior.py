import copy

import torch

from agents import ActorCritic
from diagnostics import compute_alignment_batches
from tests.test_gradient_extraction import CONFIG, FakeRollout


def test_prior_never_receives_gradients():
    target = ActorCritic(CONFIG).actor; prior = copy.deepcopy(target)
    compute_alignment_batches(target, prior, [FakeRollout()])
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in prior.parameters())

