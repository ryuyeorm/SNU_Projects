import copy

import torch

from agents import ActorCritic
from diagnostics import compute_alignment_batches


CONFIG = {"gamma": .99, "gae_lambda": .95, "actor_lr": 3e-4,
          "critic_lr": 1e-3, "hidden_sizes": [8], "max_grad_norm": 1.0}


class FakeRollout:
    observations = torch.randn(16, 2)
    actions = torch.tanh(torch.randn(16, 2))
    advantages = torch.randn(16)


def test_gradient_extraction_does_not_change_parameters_and_is_finite():
    target = ActorCritic(CONFIG).actor; prior = copy.deepcopy(target)
    before = [x.detach().clone() for x in target.parameters()]
    result = compute_alignment_batches(target, prior, [FakeRollout(), FakeRollout()])
    assert all(torch.equal(x, y) for x, y in zip(before, target.parameters()))
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


def test_all_diagnostics_use_same_state_tensor(monkeypatch):
    import diagnostics.alignment as module
    seen = []
    originals = module.LOSSES.copy()
    def wrap(function):
        def inner(prior, target, states):
            seen.append(states.data_ptr()); return function(prior, target, states)
        return inner
    monkeypatch.setattr(module, "LOSSES", {key: wrap(value) for key, value in originals.items()})
    target = ActorCritic(CONFIG).actor; prior = copy.deepcopy(target)
    compute_alignment_batches(target, prior, [FakeRollout()])
    assert len(set(seen)) == 1

