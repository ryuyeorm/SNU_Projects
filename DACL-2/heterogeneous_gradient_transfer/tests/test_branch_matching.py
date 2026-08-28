import copy

import torch

from agents import ActorCritic
from tests.test_gradient_extraction import CONFIG


def nested_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    if torch.is_tensor(left):
        return torch.equal(left, right)
    return left == right


def test_cloned_branches_match_parameters_and_optimizers():
    original = ActorCritic(CONFIG); state = copy.deepcopy(original.state_dict())
    scratch = ActorCritic(CONFIG); transfer = ActorCritic(CONFIG)
    scratch.load_state_dict(copy.deepcopy(state)); transfer.load_state_dict(copy.deepcopy(state))
    assert nested_equal(scratch.state_dict(), transfer.state_dict())

