from pathlib import Path

import pytest
import torch

from agents import ActorCritic
from experiments.run_pair import load_source


def config(root, hidden_sizes):
    return {
        "agent": {"gamma": .99, "gae_lambda": .95, "actor_lr": 3e-4,
                  "critic_lr": 1e-3, "hidden_sizes": hidden_sizes,
                  "max_grad_norm": 1.0},
        "paths": {"checkpoints": str(root)},
    }


def test_load_source_rejects_different_agent_config(tmp_path):
    smoke = config(tmp_path, [8, 8])
    checkpoint = Path(tmp_path) / "task_0" / "seed_0.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(ActorCritic(smoke["agent"]).state_dict(), checkpoint)
    with pytest.raises(ValueError, match="different agent configuration"):
        load_source(0, 0, config(tmp_path, [16, 16]))
