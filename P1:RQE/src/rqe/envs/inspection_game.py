"""Normal-form inspection game used in the RQE paper experiments.

The game is specified in Section 5.1 of "Provably Convergent
Actor-Critic for MARL through Risk-aversion" (arXiv:2602.12386).
Each call to :meth:`step` plays one complete, stateless game.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class InspectionGame:
    """Two-player, two-action normal-form inspection game.

    Inspectee actions (rows): ``0 = defect``, ``1 = comply``.
    Inspector actions (columns): ``0 = audit``, ``1 = no_audit``.

    The reward matrices reproduce the matrices reported in the paper:

    ``R_inspectee = [[0, 5], [3, 3]]``
    ``R_inspector = [[-3, -5], [0, 3]]``
    """

    observation_dim = 1
    inspectee_action_dim = 2
    inspector_action_dim = 2

    inspectee_action_names = ("defect", "comply")
    inspector_action_names = ("audit", "no_audit")

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.inspectee_payoffs = torch.tensor(
            [[0.0, 5.0], [3.0, 3.0]],
            dtype=dtype,
        )
        self.inspector_payoffs = torch.tensor(
            [[-3.0, -5.0], [0.0, 3.0]],
            dtype=dtype,
        )
        self._observation = torch.ones(1, dtype=dtype)
        self.reset_metrics()

    def reset(self) -> tuple[Tensor, Tensor]:
        """Return the constant observation seen by both players."""
        return self._observation.clone(), self._observation.clone()

    def step(
        self,
        inspectee_action: int | Tensor,
        inspector_action: int | Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], bool, dict[str, Any]]:
        """Play one game and return observations, rewards, termination, and info."""
        inspectee_index = self._action_index(
            inspectee_action,
            self.inspectee_action_dim,
            "inspectee_action",
        )
        inspector_index = self._action_index(
            inspector_action,
            self.inspector_action_dim,
            "inspector_action",
        )

        inspectee_reward = self.inspectee_payoffs[
            inspectee_index,
            inspector_index,
        ]
        inspector_reward = self.inspector_payoffs[
            inspectee_index,
            inspector_index,
        ]

        self._games_played += 1
        self._inspectee_action_counts[inspectee_index] += 1
        self._inspector_action_counts[inspector_index] += 1
        self._reward_sums += torch.stack(
            (inspectee_reward, inspector_reward)
        )

        observations = self.reset()
        rewards = inspectee_reward.clone(), inspector_reward.clone()
        info = {
            "inspectee_action": self.inspectee_action_names[inspectee_index],
            "inspector_action": self.inspector_action_names[inspector_index],
        }
        return observations, rewards, True, info

    def expected_rewards(
        self,
        inspectee_policy: Tensor,
        inspector_policy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return both expected rewards under a pair of mixed strategies."""
        self._validate_policy(inspectee_policy, self.inspectee_action_dim)
        self._validate_policy(inspector_policy, self.inspector_action_dim)

        inspectee_reward = torch.einsum(
            "...i,ij,...j->...",
            inspectee_policy,
            self.inspectee_payoffs.to(inspectee_policy),
            inspector_policy,
        )
        inspector_reward = torch.einsum(
            "...i,ij,...j->...",
            inspectee_policy,
            self.inspector_payoffs.to(inspectee_policy),
            inspector_policy,
        )
        return inspectee_reward, inspector_reward

    def metrics(self) -> dict[str, float]:
        """Return the paper's action-0 frequencies and mean rewards."""
        if self._games_played == 0:
            return {
                "inspectee_defect_probability": 0.0,
                "inspector_audit_probability": 0.0,
                "inspectee_mean_reward": 0.0,
                "inspector_mean_reward": 0.0,
            }

        count = float(self._games_played)
        return {
            "inspectee_defect_probability": (
                self._inspectee_action_counts[0].item() / count
            ),
            "inspector_audit_probability": (
                self._inspector_action_counts[0].item() / count
            ),
            "inspectee_mean_reward": self._reward_sums[0].item() / count,
            "inspector_mean_reward": self._reward_sums[1].item() / count,
        }

    def reset_metrics(self) -> None:
        """Clear cumulative action counts and rewards."""
        self._games_played = 0
        self._inspectee_action_counts = torch.zeros(2, dtype=torch.long)
        self._inspector_action_counts = torch.zeros(2, dtype=torch.long)
        self._reward_sums = torch.zeros(2, dtype=self.inspectee_payoffs.dtype)

    @staticmethod
    def _action_index(action: int | Tensor, action_dim: int, name: str) -> int:
        if isinstance(action, Tensor):
            if action.numel() != 1:
                raise ValueError(f"{name} must be a scalar")
            action = int(action.item())
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"{name} must be an integer or scalar tensor")
        if not 0 <= action < action_dim:
            raise ValueError(f"{name} must be between 0 and {action_dim - 1}")
        return action

    @staticmethod
    def _validate_policy(policy: Tensor, action_dim: int) -> None:
        if policy.shape[-1] != action_dim:
            raise ValueError(
                f"policy's final dimension must have size {action_dim}"
            )
        if not torch.isfinite(policy).all():
            raise ValueError("policy must contain only finite values")
        if (policy < 0).any():
            raise ValueError("policy probabilities must be nonnegative")
        expected_sum = torch.ones_like(policy.sum(dim=-1))
        if not torch.allclose(policy.sum(dim=-1), expected_sum):
            raise ValueError("policy probabilities must sum to one")
