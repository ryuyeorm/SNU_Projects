"""The two-agent gridworld cooperation game from the RQE paper."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


class GridworldCooperation:
    """Fully observable 5x5 cooperation/defection Markov game.

    Actions are ``up``, ``down``, ``left``, ``right``, and ``stay``. Both
    agents observe the flattened joint position ``[r0, c0, r1, c1]``.
    """

    observation_dim = 4
    action_dim = 5
    action_names = ("up", "down", "left", "right", "stay")
    action_deltas = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

    def __init__(
        self,
        grid_size: int = 5,
        horizon: int = 50,
        cooperation_stay_probability: float = 0.7,
        seed: int | None = None,
    ) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= cooperation_stay_probability <= 1.0:
            raise ValueError(
                "cooperation_stay_probability must be between 0 and 1"
            )

        self.grid_size = grid_size
        self.horizon = horizon
        self.cooperation_stay_probability = cooperation_stay_probability
        self.agent_0_defection_zone = (0, grid_size - 1)
        self.agent_1_defection_zone = (grid_size - 1, 0)
        self.cooperation_zone = (grid_size - 1, grid_size - 1)
        self._rng = random.Random(seed)
        self._positions = [(0, 0), (0, 0)]
        self._step_count = 0

    @property
    def positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return an immutable snapshot of the two agent positions."""
        return self._positions[0], self._positions[1]

    def reset(self, seed: int | None = None) -> Tensor:
        """Reset both agents to the upper-left corner."""
        if seed is not None:
            self._rng.seed(seed)
        self._positions = [(0, 0), (0, 0)]
        self._step_count = 0
        return self._observation()

    def step(
        self,
        actions: Tensor | Sequence[int],
    ) -> tuple[Tensor, Tensor, bool, dict[str, Any]]:
        """Apply simultaneous actions and return the next transition."""
        if self._step_count >= self.horizon:
            raise RuntimeError("episode is done; call reset before stepping")

        requested_actions = self._parse_actions(actions)
        applied_actions = [
            self._resolve_action(agent, requested_actions[agent])
            for agent in range(2)
        ]
        self._positions = [
            self._move(self._positions[agent], applied_actions[agent])
            for agent in range(2)
        ]
        self._step_count += 1

        rewards = torch.tensor(
            [self._reward(0), self._reward(1)],
            dtype=torch.float32,
        )
        done = self._step_count >= self.horizon
        info = {
            "step": self._step_count,
            "positions": self.positions,
            "requested_actions": tuple(requested_actions),
            "applied_actions": tuple(applied_actions),
            "social_welfare": rewards.sum().item(),
            "cooperation": tuple(
                position == self.cooperation_zone
                for position in self._positions
            ),
            "defection": tuple(
                self._positions[agent] == self._defection_zone(agent)
                for agent in range(2)
            ),
        }
        return self._observation(), rewards, done, info

    def _resolve_action(self, agent: int, requested_action: int) -> int:
        position = self._positions[agent]
        own_defection_zone = self._defection_zone(agent)

        if position == own_defection_zone:
            return 4
        if (
            position == self.cooperation_zone
            and self._rng.random() < self.cooperation_stay_probability
        ):
            return 4
        if self._is_feasible(position, requested_action):
            return requested_action

        feasible_actions = [
            action
            for action in range(self.action_dim)
            if self._is_feasible(position, action)
        ]
        return self._rng.choice(feasible_actions)

    def _reward(self, agent: int) -> float:
        position = self._positions[agent]
        other_position = self._positions[1 - agent]
        own_defection_zone = self._defection_zone(agent)
        other_defection_zone = self._defection_zone(1 - agent)

        if position == own_defection_zone:
            return 3.0 if other_position == self.cooperation_zone else 0.0

        if position == self.cooperation_zone:
            if other_position == self.cooperation_zone:
                return 2.0
            if other_position == other_defection_zone:
                return 0.5
            if self._is_blank(other_position):
                return 1.0

        return 0.0

    def _is_blank(self, position: tuple[int, int]) -> bool:
        return position not in {
            self.agent_0_defection_zone,
            self.agent_1_defection_zone,
            self.cooperation_zone,
        }

    def _defection_zone(self, agent: int) -> tuple[int, int]:
        return (
            self.agent_0_defection_zone
            if agent == 0
            else self.agent_1_defection_zone
        )

    def _is_feasible(self, position: tuple[int, int], action: int) -> bool:
        row_delta, column_delta = self.action_deltas[action]
        row = position[0] + row_delta
        column = position[1] + column_delta
        return 0 <= row < self.grid_size and 0 <= column < self.grid_size

    def _move(
        self,
        position: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        row_delta, column_delta = self.action_deltas[action]
        return position[0] + row_delta, position[1] + column_delta

    def _observation(self) -> Tensor:
        return torch.tensor(
            [*self._positions[0], *self._positions[1]],
            dtype=torch.float32,
        )

    def _parse_actions(self, actions: Tensor | Sequence[int]) -> list[int]:
        if isinstance(actions, Tensor):
            if actions.shape != (2,):
                raise ValueError("actions must have shape (2,)")
            values = actions.tolist()
        else:
            values = list(actions)
            if len(values) != 2:
                raise ValueError("actions must contain exactly two actions")

        parsed: list[int] = []
        for action in values:
            if isinstance(action, bool) or not isinstance(action, int):
                raise TypeError("each action must be an integer")
            if not 0 <= action < self.action_dim:
                raise ValueError("each action must be between 0 and 4")
            parsed.append(action)
        return parsed
