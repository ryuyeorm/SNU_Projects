"""Gymnasium 2-D PointMass with matrix-parameterized action semantics."""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install project dependencies before using PointMassEnv") from exc

from .task_registry import get_task_matrix


class PointMassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task_id: int | None = 0, dynamics_matrix=None, goal=(1.0, 0.0),
                 step_scale=0.1, episode_horizon=100, success_radius=0.1,
                 transition_noise_std=0.0, start_position_std=0.05,
                 state_bound=2.0):
        super().__init__()
        if dynamics_matrix is None:
            if task_id is None:
                raise ValueError("Provide task_id or dynamics_matrix")
            dynamics_matrix = get_task_matrix(task_id)
        matrix = np.asarray(dynamics_matrix, dtype=np.float32)
        if matrix.shape != (2, 2):
            raise ValueError("dynamics_matrix must have shape (2, 2)")
        self.task_id = task_id
        self.dynamics_matrix = matrix.copy()
        self.goal = np.asarray(goal, dtype=np.float32)
        self.step_scale = float(step_scale)
        self.episode_horizon = int(episode_horizon)
        self.success_radius = float(success_radius)
        self.transition_noise_std = float(transition_noise_std)
        self.start_position_std = float(start_position_std)
        self.state_bound = float(state_bound)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-self.state_bound, self.state_bound, (2,), np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float32).copy()
        else:
            self.state = self.np_random.normal(0.0, self.start_position_std, size=2).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {"distance": float(np.linalg.norm(self.state - self.goal))}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        noise = self.np_random.normal(0.0, self.transition_noise_std, size=2).astype(np.float32)
        self.state = np.clip(self.state + self.step_scale * (self.dynamics_matrix @ action) + noise,
                             -self.state_bound, self.state_bound).astype(np.float32)
        self.steps += 1
        distance = float(np.linalg.norm(self.state - self.goal))
        terminated = distance <= self.success_radius
        truncated = self.steps >= self.episode_horizon and not terminated
        return self.state.copy(), -distance, terminated, truncated, {
            "distance": distance, "success": bool(terminated),
        }

