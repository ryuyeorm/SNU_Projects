"""Continuous 2-D point-navigation environment used by the experiment.

Every task has the same goal ``(1, 0)`` and reward. A task is specified instead by a
dynamics angle: before affecting position, the action is multiplied by the rotation
matrix ``R(phi)``. Observations and actions are both two-element continuous vectors.
An episode ends only because of the configured time limit;
reaching the goal is reported in ``info`` but does not terminate the episode.  This
keeps every learning curve based on an equal number of interaction steps.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PointMassEnv(gym.Env):
    """A finite-horizon, two-dimensional continuous point-navigation task.

    Parameters:
        dynamics_angle: Counterclockwise rotation applied to actions, in degrees.
        step_scale: Distance multiplier ``delta`` applied to every action.
        transition_noise: Standard deviation of independent Gaussian coordinate noise.
        start_radius: Radius of the uniform start-position disk. Zero means the origin.
        goal_radius: Distance below which ``info["success"]`` is true.
        success_bonus: Reward added on every step whose endpoint is inside the goal.
        horizon: Maximum number of environment steps per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dynamics_angle: float = 0.0,
        step_scale: float = 0.1,
        transition_noise: float = 0.0,
        start_radius: float = 0.0,
        goal_radius: float = 0.1,
        success_bonus: float = 0.0,
        horizon: int = 50,
    ) -> None:
        """Initialize task geometry, dynamics, reward settings, and Gym spaces."""
        super().__init__()
        # Reward geometry is deliberately identical across every task.
        self.goal = np.array([1.0, 0.0], dtype=np.float32)
        self.dynamics_angle = float(dynamics_angle)
        angle = np.deg2rad(dynamics_angle)
        # R(phi) maps the policy's action coordinates into world displacement.
        self.action_rotation = np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]], dtype=np.float32
        )
        self.step_scale = float(step_scale)
        self.transition_noise = float(transition_noise)
        self.start_radius = float(start_radius)
        self.goal_radius = float(goal_radius)
        self.success_bonus = float(success_bonus)
        self.horizon = int(horizon)
        # Both acceleration-like action components are independently continuous.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Positions are intentionally unbounded; the finite horizon limits drift.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        """Start a new episode and return ``(observation, info)``.

        Args:
            seed: Optional seed for environment and random-action reproducibility.
            options: Reserved by the Gymnasium API; currently unused.

        Returns:
            A copy of the initial position and its distance/success metadata.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        if self.start_radius:
            # Normalized Gaussian direction plus sqrt-uniform radius samples a disk
            # uniformly by area, rather than concentrating starts near its center.
            direction = self.np_random.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-12)
            radius = self.start_radius * np.sqrt(self.np_random.uniform())
            self.state = (radius * direction).astype(np.float32)
        else:
            self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), self._info()

    def step(self, action):
        """Advance the point using one clipped continuous action.

        Args:
            action: Array-like ``[a_x, a_y]``; values are clipped into ``[-1, 1]``.

        Returns:
            Gymnasium's ``(observation, reward, terminated, truncated, info)`` tuple.
            ``terminated`` is always false and ``truncated`` marks the time limit.
        """
        # Enforce the declared action bounds even when callers bypass Gym wrappers.
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Apply s_{t+1} = s_t + delta*R(phi)*a_t + epsilon_t.
        noise = self.np_random.normal(0.0, self.transition_noise, size=2)
        world_action = self.action_rotation @ action
        self.state = (self.state + self.step_scale * world_action + noise).astype(np.float32)
        self.steps += 1
        distance = float(np.linalg.norm(self.state - self.goal))
        success = distance < self.goal_radius
        # Dense negative distance supplies a learning signal everywhere in the plane.
        reward = -distance + (self.success_bonus if success else 0.0)
        return self.state.copy(), float(reward), False, self.steps >= self.horizon, self._info()

    def _info(self):
        """Return diagnostic geometry for the current state."""
        distance = float(np.linalg.norm(self.state - self.goal))
        return {"distance": distance, "success": distance < self.goal_radius}
