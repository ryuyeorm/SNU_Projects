import numpy as np
import pytest

from envs import PointMassEnv, TASK_MATRICES


@pytest.mark.parametrize("task_id", list(TASK_MATRICES))
@pytest.mark.parametrize("action", [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
def test_matrix_displacement(task_id, action):
    env = PointMassEnv(task_id=task_id, step_scale=0.1, start_position_std=0.0)
    env.reset(seed=0, options={"state": [0.0, 0.0]})
    state, *_ = env.step(action)
    np.testing.assert_allclose(state, 0.1 * TASK_MATRICES[task_id] @ action, atol=1e-6)


@pytest.mark.parametrize("task_id", list(TASK_MATRICES))
def test_inverse_controller_solves_each_task(task_id):
    env = PointMassEnv(task_id=task_id, start_position_std=0.0)
    state, _ = env.reset(seed=0)
    inverse = np.linalg.inv(TASK_MATRICES[task_id])
    success = False
    for _ in range(100):
        action = np.clip(inverse @ (env.goal - state) / env.step_scale, -1.0, 1.0)
        state, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = info["success"]; break
    assert success

