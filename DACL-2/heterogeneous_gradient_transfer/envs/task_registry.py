"""Canonical heterogeneous PointMass task definitions."""
from __future__ import annotations

import math
import numpy as np


def rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]], dtype=np.float32)


TASK_MATRICES: dict[int, np.ndarray] = {
    0: np.eye(2, dtype=np.float32),
    1: rotation(45),
    2: rotation(90),
    3: rotation(180),
    4: np.array([[1.5, 0.0], [0.0, 0.6]], dtype=np.float32),
    5: np.array([[1.0, 0.75], [0.0, 1.0]], dtype=np.float32),
    6: np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    7: rotation(45) @ np.diag(np.array([1.3, 0.7], dtype=np.float32)),
}

TASK_NAMES = {
    0: "identity", 1: "rotation_45", 2: "rotation_90", 3: "rotation_180",
    4: "anisotropic_scale", 5: "shear", 6: "reflection",
    7: "rotation_45_anisotropic_scale",
}


def get_task_matrix(task_id: int) -> np.ndarray:
    if task_id not in TASK_MATRICES:
        raise KeyError(f"Unknown task_id {task_id}; expected one of {sorted(TASK_MATRICES)}")
    return TASK_MATRICES[task_id].copy()

