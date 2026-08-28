from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_env(task_id, environment_config, seed):
    from envs import PointMassEnv
    env = PointMassEnv(task_id=task_id, **environment_config)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate(actor, env, episodes, seed):
    returns, successes = [], []
    actor.eval()
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0; success = False
        while True:
            state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor.deterministic_action(state).squeeze(0).cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward; success = success or info["success"]
            if terminated or truncated:
                break
        returns.append(total); successes.append(success)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "success_rate": float(np.mean(successes))}


def metadata(config, seed):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "seed": seed, "config": config}


def unique_run_directory(root, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(root) / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
