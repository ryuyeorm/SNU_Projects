from __future__ import annotations

import copy
import random

import numpy as np
import torch

from utils import evaluate, make_env


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def train_steps(agent, env, total_steps, rollout_steps, prior_actor=None,
                transfer_lambda=0.0, observation=None):
    completed = 0
    while completed < total_steps:
        count = min(rollout_steps, total_steps - completed)
        rollout, observation = agent.collect_rollout(env, count, observation)
        agent.update(rollout, prior_actor=prior_actor, transfer_lambda=transfer_lambda)
        completed += count
    return observation


def train_with_evaluation(agent, task_id, config, seed, total_steps, prior_actor=None,
                          transfer_lambda=0.0):
    training = config["training"]
    env = make_env(task_id, config["environment"], seed)
    eval_env = make_env(task_id, config["environment"], seed + 1_000_000)
    interval = training["eval_interval"]
    curves = [{"step": 0, **evaluate(agent.actor, eval_env, training["eval_episodes"], seed + 2_000_000)}]
    observation = None; completed = 0
    while completed < total_steps:
        count = min(interval, total_steps - completed)
        observation = train_steps(agent, env, count, training["rollout_steps"], prior_actor,
                                  transfer_lambda, observation)
        completed += count
        curves.append({"step": completed, **evaluate(
            agent.actor, eval_env, training["eval_episodes"], seed + 2_000_000)})
    return curves


def auc(curve):
    steps = np.asarray([item["step"] for item in curve], dtype=float)
    values = np.asarray([item["mean_return"] for item in curve], dtype=float)
    return float(np.trapezoid(values, steps) if hasattr(np, "trapezoid") else np.trapz(values, steps))


def steps_to_success(curve):
    return next((item["step"] for item in curve if item["success_rate"] >= 0.5), None)

