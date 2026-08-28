"""Shared configuration, interaction, evaluation, and serialization utilities.

Keeping these operations in one module ensures source, scratch, and transfer runs use
the same environment construction and training conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from gradient_transfer_test.agents.sac import SACAgent, SACConfig, ReplayBuffer, set_seed
from gradient_transfer_test.agents.actor_critic import ActorCriticAgent, ActorCriticConfig
from gradient_transfer_test.envs.point_mass import PointMassEnv


def load_config(path):
    """Load a YAML experiment configuration from ``path`` into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config, angle):
    """Build a point-mass task using shared environment settings.

    Args:
        config: Full parsed YAML configuration.
        angle: Action-dynamics rotation in degrees for this task instance.
    """
    return PointMassEnv(dynamics_angle=angle, **config["environment"])


def make_agent(config, seed):
    """Seed global RNGs and construct the configured reinforcement-learning agent."""
    set_seed(seed)
    if config.get("algorithm", "sac") == "actor_critic":
        ac = config["actor_critic"]
        return ActorCriticAgent(ActorCriticConfig(
            gamma=ac["gamma"], learning_rate=ac["learning_rate"],
            entropy_coef=ac["entropy_coef"], value_coef=ac["value_coef"],
            max_grad_norm=ac["max_grad_norm"], rollout_steps=ac["rollout_steps"],
            gae_lambda=ac["gae_lambda"],
            hidden_sizes=tuple(ac["hidden_sizes"]),
        ), config.get("device", "cpu"))
    sac = config["sac"]
    return SACAgent(SACConfig(
        gamma=sac["gamma"], tau=sac["tau"], learning_rate=sac["learning_rate"],
        init_alpha=sac["init_alpha"], hidden_sizes=tuple(sac["hidden_sizes"]),
    ), config.get("device", "cpu"))


def interact_train(agent, env, replay, steps, batch_size, learning_starts, seed,
                   prior_actor=None, transfer_lambda=0.0, transfer_measure="kl"):
    """Collect a fixed number of interactions and train after warm-up.

    Args:
        agent: Configured actor–critic or legacy SAC learner being updated.
        env: Training environment for the current dynamics rotation.
        replay: Buffer receiving every collected transition.
        steps: Exact number of new environment interactions.
        batch_size: Transitions sampled per optimization update.
        learning_starts: Minimum replay size before policy actions and updates begin.
        seed: Seed used for the first reset and random warm-up actions.
        prior_actor: Frozen source policy in the transfer branch, otherwise ``None``.
        transfer_lambda: KL strength; zero recovers the unregularized algorithm.
        transfer_measure: Source-policy loss, either ``kl`` or ``wasserstein``.
    """
    obs, _ = env.reset(seed=seed)
    rollout = []
    for step in range(steps):
        # Uniform random actions diversify replay before learning is statistically safe.
        if getattr(agent, "on_policy", False):
            action = agent.act(obs)
        else:
            action = env.action_space.sample() if replay.size < learning_starts else agent.act(obs)
        nxt, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # A time-limit truncation is not an absorbing MDP state.
        # Actor–critic GAE must stop across episode resets. SAC, in contrast, may
        # bootstrap through a pure time-limit truncation.
        stored_done = done if getattr(agent, "on_policy", False) else terminated
        replay.add(obs, action, reward, nxt, float(stored_done))
        if getattr(agent, "on_policy", False):
            rollout.append((obs, action, reward, nxt, float(stored_done)))
        obs = nxt
        if done:
            # Subsequent resets continue the seeded environment RNG stream.
            obs, _ = env.reset()
        if getattr(agent, "on_policy", False) and len(rollout) == agent.config.rollout_steps:
            agent.update(_rollout_tensors(rollout, agent.device), prior_actor,
                         transfer_lambda, transfer_measure)
            rollout.clear()
        elif not getattr(agent, "on_policy", False) and replay.size >= max(batch_size, learning_starts):
            agent.update(replay.sample(batch_size, agent.device), prior_actor,
                         transfer_lambda, transfer_measure)
    # Update the final partial rollout rather than silently discarding its experience.
    if getattr(agent, "on_policy", False) and rollout:
        agent.update(_rollout_tensors(rollout, agent.device), prior_actor,
                     transfer_lambda, transfer_measure)


def _rollout_tensors(rollout, device):
    """Convert chronological transition tuples into actor–critic batch tensors."""
    columns = list(zip(*rollout))
    arrays = [np.asarray(column, dtype=np.float32) for column in columns]
    # Rewards and terminal flags need an explicit scalar-output dimension.
    arrays[2] = arrays[2].reshape(-1, 1)
    arrays[4] = arrays[4].reshape(-1, 1)
    return tuple(torch.as_tensor(array, device=device) for array in arrays)


def collect_diagnostic_rollout(agent, env, replay, steps, seed):
    """Collect fresh current-policy transitions without performing updates.

    This produces an on-policy dataset for actor–critic gradient measurement. The
    transitions are retained for diagnostics but actor–critic training never replays
    them later.
    """
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = agent.act(obs)
        nxt, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, nxt, float(terminated or truncated))
        obs = nxt
        if terminated or truncated:
            obs, _ = env.reset()


@torch.no_grad()
def evaluate(agent, env, episodes, seed):
    """Evaluate deterministic policy returns without altering training RNG state.

    Args:
        agent: Learner whose mean policy is evaluated.
        env: Separate evaluation environment.
        episodes: Number of fixed-horizon episodes to average.
        seed: Base seed; episode index is added for reproducible starts/noise.

    Returns:
        Pair ``(mean_return, fraction_of_episodes_with_any_success)``.
    """
    returns, successes = [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, success = 0.0, False
        while True:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            total += reward; success = success or info["success"]
            if terminated or truncated:
                break
        returns.append(total); successes.append(success)
    return float(np.mean(returns)), float(np.mean(successes))


def train_with_evaluations(agent, replay, config, angle, seed, prior_actor=None,
                           transfer_lambda=0.0, diagnostic_prior=None):
    """Train one branch and periodically record its deterministic learning curve.

    Args:
        agent: Restored scratch or transfer learner.
        replay: Independently restored probe replay buffer.
        config: Full parsed configuration.
        angle: Target task action-dynamics rotation.
        seed: Matched branch seed.
        prior_actor: Frozen source actor for transfer, or ``None`` for scratch.
        transfer_lambda: Strength of source-policy distillation.
        diagnostic_prior: Frozen source actor used only for periodic read-only
            compatibility measurements. It is supplied to both branches.

    Returns:
        Pair ``(learning_curve, alignment_curve)``. The second list is empty when
        periodic alignment is disabled.
    """
    exp = config["experiment"]
    train_env, eval_env = make_env(config, angle), make_env(config, angle)
    # Include evaluation at step zero so AUC covers the entire post-probe interval.
    points = list(range(0, exp["transfer_steps"] + 1, exp["eval_interval"]))
    if points[-1] != exp["transfer_steps"]:
        points.append(exp["transfer_steps"])
    curve, alignment_curve = [], []
    elapsed = 0
    for point in points:
        if point > elapsed:
            interact_train(agent, train_env, replay, point - elapsed, exp["batch_size"],
                           exp["learning_starts"], seed + elapsed, prior_actor,
                           transfer_lambda, exp.get("transfer_measure", "kl"))
        alignment_interval = exp.get("periodic_alignment_interval", exp["eval_interval"])
        if (exp.get("periodic_alignment", False) and diagnostic_prior is not None
                and point % alignment_interval == 0):
            # Import locally to keep general training utilities lightweight.
            from gradient_transfer_test.transfer import measure_alignment_preserving_rng
            measurement = measure_alignment_preserving_rng(
                agent,
                diagnostic_prior,
                replay,
                exp.get("periodic_alignment_batch_size", exp["probe_batch_size"]),
                exp.get("periodic_alignment_batches", exp["num_alignment_batches"]),
                num_action_samples=exp.get("diagnostic_action_samples", 1),
                recent_size=(agent.config.rollout_steps
                             if getattr(agent, "on_policy", False) else None),
                transfer_measure=exp.get("transfer_measure", "kl"),
            )
            alignment_curve.append({"step": point, **measurement})
        ret, success = evaluate(agent, eval_env, exp["eval_episodes"], seed + 100_000 + point)
        curve.append({"step": point, "return": ret, "success_rate": success})
        elapsed = point
    return curve, alignment_curve


def auc(curve):
    """Integrate evaluation return over training steps using trapezoids."""
    x = np.asarray([p["step"] for p in curve], dtype=float)
    y = np.asarray([p["return"] for p in curve], dtype=float)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def save_json(path, value):
    """Create parent directories and serialize ``value`` as indented JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
