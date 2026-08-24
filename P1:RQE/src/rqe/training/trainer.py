"""Training and evaluation loops for the gridworld experiments."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..algorithms.deep_rqe_ac import DeepRQEActorCritic
from ..envs.gridworld_cooperation import GridworldCooperation


@dataclass
class GridworldTrainingResult:
    """Per-episode measurements from one independent training run."""

    agent_returns: Tensor
    social_welfare: Tensor
    updates: int


def train_gridworld(
    agent: DeepRQEActorCritic,
    episodes: int = 20_000,
    seed: int = 0,
    updates_per_step: int = 1,
    log_interval: int = 0,
) -> GridworldTrainingResult:
    """Train one two-agent policy pair on the paper's gridworld."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if updates_per_step <= 0:
        raise ValueError("updates_per_step must be positive")

    torch.manual_seed(seed)
    environment = GridworldCooperation(horizon=50, seed=seed)
    episode_returns = torch.zeros(episodes, 2)
    updates = 0

    for episode in range(episodes):
        observation = environment.reset()
        done = False
        while not done:
            actions = agent.act(observation)
            next_observation, rewards, done, _ = environment.step(actions)
            # The 50-step horizon truncates the infinite-horizon process.
            agent.observe(
                observation,
                actions,
                rewards,
                next_observation,
                torch.tensor(False),
            )
            episode_returns[episode] += rewards
            observation = next_observation

            if len(agent.buffer) >= agent.batch_size:
                for _ in range(updates_per_step):
                    agent.update()
                    updates += 1

        if log_interval and (episode + 1) % log_interval == 0:
            start = max(0, episode + 1 - 100)
            recent = episode_returns[start : episode + 1].sum(dim=-1).mean()
            print(
                f"episode {episode + 1}/{episodes} "
                f"MA100 social welfare={recent.item():.3f}"
            )

    return GridworldTrainingResult(
        agent_returns=episode_returns,
        social_welfare=episode_returns.sum(dim=-1),
        updates=updates,
    )


@torch.no_grad()
def evaluate_gridworld(
    agent: DeepRQEActorCritic,
    episodes: int = 100,
    seed: int = 10_000,
    deterministic: bool = True,
) -> GridworldTrainingResult:
    """Evaluate a trained policy without adding data or updating networks."""
    environment = GridworldCooperation(horizon=50, seed=seed)
    episode_returns = torch.zeros(episodes, 2)
    for episode in range(episodes):
        observation = environment.reset()
        done = False
        while not done:
            actions = agent.act(observation, deterministic=deterministic)
            observation, rewards, done, _ = environment.step(actions)
            episode_returns[episode] += rewards
    return GridworldTrainingResult(
        agent_returns=episode_returns,
        social_welfare=episode_returns.sum(dim=-1),
        updates=0,
    )


def moving_average(values: Tensor, window: int = 100) -> Tensor:
    """Return a trailing average with the same length as the input."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window <= 0:
        raise ValueError("window must be positive")
    cumulative = torch.cat((values.new_zeros(1), values.cumsum(dim=0)))
    indices = torch.arange(1, values.numel() + 1, device=values.device)
    starts = (indices - window).clamp_min(0)
    totals = cumulative[indices] - cumulative[starts]
    counts = (indices - starts).to(values.dtype)
    return totals / counts
