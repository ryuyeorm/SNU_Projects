"""Compute actor, critic, adversary, entropy, and regularization losses."""
import torch
from torch import Tensor

def td_target(
    rewards: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float,
) -> Tensor:
    not_done = 1.0 - dones.to(dtype=next_values.dtype)
    return rewards + gamma * not_done * next_values # if the game is terminated, it returns bootstrapped target


def advantages(
    targets: Tensor,
    values: Tensor,
)-> Tensor:
    return targets - values

def actor_loss(
    log_probabilities: Tensor,
    advantages: Tensor,
    entropies: Tensor,
    entropy_coefficient: float,
) -> Tensor:
    policy_loss = -(log_probabilities * advantages.detach()).mean()
    
    entropy_bonus = entropies.mean()
    
    return policy_loss - entropy_coefficient * entropy_bonus

def critic_loss(
        values: Tensor,
        targets: Tensor,
) -> Tensor:
    return 0.5 * (values - targets.detach()).pow(2).mean()

def adversary_loss(objective: Tensor) -> Tensor:
    return -objective.mean() # as the loss of adversary is just the negated objective of the player

