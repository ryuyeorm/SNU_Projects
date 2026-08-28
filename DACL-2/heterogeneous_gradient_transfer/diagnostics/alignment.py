from __future__ import annotations

import numpy as np
import torch

from .gradient_utils import cosine_stats, flatten_gradients, task_gradient
from .kl import kl_loss
from .mean_mse import mse_loss
from .wasserstein import wasserstein_loss


LOSSES = {"mse": mse_loss, "wasserstein": wasserstein_loss, "kl": kl_loss}


def freeze_actor(actor):
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return actor


def compute_alignment_batches(target_actor, prior_actor, rollouts):
    """Measure every diagnostic on each rollout without changing either policy."""
    freeze_actor(prior_actor)
    before = [parameter.detach().clone() for parameter in target_actor.parameters()]
    task_gradients = []
    prior_gradients = {name: [] for name in LOSSES}
    distances = {name: [] for name in LOSSES}
    cosines = {name: [] for name in LOSSES}
    dots = {name: [] for name in LOSSES}
    task_norms = []
    prior_norms = {name: [] for name in LOSSES}
    for rollout in rollouts:
        states = rollout.observations
        task = task_gradient(target_actor, rollout)
        task_gradients.append(task)
        for name, function in LOSSES.items():
            loss = function(prior_actor, target_actor, states)
            prior = flatten_gradients(loss, target_actor.parameters())
            cosine, dot, task_norm, prior_norm = cosine_stats(task, prior)
            prior_gradients[name].append(prior)
            distances[name].append(float(loss.detach()))
            cosines[name].append(cosine); dots[name].append(dot)
            prior_norms[name].append(prior_norm)
        task_norms.append(float(task.norm()))
    result = {"task_gradient_norm": float(np.mean(task_norms))}
    average_task = torch.stack(task_gradients).mean(0)
    for name in LOSSES:
        average_prior = torch.stack(prior_gradients[name]).mean(0)
        avg_cosine, avg_dot, avg_task_norm, avg_prior_norm = cosine_stats(average_task, average_prior)
        result.update({
            f"C_{name}_avg_grad": avg_cosine,
            f"C_{name}_mean_cosine": float(np.mean(cosines[name])),
            f"{name}_gradient_norm": float(np.mean(prior_norms[name])),
            f"task_{name}_dot_product": avg_dot,
            f"raw_{name}_distance": float(np.mean(distances[name])),
        })
    if any(not torch.equal(old, new) for old, new in zip(before, target_actor.parameters())):
        raise RuntimeError("Diagnostic extraction changed target parameters")
    return result

