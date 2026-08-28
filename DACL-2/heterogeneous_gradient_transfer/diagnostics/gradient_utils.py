from __future__ import annotations

import torch


def flatten_gradients(loss, parameters):
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                      for parameter, gradient in zip(parameters, gradients)]).detach()


def cosine_stats(left, right, epsilon=1e-12):
    dot = torch.dot(left, right)
    left_norm, right_norm = left.norm(), right.norm()
    cosine = dot / (left_norm * right_norm + epsilon)
    return float(cosine), float(dot), float(left_norm), float(right_norm)


def task_gradient(actor, rollout):
    loss = -(actor.log_prob(rollout.observations, rollout.actions) * rollout.advantages).mean()
    return flatten_gradients(loss, actor.parameters())

