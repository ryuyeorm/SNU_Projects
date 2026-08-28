"""Squared 2-Wasserstein policy distance for diagonal Gaussian actors.

For diagonal Gaussians the covariance matrices commute, reducing the closed-form
Fréchet/Wasserstein expression to the squared Euclidean distance between means plus
the squared Euclidean distance between standard deviations. As with the KL option,
the calculation uses distributions before the actor's tanh transformation.
"""

from __future__ import annotations

import torch


def gaussian_wasserstein2(source_actor, target_actor, states):
    """Calculate mean squared 2-Wasserstein distance between policy Gaussians.

    Args:
        source_actor: Frozen actor defining the source policy distribution.
        target_actor: Current learner actor receiving gradients.
        states: Target rollout states shaped ``[batch, observation_dimension]``.

    Returns:
        Scalar ``W_2^2`` summed across action dimensions and averaged across states.

    Notes:
        For diagonal Gaussians ``N(mu_s, diag(sigma_s^2))`` and
        ``N(mu_t, diag(sigma_t^2))``:

        ``W_2^2 = ||mu_s-mu_t||^2 + ||sigma_s-sigma_t||^2``.
    """
    with torch.no_grad():
        source_mu, source_log_std = source_actor.distribution_params(states)
    target_mu, target_log_std = target_actor.distribution_params(states)
    mean_term = (source_mu - target_mu).pow(2)
    std_term = (source_log_std.exp() - target_log_std.exp()).pow(2)
    return (mean_term + std_term).sum(-1).mean()


def gaussian_mean_mse(source_actor, target_actor, states):
    """Calculate mean squared distance between policy Gaussian means only.

    Args:
        source_actor: Frozen source actor.
        target_actor: Current target actor receiving gradients.
        states: Target rollout states.

    Returns:
        Scalar ``E_s[||mu_source(s)-mu_target(s)||_2^2]``. Standard deviations are
        deliberately ignored, making this the action-mean matching baseline.
    """
    with torch.no_grad():
        source_mu, _ = source_actor.distribution_params(states)
    target_mu, _ = target_actor.distribution_params(states)
    return (source_mu - target_mu).pow(2).sum(-1).mean()


def policy_distance(source_actor, target_actor, states, measure="kl"):
    """Dispatch to the configured differentiable source-policy distance.

    Args:
        source_actor: Frozen source policy.
        target_actor: Current target policy.
        states: Target-task states shared with the actor-gradient diagnostic.
        measure: ``"kl"``, ``"wasserstein"``, or ``"mean_mse"``.

    Returns:
        Scalar selected policy-distance loss.
    """
    if measure == "kl":
        from .kl_loss import gaussian_kl
        return gaussian_kl(source_actor, target_actor, states)
    if measure == "wasserstein":
        return gaussian_wasserstein2(source_actor, target_actor, states)
    if measure == "mean_mse":
        return gaussian_mean_mse(source_actor, target_actor, states)
    raise ValueError(
        f"Unknown transfer_measure {measure!r}; expected 'kl', 'wasserstein', "
        "or 'mean_mse'"
    )
