"""Source-policy freezing and Gaussian policy-distillation loss.

The actor acts through tanh-squashed Gaussians. This experiment deliberately uses
the closed-form KL between the *pre-tanh* diagonal Gaussians.  The same approximation
is used for both the diagnostic gradient and transfer treatment.
"""

from __future__ import annotations

import torch


def freeze_actor(actor):
    """Put a source actor in evaluation mode and permanently disable gradients.

    Args:
        actor: Trained source :class:`Actor` that serves as the policy prior.

    Returns:
        The same actor object, frozen in-place for convenient chaining.
    """
    actor.eval()
    actor.requires_grad_(False)
    return actor


def gaussian_kl(source_actor, target_actor, states):
    """Calculate mean ``KL(source || target)`` for pre-tanh Gaussians.

    Args:
        source_actor: Frozen actor defining the distribution being distilled.
        target_actor: Current learner actor; gradients flow only into this object.
        states: Target rollout states with shape ``[batch, observation_dimension]``.

    Returns:
        Scalar mean KL, summed over action dimensions and averaged over states.
    """
    # no_grad is a second line of defense in addition to freeze_actor().
    with torch.no_grad():
        source_mu, source_log_std = source_actor.distribution_params(states)
    target_mu, target_log_std = target_actor.distribution_params(states)
    # log(sigma^2) = 2*log(sigma), so exponentiation gives each variance.
    source_var, target_var = (2 * source_log_std).exp(), (2 * target_log_std).exp()
    # Closed-form one-dimensional normal KL; diagonal dimensions add independently.
    per_dim = target_log_std - source_log_std + (
        source_var + (source_mu - target_mu).pow(2)
    ) / (2 * target_var) - 0.5
    return per_dim.sum(-1).mean()
