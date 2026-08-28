"""Actor-gradient extraction and directional compatibility measurements.

The functions in this module never call an optimizer. They compare the target
actor gradient and the source-distillation gradient over the same actor parameters
and, for each pair, exactly the same batch of target states.
"""

from __future__ import annotations

import copy
import random
import numpy as np
import torch

from .wasserstein_loss import policy_distance


def _flat_grad(loss, params):
    """Return one flat vector containing ``d(loss)/d(params)``.

    ``allow_unused`` makes the diagnostic robust to future architectures containing
    conditionally unused actor parameters; such entries are represented by zeros.
    """
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p, g in zip(params, grads)])


def gradient_pair(agent, prior_actor, batch, eps=1e-12, low_norm=1e-6,
                  num_action_samples=1, transfer_measure="kl"):
    """Measure task/prior actor gradients on one shared state batch.

    Args:
        agent: Current target agent supplying actor loss and actor parameters.
        prior_actor: Frozen source actor used in the KL loss.
        batch: Shared target-task transition batch. Actor–critic uses every field;
            source KL uses the batch's state field. SAC ignores non-state fields.
        eps: Numerical stabilizer added to the cosine denominator.
        low_norm: Gradient norm below which the cosine is flagged low-confidence.
        num_action_samples: Legacy SAC action samples averaged per state. Actor–critic
            instead uses the actions stored in its on-policy rollout.
        transfer_measure: Policy loss inducing the prior gradient: ``kl``,
            ``wasserstein`` (smooth squared ``W_2``), or mean-only ``mean_mse``.

    Returns:
        ``(task_gradient, prior_gradient, metrics)`` with detached flat vectors and
        cosine, dot product, individual norms, and confidence flag.
    """
    params = tuple(agent.actor.parameters())
    states = batch[0]
    task = _flat_grad(
        agent.diagnostic_actor_loss(batch, num_action_samples), params
    ).detach()
    prior = _flat_grad(
        policy_distance(prior_actor, agent.actor, states, transfer_measure), params
    ).detach()
    dot = torch.dot(task, prior)
    tn, pn = task.norm(), prior.norm()
    cosine = dot / (tn * pn + eps)
    return task, prior, {"cosine": float(cosine), "dot": float(dot), "task_norm": float(tn),
                         "prior_norm": float(pn), "low_confidence": bool(tn < low_norm or pn < low_norm)}


def measure_alignment(agent, prior_actor, replay, batch_size, num_batches, eps=1e-12,
                      low_norm=1e-6, num_action_samples=1, recent_size=None,
                      transfer_measure="kl"):
    """Aggregate compatibility over several independently sampled replay batches.

    Args:
        agent: Probed target learner whose actor must remain unchanged.
        prior_actor: Frozen source policy.
        replay: Target-task replay buffer providing diagnostic states.
        batch_size: Number of shared states per gradient pair.
        num_batches: Number of noisy gradient measurements to aggregate.
        eps: Cosine numerical stabilizer.
        low_norm: Threshold used to flag uninformative gradient directions.
        num_action_samples: Independent actor samples averaged within each task
            gradient measurement.
        recent_size: When set, restrict sampling to this many newest transitions.
        transfer_measure: ``kl``, ``wasserstein``, or ``mean_mse`` prior loss.

    Returns:
        Dictionary containing batchwise cosine mean/std, cosine after averaging the
        gradients, mean dot product/norms, and a combined confidence flag.
    """
    records, task_grads, prior_grads = [], [], []
    # Snapshot the actor so this read-only diagnostic can assert non-mutation.
    before = [p.detach().clone() for p in agent.actor.parameters()]
    for _ in range(num_batches):
        if getattr(agent, "on_policy", False):
            # GAE requires chronological data; initial probe and periodic windows are
            # therefore sampled as intact sequences rather than shuffled transitions.
            window = replay.size if recent_size is None else recent_size
            batch = replay.sample_sequence(batch_size, window, agent.device)
        else:
            batch = (replay.sample_recent(batch_size, recent_size, agent.device)
                     if recent_size is not None else replay.sample(batch_size, agent.device))
        tg, pg, rec = gradient_pair(
            agent, prior_actor, batch, eps, low_norm, num_action_samples,
            transfer_measure
        )
        task_grads.append(tg); prior_grads.append(pg); records.append(rec)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    # This is distinct from the mean of batch cosines and is saved separately.
    mt, mp = torch.stack(task_grads).mean(0), torch.stack(prior_grads).mean(0)
    avg_cos = float(torch.dot(mt, mp) / (mt.norm() * mp.norm() + eps))
    return {
        "mean_cosine_alignment": float(np.mean([r["cosine"] for r in records])),
        "std_cosine_alignment": float(np.std([r["cosine"] for r in records], ddof=1)) if num_batches > 1 else 0.0,
        "avg_gradient_cosine": avg_cos,
        "gradient_dot_product": float(np.mean([r["dot"] for r in records])),
        "task_gradient_norm": float(np.mean([r["task_norm"] for r in records])),
        "prior_gradient_norm": float(np.mean([r["prior_norm"] for r in records])),
        "low_confidence": any(r["low_confidence"] for r in records),
    }


def measure_alignment_preserving_rng(agent, prior_actor, replay, batch_size,
                                     num_batches, eps=1e-12, low_norm=1e-6,
                                     num_action_samples=1, recent_size=None,
                                     transfer_measure="kl"):
    """Measure compatibility without perturbing later training randomness.

    Diagnostic sampling consumes both the replay buffer's index RNG and PyTorch's
    policy-action RNG. Periodic measurement would otherwise change future minibatches
    and actions, confounding the learning curve with the act of observing it. This
    wrapper snapshots and restores Python, NumPy, CPU/CUDA PyTorch, and replay RNGs.

    Args:
        agent: Current target actor–critic or legacy SAC learner.
        prior_actor: Frozen source actor.
        replay: Current target replay buffer.
        batch_size: States per diagnostic batch.
        num_batches: Gradient pairs to aggregate.
        eps: Cosine numerical stabilizer.
        low_norm: Low-confidence gradient-norm threshold.
        num_action_samples: Actor samples averaged per task-gradient batch.
        recent_size: Optional newest-transition window for on-policy diagnostics.
        transfer_measure: ``kl``, ``wasserstein``, or ``mean_mse`` prior loss.

    Returns:
        The same aggregate dictionary as :func:`measure_alignment`.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    replay_rng_state = copy.deepcopy(replay.rng.bit_generator.state)
    try:
        return measure_alignment(
            agent, prior_actor, replay, batch_size, num_batches, eps, low_norm,
            num_action_samples, recent_size, transfer_measure
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        replay.rng.bit_generator.state = replay_rng_state
