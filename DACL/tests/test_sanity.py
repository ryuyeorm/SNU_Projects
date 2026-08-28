"""Scientific sanity tests for geometry, gradients, freezing, and branching."""

import copy

import numpy as np
import torch

from gradient_transfer_test.agents.sac import Actor, ReplayBuffer, SACAgent, SACConfig, set_seed
from gradient_transfer_test.agents.actor_critic import ActorCriticAgent, ActorCriticConfig
from gradient_transfer_test.envs.point_mass import PointMassEnv
from gradient_transfer_test.transfer.gradient_alignment import (
    gradient_pair, measure_alignment_preserving_rng,
)
from gradient_transfer_test.transfer.kl_loss import freeze_actor, gaussian_kl
from gradient_transfer_test.transfer.wasserstein_loss import (
    gaussian_mean_mse, gaussian_wasserstein2, policy_distance,
)


def small_agent(seed=0):
    """Construct a fast, deterministically initialized test-size SAC agent."""
    set_seed(seed)
    return SACAgent(SACConfig(hidden_sizes=(16, 16)))


def dummy_batch(states):
    """Build a shape-correct transition batch around supplied test states."""
    n = len(states)
    return (states, torch.zeros(n, 2), torch.zeros(n, 1), states.clone(),
            torch.zeros(n, 1))


def test_action_toward_goal_decreases_distance():
    """A noiseless action along the goal vector must improve task geometry."""
    env = PointMassEnv(dynamics_angle=45, transition_noise=0, start_radius=0)
    _, first = env.reset(seed=0)
    # The policy must undo the task rotation to produce a world action toward g.
    policy_action = env.action_rotation.T @ env.goal
    _, _, _, _, second = env.step(policy_action)
    assert second["distance"] < first["distance"]


def test_tasks_share_goal_but_rotate_action_dynamics():
    """Task angles change transitions while leaving goal and reward geometry fixed."""
    zero = PointMassEnv(dynamics_angle=0, transition_noise=0)
    ninety = PointMassEnv(dynamics_angle=90, transition_noise=0)
    zero.reset(seed=0); ninety.reset(seed=0)
    zero_state = zero.step([1, 0])[0]
    ninety_state = ninety.step([1, 0])[0]
    assert np.array_equal(zero.goal, ninety.goal)
    assert np.allclose(zero.goal, [1, 0])
    assert np.allclose(zero_state, [0.1, 0.0], atol=1e-6)
    assert np.allclose(ninety_state, [0.0, 0.1], atol=1e-6)


def test_identical_policy_has_zero_kl_and_prior_gradient():
    """An actor compared with its exact copy has no distillation direction."""
    agent = small_agent()
    prior = freeze_actor(copy.deepcopy(agent.actor))
    states = torch.randn(32, 2)
    loss = gaussian_kl(prior, agent.actor, states)
    _, prior_grad, metrics = gradient_pair(agent, prior, dummy_batch(states))
    assert abs(float(loss.detach())) < 1e-6
    assert float(prior_grad.norm()) < 1e-6
    assert metrics["low_confidence"]


def test_gradient_diagnostic_does_not_update_parameters():
    """Extracting gradients must not silently act like an optimizer step."""
    agent, prior = small_agent(), freeze_actor(Actor(hidden_sizes=(16, 16)))
    before = [p.detach().clone() for p in agent.actor.parameters()]
    gradient_pair(agent, prior, dummy_batch(torch.randn(16, 2)))
    assert all(torch.equal(x, y) for x, y in zip(before, agent.actor.parameters()))


def test_source_is_frozen_and_receives_no_gradients():
    """Backpropagation through KL must affect only the current target actor."""
    agent, prior = small_agent(), freeze_actor(Actor(hidden_sizes=(16, 16)))
    gaussian_kl(prior, agent.actor, torch.randn(16, 2)).backward()
    assert all(not p.requires_grad and p.grad is None for p in prior.parameters())


def test_identical_policy_has_zero_wasserstein_and_gradient():
    """Squared W2 is smooth and stationary when source and target are identical."""
    agent = small_agent()
    prior = freeze_actor(copy.deepcopy(agent.actor))
    states = torch.randn(32, 2)
    loss = gaussian_wasserstein2(prior, agent.actor, states)
    gradient = torch.autograd.grad(loss, tuple(agent.actor.parameters()))
    assert abs(float(loss.detach())) < 1e-8
    assert torch.cat([item.flatten() for item in gradient]).norm() < 1e-6
    assert policy_distance(prior, agent.actor, states, "wasserstein") == loss


def test_mean_mse_ignores_policy_standard_deviation():
    """Mean matching is zero when means match, even if source variances differ."""
    agent = small_agent()
    prior = copy.deepcopy(agent.actor)
    with torch.no_grad():
        prior.log_std.bias.add_(1.0)
    freeze_actor(prior)
    states = torch.randn(32, 2)
    mean_loss = gaussian_mean_mse(prior, agent.actor, states)
    wasserstein_loss = gaussian_wasserstein2(prior, agent.actor, states)
    assert abs(float(mean_loss.detach())) < 1e-8
    assert float(wasserstein_loss.detach()) > 0
    assert policy_distance(prior, agent.actor, states, "mean_mse") == mean_loss


def test_agent_and_replay_branch_snapshots_match():
    """Restoring a branch point must reproduce networks and replay exactly."""
    original = small_agent(); state = copy.deepcopy(original.state_dict())
    a, b = small_agent(1), small_agent(2)
    a.load_state_dict(copy.deepcopy(state), restore_rng=False)
    b.load_state_dict(copy.deepcopy(state), restore_rng=False)
    assert all(torch.equal(x, y) for x, y in zip(a.actor.parameters(), b.actor.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(a.critic.parameters(), b.critic.parameters()))
    ra, rb = ReplayBuffer(capacity=10, seed=0), ReplayBuffer(capacity=10, seed=1)
    ra.add([0, 0], [1, 0], -1, [.1, 0], 0)
    rb.load_state_dict(ra.state_dict())
    assert np.array_equal(ra.obs, rb.obs) and ra.size == rb.size


def test_periodic_alignment_preserves_random_states():
    """Observational periodic diagnostics must not alter subsequent randomness."""
    agent, prior = small_agent(), freeze_actor(Actor(hidden_sizes=(16, 16)))
    replay = ReplayBuffer(capacity=20, seed=7)
    for i in range(20):
        replay.add([i, -i], [0, 0], -1, [i + 1, -i], 0)
    torch_state = torch.get_rng_state().clone()
    replay_state = copy.deepcopy(replay.rng.bit_generator.state)
    measure_alignment_preserving_rng(
        agent, prior, replay, batch_size=8, num_batches=3, num_action_samples=2
    )
    assert torch.equal(torch_state, torch.get_rng_state())
    assert replay_state == replay.rng.bit_generator.state


def test_actor_critic_uses_detached_td_advantage():
    """Policy loss must not send gradients into the value baseline."""
    set_seed(3)
    agent = ActorCriticAgent(ActorCriticConfig(hidden_sizes=(16, 16)))
    states = torch.randn(12, 2)
    actions = agent.actor.sample(states)[0].detach()
    batch = (states, actions, torch.randn(12, 1), torch.randn(12, 2),
             torch.zeros(12, 1))
    agent.diagnostic_actor_loss(batch).backward()
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert any(parameter.grad is not None for parameter in agent.actor.parameters())
