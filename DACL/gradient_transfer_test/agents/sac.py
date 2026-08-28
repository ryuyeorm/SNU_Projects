"""Minimal, inspectable Soft Actor-Critic implementation for the experiment.

Only continuous two-dimensional observations/actions are needed here.  The actor is
a tanh-squashed diagonal Gaussian.  Twin critics reduce positive value bias, target
critics stabilize bootstrapping, and an automatically learned entropy temperature
balances exploration against return.  Explicit methods expose the actor objective and
complete training state so diagnostic gradients and matched experiment branches are
easy to audit.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch global random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Construct a fully connected network.

    Args:
        sizes: Width of the input, every hidden layer, and the output.
        activation: Module class inserted after hidden linear layers.
        output_activation: Module class inserted after the final linear layer.

    Returns:
        A sequential PyTorch module containing the requested layers.
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[i], sizes[i + 1]), act()])
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy.

    Args:
        obs_dim: Number of continuous observation coordinates.
        action_dim: Number of continuous action coordinates.
        hidden_sizes: Width of each shared feature layer.
    """
    def __init__(self, obs_dim=2, action_dim=2, hidden_sizes=(128, 128)):
        """Initialize the shared feature trunk and Gaussian parameter heads."""
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden_sizes], output_activation=nn.ReLU)
        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], action_dim)

    def distribution_params(self, obs):
        """Return pre-tanh Gaussian mean and bounded log standard deviation."""
        h = self.trunk(obs)
        return self.mu(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs, deterministic=False):
        """Draw differentiable actions and optionally calculate their log density.

        Args:
            obs: Batched observation tensor.
            deterministic: If true, squash the mean instead of sampling.

        Returns:
            Pair ``(actions, log_probabilities)``. Log probability is ``None`` for
            deterministic actions and includes the tanh change-of-variables term.
        """
        mu, log_std = self.distribution_params(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        # rsample uses reparameterization, allowing actor gradients through actions.
        raw = mu if deterministic else normal.rsample()
        action = torch.tanh(raw)
        log_prob = None
        if not deterministic:
            # Correct Gaussian density for the tanh transformation's Jacobian.
            log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def log_prob(self, obs, action):
        """Evaluate log density of already-squashed actions under this policy.

        Args:
            obs: Batched observations associated with ``action``.
            action: Batched bounded actions in ``[-1, 1]``.

        Returns:
            Column tensor containing the tanh-corrected action log densities.
        """
        mu, log_std = self.distribution_params(obs)
        normal = torch.distributions.Normal(mu, log_std.exp())
        # Avoid infinite inverse tanh at exact action-space boundaries.
        bounded = action.clamp(-1 + 1e-6, 1 - 1e-6)
        raw = torch.atanh(bounded)
        value = normal.log_prob(raw) - torch.log(1.0 - bounded.pow(2) + 1e-6)
        return value.sum(-1, keepdim=True)

    def entropy_estimate(self, obs):
        """Return pre-tanh diagonal-Gaussian entropy for regularization."""
        _, log_std = self.distribution_params(obs)
        return (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(-1, keepdim=True)


class Critic(nn.Module):
    """Twin action-value networks used by clipped double-Q SAC.

    Args:
        obs_dim: Observation vector width.
        action_dim: Action vector width.
        hidden_sizes: Width of each independent Q-network hidden layer.
    """
    def __init__(self, obs_dim=2, action_dim=2, hidden_sizes=(128, 128)):
        """Initialize two independent observation-action value networks."""
        super().__init__()
        self.q1 = mlp([obs_dim + action_dim, *hidden_sizes, 1])
        self.q2 = mlp([obs_dim + action_dim, *hidden_sizes, 1])

    def forward(self, obs, action):
        """Return both scalar Q estimates for each observation-action pair."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    """Fixed-capacity circular transition store with its own reproducible RNG.

    Args:
        obs_dim: Observation vector width.
        action_dim: Action vector width.
        capacity: Maximum number of transitions retained.
        seed: Seed for minibatch-index sampling.
    """
    def __init__(self, obs_dim=2, action_dim=2, capacity=100_000, seed=0):
        """Allocate transition arrays and initialize the private sampling RNG."""
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.dones = np.zeros((capacity, 1), np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        """Insert one transition, overwriting the oldest when full."""
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self.pos, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        """Sample transitions uniformly with replacement as device tensors.

        Args:
            batch_size: Number of transitions in the minibatch.
            device: PyTorch device receiving the tensors.
        """
        ids = self.rng.integers(0, self.size, size=batch_size)
        conv = lambda x: torch.as_tensor(x[ids], device=device)
        return tuple(map(conv, (self.obs, self.actions, self.rewards, self.next_obs, self.dones)))

    def sample_recent(self, batch_size, recent_size, device="cpu"):
        """Sample from only the newest ``recent_size`` transitions.

        This supports approximately on-policy periodic diagnostics without using old
        actor–critic rollouts. Sampling remains with replacement.
        """
        count = min(int(recent_size), self.size)
        offsets = self.rng.integers(0, count, size=batch_size)
        ids = (self.pos - 1 - offsets) % self.capacity
        conv = lambda x: torch.as_tensor(x[ids], device=device)
        return tuple(map(conv, (self.obs, self.actions, self.rewards, self.next_obs, self.dones)))

    def sample_sequence(self, batch_size, recent_size, device="cpu"):
        """Return one chronological sequence from the newest transition window."""
        count = min(int(recent_size), self.size)
        length = min(int(batch_size), count)
        # Offset zero denotes the oldest valid start inside the recent window.
        max_start = count - length
        start = int(self.rng.integers(0, max_start + 1)) if max_start else 0
        oldest = (self.pos - count) % self.capacity
        ids = (oldest + start + np.arange(length)) % self.capacity
        conv = lambda x: torch.as_tensor(x[ids], device=device)
        return tuple(map(conv, (self.obs, self.actions, self.rewards, self.next_obs, self.dones)))

    def state_dict(self):
        """Deep-copy all data, cursor state, and sampler RNG for exact branching."""
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        """Restore a snapshot produced by :meth:`state_dict`."""
        self.__dict__ = copy.deepcopy(state)

    def clear(self):
        """Discard stored transitions while retaining allocation and sampler RNG."""
        self.pos, self.size = 0, 0


@dataclass
class SACConfig:
    """Hyperparameters controlling SAC optimization and network capacity.

    Attributes:
        gamma: Future-reward discount factor.
        tau: Polyak interpolation fraction for target critics.
        learning_rate: Shared Adam learning rate for actor, critic, and temperature.
        init_alpha: Initial entropy-temperature value.
        hidden_sizes: Hidden widths used by actor and critics.
    """
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    init_alpha: float = 0.2
    hidden_sizes: tuple[int, ...] = (128, 128)


class SACAgent:
    """SAC networks, optimizers, losses, updates, and serializable state.

    Args:
        config: Optional :class:`SACConfig`; defaults are used when omitted.
        device: PyTorch device string or object, such as ``"cpu"`` or ``"cuda"``.
    """
    def __init__(self, config: SACConfig | None = None, device="cpu"):
        """Initialize online/target networks, optimizers, and entropy state."""
        self.config, self.device = config or SACConfig(), torch.device(device)
        c = self.config
        self.actor = Actor(hidden_sizes=c.hidden_sizes).to(self.device)
        self.critic = Critic(hidden_sizes=c.hidden_sizes).to(self.device)
        # Target critics start identical and are updated only through Polyak averaging.
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=c.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c.learning_rate)
        self.log_alpha = torch.tensor(np.log(c.init_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=c.learning_rate)
        self.target_entropy = -2.0

    @property
    def alpha(self):
        """Return positive entropy temperature by exponentiating its log parameter."""
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Convert one NumPy observation into one bounded NumPy action.

        Args:
            obs: Single environment observation.
            deterministic: Use the policy mean for evaluation when true.
        """
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.sample(x, deterministic)[0].cpu().numpy()[0]

    def actor_loss(self, obs, num_action_samples=1):
        """Calculate the standard SAC actor objective on target states.

        The critic values are differentiated through with respect to actions and actor
        parameters, but only the actor optimizer consumes this objective's gradients.

        Args:
            obs: Batched target-task observations.
            num_action_samples: Independent reparameterized actions drawn per state.
                Training uses one for efficiency; diagnostics may average several to
                reduce Monte Carlo noise in the measured task-gradient direction.
        """
        if num_action_samples < 1:
            raise ValueError("num_action_samples must be at least one")
        losses = []
        for _ in range(num_action_samples):
            action, log_prob = self.actor.sample(obs)
            q1, q2 = self.critic(obs, action)
            losses.append((self.alpha.detach() * log_prob - torch.minimum(q1, q2)).mean())
        return torch.stack(losses).mean()

    def diagnostic_actor_loss(self, batch, num_action_samples=1):
        """Adapt a replay transition batch to the shared diagnostic interface."""
        return self.actor_loss(batch[0], num_action_samples)

    def update(self, batch, prior_actor=None, transfer_lambda=0.0,
               transfer_measure="kl"):
        """Perform one critic, actor, entropy, and target-network SAC update.

        Args:
            batch: ``(obs, action, reward, next_obs, terminal)`` tensors.
            prior_actor: Optional frozen source actor for the transfer treatment.
            transfer_lambda: Weight on pre-tanh ``KL(source || target)``. Zero gives
                the scratch condition.
            transfer_measure: ``kl``, squared ``wasserstein``, or ``mean_mse``.

        Returns:
            Detached scalar actor and critic losses for optional logging.
        """
        obs, action, reward, next_obs, done = batch
        # Build a one-step entropy-regularized Bellman target without gradients.
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            tq1, tq2 = self.target_critic(next_obs, next_action)
            target = reward + self.config.gamma * (1 - done) * (
                torch.minimum(tq1, tq2) - self.alpha.detach() * next_log_prob
            )
        # Fit both critics to the same conservative target.
        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # The treatment changes only this actor objective.
        actor_loss = self.actor_loss(obs)
        if prior_actor is not None and transfer_lambda:
            from gradient_transfer_test.transfer import policy_distance
            actor_loss = actor_loss + transfer_lambda * policy_distance(
                prior_actor, self.actor, obs, transfer_measure
            )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Tune alpha toward the desired two-action-dimensional entropy.
        _, log_prob = self.actor.sample(obs)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Slowly move target critics toward the newly updated online critics.
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.mul_(1 - self.config.tau).add_(p, alpha=self.config.tau)
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}

    def state_dict(self):
        """Snapshot all trainable state plus global RNG states for fair branching."""
        return {
            "config": self.config, "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(), "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        }

    def load_state_dict(self, state, restore_rng=True):
        """Restore a complete agent snapshot.

        Args:
            state: Mapping returned by :meth:`state_dict`.
            restore_rng: Also restore Python, NumPy, and PyTorch global RNG states.
        """
        for key in ("actor", "critic", "target_critic"):
            getattr(self, key).load_state_dict(state[key])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])

    def clone(self):
        """Create an independent agent with identical model, optimizer, and RNG state."""
        state = copy.deepcopy(self.state_dict())
        other = SACAgent(self.config, self.device)
        other.load_state_dict(state)
        return other
