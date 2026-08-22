"""Deep two-agent RQE actor-critic based on Algorithm 2 of the paper."""

from copy import deepcopy

import torch
from torch import Tensor, nn

from ..buffers.replay_buffer import ReplayBuffer
from ..core.regularizers import entropy, kl_divergence
from ..models.actor import Actor
from ..models.adversary import Adversary
from ..models.joint_action_critic import JointActionCritic


class DeepRQEActorCritic(nn.Module):
    """Twin-Q, off-policy RQE actor-critic for two discrete-action agents."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_learning_rate: float = 5e-4,
        critic_learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 5.0,
        epsilon: float = 0.2,
        target_update: float = 0.002,
        batch_size: int = 256,
        replay_capacity: int = 100_000,
        risk_averse: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if not 0.0 < target_update <= 1.0:
            raise ValueError("target_update must be in (0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.target_update = target_update
        self.batch_size = batch_size
        self.risk_averse = risk_averse
        self.compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.compute_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.actors = nn.ModuleList(
            Actor(observation_dim, action_dim, hidden_dim) for _ in range(2)
        )
        self.adversaries = nn.ModuleList(
            Adversary(observation_dim, action_dim, hidden_dim) for _ in range(2)
        )
        self.critics = nn.ModuleList(
            nn.ModuleList(
                JointActionCritic(
                    observation_dim,
                    action_dim,
                    action_dim,
                    hidden_dim,
                )
                for _ in range(2)
            )
            for _ in range(2)
        )
        self.target_critics = deepcopy(self.critics)
        for parameter in self.target_critics.parameters():
            parameter.requires_grad_(False)
        self.to(self.compute_device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_learning_rate)
            for actor in self.actors
        ]
        self.adversary_optimizers = [
            torch.optim.Adam(adversary.parameters(), lr=actor_learning_rate)
            for adversary in self.adversaries
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critics.parameters(), lr=critic_learning_rate)
            for critics in self.critics
        ]
        self.buffer = ReplayBuffer(capacity=replay_capacity)

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        observation = observation.to(self.compute_device)
        actions = [
            actor.act(observation, deterministic=deterministic)[0]
            for actor in self.actors
        ]
        return torch.stack(actions).cpu()

    def observe(
        self,
        observation: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observation: Tensor,
        done: Tensor,
    ) -> None:
        self.buffer.add(
            observation,
            actions,
            rewards,
            next_observation,
            done,
        )

    def update(self) -> dict[str, float]:
        """Sample one replay batch and apply actor, adversary, and Q updates."""
        if len(self.buffer) < self.batch_size:
            raise RuntimeError(
                f"need {self.batch_size} transitions, have {len(self.buffer)}"
            )
        observations, actions, rewards, next_observations, dones = (
            self.buffer.sample(self.batch_size)
        )
        observations = observations.to(self.compute_device)
        actions = actions.to(self.compute_device)
        rewards = rewards.to(self.compute_device)
        next_observations = next_observations.to(self.compute_device)
        dones = dones.to(self.compute_device)
        dones = dones.to(dtype=observations.dtype)

        policy_probabilities = [
            torch.softmax(actor(observations), dim=-1)
            for actor in self.actors
        ]
        adversary_probabilities = [
            adversary(observations) for adversary in self.adversaries
        ]

        actor_losses: list[Tensor] = []
        adversary_losses: list[Tensor] = []
        for agent in range(2):
            q_values = torch.maximum(
                self.critics[agent][0](observations),
                self.critics[agent][1](observations),
            ).detach()
            own_policy = policy_probabilities[agent]
            opponent_policy = policy_probabilities[1 - agent]

            if agent == 0:
                actor_value = torch.einsum(
                    "bi,bij,bj->b",
                    own_policy,
                    q_values,
                    adversary_probabilities[agent].detach()
                    if self.risk_averse
                    else opponent_policy.detach(),
                )
            else:
                actor_value = torch.einsum(
                    "bi,bij,bj->b",
                    adversary_probabilities[agent].detach()
                    if self.risk_averse
                    else opponent_policy.detach(),
                    q_values,
                    own_policy,
                )
            actor_losses.append(
                (actor_value - self.epsilon * entropy(own_policy)).mean()
            )

            if self.risk_averse:
                adversary_policy = adversary_probabilities[agent]
                if agent == 0:
                    adversary_value = torch.einsum(
                        "bi,bij,bj->b",
                        own_policy.detach(),
                        q_values,
                        adversary_policy,
                    )
                else:
                    adversary_value = torch.einsum(
                        "bi,bij,bj->b",
                        adversary_policy,
                        q_values,
                        own_policy.detach(),
                    )
                adversary_losses.append(
                    (
                        -adversary_value
                        + kl_divergence(
                            adversary_policy,
                            opponent_policy.detach(),
                        )
                        / self.tau
                    ).mean()
                )

        for optimizer, loss in zip(self.actor_optimizers, actor_losses):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.risk_averse:
            for optimizer, loss in zip(
                self.adversary_optimizers,
                adversary_losses,
            ):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            next_policies = [
                torch.softmax(actor(next_observations), dim=-1)
                for actor in self.actors
            ]
            next_adversaries = [
                adversary(next_observations) for adversary in self.adversaries
            ]
            targets = [
                self._critic_target(
                    agent,
                    rewards[:, agent],
                    next_observations,
                    dones,
                    next_policies,
                    next_adversaries,
                )
                for agent in range(2)
            ]

        critic_losses: list[Tensor] = []
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        for agent in range(2):
            q_1 = self.critics[agent][0](observations)[
                batch_indices, actions[:, 0], actions[:, 1]
            ]
            q_2 = self.critics[agent][1](observations)[
                batch_indices, actions[:, 0], actions[:, 1]
            ]
            loss = (q_1 - targets[agent]).square().mean()
            loss = loss + (q_2 - targets[agent]).square().mean()
            critic_losses.append(loss)
            self.critic_optimizers[agent].zero_grad()
            loss.backward()
            self.critic_optimizers[agent].step()

        self._soft_update_targets()
        metrics = {
            "actor_0_loss": actor_losses[0].item(),
            "actor_1_loss": actor_losses[1].item(),
            "critic_0_loss": critic_losses[0].item(),
            "critic_1_loss": critic_losses[1].item(),
        }
        if self.risk_averse:
            metrics.update(
                {
                    "adversary_0_loss": adversary_losses[0].item(),
                    "adversary_1_loss": adversary_losses[1].item(),
                }
            )
        return metrics

    def _critic_target(
        self,
        agent: int,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
        policies: list[Tensor],
        adversaries: list[Tensor],
    ) -> Tensor:
        q_values = torch.maximum(
            self.target_critics[agent][0](next_observations),
            self.target_critics[agent][1](next_observations),
        )
        own_policy = policies[agent]
        opponent_policy = policies[1 - agent]
        imagined_opponent = (
            adversaries[agent] if self.risk_averse else opponent_policy
        )
        if agent == 0:
            next_value = torch.einsum(
                "bi,bij,bj->b", own_policy, q_values, imagined_opponent
            )
        else:
            next_value = torch.einsum(
                "bi,bij,bj->b", imagined_opponent, q_values, own_policy
            )
        regularized_value = next_value - self.epsilon * entropy(own_policy)
        if self.risk_averse:
            regularized_value = regularized_value - kl_divergence(
                imagined_opponent,
                opponent_policy,
            ) / self.tau
        return -rewards + self.gamma * (1.0 - dones) * regularized_value

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for critics, targets in zip(self.critics, self.target_critics):
            for critic, target in zip(critics, targets):
                for parameter, target_parameter in zip(
                    critic.parameters(), target.parameters()
                ):
                    target_parameter.lerp_(parameter, self.target_update)
