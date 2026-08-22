import torch

from rqe.models.actor import Actor
from rqe.models.critic import Critic
from rqe.training.updater import Updater


def test_updater_changes_actor_and_critic_parameters():
    actor = Actor(4, 2, hidden_dim=16)
    critic = Critic(4, hidden_dim=16)

    updater = Updater(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
    )

    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]

    observations = torch.randn(8, 4)
    next_observations = torch.randn(8, 4)
    rewards = torch.randn(8)
    dones = torch.zeros(8, dtype=torch.bool)

    with torch.no_grad():
        actions, _, _ = actor.act(observations)

    metrics = updater.update(
        observations,
        actions,
        rewards,
        next_observations,
        dones,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters())
    )
    assert set(metrics) == {"actor_loss", "critic_loss", "entropy"}