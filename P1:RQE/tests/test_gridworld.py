"""Test cooperative gridworld transitions, rewards, observations, and termination."""

import torch

from rqe.envs.gridworld_cooperation import GridworldCooperation


def test_reset_and_regular_movement():
    environment = GridworldCooperation(seed=0)
    observation = environment.reset()
    torch.testing.assert_close(observation, torch.zeros(4))

    observation, rewards, done, info = environment.step([1, 3])

    torch.testing.assert_close(
        observation,
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(rewards, torch.zeros(2))
    assert not done
    assert info["positions"] == ((1, 0), (0, 1))


def test_both_agents_in_cooperation_zone_receive_two():
    environment = GridworldCooperation(
        cooperation_stay_probability=1.0,
        seed=0,
    )
    environment.reset()
    for _ in range(3):
        environment.step([1, 1])
    for _ in range(4):
        environment.step([3, 3])
    _, rewards, _, _ = environment.step([1, 1])

    assert environment.positions == ((4, 4), (4, 4))
    torch.testing.assert_close(rewards, torch.tensor([2.0, 2.0]))


def test_defection_and_cooperation_rewards_and_absorption():
    environment = GridworldCooperation(
        cooperation_stay_probability=1.0,
        seed=0,
    )
    environment.reset()
    environment.step([1, 3])
    for _ in range(3):
        environment.step([3, 3])
    environment.step([3, 1])
    for _ in range(3):
        environment.step([4, 1])
    _, rewards, _, _ = environment.step([0, 4])

    assert environment.positions == ((0, 4), (4, 4))
    torch.testing.assert_close(rewards, torch.tensor([3.0, 0.5]))

    environment.step([2, 4])
    assert environment.positions == ((0, 4), (4, 4))


def test_cooperator_receives_one_when_other_agent_is_blank():
    environment = GridworldCooperation(
        cooperation_stay_probability=1.0,
        seed=0,
    )
    environment.reset()
    for _ in range(4):
        environment.step([1, 4])
    for _ in range(4):
        _, rewards, _, _ = environment.step([3, 4])

    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0]))


def test_horizon_terminates_episode():
    environment = GridworldCooperation(horizon=2, seed=0)
    environment.reset()
    assert not environment.step([4, 4])[2]
    assert environment.step([4, 4])[2]
