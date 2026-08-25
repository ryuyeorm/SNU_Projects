import torch

from rqe.buffers.replay_buffer import ReplayBuffer


def _add_transition(buffer: ReplayBuffer, value: float) -> None:
    scalar = torch.tensor(value)
    buffer.add(
        scalar,
        scalar.to(dtype=torch.long),
        scalar,
        scalar,
        torch.tensor(False),
    )


def test_circular_buffer_retains_newest_transitions_in_order():
    buffer = ReplayBuffer(capacity=3)
    for value in range(5):
        _add_transition(buffer, float(value))

    observations, _, _, _, _ = buffer.get()

    assert len(buffer) == 3
    torch.testing.assert_close(
        observations,
        torch.tensor([2.0, 3.0, 4.0]),
    )


def test_clear_resets_circular_write_position():
    buffer = ReplayBuffer(capacity=2)
    for value in range(3):
        _add_transition(buffer, float(value))
    buffer.clear()
    _add_transition(buffer, 10.0)

    observations, _, _, _, _ = buffer.get()

    torch.testing.assert_close(observations, torch.tensor([10.0]))
