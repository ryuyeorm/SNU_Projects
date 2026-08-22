"""Store, batch, and sample transitions or rollout data used during training."""
import torch
from torch import Tensor

class ReplayBuffer:
    def __init__(
        self,
        capacity: int | None = None,
    ) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.buffer: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []


    def add(
            self,
            observation: Tensor,
            action: Tensor,
            reward: Tensor,
            next_observation: Tensor,
            done: Tensor
    ) -> None:
        transition = (
            observation.detach().clone(),
            action.detach().clone(),
            reward.detach().clone(),
            next_observation.detach().clone(),
            done.detach().clone(),
        )
        self.buffer.append(transition)
        if self.capacity is not None and len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        

    def get(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.buffer:
            raise ValueError("Cannot get data from an empty buffer")
        observations, actions, rewards, next_observations, dones = zip(
            *self.buffer
        )

        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_observations),
            torch.stack(dones),
        )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample transitions uniformly without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot exceed the buffer length")
        indices = torch.randperm(len(self.buffer))[:batch_size].tolist()
        transitions = [self.buffer[index] for index in indices]
        fields = zip(*transitions)
        return tuple(torch.stack(field) for field in fields)  # type: ignore[return-value]

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
