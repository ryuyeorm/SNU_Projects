"""Implement tabular RQE actor-critic training and policy updates."""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

class TabularRqeActorCritic(nn.Module):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_opponent_actions: int
    ) -> None:
        super().__init__()
        
        self.policy_logits = nn.Parameter(
            torch.zeros(num_states, num_actions)
        )
        
        self.values = nn.Parameter(
            torch.zeros(num_states)
        )
        
        self.adversary_logits = nn.Parameter(
            torch.zeros(num_states,num_opponent_actions)
        )
        
    
    def policy(self, states: Tensor) -> Categorical:
        logits = self.policy_logits[states]
        return Categorical(logits=logits)
    
    def adversary(self, states: Tensor) -> Categorical:
        logits = self.adversary_logits[states]
        return Categorical(logits= logits)
    
    def value(self, states: Tensor) -> Tensor:
        return self.values[states]
    
    def act(
        self,
        states: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        
        distribution = self.policy(states)
        
        if deterministic:
            actions = distribution.logits.argmax(dim=-1)
        else:
            actions = distribution.sample()
            
        
        return actions, distribution.log_prob(actions)
    
        
    def adversary_act(
        self,
        states: Tensor,
        deterministic=False,
    ) -> tuple[Tensor, Tensor]:
        
        distribution = self.adversary(states)
        
        if deterministic:
            actions = distribution.logits.argmax(dim=-1)
        else:
            actions = distribution.sample()
            
        
        return actions, distribution.log_prob(actions)
            
