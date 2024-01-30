from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from noise_process import ColoredNoiseProcess


class ColoredActor(nn.Module):
    def __init__(self,
                 beta: float,
                 seq_len: int,
                 batch_size: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 edac_init: bool = False,
                 max_action: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        '''
        batch_size can also be considered as number of envs running in parallel
        '''
        super().__init__()
    
        # setup actor
        self.action_dim = action_dim
        self.max_action = max_action

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        if edac_init:
            # init as in the EDAC paper
            for layer in self.trunk[::2]:
                nn.init.constant_(layer.bias, 0.1)

            nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
            nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)

        # setup noise process
        self.beta = beta
        self.noise = ColoredNoiseProcess(beta=self.beta, size=[batch_size, action_dim, seq_len], rng=rng)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            I am not sure but I hope it works as it should do
        '''
        hidden = self.trunk(state)
        mu, log_std = self.mu(hidden), self.log_std(hidden)
        log_std = torch.clip(log_std, -20, 2)  # log_std = torch.clip(log_std, -5, 2) EDAC clipping

        eps = torch.from_numpy(self.noise.sample()).float().to(state.device)
        action = mu + eps * log_std

        log_prob = self.log_prob(mu, log_std, action)

        return torch.tanh(action) * self.max_action, log_prob
    
    def log_prob(self,
                 mu: torch.Tensor,
                 log_std: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        policy_distribution = Normal(mu, log_std.exp())

        log_prob = policy_distribution.log_prob(action).sum(-1)
        log_prob = log_prob - torch.log(1 - action.tanh().pow(2) + 1e-6).sum(-1)
        return log_prob


class PinkActor(ColoredActor):
    def __init__(self,
                 seq_len: int,
                 batch_size: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 edac_init: bool = False,
                 max_action: float = 1,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(1.0,
                         seq_len,
                         batch_size,
                         state_dim,
                         action_dim,
                         hidden_dim,
                         edac_init,
                         max_action,
                         rng)
