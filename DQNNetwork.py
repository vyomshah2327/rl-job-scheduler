"""
DQNNetwork.py
=============
Neural network approximating Q(s, a).

Paper specs (§4):
  "a neural network with a fully connected hidden layer with 20 neurons
   and a total of 89,451 parameters"

The paper uses policy gradient (REINFORCE) with a large input image.
We use DQN (value-based) with the compact state vector (91 dims).
To match roughly the same parameter scale with our smaller input we use
two hidden layers of 256 neurons → ~113k parameters.

Architecture:  91 → 256 → 256 → 11  (11 = M + 1 = 10 slots + void)
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
