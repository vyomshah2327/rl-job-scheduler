"""
PPONetwork.py
=============
Actor-Critic neural network for PPO.

Architecture based on DeepRM/Decima research:
- Shared feature extractor
- Separate actor (policy) and critic (value) heads
- Support for action masking
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic architecture for PPO job scheduling.
    
    Input: state (101-dim vector with timeline information)
    Outputs:
      - Actor: action distribution (masked softmax over 11 actions)
      - Critic: state value estimate (scalar)
    """
    
    def __init__(self, state_dim=101, action_dim=11, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        # Two hidden layers as recommended by Decima paper
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy network)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights (important for stable training)
        self._init_weights()
    
    def _init_weights(self):
        """
        Orthogonal initialization as used in PPO papers.
        Helps with training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Actor head gets smaller initialization (prevents overconfident early policy)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
    
    def forward(self, state, action_mask=None):
        """
        Forward pass through actor-critic network.
        
        Args:
            state: (batch_size, state_dim) tensor
            action_mask: (batch_size, action_dim) boolean tensor
                        True = valid action, False = invalid
        
        Returns:
            dist: Categorical distribution over valid actions
            value: State value estimates (batch_size,)
        """
        # Shared features
        features = self.shared(state)
        
        # Actor: compute action logits
        logits = self.actor(features)
        
        # Apply action mask (critical for scheduling!)
        # Invalid actions get -inf logits → 0 probability
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Create categorical distribution
        dist = Categorical(logits=logits)
        
        # Critic: compute state value
        value = self.critic(features)
        
        return dist, value.squeeze(-1)
    
    def get_value(self, state):
        """Get only the value estimate (used for advantage computation)."""
        features = self.shared(state)
        value = self.critic(features)
        return value.squeeze(-1)
    
    def evaluate_actions(self, state, action, action_mask=None):
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: log probability of taken actions
            value: state value estimates
            entropy: policy entropy (for exploration bonus)
        """
        dist, value = self.forward(state, action_mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy


# Need numpy for weight initialization
import numpy as np
