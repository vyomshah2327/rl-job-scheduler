"""
RolloutBuffer.py
================
Experience buffer for PPO on-policy learning.

Stores full trajectories and computes advantages using GAE.
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing and processing PPO rollouts.
    
    PPO is on-policy, so we collect a batch of experience,
    update the policy, then discard the data.
    """
    
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, action_dim), dtype=bool)
        
        self.ptr = 0
        self.full = False
    
    def add(self, state, action, reward, value, log_prob, done, action_mask):
        """Add one transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0
    
    def get(self):
        """
        Get all data from buffer and compute advantages using GAE.
        
        Returns:
            All arrays as PyTorch tensors on the correct device.
        """
        # Get actual data size
        size = self.buffer_size if self.full else self.ptr
        
        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages, returns = self._compute_gae(size)
        
        # Convert to tensors
        data = {
            'states': torch.FloatTensor(self.states[:size]).to(self.device),
            'actions': torch.LongTensor(self.actions[:size]).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs[:size]).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'returns': torch.FloatTensor(returns).to(self.device),
            'action_masks': torch.BoolTensor(self.action_masks[:size]).to(self.device),
        }
        
        return data
    
    def _compute_gae(self, size, gamma=0.99, lambda_=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE-Lambda smoothly interpolates between:
        - TD(0): low variance, high bias
        - Monte Carlo: high variance, low bias
        
        Args:
            size: number of transitions to process
            gamma: discount factor
            lambda_: GAE parameter (higher = more Monte Carlo)
        
        Returns:
            advantages: (size,) array
            returns: (size,) array for training critic
        """
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0
        
        # Work backwards through the trajectory
        for t in reversed(range(size)):
            if t == size - 1:
                # Last step in buffer
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[t] + 
                gamma * next_value * next_non_terminal - 
                self.values[t]
            )
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        # Returns for training the critic
        returns = advantages + self.values[:size]
        
        return advantages, returns
    
    def clear(self):
        """Reset buffer for next rollout."""
        self.ptr = 0
        self.full = False
    
    def __len__(self):
        """Current buffer size."""
        return self.buffer_size if self.full else self.ptr
