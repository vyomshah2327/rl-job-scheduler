"""
PPOAgent.py
===========
Proximal Policy Optimization agent for job scheduling.

Implementation based on:
- PPO paper (Schulman et al., 2017)
- Decima scheduling paper (Mao et al., 2019)
- OpenAI Spinning Up guidelines

Key features:
- Action masking for valid scheduling decisions
- Clipped surrogate objective for stable updates
- GAE for advantage estimation
- Entropy bonus for exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from PPONetwork import ActorCriticNetwork
from RolloutBuffer import RolloutBuffer


class PPOAgent:
    """
    PPO agent for job scheduling with action masking.
    """
    
    def __init__(
        self,
        state_dim=101,
        action_dim=11,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=10000,
    ):
        """
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space
            hidden_dim: width of hidden layers
            lr: learning rate
            gamma: discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter (typically 0.1-0.3)
            entropy_coef: entropy bonus coefficient (encourages exploration)
            value_coef: value loss coefficient
            max_grad_norm: gradient clipping threshold
            ppo_epochs: number of epochs to train on each batch
            batch_size: minibatch size for updates
            buffer_size: size of rollout buffer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size, state_dim, action_dim, self.device
        )
        
        # Training stats
        self.update_count = 0
        
        print(f"[PPO] Device: {self.device}")
        print(f"[PPO] Network params: {sum(p.numel() for p in self.network.parameters()):,}")
    
    def select_action(self, state, action_mask, deterministic=False):
        """
        Select action using current policy.
        
        Args:
            state: (state_dim,) numpy array
            action_mask: (action_dim,) boolean numpy array
            deterministic: if True, pick argmax (for evaluation)
        
        Returns:
            action: int
            log_prob: float (for training)
            value: float (for training)
        """
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist, value = self.network(state, action_mask)
            
            if deterministic:
                # Greedy action (for evaluation)
                action = dist.probs.argmax(dim=-1)
            else:
                # Sample from distribution (for training)
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store(self, state, action, reward, value, log_prob, done, action_mask):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, reward, value, log_prob, done, action_mask)
    
    def update(self):
        """
        Update policy using collected experience (PPO update).
        
        Returns:
            dict with training metrics
        """
        # Get data from buffer with computed advantages
        data = self.buffer.get()
        
        states = data['states']
        actions = data['actions']
        old_log_probs = data['old_log_probs']
        advantages = data['advantages']
        returns = data['returns']
        action_masks = data['action_masks']
        
        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []
        
        # Number of samples
        n_samples = len(states)
        
        # PPO update loop: train on same data multiple times
        for epoch in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = torch.randperm(n_samples, device=self.device)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # Get current policy outputs
                new_log_probs, new_values, entropy = self.network.evaluate_actions(
                    states[batch_idx],
                    actions[batch_idx],
                    action_masks[batch_idx]
                )
                
                # ──────────────────────────────────────────────────────────
                # Policy loss (PPO clipped objective)
                # ──────────────────────────────────────────────────────────
                
                # Probability ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                
                # Surrogate objectives
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * advantages[batch_idx]
                
                # Policy loss: take minimum (conservative update)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ──────────────────────────────────────────────────────────
                # Value loss (MSE between predicted and actual returns)
                # ──────────────────────────────────────────────────────────
                value_loss = nn.functional.mse_loss(new_values, returns[batch_idx])
                
                # ──────────────────────────────────────────────────────────
                # Total loss
                # ──────────────────────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()  # Entropy bonus
                )
                
                # ──────────────────────────────────────────────────────────
                # Optimization step
                # ──────────────────────────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients)
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # ──────────────────────────────────────────────────────────
                # Record metrics
                # ──────────────────────────────────────────────────────────
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                # KL divergence (for monitoring policy change)
                with torch.no_grad():
                    kl = (old_log_probs[batch_idx] - new_log_probs).mean()
                    kl_divs.append(kl.item())
                    
                    # Clip fraction (how often clipping is active)
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_frac.item())
        
        # Clear buffer for next rollout
        self.buffer.clear()
        self.update_count += 1
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_div': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'update_count': self.update_count,
        }
    
    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, filepath)
        print(f"[PPO] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        print(f"[PPO] Model loaded from {filepath}")
