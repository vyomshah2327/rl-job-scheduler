"""
DQNAgent.py
===========
Double DQN agent with Huber loss.

Key design:
  - Double DQN: online net selects action, target net evaluates it
  - SmoothL1 (Huber) loss: robust to outlier rewards
  - Reward is already normalised in SchedulerEnv._get_reward()
    so no extra scaling needed here
  - Epsilon decays per episode using harmonic schedule 1/(1+k*t)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DQNNetwork   import DQNNetwork
from ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim:          int,
        action_dim:         int,
        lr:                 float = 5e-4,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        buffer_capacity:    int   = 50_000,
        batch_size:         int   = 256,
        target_update_freq: int   = 500,
        hidden_dim:         int   = 256,
    ):
        self.action_dim         = action_dim
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done         = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DQNAgent] Device: {self.device}")

        self.online_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    # ──────────────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(s).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ──────────────────────────────────────────────────────────────────────
    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q
        current_q = (self.online_net(states)
                     .gather(1, actions.unsqueeze(1))
                     .squeeze(1))

        # Double DQN target
        with torch.no_grad():
            next_acts = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q    = (self.target_net(next_states)
                         .gather(1, next_acts)
                         .squeeze(1))
            target_q  = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ──────────────────────────────────────────────────────────────────────
    def decay_epsilon(self, episode: int):
        """
        Harmonic epsilon decay: ε(t) = max(ε_end, 1 / (1 + k * t))

        Why harmonic over multiplicative (ε × 0.97):
          - Multiplicative falls exponentially — hits minimum too fast,
            leaving hundreds of near-identical low-epsilon episodes
          - Harmonic falls slowly at first (high exploration early),
            then stabilises — matches theoretical Q-learning convergence
            requirement that exploration decays but not too fast

        With k=0.05:
          ep=1  → ε=0.95   (lots of exploration)
          ep=10 → ε=0.67
          ep=50 → ε=0.29
          ep=100→ ε=0.17
          ep=200→ ε=0.09
          ep=300→ ε=0.06
          ep=400→ ε=0.05  (hits floor, stays there)

        This gives ~400 episodes of gradually decreasing exploration
        over a 500-episode run, much richer than multiplicative which
        hits 0.05 by episode 100 and stays flat for the remaining 400.
        """
        harmonic      = 1.0 / (1.0 + 0.05 * float(episode))
        self.epsilon  = max(self.epsilon_end, harmonic)

    def save(self, filepath: str):
        torch.save(self.online_net.state_dict(), filepath)
        print(f"[DQNAgent] Saved → {filepath}")

    def load(self, filepath: str):
        self.online_net.load_state_dict(
            torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"[DQNAgent] Loaded ← {filepath}")
