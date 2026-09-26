"""
PPOTrainer.py
=============
Training harness for PPO job scheduling agent.

Features:
- Rolling episode collection
- Periodic policy updates
- Detailed progress logging
- Evaluation on test set
- Model checkpointing
"""

import numpy as np
import pandas as pd
import time
from collections import deque

from SchedulerEnv import SchedulerEnv
from PPOAgent import PPOAgent
from Data import CPU_CAP, MEM_CAP


class PPOTrainer:
    """
    PPO training manager with comprehensive logging.
    """
    
    def __init__(
        self,
        train_df,
        test_df,
        num_episodes=500,
        num_machines=10,
        max_jobs=5000,
        update_freq=5,  # Update policy every N episodes
        # PPO hyperparameters
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=10000,
        hidden_dim=256,
    ):
        """
        Args:
            train_df: training dataset
            test_df: test dataset
            num_episodes: number of training episodes
            num_machines: number of machines in cluster
            max_jobs: subsample jobs per episode (None = use all)
            update_freq: update policy every N episodes
            lr: learning rate
            gamma: discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip parameter
            entropy_coef: entropy bonus coefficient
            ppo_epochs: number of update epochs per batch
            batch_size: minibatch size
            buffer_size: rollout buffer capacity
            hidden_dim: network hidden dimension
        """
        self.train_df = train_df
        self.test_df = test_df
        self.num_episodes = num_episodes
        self.num_machines = num_machines
        self.max_jobs = max_jobs
        self.update_freq = update_freq
        
        # Create training environment
        self.env = SchedulerEnv(
            train_df,
            num_machines=num_machines,
            cpu_cap=CPU_CAP,
            mem_cap=MEM_CAP,
            max_jobs=max_jobs,
        )
        
        # Get state/action dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        print(f"[PPOTrainer] State dim  : {state_dim}")
        print(f"[PPOTrainer] Action dim : {action_dim}")
        print(f"[PPOTrainer] Train jobs : {len(train_df):,}")
        if max_jobs:
            print(f"[PPOTrainer] Episode window: {max_jobs:,} jobs")
        print(f"[PPOTrainer] Test jobs  : {len(test_df):,}")
        print(f"[PPOTrainer] Update freq: every {update_freq} episodes")
        
        # Create PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.update_metrics = []
        
        # Rolling statistics (for logging)
        self.recent_rewards = deque(maxlen=10)
        self.recent_waits = deque(maxlen=10)
    
    def train(self):
        """
        Main training loop.
        
        Process:
        1. Collect rollout over N episodes
        2. Update policy using PPO
        3. Log progress
        4. Repeat
        """
        print(f"\n{'='*80}")
        print(f"PPO TRAINING START - {self.num_episodes} episodes")
        print(f"{'='*80}\n")
        
        global_step = 0
        
        for episode in range(1, self.num_episodes + 1):
            # ──────────────────────────────────────────────────────────────
            # Collect one episode of experience
            # ──────────────────────────────────────────────────────────────
            ep_start_time = time.time()
            state, info = self.env.reset()
            done = False
            truncated = False
            
            ep_reward = 0.0
            ep_steps = 0
            
            while not done and not truncated:
                # Get action mask
                action_mask = info['action_mask']
                
                # Select action (stochastic during training)
                action, log_prob, value = self.agent.select_action(
                    state, action_mask, deterministic=False
                )
                
                # Step environment
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Store transition
                self.agent.store(
                    state, action, reward, value, log_prob, 
                    done or truncated, action_mask
                )
                
                ep_reward += reward
                ep_steps += 1
                global_step += 1
                state = next_state
            
            ep_time = time.time() - ep_start_time
            
            # Get scheduling metrics
            env_metrics = self.env.get_metrics()
            
            # Record episode stats
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_steps)
            self.episode_metrics.append(env_metrics)
            
            self.recent_rewards.append(ep_reward)
            self.recent_waits.append(env_metrics.get('avg_waiting_time', 0))
            
            # ──────────────────────────────────────────────────────────────
            # Update policy every N episodes
            # ──────────────────────────────────────────────────────────────
            if episode % self.update_freq == 0:
                update_start = time.time()
                update_stats = self.agent.update()
                update_time = time.time() - update_start
                
                self.update_metrics.append(update_stats)
                
                # ──────────────────────────────────────────────────────────
                # Logging
                # ──────────────────────────────────────────────────────────
                avg_reward = np.mean(self.recent_rewards)
                avg_wait = np.mean(self.recent_waits)
                
                print(
                    f"Ep {episode:4d}/{self.num_episodes} | "
                    f"Steps: {ep_steps:6,} | "
                    f"Reward: {ep_reward:8.2f} (avg: {avg_reward:7.2f}) | "
                    f"AvgWait: {env_metrics['avg_waiting_time']:7.2f}s (avg: {avg_wait:7.2f}s) | "
                    f"Jobs: {env_metrics['total_jobs']:5,} | "
                    f"EpTime: {ep_time:5.1f}s"
                )
                print(
                    f"           | "
                    f"PolicyLoss: {update_stats['policy_loss']:7.4f} | "
                    f"ValueLoss: {update_stats['value_loss']:7.4f} | "
                    f"Entropy: {update_stats['entropy']:6.4f} | "
                    f"KL: {update_stats['kl_div']:7.4f} | "
                    f"ClipFrac: {update_stats['clip_fraction']:5.3f} | "
                    f"UpdateTime: {update_time:5.1f}s"
                )
            else:
                # Log episode only (no update)
                if episode % 1 == 0:  # Log every episode for visibility
                    print(
                        f"Ep {episode:4d}/{self.num_episodes} | "
                        f"Steps: {ep_steps:6,} | "
                        f"Reward: {ep_reward:8.2f} | "
                        f"AvgWait: {env_metrics['avg_waiting_time']:7.2f}s | "
                        f"Jobs: {env_metrics['total_jobs']:5,}"
                    )
            
            # ──────────────────────────────────────────────────────────────
            # Checkpoint every 100 episodes
            # ──────────────────────────────────────────────────────────────
            if episode % 100 == 0:
                self.save_checkpoint(f'ppo_checkpoint_ep{episode}.pth')
        
        print(f"\n{'='*80}")
        print(f"PPO TRAINING COMPLETE")
        print(f"{'='*80}\n")
    
    def evaluate(self, num_eval_episodes=1):
        """
        Evaluate trained agent on test set.
        
        Args:
            num_eval_episodes: number of evaluation runs
        
        Returns:
            dict with average metrics
        """
        print(f"\n{'='*80}")
        print(f"PPO EVALUATION - {num_eval_episodes} episode(s) on test set")
        print(f"{'='*80}\n")
        
        # Create test environment (no job subsampling)
        test_env = SchedulerEnv(
            self.test_df,
            num_machines=self.num_machines,
            cpu_cap=CPU_CAP,
            mem_cap=MEM_CAP,
            max_jobs=None,  # Use full test set
        )
        
        all_metrics = []
        
        for eval_ep in range(num_eval_episodes):
            print(f"  Eval episode {eval_ep + 1}/{num_eval_episodes}...")
            
            state, info = test_env.reset()
            done = False
            truncated = False
            
            ep_reward = 0.0
            ep_steps = 0
            
            while not done and not truncated:
                action_mask = info['action_mask']
                
                # Greedy action selection (deterministic=True)
                action, _, _ = self.agent.select_action(
                    state, action_mask, deterministic=True
                )
                
                state, reward, done, truncated, info = test_env.step(action)
                
                ep_reward += reward
                ep_steps += 1
                
                # Progress update every 50k steps
                if ep_steps % 50000 == 0:
                    running = sum(len(m.running_jobs) for m in test_env.machines)
                    print(
                        f"    Step {ep_steps:,} | "
                        f"Queue: {len(test_env.waiting_queue)} | "
                        f"Running: {running} | "
                        f"Scheduled: {len(test_env.results):,}"
                    )
            
            metrics = test_env.get_metrics()
            all_metrics.append(metrics)
            
            print(
                f"    Finished: {ep_steps:,} steps | "
                f"Reward: {ep_reward:.2f} | "
                f"AvgWait: {metrics['avg_waiting_time']:.2f}s | "
                f"Jobs: {metrics['total_jobs']:,}"
            )
        
        # Average over all eval episodes
        avg_metrics = {
            'total_jobs': int(np.mean([m['total_jobs'] for m in all_metrics])),
            'avg_waiting_time': float(np.mean([m['avg_waiting_time'] for m in all_metrics])),
            'makespan': float(np.mean([m['makespan'] for m in all_metrics])),
        }
        
        print(f"\n  {'─'*60}")
        print(f"  PPO Evaluation Results (average over {num_eval_episodes} run(s)):")
        print(f"  {'─'*60}")
        print(f"  Total jobs       : {avg_metrics['total_jobs']:,}")
        print(f"  Avg waiting time : {avg_metrics['avg_waiting_time']:.4f} s")
        print(f"  Makespan         : {avg_metrics['makespan']:.4f} s")
        print(f"  {'─'*60}\n")
        
        return avg_metrics
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        self.agent.save(filepath)
    
    def save_model(self, filepath):
        """Save final model."""
        self.agent.save(filepath)
        print(f"[PPOTrainer] Final model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from checkpoint."""
        self.agent.load(filepath)
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame.
        
        Returns:
            DataFrame with episode-level metrics
        """
        rows = []
        for i, (r, l, m) in enumerate(
            zip(self.episode_rewards, self.episode_lengths, self.episode_metrics), 1
        ):
            rows.append({
                'episode': i,
                'total_reward': r,
                'episode_length': l,
                'avg_waiting_time': m.get('avg_waiting_time'),
                'makespan': m.get('makespan'),
                'total_jobs': m.get('total_jobs'),
            })
        return pd.DataFrame(rows)
