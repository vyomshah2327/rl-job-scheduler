"""
Trainer.py
==========
Training and evaluation harness for the DQN scheduler.

Training:
  - Each episode uses a random contiguous window of max_jobs from train set
  - Trains every step
  - Epsilon decays once per episode
  - Logs every episode

Evaluation:
  - Uses the FULL test set (max_jobs=None)
  - Epsilon = 0 (greedy)
  - Event-driven env means eval finishes in reasonable steps
  - MAX_EVAL is a safety cap — with the correct env it should not be hit
"""

import numpy as np
import pandas as pd
import time

from SchedulerEnv import SchedulerEnv
from DQNAgent     import DQNAgent
from Data         import CPU_CAP, MEM_CAP


class Trainer:
    def __init__(
        self,
        train_df:      pd.DataFrame,
        test_df:       pd.DataFrame,
        num_episodes:  int   = 200,
        num_machines:  int   = 10,
        lr:            float = 1e-3,
        gamma:         float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.05,
        buffer_cap:    int   = 20_000,
        batch_size:    int   = 128,
        target_update: int   = 200,
        hidden_dim:    int   = 128,
        max_jobs:      int   = None,
    ):
        self.train_df     = train_df
        self.test_df      = test_df
        self.num_episodes = num_episodes
        self.num_machines = num_machines
        self.max_jobs     = max_jobs

        self.env = SchedulerEnv(
            train_df,
            num_machines = num_machines,
            cpu_cap      = CPU_CAP,
            mem_cap      = MEM_CAP,
            max_jobs     = max_jobs,
        )

        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        print(f"[Trainer] State dim  : {state_dim}")
        print(f"[Trainer] Action dim : {action_dim}")
        if max_jobs:
            print(f"[Trainer] Train jobs : {len(train_df):,} "
                  f"(episode window: {max_jobs:,})")
        else:
            print(f"[Trainer] Train jobs : {len(train_df):,} (full dataset per episode)")
        print(f"[Trainer] Test jobs  : {len(test_df):,}")

        self.agent = DQNAgent(
            state_dim          = state_dim,
            action_dim         = action_dim,
            lr                 = lr,
            gamma              = gamma,
            epsilon_start      = epsilon_start,
            epsilon_end        = epsilon_end,
            buffer_capacity    = buffer_cap,
            batch_size         = batch_size,
            target_update_freq = target_update,
            hidden_dim         = hidden_dim,
        )

        self.episode_rewards = []
        self.episode_losses  = []
        self.episode_metrics = []

    # ──────────────────────────────────────────────────────────────────────
    def train(self):
        window_str = (f"{self.max_jobs:,}" if self.max_jobs
                      else f"{len(self.train_df):,}")
        print(f"\n[Trainer] Training for {self.num_episodes} episodes "
              f"(~{window_str} jobs/episode)...\n")

        MAX_STEPS_PER_EP = 500_000   # safety cap — should never be hit

        for ep in range(1, self.num_episodes + 1):
            t0        = time.time()
            state, _  = self.env.reset()
            done      = False
            ep_reward = 0.0
            losses    = []
            steps     = 0

            while not done and steps < MAX_STEPS_PER_EP:
                steps += 1
                action                         = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.store(state, action, reward, next_state, done)

                loss = None 
                if steps % 4 == 0:
                    loss = self.agent.train_step()
                if loss is not None:
                    losses.append(loss)

                state      = next_state
                ep_reward += reward

            if steps >= MAX_STEPS_PER_EP and not done:
                q_len   = len(self.env.waiting_queue)
                running = sum(len(m.running_jobs) for m in self.env.machines)
                print(f"  ⚠️  Ep {ep}: hit MAX_STEPS — "
                      f"queue={q_len} running={running} "
                      f"ptr={self.env.job_ptr}/{len(self.env.all_jobs)}")

            self.agent.decay_epsilon(ep)

            avg_loss = float(np.mean(losses)) if losses else 0.0
            metrics  = self.env.get_metrics()
            elapsed  = time.time() - t0

            self.episode_rewards.append(ep_reward)
            self.episode_losses.append(avg_loss)
            self.episode_metrics.append(metrics)

            print(f"  Ep {ep:4d}/{self.num_episodes} | "
                  f"Reward: {ep_reward:9.2f} | "
                  f"Loss: {avg_loss:.5f} | "
                  f"ε: {self.agent.epsilon:.4f} | "
                  f"AvgWait: {metrics.get('avg_waiting_time', 'N/A'):>10} s | "
                  f"Jobs: {metrics.get('total_jobs', 0):,} | "
                  f"Steps: {steps:,} | "
                  f"Time: {elapsed:.1f}s")

        print("\n[Trainer] Training complete.")

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self) -> dict:
        """
        Run the trained agent greedily on the full test set.
        Uses the same SchedulerEnv with max_jobs=None (no subsampling).
        No job is ever dropped — every job gets scheduled.
        """
        print("\n[Trainer] Evaluating on full test set (greedy, ε=0)...")

        saved_eps          = self.agent.epsilon
        self.agent.epsilon = 0.0

        test_env = SchedulerEnv(
            self.test_df,
            num_machines = self.num_machines,
            cpu_cap      = CPU_CAP,
            mem_cap      = MEM_CAP,
            max_jobs     = None,   # always use full test set for eval
        )

        state, _ = test_env.reset()
        done     = False
        steps    = 0
        MAX_EVAL = 2_000_000   # safety cap

        while not done and steps < MAX_EVAL:
            steps += 1
            action                         = self.agent.select_action(state)
            state, _, done, _, _           = test_env.step(action)

            if steps % 50_000 == 0:
                running = sum(len(m.running_jobs) for m in test_env.machines)
                print(f"  [Eval] Step {steps:,} | "
                      f"Running: {running} | "
                      f"Queue: {len(test_env.waiting_queue)} | "
                      f"Scheduled: {len(test_env.results):,} | "
                      f"Ptr: {test_env.job_ptr}/{len(test_env.all_jobs)}")

        if steps >= MAX_EVAL:
            print(f"  ⚠️  Eval hit MAX_STEPS ({MAX_EVAL:,}) — "
                  f"queue={len(test_env.waiting_queue)} "
                  f"running={sum(len(m.running_jobs) for m in test_env.machines)}")

        self.agent.epsilon = saved_eps
        metrics = test_env.get_metrics()

        print(f"\n  [DQN Evaluation Results]")
        print(f"  Total jobs       : {metrics.get('total_jobs', 0):,}")
        print(f"  Avg waiting time : {metrics.get('avg_waiting_time', 'N/A')} s")
        print(f"  Makespan         : {metrics.get('makespan', 'N/A')} s")
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    def save_model(self, filepath: str = 'dqn_scheduler.pth'):
        self.agent.save(filepath)

    def get_training_history(self) -> pd.DataFrame:
        rows = []
        for i, (r, l, m) in enumerate(
                zip(self.episode_rewards, self.episode_losses,
                    self.episode_metrics), 1):
            rows.append({
                'episode':          i,
                'total_reward':     r,
                'avg_loss':         l,
                'avg_waiting_time': m.get('avg_waiting_time'),
                'makespan':         m.get('makespan'),
            })
        return pd.DataFrame(rows)