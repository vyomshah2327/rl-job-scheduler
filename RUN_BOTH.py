"""
RUN_BOTH.py - Train DQN + PPO with SAME Environment
====================================================
Both use DQN's proven environment:
- State: 35 dimensions (4 aggregate machine + 30 job features + 1 queue ratio)
- Reward: Delta waiting time (change in total wait per step)
- Episodes: 5,000 jobs each

This is the PROVEN configuration that achieved 506s with DQN.
"""

import time
import pandas as pd
import json

from Data import Data, CPU_CAP, MEM_CAP
from FIFO import FIFOScheduler
from SJF import SJFScheduler
from Trainer import Trainer as DQNTrainer
from PPOTrainer import PPOTrainer


def main():
    print("="*70)
    print("DQN + PPO TRAINING (SAME ENVIRONMENT)")
    print("="*70)
    print("Environment: 35-dim state, delta reward (DQN's proven config)")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n[1/6] Loading data...")
    pipeline = Data('batch_task.csv')
    pipeline.run()
    train_df, test_df = pipeline.split()
    
    print(f"✓ Train: {len(train_df):,} jobs")
    print(f"✓ Test: {len(test_df):,} jobs")
    
    # ========================================================================
    # STEP 2: RUN BASELINES
    # ========================================================================
    print("\n[2/6] Running baselines...")
    
    # FIFO
    fifo = FIFOScheduler(test_df, num_machines=10, cpu_cap=CPU_CAP, mem_cap=MEM_CAP)
    fifo_results = fifo.run()
    fifo_metrics = fifo.get_metrics(fifo_results)
    print(f"✓ FIFO: {fifo_metrics['avg_waiting_time']:.2f}s")
    
    # SJF
    sjf = SJFScheduler(test_df, num_machines=10, cpu_cap=CPU_CAP, mem_cap=MEM_CAP)
    sjf_results = sjf.run()
    sjf_metrics = sjf.get_metrics(sjf_results)
    print(f"✓ SJF: {sjf_metrics['avg_waiting_time']:.2f}s")
    
    # ========================================================================
    # STEP 3: TRAIN DQN
    # ========================================================================
    print("\n" + "="*70)
    print("[3/6] TRAINING DQN")
    print("="*70)
    print("Episodes: 500 | Jobs per episode: 5,000")
    print("State: 35-dim | Reward: Delta waiting time")
    print("="*70 + "\n")
    
    dqn_start = time.time()
    
    dqn_trainer = DQNTrainer(
        train_df=train_df,
        test_df=test_df,
        num_episodes=500,
        num_machines=10,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        buffer_cap=20000,
        batch_size=128,
        target_update=500,
        hidden_dim=128,
        max_jobs=5000
    )
    
    dqn_trainer.train()
    dqn_metrics = dqn_trainer.evaluate()
    dqn_trainer.save_model('dqn_final.pth')
    
    dqn_time = time.time() - dqn_start
    print(f"\n✓ DQN completed in {dqn_time/60:.1f} minutes")
    print(f"✓ DQN result: {dqn_metrics['avg_waiting_time']:.2f}s")
    
    # ========================================================================
    # STEP 4: TRAIN PPO (SAME ENVIRONMENT)
    # ========================================================================
    print("\n" + "="*70)
    print("[4/6] TRAINING PPO (SAME ENVIRONMENT AS DQN)")
    print("="*70)
    print("Episodes: 300 | Jobs per episode: 5,000")
    print("State: 35-dim | Reward: Delta waiting time (SAME AS DQN)")
    print("="*70 + "\n")
    
    ppo_start = time.time()
    
    ppo_trainer = PPOTrainer(
        train_df=train_df,
        test_df=test_df,
        num_episodes=300,
        num_machines=10,
        max_jobs=5000,
        update_freq=5,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=10000,
        hidden_dim=256
    )
    
    ppo_trainer.train()
    ppo_metrics = ppo_trainer.evaluate()
    ppo_trainer.save_model('ppo_final.pth')
    
    ppo_time = time.time() - ppo_start
    print(f"\n✓ PPO completed in {ppo_time/60:.1f} minutes")
    print(f"✓ PPO result: {ppo_metrics['avg_waiting_time']:.2f}s")
    
    # ========================================================================
    # STEP 5: FINAL RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    fifo_wait = fifo_metrics['avg_waiting_time']
    sjf_wait = sjf_metrics['avg_waiting_time']
    dqn_wait = dqn_metrics['avg_waiting_time']
    ppo_wait = ppo_metrics['avg_waiting_time']
    
    print(f"\n{'Algorithm':<15} {'Avg Wait (s)':>15} {'vs FIFO':>12}")
    print("-"*44)
    print(f"{'FIFO':<15} {fifo_wait:>15.2f} {'baseline':>12}")
    print(f"{'SJF':<15} {sjf_wait:>15.2f} {((fifo_wait-sjf_wait)/fifo_wait*100):>11.1f}%")
    print(f"{'DQN':<15} {dqn_wait:>15.2f} {((fifo_wait-dqn_wait)/fifo_wait*100):>11.1f}%")
    print(f"{'PPO':<15} {ppo_wait:>15.2f} {((fifo_wait-ppo_wait)/fifo_wait*100):>11.1f}%")
    
    print(f"\n{'':15} {'Both trained on SAME 35-dim environment':>44}")
    
    winner = "DQN" if dqn_wait < ppo_wait else "PPO"
    print(f"\n✓ Best RL algorithm: {winner}")
    print(f"✓ DQN training time: {dqn_time/60:.1f} min")
    print(f"✓ PPO training time: {ppo_time/60:.1f} min")
    print(f"✓ Total time: {(dqn_time + ppo_time)/60:.1f} min")
    print(f"✓ Models saved: dqn_final.pth, ppo_final.pth")
    print("="*70)
    
    # Save results
    results = {
        'FIFO': fifo_metrics,
        'SJF': sjf_metrics,
        'DQN': dqn_metrics,
        'PPO': ppo_metrics,
        'environment': '35-dim state, delta reward (DQN config)',
        'training_time': {
            'dqn_minutes': round(dqn_time/60, 2),
            'ppo_minutes': round(ppo_time/60, 2),
            'total_minutes': round((dqn_time + ppo_time)/60, 2)
        }
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to results.json\n")


if __name__ == "__main__":
    main()
