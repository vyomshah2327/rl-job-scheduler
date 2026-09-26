"""
EVALUATION SCRIPT - Deep RL Job Scheduling
==========================================
Loads pre-trained DQN and PPO models and evaluates them on test data.

Outputs:
1. Table 1: Training Scale (5,000 jobs) - printed to console
2. Table 2: Full Test Set (13,363 jobs) - printed to console  
3. Figure: Cumulative waiting time comparison - saved as 'cumulative_wait.png'
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================
from Data import Data, CPU_CAP, MEM_CAP
from FIFO import FIFOScheduler
from SJF import SJFScheduler
from SchedulerEnv import SchedulerEnv
from DQNAgent import DQNAgent
from PPOAgent import PPOAgent

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_MACHINES = 10
BATCH_SIZE_5K = 5000
BATCH_SIZE_13K = 13363
DQN_MODEL_PATH = 'dqn_final.pth'
PPO_MODEL_PATH = 'ppo_final.pth'
STATE_DIM = 35
ACTION_DIM = 11
DQN_HIDDEN_DIM = 128
PPO_HIDDEN_DIM = 128  # Match saved model architecture

# Color scheme
COLORS = {
    'FIFO': '#FF6B6B',  # Coral red
    'SJF': '#4ECDC4',   # Turquoise
    'DQN': '#FFE66D',   # Yellow
    'PPO': '#95E1D3'    # Mint green
}

print("="*80)
print("EVALUATION: Deep RL Job Scheduling")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")
csv_path = 'batch_task.csv'
pipeline = Data(csv_path)
pipeline.run()
train_df, test_df = pipeline.split()

# Get batches
batch_5k = test_df.iloc[:BATCH_SIZE_5K].copy()
batch_13k = test_df.iloc[:BATCH_SIZE_13K].copy()
print(f"✓ Training scale batch: {len(batch_5k):,} jobs")
print(f"✓ Full test set: {len(batch_13k):,} jobs")

# ============================================================================
# STEP 2: LOAD MODELS
# ============================================================================
print("\n[2/5] Loading trained models...")

# DQN
dqn_agent = DQNAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=0.0,
    epsilon_end=0.0,
    hidden_dim=DQN_HIDDEN_DIM
)
dqn_agent.load(DQN_MODEL_PATH)
print(f"✓ DQN loaded from {DQN_MODEL_PATH}")

# PPO
ppo_agent = PPOAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    hidden_dim=PPO_HIDDEN_DIM
)
ppo_agent.load(PPO_MODEL_PATH)
print(f"✓ PPO loaded from {PPO_MODEL_PATH}")

# ============================================================================
# HELPER FUNCTION: RUN EVALUATION
# ============================================================================
def evaluate_all_algorithms(batch_df, batch_name):
    """Run all 4 algorithms on given batch and return results."""
    print(f"\n[{batch_name}] Running algorithms...")
    results = {}
    
    # FIFO
    print(f"  Running FIFO...")
    fifo = FIFOScheduler(batch_df, NUM_MACHINES, CPU_CAP, MEM_CAP)
    fifo_results_df = fifo.run().sort_values('start_time').reset_index(drop=True)
    fifo_results_df['cumavg_wait'] = fifo_results_df['wait_time'].expanding().mean()
    fifo_metrics = fifo.get_metrics(fifo.run())
    results['FIFO'] = {
        'df': fifo_results_df,
        'avg_wait': fifo_metrics['avg_waiting_time'],
        'makespan': fifo_metrics['makespan']
    }
    
    # SJF
    print(f"  Running SJF...")
    sjf = SJFScheduler(batch_df, NUM_MACHINES, CPU_CAP, MEM_CAP)
    sjf_results_df = sjf.run().sort_values('start_time').reset_index(drop=True)
    sjf_results_df['cumavg_wait'] = sjf_results_df['wait_time'].expanding().mean()
    sjf_metrics = sjf.get_metrics(sjf.run())
    results['SJF'] = {
        'df': sjf_results_df,
        'avg_wait': sjf_metrics['avg_waiting_time'],
        'makespan': sjf_metrics['makespan']
    }
    
    # DQN
    print(f"  Running DQN...")
    dqn_env = SchedulerEnv(batch_df, NUM_MACHINES, CPU_CAP, MEM_CAP)
    state, _ = dqn_env.reset()
    done = False
    steps = 0
    
    while not done and steps < 500_000:
        action = dqn_agent.select_action(state)
        state, _, done, _, _ = dqn_env.step(action)
        steps += 1
    
    dqn_results_df = pd.DataFrame(dqn_env.results).sort_values('start_time').reset_index(drop=True)
    dqn_results_df['cumavg_wait'] = dqn_results_df['wait_time'].expanding().mean()
    dqn_metrics = dqn_env.get_metrics()
    results['DQN'] = {
        'df': dqn_results_df,
        'avg_wait': dqn_metrics['avg_waiting_time'],
        'makespan': dqn_metrics['makespan']
    }
    
    # PPO
    print(f"  Running PPO...")
    ppo_env = SchedulerEnv(batch_df, NUM_MACHINES, CPU_CAP, MEM_CAP)
    state, _ = ppo_env.reset()
    done = False
    steps = 0
    
    while not done:
        action_mask = ppo_env.get_action_mask()
        action, _, _ = ppo_agent.select_action(state, action_mask)
        state, _, done, _, _ = ppo_env.step(action)
        steps += 1
    
    ppo_results_df = pd.DataFrame(ppo_env.results).sort_values('start_time').reset_index(drop=True)
    ppo_results_df['cumavg_wait'] = ppo_results_df['wait_time'].expanding().mean()
    ppo_metrics = ppo_env.get_metrics()
    results['PPO'] = {
        'df': ppo_results_df,
        'avg_wait': ppo_metrics['avg_waiting_time'],
        'makespan': ppo_metrics['makespan']
    }
    
    return results

# ============================================================================
# STEP 3: EVALUATE ON 5K JOBS (TRAINING SCALE)
# ============================================================================
print("\n[3/5] Evaluating on 5,000 jobs (training scale)...")
results_5k = evaluate_all_algorithms(batch_5k, "5K")

# ============================================================================
# STEP 4: EVALUATE ON 13K JOBS (FULL TEST SET)
# ============================================================================
print("\n[4/5] Evaluating on 13,363 jobs (full test set)...")
results_13k = evaluate_all_algorithms(batch_13k, "13K")

# ============================================================================
# STEP 5: PRINT TABLES
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)

# TABLE 1: Training Scale (5,000 jobs)
print("\nTable 1: Training Scale (5,000 jobs)")
print("-" * 60)
print(f"{'Algorithm':<12} {'Avg Wait (s)':>15} {'Makespan (s)':>15}")
print("-" * 60)
for algo in ['FIFO', 'SJF', 'DQN', 'PPO']:
    wait = results_5k[algo]['avg_wait']
    makespan = results_5k[algo]['makespan']
    print(f"{algo:<12} {wait:>15.2f} {makespan:>15.0f}")
print("-" * 60)

# TABLE 2: Full Test Set (13,363 jobs)
print("\nTable 2: Full Test Set (13,363 jobs)")
print("-" * 60)
print(f"{'Algorithm':<12} {'Avg Wait (s)':>15} {'Makespan (s)':>15}")
print("-" * 60)
for algo in ['FIFO', 'SJF', 'DQN', 'PPO']:
    wait = results_13k[algo]['avg_wait']
    makespan = results_13k[algo]['makespan']
    print(f"{algo:<12} {wait:>15.2f} {makespan:>15.0f}")
print("-" * 60)

# ============================================================================
# STEP 6: CREATE FIGURE (5K JOBS ONLY)
# ============================================================================
print("\n[5/5] Generating cumulative waiting time figure...")

# Set matplotlib parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(12, 7))

algorithms = ['FIFO', 'SJF', 'DQN', 'PPO']

# Plot all 4 algorithms
for algo in algorithms:
    df = results_5k[algo]['df']
    avg_wait = results_5k[algo]['avg_wait']
    
    ax.plot(range(1, len(df)+1), df['cumavg_wait'],
            label=f"{algo}  ({avg_wait:.1f}s)",
            color=COLORS[algo],
            linewidth=2.5,
            alpha=0.9)

# Formatting
ax.set_xlabel('Scheduled Jobs', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Average Waiting Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Average Waiting Time Comparison\nTraining Scale (5,000 jobs)',
             fontsize=15, fontweight='bold', pad=20)

# Legend
legend = ax.legend(fontsize=12, title='Algorithm (Avg Wait)', 
                   title_fontsize=13, loc='upper left',
                   frameon=True, fancybox=True, shadow=True,
                   edgecolor='black', facecolor='white', framealpha=0.95)
legend.get_title().set_fontweight('bold')

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Format axes with commas
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Set limits
ax.set_xlim(0, BATCH_SIZE_5K)
ax.set_ylim(0, max([results_5k[algo]['df']['cumavg_wait'].max() for algo in algorithms]) * 1.05)

plt.tight_layout()
plt.savefig('cumulative_wait.png', bbox_inches='tight', dpi=300)
print("✓ Saved: cumulative_wait.png")
plt.close()

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated outputs:")
print("  1. Table 1 (5,000 jobs) - printed above")
print("  2. Table 2 (13,363 jobs) - printed above")
print("  3. cumulative_wait.png - saved to current directory")
print("="*80)
