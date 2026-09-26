from Data import Data, CPU_CAP, MEM_CAP
from FIFO import FIFOScheduler
from SJF import SJFScheduler
from Trainer import Trainer


class Main:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.train_df = None
        self.test_df  = None

    def run(self):
        # --- Load and preprocess ---
        pipeline = Data(self.filepath)
        pipeline.run()

        # --- Train/test split ---
        self.train_df, self.test_df = pipeline.split()

        # --- Run FIFO on test set ---
        fifo       = FIFOScheduler(self.test_df, num_machines=10, cpu_cap=CPU_CAP, mem_cap=MEM_CAP)
        results_df = fifo.run()
        fifo_metrics    = fifo.get_metrics(results_df)
 
        # --- Run SJF on test set ---
        sjf         = SJFScheduler(self.test_df, num_machines=10, cpu_cap=CPU_CAP, mem_cap=MEM_CAP)
        sjf_results = sjf.run()
        sjf_metrics = sjf.get_metrics(sjf_results)
 
        # -------------------------------------------------------
        # 4. Train DQN agent on train set, evaluate on test set
        # -------------------------------------------------------
        trainer = Trainer(
            train_df      = self.train_df,
            test_df       = self.test_df,
            num_episodes  = 500,
            num_machines  = 10,
            lr            = 1e-4,     # ↓ prevents Q-value divergence
            gamma         = 0.99,
            epsilon_start = 1.0,
            epsilon_end   = 0.02,
            buffer_cap    = 20_000,
            batch_size    = 128,       # ✅
            target_update = 500,      # ✅
            hidden_dim    = 128,       # ✅
            max_jobs      = 5_000,
        )
        trainer.train()
        dqn_metrics = trainer.evaluate()
        trainer.save_model('dqn_scheduler.pth')
 
        # -------------------------------------------------------
        # 5. Final comparison table
        # -------------------------------------------------------
        print(f"\n{'='*55}")
        print(f"{'Metric':<25} {'FIFO':>10} {'SJF':>10} {'DQN':>10}")
        print(f"{'='*55}")
        print(f"{'Total jobs':<25} "
              f"{fifo_metrics['total_jobs']:>10,} "
              f"{sjf_metrics['total_jobs']:>10,} "
              f"{dqn_metrics.get('total_jobs', 0):>10,}")
        print(f"{'Avg waiting time (s)':<25} "
              f"{fifo_metrics['avg_waiting_time']:>10} "
              f"{sjf_metrics['avg_waiting_time']:>10} "
              f"{dqn_metrics.get('avg_waiting_time', 'N/A'):>10}")
        print(f"{'Makespan (s)':<25} "
              f"{fifo_metrics['makespan']:>10} "
              f"{sjf_metrics['makespan']:>10} "
              f"{dqn_metrics.get('makespan', 'N/A'):>10}")
        print(f"{'='*55}\n")
 
        return fifo_metrics, sjf_metrics, dqn_metrics, trainer
 
 
if __name__ == '__main__':
    main = Main('/kaggle/input/datasets/vyomshah/rl_dataset/batch_task.csv')
    main.run()
 