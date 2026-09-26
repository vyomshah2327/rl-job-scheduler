import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Column layout of the Alibaba batch_task trace
COLUMN_NAMES = [
    'start_time', 'end_time', 'job_id', 'task_id',
    'inst_num', 'status', 'plan_cpu', 'plan_mem'
]

STATUS_ENCODING = {'Terminated': 0, 'Waiting': 1, 'Failed': 2, 'Running': 3}

# ─────────────────────────────────────────────────────────────────────────────
# Machine capacity constants
#   CPU_CAP=1000  → allows ~20 concurrent jobs per machine (median job = 50 CPU)
#   MEM_CAP=1.0   → memory is rarely the bottleneck (max job mem ≈ 0.127)
# ─────────────────────────────────────────────────────────────────────────────
CPU_CAP = 1000
MEM_CAP = 1.0


class Data:
    """Loads, cleans, and splits the Alibaba cluster trace."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.scaler   = MinMaxScaler()
        self.df       = None

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, header=None, names=COLUMN_NAMES)

        # Keep only jobs with known, valid outcomes
        df = df[df['status'] == 'Terminated']
        df = df.drop_duplicates()
        df = df[df['start_time'] >= 0]
        df = df[df['end_time'] > df['start_time']]
        df = df.dropna(subset=['plan_cpu', 'plan_mem'])

        # Drop jobs that a single machine can never fit (would loop forever)
        before = len(df)
        df = df[(df['plan_cpu'] <= CPU_CAP) & (df['plan_mem'] <= MEM_CAP)]
        print(f"[Data] Dropped {before - len(df):,} jobs exceeding machine capacity")

        df = df.sort_values('start_time')
        df['duration_s']  = df['end_time'] - df['start_time']
        df['status_code'] = df['status'].map(STATUS_ENCODING)

        # Scale inst_num only; plan_cpu / plan_mem stay raw for scheduler checks
        df['inst_num'] = self.scaler.fit_transform(df[['inst_num']])

        self.df = df.reset_index(drop=True)
        print(f"[Data] Loaded {len(self.df):,} Terminated jobs")
        return self.df

    def split(self, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/test split — no shuffling (jobs are time-ordered).
        Scaler is re-fit on train only to avoid data leakage.
        """
        if self.df is None:
            raise RuntimeError("Call .run() first.")

        split_idx = int(len(self.df) * ratio)
        train_df  = self.df.iloc[:split_idx].reset_index(drop=True)
        test_df   = self.df.iloc[split_idx:].reset_index(drop=True)

        train_df['inst_num'] = self.scaler.fit_transform(train_df[['inst_num']])
        test_df['inst_num']  = self.scaler.transform(test_df[['inst_num']])

        print(f"[Data] Train: {len(train_df):,} | Test: {len(test_df):,}")
        return train_df, test_df
