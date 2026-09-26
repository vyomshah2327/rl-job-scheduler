import pandas as pd
from Machine import Job, Machine


class FIFOScheduler:
    """
    First-In-First-Out scheduler baseline.
    Assigns jobs in arrival order to any machine with sufficient capacity.
    Machines are concurrent: multiple jobs can run simultaneously.
    """

    def __init__(self, df: pd.DataFrame, num_machines: int = 10,
                 cpu_cap: float = 1000, mem_cap: float = 1.0):
        self.df           = df
        self.num_machines = num_machines
        self.cpu_cap      = cpu_cap
        self.mem_cap      = mem_cap

    def run(self) -> pd.DataFrame:
        df   = self.df.sort_values('start_time')
        jobs = [
            Job(job_id=r['job_id'], arrival=r['start_time'],
                duration=r['duration_s'], cpu=r['plan_cpu'], mem=r['plan_mem'])
            for _, r in df.iterrows()
        ]

        machines = [Machine(i, self.cpu_cap, self.mem_cap)
                    for i in range(self.num_machines)]
        queue    = []
        results  = []
        i        = 0
        t        = jobs[0].arrival if jobs else 0

        while i < len(jobs) or queue or any(m.running_jobs for m in machines):
            for m in machines:
                m.update(t)

            while i < len(jobs) and jobs[i].arrival <= t:
                queue.append(jobs[i])
                i += 1

            unscheduled = []
            for job in queue:
                assigned = False
                for m in machines:
                    if m.can_run(job):
                        end_t = m.assign(job, t)
                        results.append({
                            'job_id': job.job_id, 'arrival_time': job.arrival,
                            'start_time': t, 'end_time': end_t,
                            'wait_time': t - job.arrival,
                        })
                        assigned = True
                        break
                if not assigned:
                    unscheduled.append(job)
            queue = unscheduled

            next_times = []
            if i < len(jobs):
                next_times.append(jobs[i].arrival)
            for m in machines:
                if m.running_jobs:
                    next_times.append(m.next_free_at())
            if next_times:
                t = min(next_times)

        return pd.DataFrame(results)

    def get_metrics(self, results_df: pd.DataFrame) -> dict:
        return {
            'total_jobs':       len(results_df),
            'avg_waiting_time': round(results_df['wait_time'].mean(), 4),
            'makespan':         round(
                results_df['end_time'].max() - results_df['arrival_time'].min(), 4),
        }
