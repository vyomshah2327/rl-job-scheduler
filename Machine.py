class Job:
    """A single schedulable job."""
    def __init__(self, job_id, arrival, duration, cpu, mem):
        self.job_id   = job_id
        self.arrival  = arrival
        self.duration = duration   # seconds (raw, not normalised)
        self.cpu      = cpu        # plan_cpu  (raw)
        self.mem      = mem        # plan_mem  (raw)


class Machine:
    """
    One machine with CPU and memory capacity.
    Supports concurrent jobs as long as resources permit.
    """
    def __init__(self, machine_id: int, cpu_capacity: float = 1000,
                 mem_capacity: float = 1.0):
        self.machine_id   = machine_id
        self.cpu_total    = cpu_capacity
        self.mem_total    = mem_capacity
        self.cpu_used     = 0.0
        self.mem_used     = 0.0
        self.running_jobs = []   # list of (job, end_time)

    def can_run(self, job: Job) -> bool:
        return (self.cpu_used + job.cpu <= self.cpu_total and
                self.mem_used + job.mem <= self.mem_total)

    def assign(self, job: Job, current_time: float) -> float:
        self.cpu_used += job.cpu
        self.mem_used += job.mem
        end_time = current_time + job.duration
        self.running_jobs.append((job, end_time))
        return end_time

    def update(self, current_time: float):
        """Release resources for all jobs that have finished by current_time."""
        still_running = []
        for job, end_time in self.running_jobs:
            if end_time <= current_time:
                self.cpu_used -= job.cpu
                self.mem_used -= job.mem
            else:
                still_running.append((job, end_time))
        self.running_jobs = still_running

    def next_free_at(self) -> float:
        """Earliest time a slot opens on this machine (0 if idle)."""
        if not self.running_jobs:
            return 0.0
        return min(end_time for _, end_time in self.running_jobs)
