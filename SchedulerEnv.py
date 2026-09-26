"""
SchedulerEnv.py
===============
DQN scheduling environment — fair comparison with FIFO and SJF.

Core design principle:
──────────────────────
NO JOB IS EVER DROPPED. Every job that arrives joins an unlimited
waiting queue and stays there until it is scheduled onto a machine.
This matches exactly how FIFO and SJF work.

Structure:
──────────
  all_jobs (sorted by arrival)
      │
      ▼  jobs arrive as curr_time passes their arrival time
  waiting_queue  (unlimited Python list — all unscheduled jobs in order)
      │
      ▼  front 10 jobs of the queue are the "visible window"
  visible slots [0..9]  (what the agent sees and acts on)
      │
      ▼  agent picks action 0-9 (schedule slot i) or 10 (void)
  Machine  (10 machines, each runs concurrent jobs if CPU+MEM fits)

When the agent schedules slot i:
  - That job is removed from position i in the queue
  - The next job in queue automatically becomes visible (queue slides)
  - No time advance — agent can schedule again immediately

When the agent picks void OR no slot job fits any machine:
  - Time jumps to the next event: next arrival OR next machine finishing
  - New arrivals are appended to the waiting queue
  - Machines release finished jobs

Done condition:
  - All jobs have arrived (job_ptr >= len(all_jobs))
  - AND waiting_queue is empty (all scheduled or running)
  - AND no machines are running anything

State vector (35 dims):
  [0]     free_machine_ratio  — machines with no running jobs / total machines
  [1]     avg_cpu_util        — mean(cpu_used / cpu_cap) across all machines
  [2]     avg_mem_util        — mean(mem_used / mem_cap) across all machines
  [3]     avg_running_ratio   — mean(running_jobs / MAX_JOBS_PER_MACHINE) per machine
                                MAX_JOBS_PER_MACHINE from median job demand in dataset
  [4..33] visible slot features × 10:
            norm_duration = job.duration / MAX_DURATION
            cpu_frac      = job.cpu / cpu_cap
            mem_frac      = job.mem / mem_cap
  [34]    queue_ratio — len(waiting_queue) / len(all_jobs)

Reward (delta waiting time):
  r = -(current_total_wait - prev_total_wait) / (total_jobs * episode_time_span)

  Measures how much total waiting time INCREASED this step.
  Stays bounded and consistent throughout the episode — no curr_time drift.
  Range: approximately [-1, 0] per step.
  episode_time_span = last_arrival - first_arrival of the episode window.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from Machine import Job, Machine


class SchedulerEnv(gym.Env):

    NUM_VISIBLE   = 10
    DELAY_PENALTY = -1.0
    HOLD_PENALTY  = -1.0

    def __init__(
        self,
        df:           pd.DataFrame,
        num_machines: int   = 10,
        cpu_cap:      float = 1000.0,
        mem_cap:      float = 1.0,
        max_jobs:     int   = None,
    ):
        super().__init__()

        self.full_df      = df
        self.num_machines = num_machines
        self.cpu_cap      = cpu_cap
        self.mem_cap      = mem_cap
        self.max_jobs     = max_jobs

        self.all_jobs_full = self._build_jobs(df)
        self.MAX_DURATION  = float(df['duration_s'].max())

        # MAX_JOBS_PER_MACHINE — used only in state normalisation
        # Computed from median job demand (not min) so outlier tiny jobs
        # do not produce an unrealistically large denominator.
        # Bottleneck resource (min of CPU and MEM capacity) determines
        # how many median-sized jobs can actually run simultaneously.
        median_cpu = float(df['plan_cpu'].median())
        median_mem = float(df['plan_mem'].median())
        cap_by_cpu = (cpu_cap / median_cpu) if median_cpu > 0 else 1.0
        cap_by_mem = (mem_cap / median_mem) if median_mem > 0 else 1.0
        self.MAX_JOBS_PER_MACHINE = max(1, int(min(cap_by_cpu, cap_by_mem)))

        # State: 4 machine aggregates + 10 job slots × 3 + 1 queue = 35
        M         = self.NUM_VISIBLE
        state_dim = 4 + M * 3 + 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(M + 1)  # 0..9 schedule, 10 void

        # Per-episode state — initialised in reset()
        self.all_jobs           = None
        self.machines           = None
        self.waiting_queue      = None
        self.job_ptr            = 0
        self.curr_time          = 0.0
        self.results            = None
        self.episode_time_span  = 1.0   # set in reset()
        self._prev_total_wait   = 0.0   # for delta reward

    # ──────────────────────────────────────────────────────────────────────
    def _build_jobs(self, df: pd.DataFrame) -> list:
        """Convert DataFrame rows to Job objects, sorted by arrival time."""
        df = df.sort_values('start_time')
        return [
            Job(
                job_id   = row['job_id'],
                arrival  = float(row['start_time']),
                duration = float(row['duration_s']),
                cpu      = float(row['plan_cpu']),
                mem      = float(row['plan_mem']),
            )
            for _, row in df.iterrows()
        ]

    # ──────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """Reset for a new episode. Optionally subsample jobs for training."""
        super().reset(seed=seed)

        if self.max_jobs and len(self.all_jobs_full) > self.max_jobs:
            hi  = len(self.all_jobs_full) - self.max_jobs
            idx = np.random.randint(0, hi + 1)
            self.all_jobs = self.all_jobs_full[idx: idx + self.max_jobs]
        else:
            self.all_jobs = self.all_jobs_full

        self.machines           = [Machine(i, self.cpu_cap, self.mem_cap)
                                   for i in range(self.num_machines)]
        self.waiting_queue      = []
        self.job_ptr            = 0
        self.curr_time          = float(self.all_jobs[0].arrival) if self.all_jobs else 0.0
        self.results            = []
        self._prev_total_wait   = 0.0

        # episode_time_span: used to normalise the waiting-time reward
        # = time between first and last job arrival in this episode window
        # If all jobs arrive at the same time, use 1.0 to avoid division by zero
        if len(self.all_jobs) > 1:
            self.episode_time_span = max(
                1.0,
                float(self.all_jobs[-1].arrival - self.all_jobs[0].arrival)
            )
        else:
            self.episode_time_span = 1.0

        self._load_arriving_jobs()
        obs = self._observe()
        info = {'action_mask': self.get_action_mask()}
        return obs, info

    # ──────────────────────────────────────────────────────────────────────
    # def step(self, action: int):
    #     """
    #     Execute one agent action.

    #     Actions 0-9: schedule the job at queue position i.
    #     Action 10: void — advance time to next event.

    #     Work-conserving constraint:
    #       If agent picks void but a visible job fits a free machine,
    #       override to that job. Matches FIFO/SJF behaviour — no machine
    #       should sit idle when schedulable work exists.
    #       This is the fix for the eval freeze (Queue:5, Running:0 loop).
    #     """
    #     M         = self.NUM_VISIBLE
    #     scheduled = False

    #     # Work-conserving override
    #     if action == M or action >= len(self.waiting_queue):
    #         for i, job in enumerate(self.waiting_queue[:M]):
    #             fits = any(m.can_run(job) for m in self.machines)
    #             if fits:
    #                 action = i
    #                 break

    #     if action < M and action < len(self.waiting_queue):
    #         job      = self.waiting_queue[action]
    #         assigned = self._try_assign(job)
    #         if assigned:
    #             print("[ERROR] agent choose void even if machines were free")
    #             self.waiting_queue.pop(action)
    #             scheduled = True

    #     if not scheduled:
    #         self._advance_to_next_event()

    #     reward = self._get_reward()

    #     done = (
    #         self.job_ptr >= len(self.all_jobs)
    #         and len(self.waiting_queue) == 0
    #         and not any(m.running_jobs for m in self.machines)
    #     )

    #     return self._observe(), reward, done, False, {}

    def step(self, action: int):
        """
        Execute one agent action — no overrides, pure RL.
 
        Actions 0-9: schedule the job at queue position i.
          - If that position exists AND a machine can fit it → scheduled.
          - Otherwise treated as void (agent gets void penalty if applicable).
        Action 10: explicit void → advance time to next event.
 
        Void penalty:
          If the agent picks void BUT there is at least one visible job
          that could fit a free machine RIGHT NOW, the agent receives an
          extra penalty of -1.0 on top of the normal step reward.
          This teaches the agent that leaving a machine idle when work
          is available is very costly. The agent is NOT overridden —
          it still takes void and time advances — but it feels the pain
          and learns to avoid this over episodes.
 
          Why -1.0? Normal per-step reward ≈ -0.14. A penalty of -1.0
          makes unnecessary void ~8x worse than a normal step, giving
          a strong but not catastrophic learning signal.
 
          When void IS genuinely needed (all machines too full for any
          visible job), no penalty is applied — void is the correct action.
        """
        M         = self.NUM_VISIBLE
        scheduled = False
 
        # ── Detect unnecessary void BEFORE executing ──────────────────────
        # An unnecessary void is: agent picked void (or invalid slot)
        # AND at least one visible job fits at least one free machine.
        agent_voided = (action == M or action >= len(self.waiting_queue))
        unnecessary_void = False
        if agent_voided:
            for job in self.waiting_queue[:M]:
                if any(m.can_run(job) for m in self.machines):
                    unnecessary_void = True
                    break
 
        # ── Execute the agent's actual chosen action ───────────────────────
        if action < M and action < len(self.waiting_queue):
            job      = self.waiting_queue[action]
            assigned = self._try_assign(job)
            if assigned:
                self.waiting_queue.pop(action)
                scheduled = True
            # If assign failed (no machine fits this specific job),
            # fall through to void — no penalty since agent tried to schedule
 
        if not scheduled:
            self._advance_to_next_event()
 
        # ── Compute reward with optional void penalty ─────────────────────
        reward = self._get_reward()
        if unnecessary_void:
            reward -= 0.1   # small penalty for unnecessary void, scaled to delta reward

        # Clip total reward to prevent extreme values
        reward = max(-2.0, reward)
        done = (
            self.job_ptr >= len(self.all_jobs)
            and len(self.waiting_queue) == 0
            and not any(m.running_jobs for m in self.machines)
        )
 
        obs = self._observe()
        info = {'action_mask': self.get_action_mask()}
        return obs, reward, done, False, info

    # ──────────────────────────────────────────────────────────────────────
    def get_action_mask(self) -> np.ndarray:
        """
        Return boolean mask indicating which actions are valid.
        
        Returns:
            mask: (11,) boolean array where True = valid action
        """
        M = self.NUM_VISIBLE
        mask = np.zeros(M + 1, dtype=bool)
        
        # Check each visible slot
        for i in range(min(M, len(self.waiting_queue))):
            job = self.waiting_queue[i]
            # Action is valid if job fits at least one machine
            if any(m.can_run(job) for m in self.machines):
                mask[i] = True
        
        # Void action is always valid
        mask[M] = True
        
        return mask

    # ──────────────────────────────────────────────────────────────────────
    def _try_assign(self, job: Job) -> bool:
        """
        Try to assign job to the first machine with enough free CPU and MEM.
        Returns True if assigned, False if no machine can take it right now.
        Records the result for metrics.
        """
        for m in self.machines:
            if m.can_run(job):
                end_time = m.assign(job, self.curr_time)
                self.results.append({
                    'job_id':       job.job_id,
                    'arrival_time': job.arrival,
                    'start_time':   self.curr_time,
                    'end_time':     end_time,
                    'wait_time':    self.curr_time - job.arrival,
                })
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    def _advance_to_next_event(self):
        """
        Jump curr_time to the earliest of:
          a) next job arrival   — new work enters the queue
          b) soonest machine finish — resources freed up

        Always moves forward by at least 1 second.
        After jumping: release finished machines and load newly arrived jobs.
        """
        candidates = []

        if self.job_ptr < len(self.all_jobs):
            candidates.append(float(self.all_jobs[self.job_ptr].arrival))

        for m in self.machines:
            if m.running_jobs:
                candidates.append(m.next_free_at())

        if candidates:
            next_t         = min(candidates)
            self.curr_time = max(self.curr_time + 1.0, next_t)
        else:
            self.curr_time += 1.0

        self._release_finished()
        self._load_arriving_jobs()

    # ──────────────────────────────────────────────────────────────────────
    def _release_finished(self):
        """Release resources on machines for jobs that finished by curr_time."""
        for m in self.machines:
            m.update(self.curr_time)

    # ──────────────────────────────────────────────────────────────────────
    def _load_arriving_jobs(self):
        """
        Append all jobs with arrival <= curr_time to the waiting_queue.
        No job is ever dropped — the queue is unlimited.
        """
        while self.job_ptr < len(self.all_jobs):
            job = self.all_jobs[self.job_ptr]
            if job.arrival > self.curr_time:
                break
            self.waiting_queue.append(job)
            self.job_ptr += 1

    # ──────────────────────────────────────────────────────────────────────
    def _get_reward(self) -> float:
        """
        Reward = delta waiting-time reward.

        Measures how much total waiting time INCREASED this step, normalised
        by (total_jobs * episode_time_span) to keep values in a consistent
        range regardless of where we are in the episode.

        Why delta instead of absolute waiting time:
          Absolute waiting time = sum(curr_time - job.arrival) grows
          unboundedly as curr_time increases, causing reward scale variance
          that blows up DQN loss (0.04 -> 168) and Q-values.

          Delta reward captures only the *change* this step, so it stays
          bounded and consistent throughout the episode:
            - Agent schedules a job well  -> its wait stops accumulating
                                          -> smaller (less negative) delta
            - Agent idles unnecessarily   -> all waiting jobs keep accruing
                                          -> larger (more negative) delta
            - Agent voids when queue empty -> delta ~= 0 (correct)

        Range: approximately [-1, 0] per step, stable across the episode.
        """
        total_jobs = len(self.all_jobs)
        if total_jobs == 0:
            return 0.0

        current_total_wait = sum(
            self.curr_time - job.arrival for job in self.waiting_queue
        )
        delta = current_total_wait - self._prev_total_wait
        self._prev_total_wait = current_total_wait

        normaliser = float(total_jobs) * float(self.episode_time_span)
        return -delta / normaliser if normaliser > 0 else 0.0
    # ──────────────────────────────────────────────────────────────────────
    def _observe(self) -> np.ndarray:
        """
        Build the 35-dim state vector.

        Machine section — 4 aggregate dims:
          [0] free_machine_ratio  = machines with no running jobs / num_machines
          [1] avg_cpu_util        = mean(cpu_used / cpu_cap) across machines
          [2] avg_mem_util        = mean(mem_used / mem_cap) across machines
          [3] avg_running_ratio   = mean(running_jobs / MAX_JOBS_PER_MACHINE)
                                    clipped to [0,1]

          Aggregate stats are used here (not per-machine) to keep the state
          vector compact at 35 dims. Smaller state = fewer parameters =
          faster learning on CPU with limited training episodes.

        Visible job section — 30 dims (10 jobs × 3):
          For each of the front 10 jobs in waiting_queue:
            norm_duration = duration / MAX_DURATION   ∈ [0, 1]
            cpu_frac      = cpu / cpu_cap             ∈ [0, 1]
            mem_frac      = mem / mem_cap             ∈ [0, 1]
          Empty slots (queue < 10) are zeros.

        Queue section — 1 dim:
          queue_ratio = len(waiting_queue) / len(all_jobs)  ∈ [0, 1]
          Tells the agent how much work remains relative to total.
        """
        M   = self.NUM_VISIBLE
        obs = np.zeros(4 + M * 3 + 1, dtype=np.float32)

        # ── Machine aggregate stats ───────────────────────────────────────
        num_free      = sum(1 for m in self.machines if not m.running_jobs)
        total_cpu     = sum(m.cpu_used for m in self.machines)
        total_mem     = sum(m.mem_used for m in self.machines)
        total_running = sum(len(m.running_jobs) for m in self.machines)

        obs[0] = num_free / float(self.num_machines)
        obs[1] = total_cpu / float(self.num_machines * self.cpu_cap)
        obs[2] = total_mem / float(self.num_machines * self.mem_cap)
        obs[3] = min(total_running,
                     self.num_machines * self.MAX_JOBS_PER_MACHINE) / float(
                     self.num_machines * self.MAX_JOBS_PER_MACHINE)

        # ── Visible job features ──────────────────────────────────────────
        ptr = 4
        for job in self.waiting_queue[:M]:
            obs[ptr]     = min(job.duration, self.MAX_DURATION) / self.MAX_DURATION
            obs[ptr + 1] = job.cpu / self.cpu_cap
            obs[ptr + 2] = job.mem / self.mem_cap
            ptr += 3
        ptr += (M - min(len(self.waiting_queue), M)) * 3

        # ── Queue ratio ───────────────────────────────────────────────────
        total    = len(self.all_jobs)
        obs[ptr] = len(self.waiting_queue) / float(total) if total > 0 else 0.0

        return obs

    # ──────────────────────────────────────────────────────────────────────
    def get_metrics(self) -> dict:
        """Scheduling metrics — matches FIFO and SJF output format exactly."""
        if not self.results:
            return {'total_jobs': 0, 'avg_waiting_time': 0.0, 'makespan': 0.0}
        df = pd.DataFrame(self.results)
        return {
            'total_jobs':       len(df),
            'avg_waiting_time': round(float(df['wait_time'].mean()), 4),
            'makespan':         round(
                float(df['end_time'].max() - df['arrival_time'].min()), 4),
        }