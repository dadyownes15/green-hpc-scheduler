import os
import ast
import configparser
import numpy as np
from gymnasium import Env, spaces
from typing import List
import random
import sys
from cluster import Cluster
from workloads import Workloads
from job import Job
from reward import Reward
from carbon_intensity import CarbonIntensity

# Load config with explicit path and typed parsing
config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config_file', 'config.ini')
config.read(config_path)

MAX_QUEUE_SIZE = config.getint('architecture', 'max_queue_size')
RUN_WIN_LENGTH = config.getint('architecture', 'run_win_length')
GREEN_FORECAST_LENGTH = config.getint('architecture', 'green_forecast_length')

DELAY_TIME_LIST = ast.literal_eval(config.get('architecture', 'delay_time_list'))
DELAY_TIME_LIST_LENGTH = len(DELAY_TIME_LIST)

JOB_FEATURE = config.getint('architecture', 'job_feature')
RUN_FEATURE = config.getint('architecture', 'run_feature')
GREEN_FEATURE_PR_TIMESLOT = config.getint('architecture', 'green_feature_pr_timeslot')
GREEN_FEATURE_CONSTANT = config.getint('architecture', 'green_feature_constant')
MAX_WAIT_N_JOBS = config.getint('architecture', 'max_wait_n_jobs')

# Power / carbon settings (use correct option names)
IDLE_POWER = config.getfloat('power settings', 'idle_power')
NUM_PROCS_PER_NODE = config.getint('power settings', 'procs_per_node')
CARBON_YEAR = config.getint('power settings', 'carbon_year')
VARIABLE_CARBON_INTENSITIES = config.getboolean('power settings', 'variable_carbon_intensities')
EPISODE_LENGTH = config.getint('training', 'episode_length')


class HPCenv(Env):
    def __init__(self, workload_path, debug=False):
        self.debug = debug

        # Flat action space
        num_job_actions = MAX_QUEUE_SIZE
        do_nothing_actions = DELAY_TIME_LIST_LENGTH + MAX_WAIT_N_JOBS
        self.action_space_size = num_job_actions + do_nothing_actions
        self.action_space = spaces.Discrete(self.action_space_size)

        # Observation space: flattened 1-D vector
        obs_len = (MAX_QUEUE_SIZE * JOB_FEATURE) + (RUN_WIN_LENGTH * RUN_FEATURE) + GREEN_FEATURE_CONSTANT + ((GREEN_FORECAST_LENGTH-1) * GREEN_FEATURE_PR_TIMESLOT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # Gymnasium env
        self.state = None
        self.current_step = 0
 
        # HPC Env variables
        self.job_queue: List[Job] = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_jobs = 0 

        # Load workloads and cluster
        self.loads = Workloads(workload_path)
        self.cluster = Cluster(self.loads.max_nodes, NUM_PROCS_PER_NODE, IDLE_POWER)
        self.reward = Reward()
        self.carbon_intensity = CarbonIntensity(year=CARBON_YEAR, green_win_length=GREEN_FORECAST_LENGTH)

        # Episode-specific carbon-timeline offset (in hours)
        self.episode_start_hour_offset = 0

    def step(self, action):
        # Initialize default outputs
        scheduled_job = None
        terminated = False
        truncated = False

        # Schedule job [0 ... MAX_QUEUE_SIZE-1]
        # Delay fixed amount [MAX_QUEUE_SIZE ... MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH - 1]
        # Wait until N jobs are finished [MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH ... MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH + MAX_WAIT_N_JOBS - 1]

        if 0 <= action < MAX_QUEUE_SIZE:
            scheduled_job = self.schedule_job(action)

        elif MAX_QUEUE_SIZE <= action < (MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH):
            delay_idx = action - MAX_QUEUE_SIZE
            skip_time = DELAY_TIME_LIST[delay_idx]
            self.delay_fixed_amount(skip_time=skip_time)

        elif (MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH) <= action < (MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH + MAX_WAIT_N_JOBS):
            # action corresponds to waiting until `jobs_to_finish` running jobs complete
            jobs_to_finish = action - (MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH) + 1
            self.delay_to_finished_job(jobs_to_finish=jobs_to_finish)

        else:
            # Should not happen if action_space is correct
            raise AssertionError("Action index out of predefined categories in env.step")

        # TODO: potential truncated logic or episode termination conditions
        
        reward = self.reward.get_reward(scheduled_job=scheduled_job, carbon_intensity=self.carbon_intensity)
        obs = self.build_observation()
        info = {}

        terminated = self.should_terminate()

        return obs, reward, terminated, truncated, info

    def should_terminate(self): 
        if self.scheduled_jobs == EPISODE_LENGTH: 
            print(EPISODE_LENGTH, " jobs scheduled, terminating episode")
            return True
        else:
            return False

    def reset(self, seed, options):
        # Randomize carbon offset and reset components
        random.seed(seed)
        random_offset = random.randint(0, 8760)
        self.episode_start_hour_offset = random_offset
        
        if VARIABLE_CARBON_INTENSITIES == True: 
            self.carbon_intensity.reset(start_offset=random_offset)
        else:
            self.carbon_intensity.reset(start_offset=0)


        self.start_job_offset = random.randint(0, max(0, (self.loads.size() - EPISODE_LENGTH - 1)))

        self.cluster.reset()
        self.loads.reset(start_job_offset=self.start_job_offset)
        
        self.current_step = 0
        self.start = 0
        # Reset env variables
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.scheduled_jobs = 0 
        self.current_timestamp = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        self.num_job_in_batch = EPISODE_LENGTH
        self.last_job_in_batch = self.num_job_in_batch
        self.next_arriving_job_idx = 0

        return self.build_observation(), {}

    def build_observation(self) -> np.ndarray:
        # Creating queued jobs encoding
        self.job_queue.sort(key=lambda job: job.submit_time)
        queue_vector = []

        for job in self.job_queue:
            job_vector = job.encode_vector(self.current_timestamp)
            queue_vector.append(job_vector)

        # Fill in with empty jobs to maintain size
        for _ in range(len(self.job_queue), MAX_QUEUE_SIZE):
            empty_job_encoding = np.zeros((JOB_FEATURE,), dtype=np.float32)
            queue_vector.append(empty_job_encoding)

        # Flatten queue vector
        queue_flat = np.concatenate(queue_vector).astype(np.float32)

        # Creating running jobs encoding
        # Sort by completion time (scheduled_time + run_time)
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        running_jobs_vector = []

        for running_job in self.running_jobs:
            allocated_processors = running_job.request_number_of_processors
            remaining = (running_job.scheduled_time + running_job.run_time) - self.current_timestamp
            time_until_finish = max(remaining, 0)
            running_job_encoding = np.array([allocated_processors, time_until_finish], dtype=np.float32)
            running_jobs_vector.append(running_job_encoding)
            # Cap the encoding length to RUN_WIN_LENGTH earliest finishing jobs
            if len(running_jobs_vector) == RUN_WIN_LENGTH:
                break

        # Fill in with empty running job encodings to maintain size
        for _ in range(len(running_jobs_vector), RUN_WIN_LENGTH):
            empty_running_job_encoding = np.zeros((RUN_FEATURE,), dtype=np.float32)
            running_jobs_vector.append(empty_running_job_encoding)

        running_flat = np.concatenate(running_jobs_vector).astype(np.float32)

        # Create carbon intensity encoding
        # Note: CarbonIntensity.create_carbon_forecast_enconding returns a 1-D numpy array
        carbon_vector = self.carbon_intensity.create_carbon_forecast_enconding(self.current_timestamp).astype(np.float32)

        # Concatenate everything into a single observation vector
        obs = np.concatenate((queue_flat, carbon_vector, running_flat)).astype(np.float32)

        return obs

    def schedule_job(self, queue_index):
        # Guard index
        if queue_index < 0 or queue_index >= len(self.job_queue):
            raise IndexError("schedule_job: queue_index out of range")

        job = self.job_queue[queue_index]

        if not self.cluster.can_allocated(job):
            raise AssertionError("Tried to schedule an invalid scheduling. This should be masked out")

     #   if self.debug:
            #EPI("Current step: ", self.current_step)
            #print("Current timestamp: ", self.current_timestamp)
            #print("Trying to schedule job: ", job.job_id, ". Submitted to queue at: ", job.submit_time )

        assert job.scheduled_time == -1
        assert job.submit_time <= self.current_timestamp
        assert job.power_usage != -1

        job.scheduled_time = self.current_timestamp
        allocated_nodes = self.cluster.allocate(job_id=job.job_id, request_num_procs=job.request_number_of_processors)

        # Save allocated machines on the job to allow release later
        job.allocated_machines = allocated_nodes
        self.running_jobs.append(job)
        self.job_queue.remove(job)
        self.scheduled_jobs += 1 

        return job

    def delay_to_finished_job(self, jobs_to_finish):
        """
        Advances the simulation time until a specific number of running jobs have finished,
        while processing intermediate events like other job completions or new arrivals.
        jobs_to_finish is treated as a count (1 = wait for the next job to finish).
        """
        if jobs_to_finish <= 0:
            return

        # Sort running jobs by completion time
        self.running_jobs.sort(key=lambda job: (job.scheduled_time + job.run_time))

        # Convert count to index (1-based count -> 0-based index)
        target_index = jobs_to_finish - 1
        if target_index >= len(self.running_jobs):
            # If asking for more than available, skip until at most available jobs finish (or cap to 1 hour)
            next_time_after_skip = min(self.current_timestamp + 3600, sys.maxsize)
        else:
            release_time = (self.running_jobs[target_index].scheduled_time + self.running_jobs[target_index].run_time)
            next_time_after_skip = min(release_time, self.current_timestamp + 3600)

        self._process_events_until(next_time_after_skip)

    def delay_fixed_amount(self, skip_time):
        """
        Advances the simulation time by a fixed amount, while processing intermediate events.
        """
        next_time_after_skip = self.current_timestamp + skip_time
        self._process_events_until(next_time_after_skip)

    def _update_next_job_submit_time(self):
        """Gets the submit time of the next job in the batch, or sys.maxsize if none."""
        if self.next_arriving_job_idx < self.last_job_in_batch and self.next_arriving_job_idx < self.loads.size():
            return self.loads[self.next_arriving_job_idx].submit_time
        return sys.maxsize

    def _update_next_resource_release(self):
        """
        Gets the completion time and machines of the next-to-finish running job.
        Returns (release_time, list_of_machines) or (sys.maxsize, []) if none.
        """
        if self.running_jobs:
            next_job = self.running_jobs[0]
            release_time = next_job.scheduled_time + next_job.run_time
            release_machines = next_job.allocated_machines if next_job.allocated_machines is not None else []
            return release_time, release_machines
        return sys.maxsize, []

    def _process_events_until(self, next_time_after_skip):
        """
        Event loop that processes job arrivals and completions until next_time_after_skip.
        """
        # Sort running jobs by completion time to process them in order
        self.running_jobs.sort(key=lambda job: (job.scheduled_time + job.run_time))

        next_resource_release_time, next_resource_release_machines = self._update_next_resource_release()
        next_job_submit_time = self._update_next_job_submit_time()

        while True:
            next_event_time = min(next_job_submit_time, next_resource_release_time)

            # If the skip time is before the next event, advance time and exit.
            if next_time_after_skip < next_event_time:
                self.current_timestamp = max(self.current_timestamp, next_time_after_skip)
                return

            # If the next event is a job arrival
            if next_job_submit_time <= next_resource_release_time:
                # Guard: ensure index is valid before appending
                if self.next_arriving_job_idx < self.last_job_in_batch and self.next_arriving_job_idx < self.loads.size():
                    self.current_timestamp = max(self.current_timestamp, next_job_submit_time)
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                    next_job_submit_time = self._update_next_job_submit_time()
                else:
                    # No more arrivals; set next_job_submit_time to infinity and continue to process releases
                    next_job_submit_time = sys.maxsize
            # If the next event is a resource release (job completion)
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                # Release cluster resources
                if next_resource_release_machines:
                    self.cluster.release(next_resource_release_machines)
                # Remove the completed job from running_jobs if present
                if self.running_jobs:
                    self.running_jobs.pop(0)
                next_resource_release_time, next_resource_release_machines = self._update_next_resource_release()

    def valid_action_mask(self):
        mask = np.full(self.action_space_size, True, dtype=bool)
       
        # Remove jobs empty jobslots
        queue_length = len(self.job_queue)
        for i in range(queue_length,MAX_QUEUE_SIZE):
            mask[i] = False

        # Remove jobs that cannot fit
        for idx, job in enumerate(self.job_queue):
            if not self.cluster.can_allocated(job):
                mask[idx] = False
        
        # Remove options to wait n jobs to complete if there are fewer than n running jobs
        if len(self.running_jobs) < MAX_WAIT_N_JOBS:
            for i in range(len(self.running_jobs), MAX_WAIT_N_JOBS):
                mask[MAX_QUEUE_SIZE + DELAY_TIME_LIST_LENGTH + i] = False
         
        return mask