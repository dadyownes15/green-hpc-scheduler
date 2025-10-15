import configparser
import numpy as np
from gymnasium import Env, spaces
from typing import List, Dict, Any
import random
import sys
from src.cluster import Cluster
from src.workloads import Workloads
from src.job import Job
from src.carbon_intensity import CarbonIntensity
from src.utils import create_directory_if_not_exists, get_config_as_dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class HPCenv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_dict, mode = "training", debug=False, name = None, trace_enabled: bool = True):
        
        self.debug = debug
        self.config_dict = config_dict 
        self.name = name
        self.mode = mode
        self.trace_enabled = bool(trace_enabled)
        self.state = None
        self.current_step = 0
        self.job_queue: List[Job] = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.new_job_arrived_in_step = False
        self.last_action_info: Dict[str, Any] = {'type': None, 'is_delay': False}
        self.action_trace = []  # list of dict entries, one per env.step()
        self.step_counter = 0
        self.episode_start_hour_offset = 0
        self.carbon_reward_booster = config_dict["carbon_reward_booster"]
        self.wait_reward_booster = config_dict["wait_reward_booster"]
        
        ## ------ Reward config -------
        self.reward_type = config_dict["reward_type"]
        self.eta = config_dict["eta"]

        ## ------ Data config -------
        if self.mode == "training":
            self.workload_path = "data/workloads/training_workload.swf"
        if self.mode == "validation":
            self.config_dict["episode_length"] = 8174  # cover entire validation trace
            self.workload_path = "data/workloads/validation_workload.swf"
        if self.mode == "test":
            self.config_dict["episode_length"] = 22341  # cover entire test trace
            self.workload_path = "data/workloads/test_workload.swf"

        ## -------- Action and observation space def -------
        # Action space
        max_queue_size = self.config_dict['max_queue_size']
        delay_action_count = self.config_dict['delay_time_list_length']
        wait_action_count = self.config_dict['max_wait_n_jobs']

        job_action_count = max_queue_size
        noop_action_count = delay_action_count + wait_action_count

        self.action_space_size = job_action_count + noop_action_count
        self.action_space = spaces.Discrete(self.action_space_size)

        # Observation space
        job_feature_count = self.config_dict['job_feature']
        run_window_length = self.config_dict['run_win_length']
        run_feature_count = self.config_dict['run_feature']
        green_constant_features = self.config_dict['green_feature_constant']
        green_forecast_length = self.config_dict['green_forecast_length']
        green_features_per_slot = self.config_dict['green_feature_pr_timeslot']

        queue_features = max_queue_size * job_feature_count
        run_window_features = run_window_length * run_feature_count
        # Forecast includes current slot separately; historical/forecasted slots are (length - 1)
        green_forecast_slots = max(0, green_forecast_length - 1)
        green_forecast_features = green_forecast_slots * green_features_per_slot

        obs_len = (
            queue_features
            + run_window_features
            + green_constant_features
            + green_forecast_features
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(int(obs_len),), dtype=np.float32)


        # Load workloads and cluster
        self.loads = Workloads(self.workload_path, config_dict=self.config_dict)
        self.cluster = Cluster(self.loads.max_nodes, self.config_dict['procs_per_node'], self.config_dict['idle_power'])
        self.carbon_intensity = CarbonIntensity(
                normalize= self.mode == "training",
             green_win_length=self.config_dict['green_forecast_length'], custom_intensity=config_dict['custom_intensity'])
        self.carbon_intensity.set_mode(self.mode)
        self.total_processors = self.loads.max_procs

        assert self.config_dict is not None, "Config dict, did not parse"
        assert self.mode in ["training", "validation", "test"]
        assert self.reward_type in ["wait_abs_ems", "wait_abs_ems_clip","wait_abs_ems_ci_clip","bd_abs_ems","wait_relative_ems", "bd_relative_ems","wait_relative_compute_ems","bd_abs_ems_clip"]
 
    def step(self, action):
        self.new_job_arrived_in_step = False
        self.last_action_info['type'] = action
        scheduled_job = None
        terminated = False
        truncated = False
        self.step_counter += 1
        
        if self.step_counter % 10_000 == 0 and (self.mode == "test" or self.mode == "validation"):
            print("Current secounds after start: ", self.current_timestamp)

        

        # Auto-advance: if the queue is empty and there are future arrivals,
        # advance time to the next job submission so the agent doesn't need
        # to issue a manual delay action.
        if len(self.job_queue) == 0:
            next_submit_time = self._update_next_job_submit_time()
            if next_submit_time != sys.maxsize:
            #   print("After scheduling jobs: ", self.scheduled_jobs)
            #    print("Queue is empty and future jobs are appending")
                
                if next_submit_time != sys.maxsize and next_submit_time > self. current_timestamp:
                    self._process_events_until(next_submit_time)

        # Snapshot for logging and reward attribution
        t_before = self.current_timestamp
        qlen_before = len(self.job_queue)
        trace_entry = self._init_trace_entry(int(action)) if self.trace_enabled else None


        ## Scheduling action
        if 0 <= action < self.config_dict['max_queue_size']:
            # Perform action
            scheduled_job = self.schedule_job(int(action))
            # Log trace
            self._log_trace_schedule(scheduled_job=scheduled_job, trace_entry=trace_entry)
            self.last_action_info['is_delay'] = False

        ## Fixed delay action
        elif self.config_dict['max_queue_size'] <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']):
            delay_idx = action - self.config_dict['max_queue_size']
            skip_time = int(self.config_dict['delay_time_list'][delay_idx])
            events = {'arrivals': [], 'completions': []}
            # Perform action
            self.delay_fixed_amount(skip_time=skip_time, events=events)
            # Log trace
            self._log_trace_delay_fixed(skip_time=skip_time, events=events, trace_entry=trace_entry)
            self.last_action_info['is_delay'] = True

        # Wait til n jobs are finished action
        elif (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length'] + self.config_dict['max_wait_n_jobs']):
            # action corresponds to waiting until `jobs_to_finish` running jobs complete
            jobs_to_finish = int(action - (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) + 1)
            events = {'arrivals': [], 'completions': []}
            # Perform action
            self.delay_to_finished_job(jobs_to_finish=jobs_to_finish, events=events)
            # Log trace
            self._log_trace_entry_wait_jobs(jobs_to_finish=jobs_to_finish, events=events, trace_entry=trace_entry)
            self.last_action_info['is_delay'] = True

        else:
            # Should not happen if action_space is correct
            raise AssertionError("Action index out of predefined categories in env.step")

        
        # Compute reward for the action
        # Note: for delay actions, time may have advanced. We capture the delta
        # to attribute a small waiting penalty per-second to all queued jobs.
        dt = self.current_timestamp - t_before

        obs = self.build_observation()
        info = {}
        if self.current_timestamp - self.time_offset > 57303812: ## episode duration for 5th percentile
            print("Timestamp at exit: ", self.current_timestamp)
            if len(self.scheduled_job_history) > 0:
                print("Last job: ", self.scheduled_job_history[-1])
                print("FIrst job: ", self.scheduled_job_history[0])
            else:
                print("No scheduled jobs")
            truncated = True;
            print("Truncating episode")
            print("Jobs in queue: ", len(self.job_queue))
            print("Jobs left in episode: ",self.config_dict["episode_length"]  - self.scheduled_jobs) 
            print("Jobs unseen in episode: ",self.config_dict["episode_length"]  - self.scheduled_jobs - len(self.job_queue)) 
            reward = -10000*self.config_dict["episode_length"] 
            return obs, reward, terminated, truncated, info

        reward, components = self.get_reward(
            scheduled_job=scheduled_job,
            current_timestamp=self.current_timestamp,
            time_advanced=dt,
            was_delay=self.last_action_info['is_delay'],
        )
       

        terminated = self.should_terminate()


        # Expose reward breakdown for logging/analysis
        info.update({
            'reward_total': float(components.get('total', 0.0)),
            'reward_wait': float(components.get('wait', 0.0)),
            'reward_carbon': float(components.get('carbon', 0.0)),
        })

        
                # TODO: potential truncated logic or episode termination conditions

        if self.trace_enabled:
            self._finalize_trace_entry(trace_entry)

        return obs, reward, terminated, truncated, info

    def should_terminate(self): 
        if self.scheduled_jobs == self.config_dict['episode_length']: 
            return True
        else:
            return False

    def reset(self, seed, options = {}):

           
        # Randomize carbon offset and reset components
        super().reset(seed=seed)
        random.seed(seed)

        # Ensure deterministic start offset for validation/test only
        if self.mode in ["validation", "test"]:
            self.start_job_offset = 0        
        else:
            self.start_job_offset = random.randint(0, max(0, (self.loads.size() - self.config_dict['episode_length'] - 1)))
        
        self.loads.reset(start_job_offset=self.start_job_offset)

        first_job = self.loads.get_job(0)
        self.time_offset = first_job.submit_time
        self.episode_start_hour_offset = self.time_offset // 3600 
        self.action_trace = []
        self.step_counter = 0


        self.cluster.reset()
        self.loads.reset(start_job_offset=self.start_job_offset)
        
        self.current_step = 0
        self.start = 0
        # Reset env variables
        self.job_queue = []
        self.running_jobs = []
        self.scheduled_job_history = []
        self.delay_history = []
        
        # Rendering
        self.new_job_arrived_in_step = True # True on reset to force first render
        self.last_action_info = {'type': None, 'is_delay': False}
        # First job
      
        self.job_queue.append(first_job)


        self.scheduled_jobs = 0 
        self.current_timestamp = first_job.submit_time
        self.next_arriving_job_idx =  1
        self.num_job_in_batch = 1

        self.num_job_in_batch = self.config_dict['episode_length']
        self.last_job_in_batch = self.num_job_in_batch
        self.next_arriving_job_idx = 1

        return self.build_observation(), {}

    def build_observation(self) -> np.ndarray:
        # Creating queued jobs encoding
        self.job_queue.sort(key=lambda job: job.submit_time)
        queue_vector = []
        max_visible_queue_length = self.config_dict['max_queue_size']
    
        for i in range(min(max_visible_queue_length, len(self.job_queue))):
            job = self.job_queue[i]
            job_vector = job.encode_vector(self.current_timestamp)
            queue_vector.append(job_vector)

        # Fill in with empty jobs to maintain size
        for _ in range(len(queue_vector), max_visible_queue_length):
            empty_job_encoding = np.zeros((self.config_dict['job_feature'],), dtype=np.float32)
            queue_vector.append(empty_job_encoding)
            
        assert len(queue_vector) <= self.config_dict["max_queue_size"]
        # Flatten queue vector
        queue_flat = np.concatenate(queue_vector).astype(np.float32)

        # Creating running jobs encoding
        # Sort by completion time (scheduled_time + run_time)
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        running_jobs_vector = []

        for running_job in self.running_jobs:
            remaining = (running_job.scheduled_time + running_job.run_time) - self.current_timestamp
            time_until_finish = max(remaining / self.config_dict['max_run_time'], 0)
            running_job_encoding = np.array([(
                running_job.request_number_of_processors - running_job.processor_mean) / running_job.processor_std, time_until_finish], dtype=np.float32)
            running_jobs_vector.append(running_job_encoding)
            # Cap the encoding length to run_win_length earliest finishing jobs
            if len(running_jobs_vector) == self.config_dict['run_win_length']:
                break

        # Fill in with empty running job encodings to maintain size
        for _ in range(len(running_jobs_vector), self.config_dict['run_win_length']):
            empty_running_job_encoding = np.zeros((self.config_dict['run_feature'],), dtype=np.float32)
            running_jobs_vector.append(empty_running_job_encoding)

        running_flat = np.concatenate(running_jobs_vector).astype(np.float32)

        # Create carbon intensity encoding
        # Note: CarbonIntensity.create_carbon_forecast_enconding returns a 1-D numpy array
        carbon_vector = self.carbon_intensity.create_carbon_forecast_enconding(self.current_timestamp).astype(np.float32)
        if self.debug:
            print("Average carbon next 24 hours: ", np.mean(carbon_vector[8:]))
            print("Current CI: ", carbon_vector[0])
        # Concatenate everything into a single observation vector
        obs = np.concatenate((queue_flat, running_flat, carbon_vector)).astype(np.float32)

        return obs

    def schedule_job(self, queue_index):
        # Guard index
        if queue_index < 0 or queue_index >= len(self.job_queue):
            # Check if the the queue index was block
            raise IndexError("schedule_job: queue_index out of range")

        job = self.job_queue[queue_index]

        if not self.cluster.can_allocated(job):
            raise AssertionError("Tried to schedule an invalid scheduling. This should be masked out")
        if self.debug:
            print("Job scheduled: ", job.job_id)
            print("Current timestamp (env): ", self.current_timestamp)

        assert job.scheduled_time == -1
        assert job.submit_time <= self.current_timestamp
        assert job.power_usage != -1

        job.scheduled_time = self.current_timestamp
        allocated_nodes = self.cluster.allocate(job_id=job.job_id, request_num_procs=job.request_number_of_processors)

        # Save allocated machines on the job to allow release later
        job.allocated_machines = allocated_nodes
        self.running_jobs.append(job)
        # Track scheduled jobs for validation metrics
        self.scheduled_job_history.append(job)
        self.job_queue.remove(job)
        self.scheduled_jobs += 1 


        return job

    def delay_to_finished_job(self, jobs_to_finish, events: Dict[str, list] | None = None):
        """
        Advances the simulation time until a specific number of running jobs have finished,
        while processing intermediate events like other job completions or new arrivals.
        jobs_to_finish is treated as a count (1 = wait for the next job to finish).
        """
        start_delay_time = self.current_timestamp 

        if jobs_to_finish <= 0:
            if self.debug:
                print("jobs to finished was zero or negative")
            return

        if self.debug:
            print("Delay to finished job: ")
            print("Current timestamp (env): ", self.current_timestamp)
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

        self._process_events_until(next_time_after_skip, events=events)
        if self.current_timestamp > start_delay_time:
            self.delay_history.append((start_delay_time, self.current_timestamp))


    def delay_fixed_amount(self, skip_time, events: Dict[str, list] | None = None):
        """
        Advances the simulation time by a fixed amount, while processing intermediate events.
        """

        if self.debug:
            print("Delay fixed amount: ", skip_time)
            print("current timestamp: ", self.current_timestamp)
            print("Job left: ", (self.config_dict['episode_length'] - self.scheduled_jobs) )
            job_ids = [job.job_id for job in self.job_queue ]
            print("queue, ", job_ids)
            print("Nodes free, ", self.cluster.free_node)

        start_delay_time = self.current_timestamp
        next_time_after_skip = self.current_timestamp + skip_time
        self._process_events_until(next_time_after_skip, events=events)

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
        self.running_jobs.sort(key=lambda job: (job.scheduled_time + job.run_time))
        if self.running_jobs:
            next_job = self.running_jobs[0]
            release_time = next_job.scheduled_time + next_job.run_time
            release_machines = next_job.allocated_machines if next_job.allocated_machines is not None else []
            return release_time, release_machines
        return sys.maxsize, []

    def _process_events_until(self, next_time_after_skip, events: Dict[str, list] | None = None):
        """
        Event loop that processes job arrivals and completions until next_time_after_skip.
        """

        next_resource_release_time, next_resource_release_machines = self._update_next_resource_release()
        next_job_submit_time = self._update_next_job_submit_time()

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
                arriving_job = self.loads[self.next_arriving_job_idx]
                self.job_queue.append(arriving_job)
                if events is not None:
                    events['arrivals'].append(arriving_job.job_id)
                self.next_arriving_job_idx += 1
                # Rendering
                self.new_job_arrived_in_step = True
                next_job_submit_time = self._update_next_job_submit_time()

            else:
                # No more arrivals; set next_job_submit_time to infinity and continue to process releases
                next_job_submit_time = sys.maxsize
        # If the next event is a resource release (job completion)
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            # Release cluster resources
            if self.running_jobs:
                finished_job = self.running_jobs[0]
                if events is not None:
                    events['completions'].append(finished_job.job_id)
                if next_resource_release_machines:
                    self.cluster.release(next_resource_release_machines)
                # Remove the completed job from running_jobs if present
                self.running_jobs.pop(0)
            next_resource_release_time, next_resource_release_machines = self._update_next_resource_release()

    def get_action_trace(self):
        return list(self.action_trace)

    # -----------------
    # Logging helpers
    # -----------------
    def _init_trace_entry(self, action_index: int) -> Dict[str, Any]:
        t_before = self.current_timestamp
        qlen_before = len(self.job_queue)
        rlen_before = len(self.running_jobs)
        return {
            'step': self.step_counter,
            'action_index': int(action_index),
            'action_type': None,
            'timestamp_before': t_before,
            'timestamp_after': None,
            'scheduled_job_id': None,
            'scheduled_job_procs': None,
            'scheduled_job_run_time': None,
            'delay_kind': None,
            'delay_value': None,
            'events': {'arrivals': [], 'completions': []},
            'queue_len_before': qlen_before,
            'running_len_before': rlen_before,
            'queue_len_after': None,
            'running_len_after': None,
        }

    def _finalize_trace_entry(self, trace_entry: Dict[str, Any]) -> None:
        trace_entry['timestamp_after'] = self.current_timestamp
        trace_entry['queue_len_after'] = len(self.job_queue)
        trace_entry['running_len_after'] = len(self.running_jobs)
        self.action_trace.append(trace_entry)

    def _compute_episode_metrics(self) -> Dict[str, float]:
        jobs = getattr(self, 'scheduled_job_history', [])
        if not jobs:
            return {
                'avg_wait': 0.0,
                'avg_emissions': 0.0,
                'span_seconds': 0.0,
                'job_count': 0,
            }

        waits = []
        emissions = []
        first_submit = None
        last_finish = None

        for job in jobs:
            if job.scheduled_time == -1:
                continue

            wait = max(0, job.scheduled_time - job.submit_time)
            waits.append(wait)

            start = job.scheduled_time
            end = start + job.run_time

            try:
                emission = self.carbon_intensity.getCarbonEmissions(job.power_usage, start, end)
                emissions.append(emission)
            except Exception:
                pass

            first_submit = job.submit_time if first_submit is None else min(first_submit, job.submit_time)
            last_finish = end if last_finish is None else max(last_finish, end)

        avg_wait = float(np.mean(waits)) if waits else 0.0
        avg_emissions = float(np.mean(emissions)) if emissions else 0.0
        span_seconds = float((last_finish - first_submit) if first_submit is not None and last_finish is not None else 0.0)

        return {
            'avg_wait': avg_wait,
            'avg_emissions': avg_emissions,
            'span_seconds': span_seconds,
            'job_count': len(waits),
        }

    def _log_trace_schedule(self, scheduled_job: Job | None, trace_entry: Dict[str, Any]) -> None:
        if trace_entry is None:
            return
        trace_entry['action_type'] = 'schedule'
        if scheduled_job is not None:
            trace_entry['scheduled_job_id'] = scheduled_job.job_id
            trace_entry['scheduled_job_procs'] = scheduled_job.request_number_of_processors
            trace_entry['scheduled_job_run_time'] = scheduled_job.run_time
            # Add submit and wait time to enable downstream wait-time plotting
            try:
                trace_entry['scheduled_job_submit_time'] = int(scheduled_job.submit_time)
                trace_entry['scheduled_job_wait_time'] = int(max(0, self.current_timestamp - scheduled_job.submit_time))
            except Exception:
                pass
            # Include nodes for precise utilization reconstruction downstream
            try:
                trace_entry['scheduled_job_nodes'] = scheduled_job.request_number_of_nodes
            except Exception:
                pass

    def _log_trace_delay_fixed(self, skip_time: int, events: Dict[str, list], trace_entry: Dict[str, Any]) -> None:
        if trace_entry is None:
            return
        trace_entry['action_type'] = 'delay'
        trace_entry['delay_kind'] = 'fixed'
        trace_entry['delay_value'] = int(skip_time)
        trace_entry['events'] = events

    def _log_trace_entry_wait_jobs(self, jobs_to_finish: int, events: Dict[str, list], trace_entry: Dict[str, Any]) -> None:
        if trace_entry is None:
            return
        trace_entry['action_type'] = 'delay'
        trace_entry['delay_kind'] = 'wait'
        trace_entry['delay_value'] = int(jobs_to_finish)
        trace_entry['events'] = events


    def valid_action_mask(self):
        mask = np.full(self.action_space_size, True, dtype=bool)
        
        # Remove jobs empty jobslots
        queue_length = len(self.job_queue)
        for i in range(queue_length, self.config_dict['max_queue_size']):
            mask[i] = False

        # Remove jobs that cannot fit
        for idx, job in enumerate(self.job_queue[:self.config_dict['max_queue_size']]):
            if not self.cluster.can_allocated(job):
                mask[idx] = False
        
        # Remove options to wait n jobs to complete if there are fewer than n running jobs
        if len(self.running_jobs) < self.config_dict['max_wait_n_jobs']:
            for i in range(len(self.running_jobs), self.config_dict['max_wait_n_jobs']):
                mask[self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length'] + i] = False
        
        # Prevent indefinite stalling at the end of the episode during eval
        # If no more arrivals remain and at least one job can be scheduled now,
        # mask out all delay actions (both fixed delays and wait-for-N-jobs).

        """         
        ## TO DO: Remove this, and figure out why we need this
        no_more_arrivals = not (self.next_arriving_job_idx < self.last_job_in_batch and self.next_arriving_job_idx < self.loads.size())
        any_schedulable = any(
            self.cluster.can_allocated(job) for job in self.job_queue[:self.config_dict['max_queue_size']]
        )
        if self.mode in ["validation", "test"] and no_more_arrivals and any_schedulable:
            # Mask all delay actions: indices from max_queue_size to end
            mask[self.config_dict['max_queue_size']:] = False
        """
        assert np.all(mask[self.config_dict['max_queue_size']:self.config_dict["delay_time_list_length"]])
        return mask

    def get_reward(self,scheduled_job : Job | None, current_timestamp, time_advanced: int = 0, was_delay: bool = False):
        """
        Returns total reward and a components dict.
        Components keys:
          - carbon: carbon-related reward (>=0 if improvement)
          - wait_schedule: negative penalty based on actual wait at scheduling time
          - delay_wait: incremental negative penalty for delaying (time_advanced>0)
          - total: sum of the above
        """
        reward = 0.0
        components = {'carbon': 0.0, 'wait': 0.0}
      
        if self.reward_type == "wait_relative_compute_ems":
            if scheduled_job: 

                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                # Carbon reward calcuation
                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                carbon_emission_initial = self.carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                carbon_ratio_reward = ((carbon_emission_initial-carbon_emission))/(carbon_emission_initial + 0.1)

                compute_time = scheduled_job.request_number_of_nodes * scheduled_job.request_time

                compute_time_norm = (compute_time - 143567.71930292607)/587102.0877243354
                

                carbon_ratio_reward = np.clip(carbon_ratio_reward*compute_time_norm,-self.config_dict["reward_clip"],self.config_dict["reward_clip"])

                # Waittime calculation
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                components['carbon'] = - float(carbon_ratio_reward)*(1-self.eta)
                components['wait'] = - (actual_wait / self.config_dict["max_wait_time"])*self.eta
                reward = components['wait'] + components['carbon']

            else: 
                reward = 0                
        if self.reward_type == "wait_relative_ems":
            if scheduled_job: 

                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                # Carbon reward calcuation
                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                carbon_emission_initial = self.carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                carbon_ratio_reward = ((carbon_emission_initial-carbon_emission))/(carbon_emission_initial + 1)

                carbon_ratio_reward_clipped = max(-self.config_dict["reward_clip"], min(self.config_dict["reward_clip"], carbon_ratio_reward))
                # Waittime calculation
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                components['carbon'] = - float(carbon_ratio_reward_clipped)*(1 - self.eta)
                components['wait'] = - (actual_wait / self.config_dict["max_wait_time"])*self.eta
                reward = components['wait'] + components['carbon']

            else: 
                reward = 0               
        if self.reward_type == "wait_abs_ems":
            if scheduled_job: 

                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                # Carbon reward calcuation
                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)
                components['carbon'] = - carbon_emission*(1-self.eta)
                components['wait'] = - (actual_wait / self.config_dict["max_wait_time"])*self.eta
    
                reward = components['wait'] + components['carbon']

            else: 
                reward = 0               
        if self.reward_type == "wait_abs_ems_clip":
            if scheduled_job: 

                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                # Carbon reward calcuation
                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                carbon_reward = - np.clip(carbon_emission * self.carbon_reward_booster, -self.config_dict["abs_carbon_reward_clip"],self.config_dict["abs_carbon_reward_clip"] )

                wait = - (actual_wait / self.config_dict["max_wait_time"])
                wait_reward = np.clip(wait*self.wait_reward_booster, -self.config_dict['wait_reward_clip'], 0)
                components['carbon'] = carbon_reward*(1-self.eta)
                components['wait'] = wait_reward*self.eta
    
                reward = components['wait'] + components['carbon']

            else: 
                reward = 0   
        if self.reward_type == "wait_abs_ems_ci_clip":
            if scheduled_job:
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                assert self.carbon_reward_booster is not None and self.carbon_reward_booster != 0
                carbon_reward = -np.clip(
                    carbon_emission * self.carbon_reward_booster,
                    -self.config_dict["abs_carbon_reward_clip"],
                    self.config_dict["abs_carbon_reward_clip"],
                )

                wait = -(actual_wait / self.config_dict["max_wait_time"])
                wait_reward = np.clip(wait, -self.config_dict["wait_reward_clip"], 0)

                job_ci = float(np.clip(getattr(scheduled_job, "carbon_consideration", 1.0), 0.0, 1.0))
                components['carbon'] = carbon_reward * (1.0 - job_ci)
                components['wait'] = wait_reward * job_ci
                reward = components['wait'] + components['carbon']
            else:
                reward = 0
                                
        if self.reward_type == "bd_abs_ems_clip":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time

                # Carbon reward calcuation
                power_usage = scheduled_job.power_usage
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)
                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                carbon_reward = np.clip(carbon_emission, -self.config_dict["abs_carbon_reward_clip"],self.config_dict["abs_carbon_reward_clip"] )

                actual_wait = max(0, current_timestamp - scheduled_job.submit_time)

                bounded_slowdown = (actual_wait + scheduled_job.run_time) / max([0, scheduled_job.run_time])

                components['carbon'] = carbon_reward*(1-self.eta)
                components['wait'] = -float(bounded_slowdown)*self.eta
                reward = components['carbon'] + components['wait']
            else: 
                reward = 0 
      
        components['total'] = float(reward)
        return float(reward), components


def _assert_finite(x, name):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        bad = np.where(~np.isfinite(x))
        raise ValueError(f"Non-finite {name} detected at indices {bad}")
