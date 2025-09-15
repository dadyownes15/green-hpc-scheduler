import os
import ast
import configparser
import numpy as np
from gymnasium import Env, spaces
from typing import List, Dict, Any
import random
import sys
from src.cluster import Cluster
from src.workloads import Workloads
from src.job import Job
from src.reward import Reward
from src.carbon_intensity import CarbonIntensity
from src.utils import create_directory_if_not_exists, get_config_as_dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class HPCenv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_dict, mode = "training", debug=False, generate_rendering = False, name = None, ):
        self.debug = debug
        self.config_dict = config_dict 
        assert self.config_dict is not None, "Config dict, did not parse"
        assert mode in ["training", "validation", "test"]
        assert generate_rendering == False or (generate_rendering == True and name != None), "You must name the env, to be able to generate renderings" 

        self.name = name
        self.generate_rendering = generate_rendering
        self.mode = mode

        ## ------ Reward config --------
        self.reward_type = config_dict["reward_type"]
        assert self.reward_type in ["CO2_direct", "delay_vs_now_reward", "CO2_direct_c", "delay_vs_now_reward_n","carbon_ratio_plus"]
        self.eta = config_dict["eta"]
        self.bounded_slowdown_threshold = config_dict["bounded_slowdown_threshold"]
        self.alpha = config_dict["alpha"]
        # Flat action space
        num_job_actions = self.config_dict['max_queue_size']
        do_nothing_actions = self.config_dict['delay_time_list_length'] + self.config_dict['max_wait_n_jobs']
        self.action_space_size = num_job_actions + do_nothing_actions
        self.action_space = spaces.Discrete(self.action_space_size)

        # Observation space: flattened 1-D vector
        obs_len = (self.config_dict['max_queue_size'] * self.config_dict['job_feature']) + (self.config_dict['run_win_length'] * self.config_dict['run_feature']) + self.config_dict['green_feature_constant'] + ((self.config_dict['green_forecast_length']-1) * self.config_dict['green_feature_pr_timeslot'])
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


        # The idea is that we train from data in 2021 - 01 - 01 to 2023 - 31 - 12 only, and ensure that this is synched with the cycles defined in the lublin trace. 
        if self.mode == "training":
            self.workload_path = "data/workloads/training_workload.swf"
        if self.mode == "validation":
            self.config_dict["episode_length"] = 5643 # This value is hardcoded to be the length of the whole validation set
            self.workload_path = "data/workloads/validation_workload.swf"
        if self.mode == "test":
            self.config_dict["episode_length"] = 54811 # this is hardcoded to be the length of the whole test set
            self.workload_path = "data/workloads/test_workload.swf"

        # Load workloads and cluster
        self.loads = Workloads(self.workload_path, config_dict=self.config_dict)
        self.cluster = Cluster(self.loads.max_nodes, self.config_dict['procs_per_node'], self.config_dict['idle_power'])
        self.reward = Reward(config_dict=config_dict)
        self.carbon_intensity = CarbonIntensity( green_win_length=self.config_dict['green_forecast_length'], custom_intensity=config_dict['custom_intensity'])
        self.carbon_intensity.set_mode(self.mode)

        # For visualization
        self.total_processors = self.loads.max_procs
        self.scheduled_job_history = []
        self.delay_history = []
        self.new_job_arrived_in_step = False
        self.last_action_info: Dict[str, Any] = {'type': None, 'is_delay': False}
        
        #logging
        self.episode_start_time = 0
        # From simple counters to a dictionary holding lists for indexed actions
        self.action_log = {
            'schedule': 0,
            'delay_fixed': 0,
            'delay_wait': 0,
            'delay_fixed_indices': [0] * self.config_dict['delay_time_list_length'],
            'delay_wait_indices': [0] * self.config_dict['max_wait_n_jobs']
        }
        # Step-by-step action trace for exact replay
        self.action_trace = []  # list of dict entries, one per env.step()
        self.step_counter = 0
        # Internal event recording during delays
        self._record_events = False
        self._recorder_arrivals = []
        self._recorder_completions = []
        # Episode-specific carbon-timeline offset (in hours)
        self.episode_start_hour_offset = 0

    def step(self, action):
        self.new_job_arrived_in_step = False
        self.last_action_info['type'] = action
        # Initialize default outputs
        scheduled_job = None
        terminated = False
        truncated = False
        # For action trace
        t_before = self.current_timestamp
        qlen_before = len(self.job_queue)
        rlen_before = len(self.running_jobs)
        trace_entry: Dict[str, Any] = {
            'step': self.step_counter,
            'action_index': int(action),
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

        # Schedule job [0 ... max_queue_size-1]
        # Delay fixed amount [max_queue_size ... max_queue_size + delay_time_list_length - 1]
        # Wait until N jobs are finished [max_queue_size + delay_time_list_length ... max_queue_size + delay_time_list_length + max_wait_n_jobs - 1]

        if 0 <= action < self.config_dict['max_queue_size']:
            scheduled_job = self.schedule_job(action)
            trace_entry['action_type'] = 'schedule'
            self.last_action_info['is_delay'] = False
            if scheduled_job is not None:
                trace_entry['scheduled_job_id'] = scheduled_job.job_id
                trace_entry['scheduled_job_procs'] = scheduled_job.request_number_of_processors
                trace_entry['scheduled_job_run_time'] = scheduled_job.run_time

        elif self.config_dict['max_queue_size'] <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']):
            delay_idx = action - self.config_dict['max_queue_size']
            skip_time = self.config_dict['delay_time_list'][delay_idx]
            # Record intermediate events during delay
            self._record_events = True
            self._recorder_arrivals = []
            self._recorder_completions = []
            self.delay_fixed_amount(skip_time=skip_time)
            trace_entry['action_type'] = 'delay'
            trace_entry['delay_kind'] = 'fixed'
            trace_entry['delay_value'] = int(skip_time)
            trace_entry['events']['arrivals'] = list(self._recorder_arrivals)
            trace_entry['events']['completions'] = list(self._recorder_completions)
            self._record_events = False
            self.last_action_info['is_delay'] = True

            # Logging
            self.action_log['delay_fixed'] += 1
            self.action_log['delay_fixed_indices'][delay_idx] += 1

        elif (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length'] + self.config_dict['max_wait_n_jobs']):
            # action corresponds to waiting until `jobs_to_finish` running jobs complete
            jobs_to_finish = action - (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) + 1
            # Record intermediate events during delay
            self._record_events = True
            self._recorder_arrivals = []
            self._recorder_completions = []
            self.delay_to_finished_job(jobs_to_finish=jobs_to_finish)
            trace_entry['action_type'] = 'delay'
            trace_entry['delay_kind'] = 'wait'
            trace_entry['delay_value'] = int(jobs_to_finish)
            trace_entry['events']['arrivals'] = list(self._recorder_arrivals)
            trace_entry['events']['completions'] = list(self._recorder_completions)
            self._record_events = False
            self.last_action_info['is_delay'] = True

            # logging
            self.action_log['delay_wait'] += 1
            wait_idx = jobs_to_finish - 1 # Convert 1-based count to 0-based index
            self.action_log['delay_wait_indices'][wait_idx] += 1

        else:
            # Should not happen if action_space is correct
            raise AssertionError("Action index out of predefined categories in env.step")

        # TODO: potential truncated logic or episode termination conditions
        
        reward = self.get_reward(scheduled_job=scheduled_job, current_timestamp=self.current_timestamp)
        obs = self.build_observation()
        info = {}

        assert obs.shape == self.observation_space.shape , "Shape mismatch between actual shape, and defined shape of observation space" 
        terminated = self.should_terminate()

        info['new_job_arrived'] = self.new_job_arrived_in_step
        info['action_is_delay'] = self.last_action_info['is_delay']

        # finalize trace entry
        trace_entry['timestamp_after'] = self.current_timestamp
        trace_entry['queue_len_after'] = len(self.job_queue)
        trace_entry['running_len_after'] = len(self.running_jobs)
        self.action_trace.append(trace_entry)
        self.step_counter += 1

        return obs, reward, terminated, truncated, info

    def should_terminate(self): 
        if self.scheduled_jobs == self.config_dict['episode_length']: 
            return True
        else:
            return False

    def reset(self, seed, options):

        if self.generate_rendering:
            self.dir_path = "renderings/" + str(self.name) + "/" + "seed_" + str(seed)
            create_directory_if_not_exists(directory_path=self.dir_path)
            
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
        time_offset = first_job.submit_time
        
        self.episode_start_hour_offset = time_offset // 3600 

        self.action_log = {
            'schedule': 0,
            'delay_fixed': 0,
            'delay_wait': 0,
            'delay_fixed_indices': [0] * self.config_dict['delay_time_list_length'],
            'delay_wait_indices': [0] * self.config_dict['max_wait_n_jobs']
        }
        self.action_trace = []
        self.step_counter = 0
        self._record_events = False
        self._recorder_arrivals = []
        self._recorder_completions = []


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
        first_job = self.loads.get_job(0)
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
            masked =  self.valid_action_mask()[queue_index]
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
        self.job_queue.remove(job)
        self.scheduled_jobs += 1 

        # logging
        if 'schedule' in self.action_log:
            self.action_log['schedule'] += 1

        # rendering
        self.scheduled_job_history.append(job)

        return job

    def delay_to_finished_job(self, jobs_to_finish):
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

        self._process_events_until(next_time_after_skip)
        if self.current_timestamp > start_delay_time:
            self.delay_history.append((start_delay_time, self.current_timestamp))


    def delay_fixed_amount(self, skip_time):
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
        self._process_events_until(next_time_after_skip)
        if self.current_timestamp > start_delay_time:
            self.delay_history.append((start_delay_time, self.current_timestamp))
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
                    arriving_job = self.loads[self.next_arriving_job_idx]
                    self.job_queue.append(arriving_job)
                    if self._record_events:
                        self._recorder_arrivals.append(arriving_job.job_id)
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
                    if self._record_events:
                        self._recorder_completions.append(finished_job.job_id)
                    if next_resource_release_machines:
                        self.cluster.release(next_resource_release_machines)
                    # Remove the completed job from running_jobs if present
                    self.running_jobs.pop(0)
                next_resource_release_time, next_resource_release_machines = self._update_next_resource_release()

    def get_action_trace(self):
        return list(self.action_trace)

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

        ## TO DO: Remove this, and figure out why we need this
        no_more_arrivals = not (self.next_arriving_job_idx < self.last_job_in_batch and self.next_arriving_job_idx < self.loads.size())
        any_schedulable = any(
            self.cluster.can_allocated(job) for job in self.job_queue[:self.config_dict['max_queue_size']]
        )
        if self.mode in ["validation", "test"] and no_more_arrivals and any_schedulable:
            # Mask all delay actions: indices from max_queue_size to end
            mask[self.config_dict['max_queue_size']:] = False
            
        return mask

    def get_reward(self,scheduled_job : Job | None, current_timestamp):
        reward = 0

        if self.reward_type == "CO2_direct":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)


                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshold, scheduled_job.run_time])

                reward = - (carbon_emission + bounded_slowdown*self.eta)

            else: 
                reward = 0
        if self.reward_type == "CO2_direct_c":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                compute_req = scheduled_job.request_number_of_nodes * scheduled_job.request_time

                normalized_carbon_emission = (carbon_emission/compute_req) * 100000
                
                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshold, scheduled_job.run_time])
                reward = - (normalized_carbon_emission + bounded_slowdown*self.eta)

            else: 
                reward = 0

        if self.reward_type == "delay_vs_now_reward":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
                
                carbon_emission_actual = self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time)

                carbon_emission_initial = self.carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                
                carbon_ratio_reward = ((carbon_emission_initial-carbon_emission_actual) +0.1)/(carbon_emission_initial + 0.1)

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshold, scheduled_job.run_time])

                print("carbon ratio reward")
                print("wait reward: bounded_slowdown: ", bounded_slowdown*self.eta)

                reward = carbon_ratio_reward - bounded_slowdown*self.eta
            else: 
                reward = 0 

        if self.reward_type == "delay_vs_now_reward_n":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
  
                compute_req = scheduled_job.request_number_of_nodes * scheduled_job.request_time

                              
                carbon_emission_actual_n = (self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time) / compute_req) * 100000 


                carbon_emission_initial_n = (self.carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                /compute_req) * 100000

                carbon_ratio_reward = ((carbon_emission_initial_n-carbon_emission_actual_n) +0.1)/(carbon_emission_initial_n + 0.1)

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshold, scheduled_job.run_time])

                reward = carbon_ratio_reward - bounded_slowdown*self.eta
            else: 
                reward = 0 
        if self.reward_type == "carbon_ratio_plus":
            if scheduled_job: 
                start_time = current_timestamp
                end_time = start_time + scheduled_job.run_time
                power_usage = scheduled_job.power_usage
  
                compute_req = scheduled_job.request_number_of_nodes * scheduled_job.request_time

                              
                carbon_emission_actual_n = (self.carbon_intensity.getCarbonEmissions(power_usage, start_time, end_time) / compute_req) * 100000 


                carbon_emission_initial_n = (self.carbon_intensity.getCarbonEmissions(power_usage, scheduled_job.submit_time, scheduled_job.submit_time+scheduled_job.run_time)
                /compute_req) * 100000

                carbon_ratio_reward = ((carbon_emission_initial_n-carbon_emission_actual_n) +0.1)/(carbon_emission_initial_n + 0.1)

                bounded_slowdown = (scheduled_job.wait_time + scheduled_job.run_time) / max([self.bounded_slowdown_threshold, scheduled_job.run_time])

                reward = self.alpha*carbon_ratio_reward - bounded_slowdown*self.eta
            else: 
                reward = 0 
            


        return reward 
