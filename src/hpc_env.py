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

    def __init__(self, workload_path, config : configparser.ConfigParser, debug=False, generate_rendering = False, name = None,):
        self.debug = debug

        self.config_dict = get_config_as_dict(config)
        assert self.config_dict is not None, "Config dict, did not parse"

        assert generate_rendering == False or (generate_rendering == True and name != None), "You must name the env, to be able to generate renderings" 

        self.name = name
        self.generate_rendering = generate_rendering


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

                # Load workloads and cluster
        self.loads = Workloads(workload_path, config_dict=self.config_dict)
        self.cluster = Cluster(self.loads.max_nodes, self.config_dict['procs_per_node'], self.config_dict['idle_power'])
        self.reward = Reward()
        self.carbon_intensity = CarbonIntensity(year=self.config_dict['carbon_year'], green_win_length=self.config_dict['green_forecast_length'])

        # For visualization
        self.total_processors = self.loads.max_procs
        self.scheduled_job_history = []
        self.delay_history = []
        self.new_job_arrived_in_step = False
        self.last_action_info: Dict[str, Any] = {'type': None, 'is_delay': False}


        # Episode-specific carbon-timeline offset (in hours)
        self.episode_start_hour_offset = 0

    def step(self, action):
        self.new_job_arrived_in_step = False
        self.last_action_info['type'] = action
        # Initialize default outputs
        scheduled_job = None
        terminated = False
        truncated = False

        # Schedule job [0 ... max_queue_size-1]
        # Delay fixed amount [max_queue_size ... max_queue_size + delay_time_list_length - 1]
        # Wait until N jobs are finished [max_queue_size + delay_time_list_length ... max_queue_size + delay_time_list_length + max_wait_n_jobs - 1]

        if 0 <= action < self.config_dict['max_queue_size']:
            scheduled_job = self.schedule_job(action)

        elif self.config_dict['max_queue_size'] <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']):
            delay_idx = action - self.config_dict['max_queue_size']
            skip_time = self.config_dict['delay_time_list'][delay_idx]
            self.delay_fixed_amount(skip_time=skip_time)

        elif (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) <= action < (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length'] + self.config_dict['max_wait_n_jobs']):
            # action corresponds to waiting until `jobs_to_finish` running jobs complete
            jobs_to_finish = action - (self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length']) + 1
            self.delay_to_finished_job(jobs_to_finish=jobs_to_finish)

        else:
            # Should not happen if action_space is correct
            raise AssertionError("Action index out of predefined categories in env.step")

        # TODO: potential truncated logic or episode termination conditions
        
        reward = self.reward.get_reward(scheduled_job=scheduled_job, carbon_intensity=self.carbon_intensity, current_timestamp=self.current_timestamp)
        obs = self.build_observation()
        info = {}

        terminated = self.should_terminate()

        info['new_job_arrived'] = self.new_job_arrived_in_step
        info['action_is_delay'] = self.last_action_info['is_delay']
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

        #random_offset = random.randint(0, 8760)
        random_offset = 0
        self.episode_start_hour_offset = random_offset

        if self.config_dict['variable_carbon_intensities'] == True: 
            self.carbon_intensity.reset(start_offset=random_offset)
        else:
            self.carbon_intensity.reset(start_offset=0)


        self.start_job_offset = random.randint(0, max(0, (self.loads.size() - self.config_dict['episode_length'] - 1)))

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

        for job in self.job_queue:
            job_vector = job.encode_vector(self.current_timestamp)
            queue_vector.append(job_vector)

        # Fill in with empty jobs to maintain size
        for _ in range(len(self.job_queue), self.config_dict['max_queue_size']):
            empty_job_encoding = np.zeros((self.config_dict['job_feature'],), dtype=np.float32)
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

        # Concatenate everything into a single observation vector
        obs = np.concatenate((queue_flat, running_flat, carbon_vector)).astype(np.float32)

        return obs

    def schedule_job(self, queue_index):
        # Guard index
        if queue_index < 0 or queue_index >= len(self.job_queue):
            # Check if the the queue index was block
            print(self.valid_action_mask())
            masked =  self.valid_action_mask()[queue_index]
            print("mask at queue index: ", masked)
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
        if self.current_timestamp > start_delay_time:
            self.delay_history.append((start_delay_time, self.current_timestamp))


    def delay_fixed_amount(self, skip_time):
        """
        Advances the simulation time by a fixed amount, while processing intermediate events.
        """
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
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
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
        for i in range(queue_length, self.config_dict['max_queue_size']):
            mask[i] = False

        # Remove jobs that cannot fit
        for idx, job in enumerate(self.job_queue):
            if not self.cluster.can_allocated(job):
                mask[idx] = False
        
        # Remove options to wait n jobs to complete if there are fewer than n running jobs
        if len(self.running_jobs) < self.config_dict['max_wait_n_jobs']:
            for i in range(len(self.running_jobs), self.config_dict['max_wait_n_jobs']):
                mask[self.config_dict['max_queue_size'] + self.config_dict['delay_time_list_length'] + i] = False
            
        return mask

    def render(self, step_count, window_hours=12, show_median_forecast=True):
        if not self.generate_rendering:
            # print("Rendering is disabled for this environment instance.")
            return

        save_path = os.path.join(self.dir_path, f"{str(step_count).zfill(4)}.png")
        
        # Use a layout that reserves space on the left for the job queue
        fig = plt.figure(figsize=(20, 8), constrained_layout=True)
        gs = fig.add_gridspec(1, 5)
        ax_queue = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1:])
        ax2 = ax1.twinx()

        # --- 1. Draw the Job Queue Panel ---
        ax_queue.set_title("Job Queue")
        ax_queue.set_xlim(0, 1)
        ax_queue.set_ylim(0, self.config_dict['max_queue_size']*2)
        ax_queue.set_xticks([])
        ax_queue.set_yticks([])
        
        for i, job in enumerate(self.job_queue):
            y_pos = 2*(self.config_dict['max_queue_size'] - i - 1)
            wait_time = self.current_timestamp - job.submit_time
            
            # Color from green to red based on wait time
            norm_wait = min(wait_time / self.config_dict['max_wait_time'], 1.0)
            color = (norm_wait, 1 - norm_wait, 0) # (R, G, B)
            
            # Draw job box
            rect = patches.Rectangle((0, y_pos), 1, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
            ax_queue.add_patch(rect)
            
            # Add text
            job_text = (
                f"Wait: {int(wait_time // 60)}m\n"
                f"Procs: {job.request_number_of_processors}\n"
                f"Runtime: {int(job.run_time / 60)}m"
            )
            ax_queue.text(0.5, y_pos + 0.5, job_text, ha='center', va='center', fontsize=9, color='black')

        # --- 2. Draw the Main Timeline Plot ---
        now = self.current_timestamp
        time_window_seconds = window_hours * 3600
        start_time = now - time_window_seconds
        end_time = now + time_window_seconds

        # Plot Corrected System Utilization History
        if self.total_processors > 0:
            events = []
            # Use the permanent history for a correct representation of the past
            for job in self.scheduled_job_history:
                events.append((job.scheduled_time, job.request_number_of_processors))
                events.append((job.scheduled_time + job.run_time, -job.request_number_of_processors))
            events.sort()

            if events:
                # Filter events to be within our drawing window for efficiency
                visible_events = [e for e in events if e[0] > start_time or (e[0] < start_time and e[1] > 0)]
                
                util_times = [start_time]
                initial_procs = sum(j.request_number_of_processors for j in self.scheduled_job_history if j.scheduled_time < start_time and (j.scheduled_time + j.run_time) > start_time)
                current_procs = initial_procs
                util_values = [(current_procs / self.total_processors) * 100]

                for t, proc_change in visible_events:
                    if t > start_time and t < now:
                        util_times.append(t)
                        util_values.append((current_procs / self.total_processors) * 100)
                        current_procs += proc_change
                        util_times.append(t)
                        util_values.append((current_procs / self.total_processors) * 100)
                
                util_times.append(now)
                util_values.append((current_procs / self.total_processors) * 100)
                ax1.fill_between(util_times, util_values, step='post', color='lightgreen', alpha=0.7, label='System Utilization %')

        # Plot Carbon Intensity
        carbon_times, carbon_values = [], []
        start_hour = int(start_time // 3600)
        num_hours_to_plot = (window_hours * 2) + 1
        
        for h_offset in range(num_hours_to_plot):
            abs_hour = start_hour + h_offset
            timestamp = abs_hour * 3600
            hour_index = (self.episode_start_hour_offset + abs_hour) % len(self.carbon_intensity.carbonIntensityList)
            ci_value = self.carbon_intensity.carbonIntensityList[hour_index]
            carbon_times.append(timestamp)
            carbon_values.append(ci_value)
        ax2.plot(carbon_times, carbon_values, color='seagreen', label='Carbon Intensity')

        # Visualize Median Forecast
        if show_median_forecast:
            future_ci_values = [val for ts, val in zip(carbon_times, carbon_values) if ts >= now]
            if future_ci_values:
                median_ci = np.median(future_ci_values)
                # xmin=0.5 means start plotting the line halfway across the x-axis (at "now")
                ax2.axhline(y=median_ci, xmin=0.5, xmax=1.0, color='darkorange', linestyle='--', label=f'Forecast Median ({median_ci:.1f})')

        # Visualize Skips/Delays
        for delay_start, delay_end in self.delay_history:
            if delay_end > start_time and delay_start < now:
                rect = patches.Rectangle((delay_start, 0), delay_end - delay_start, 100, linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.2, zorder=10)
                ax1.add_patch(rect)
        
        # --- Formatting the Plot ---
        ax1.axvline(x=now, color='r', linestyle='--', linewidth=2, label='Now')
        ax1.set_xlim(start_time, end_time)
        ax1.set_xticks([start_time, now, end_time])
        ax1.set_xticklabels([f'{window_hours} hours ago', 'Now', f'{window_hours} hours in the future'])
        ax1.set_xlabel('Timeline')
        ax1.set_ylim(0, 101)
        ax1.set_ylabel('System Utilization (%)', color='green')
        ax2.set_ylabel('Carbon Intensity (gCO2/kWh)', color='seagreen')
        fig.suptitle('HPC Scheduler State', fontsize=16)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.savefig(save_path, dpi=150)
        plt.close(fig)