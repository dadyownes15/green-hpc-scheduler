import sys
import os
import csv
import configparser
import copy
from src.job import Job
from typing import Dict
import re
import numpy as np


class Workloads:
    def __init__(self, path, config_dict : Dict):
        # Loaded jobs are the array of the workfile
        self.loaded_jobs = []
        # Episode jobs are the jobs that are used in the current episode
        self.episode_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0
        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0
        self.max_power = 0
        self.min_submitTime = -sys.maxsize - 1
        self.offset = None
        # Load configuration for power settings
        self.config_dict = config_dict

       #TO DO: Optimize this, to avoid multiple loops. The aggregate calculates are nesacarry for normalization
        with open(path) as fp:
            processor_list = []  # Use standard lists for appending
            run_time_list = []   # This is more efficient for building the data

            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue
                line = line.strip()
                s_array = re.split("\\s+", line)
                
                # Append to standard Python lists
                processor_list.append(int(s_array[3]))
                run_time_list.append(int(s_array[4]))

        # Convert to numpy arrays outside the loop after all data is collected
        processor_list = np.array(processor_list)
        run_time_list = np.array(run_time_list)

        self.run_time_mean = np.mean(run_time_list)
        self.run_time_std = np.std(run_time_list)

        self.processor_mean = np.mean(processor_list)
        self.processor_std = np.mean(processor_list)

        print("run time mean: ", self.run_time_mean)
        print("run time std: ", self.run_time_std)
        with open(path) as fp:
            for line in fp:

                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                j = Job(self.config_dict, self.run_time_mean, self.run_time_std, self.processor_mean, self.processor_std, line)
                
                # Set power based on configuration
                j.power_usage = self.config_dict['constant_power_per_processor'] * j.request_number_of_processors

                # Default schedueld time to -1 
                j.scheduled_time = -1
                if self.min_submitTime == -sys.maxsize - 1:
                    self.min_submitTime = j.submit_time
                j.submit_time = j.submit_time - self.min_submitTime
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.loaded_jobs.append(j)

                    if j.request_number_of_processors > self.max:
                        self.max = j.request_number_of_processors

        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print("Max Allocated Processors:", str(self.max), ";max node:", self.max_nodes,
              ";max procs:", self.max_procs,
              ";max execution time:", self.max_exec_time)

        self.loaded_jobs.sort(key=lambda job: job.job_id)


    def size(self):
        return len(self.loaded_jobs)

    def reset(self, start_job_offset):
        self.episode_jobs = []
        start_submit = self.loaded_jobs[start_job_offset].submit_time
        assert start_submit >= 0

        for idx in range(start_job_offset, start_job_offset + self.config_dict['episode_length']):
            job = self.loaded_jobs[idx]
            job_copy = copy.deepcopy(job)

            # Ensure that submit times are relative to the first job
            job_copy.submit_time -= start_submit
            assert job_copy.submit_time >= 0
            assert job_copy.scheduled_time == -1

            self.episode_jobs.append(job_copy)

        assert len(self.episode_jobs) == self.config_dict['episode_length']
            
    def get_job(self,idx) -> Job:
        return self.episode_jobs[idx]
    
    def __getitem__(self, item):
        return self.episode_jobs[item]

    def read_csv_to_list(self, filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data_list = [row[0] for row in reader]
        return data_list

