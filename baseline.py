import numpy as np
import time
import random
from workloads import Workloads
from carbon_intensity import CarbonIntensity
from job import Job
# --- System Configuration Constants ---
# These were missing and caused errors.
TOTAL_NODES = 256 
PROCESSOR_PER_MACHINE = 1
POWER_PR_PROCESSOR = 15 # Watts per processor



class Baseline():
    def __init__(self, workload_path, year, start_offset = 0, start_job_id = None):
        # Start_offset is for carbon intensity
        self.loads = Workloads(workload_path)
        self.carbon_intensity = CarbonIntensity(year=year, start_offset = start_offset)
        self.reserved_list = None
        self.start_offset = start_offset
        self.total_carbon_emissions = 0
        self.start_job_id = start_job_id
            
    def load_jobs(self, n_jobs):
        if self.start_job_id == None:
            job_offset = random.randint(0,len(self.loads.all_jobs)-n_jobs)
        else:
            job_offset = self.start_job_id

        jobs = self.loads.all_jobs[job_offset:job_offset+n_jobs]
        
        ## Normalize submit time
        first_submit_time = jobs[0].submit_time        
        for job in jobs:
            job.submit_time = job.submit_time - first_submit_time
        
        assert(jobs[0].submit_time == 0)

        return jobs

    def minimal_carbon_calc(self, end_hour, n_jobs=100, heuristic = "biggest_first"):
        """
        Calculates the minimal carbon schedule for a given number of jobs
        within a time window.

        possible heuristics : "biggest_first", "smallest_first", "arrival_first"
        """
        start_time = time.time() # Start the timer

        # The scheduling period is in hours.
        period_hours = end_hour - self.start_offset 
        
        # This was causing a NameError.
        assert (PROCESSOR_PER_MACHINE == 1), "Processor per machine different than 1 is not implemented"

        # Initialize an array to track node usage for each hour in the period.
        self.reserved_list = np.zeros(period_hours)
        
        jobs = self.load_jobs(n_jobs = n_jobs) 

        # Sort jobs by their power consumption (smallest first).

        
        sorted_jobs = sorted(jobs, key=lambda job: job.request_number_of_processors * POWER_PR_PROCESSOR)


        
        assert(sorted_jobs[-1].submit_time < end_hour*3600), "Last job is scheduled after end_hour"
        
        # Consider only the first n_jobs.
        jobs_to_schedule = sorted_jobs[:n_jobs]
        scheduled_count = 0

        for job in jobs_to_schedule:
            min_carbon_for_job = float('inf')
            best_start_time = None
            
            power_usage = job.request_number_of_processors * POWER_PR_PROCESSOR

            for t in range(period_hours - job.run_time + 1):
                if self._can_allocate(job, t):
                    # Calculate carbon emission for this potential time slot.
                    current_carbon = self.carbon_intensity.getCarbonEmissions(
                        power_usage, t, t + job.run_time
                    )
                    
                    if current_carbon < min_carbon_for_job:
                        min_carbon_for_job = current_carbon
                        best_start_time = t
            
            if best_start_time is not None:
                self._allocate(job, best_start_time)
                self.total_carbon_emissions += min_carbon_for_job
                scheduled_count += 1
                # print(f"Scheduled {job} at hour {best_start_time}.") # Optional: for debugging
            else:
                print("DId not schedule  job ", {job.job_id})
        
        end_time = time.time() # Stop the timer
        execution_speed = end_time - start_time
        
        # --- Print Results ---
        print("\n--- Scheduling Complete ---")
        print(f"Scheduled {scheduled_count} out of {n_jobs} jobs.")
        print(f"Total Carbon Emissions: {self.total_carbon_emissions:.2f} gCO2eq")
        print(f"Execution Speed: {execution_speed:.4f} seconds")
        print("---------------------------\n")


    def _can_allocate(self, job: Job, t: int):
        """
        Checks if a job can be allocated at a given start time `t`.
        """
        if job.submit_time < t * 3600:
            # print("Cannot submit job with job id ", job.job_id)
            return False
        
        for i in range(job.run_time):
            if self.reserved_list[t + i] + job.request_number_of_nodes > TOTAL_NODES:
                return False # Not enough nodes available
        return True # Allocation is possible


    def _allocate(self, job: Job, reserve_time: int):
        """
        Reserves the nodes for a job at the specified time.
        """
        for i in range(job.run_time):
            self.reserved_list[i + reserve_time] += job.request_number_of_nodes
