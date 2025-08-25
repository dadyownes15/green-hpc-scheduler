import re
import numpy as np
import configparser
import os
import math
config = configparser.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config_file', 'config.ini')
config.read(config_path)

JOB_FEATURE = config.getint('architecture', 'job_feature')
# Power / carbon settings
MAX_WAIT_TIME = config.getint('normalization constants', 'max_wait_time')
MAX_RUN_TIME  = eval(config.get('normalization constants', 'max_run_time'))
MAX_REQUESTED_PROCESSORS = eval(config.get('normalization constants', 'max_requested_processors')) 
MAX_POWER_PR_PROCESSOR = eval(config.get('power settings', 'constant_power_per_processor')) 



class Job:
    """
    1. Job Number -- a counter field, starting from 1.
    2. Submit Time -- in seconds. The earliest time the log refers to is zero, and is usually the submittal time of the first job. The lines in the log are sorted by ascending submittal times. It makes sense for jobs to also be numbered in this order.
    3. Wait Time -- in seconds. The difference between the job's submit time and the time at which it actually began to run. Naturally, this is only relevant to real logs, not to models.
    4. Run Time -- in seconds. The wall clock time the job was running (end time minus start time).
    We decided to use ``wait time'' and ``run time'' instead of the equivalent ``start time'' and ``end time'' because they are directly attributable to the Scheduler and application, and are more suitable for models where only the run time is relevant.
    Note that when values are rounded to an integral number of seconds (as often happens in logs) a run time of 0 is possible and means the job ran for less than 0.5 seconds. On the other hand it is permissable to use floating point values for time fields.
    5. Number of Allocated Processors -- an integer. In most cases this is also the number of processors the job uses; if the job does not use all of them, we typically don't know about it.

The rest are not used in are simplified envoriemnt


    6. Average CPU Time Used -- both user and system, in seconds. This is the average over all processors of the CPU time used, and may therefore be smaller than the wall clock runtime. If a log contains the total CPU time used by all the processors, it is divided by the number of allocated processors to derive the average.
    7. Used Memory -- in kilobytes. This is again the average per processor.
    8. Requested Number of Processors.
    9. Requested Time. This can be either runtime (measured in wallclock seconds), or average CPU time per processor (also in seconds) -- the exact meaning is determined by a header comment. In many logs this field is used for the user runtime estimate (or upper bound) used in backfilling. If a log contains a request for total CPU time, it is divided by the number of requested processors.
    10. Requested Memory (again kilobytes per processor).
    11. Status 1 if the job was completed, 0 if it failed, and 5 if cancelled. If information about chekcpointing or swapping is included, other values are also possible. See usage note below. This field is meaningless for models, so would be -1.
    12. User ID -- a natural number, between one and the number of different users.
    13. Group ID -- a natural number, between one and the number of different groups. Some systems control resource usage by groups rather than by individual users.
    14. Executable (Application) Number -- a natural number, between one and the number of different applications appearing in the workload. in some logs, this might represent a script file used to run jobs rather than the executable directly; this should be noted in a header comment.
    15. Queue Number -- a natural number, between one and the number of different queues in the system. The nature of the system's queues should be explained in a header comment. This field is where batch and interactive jobs should be differentiated: we suggest the convention of denoting interactive jobs by 0.
    16. Partition Number -- a natural number, between one and the number of different partitions in the systems. The nature of the system's partitions should be explained in a header comment. For example, it is possible to use partition numbers to identify which machine in a cluster was used.
    17. Preceding Job Number -- this is the number of a previous job in the workload, such that the current job can only start after the termination of this preceding job. Together with the next field, this allows the workload to include feedback as described below.
    18. Think Time from Preceding Job -- this is the number of seconds that should elapse between the termination of the preceding job and the submittal of this one.
    """

    def __init__(self, config_dict, request_time_mean, request_time_std, processor_mean, processor_std, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0", ):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.config_dict = config_dict 

        # For normalization
        self.request_time_mean = request_time_mean 
        self.request_time_std = request_time_std 
        self.processor_mean = processor_mean
        self.processor_std = processor_std
        # No difference between run time and request time for simplification
        self.request_time = self.run_time 
        
        # No difference between requested and allocated for simplification
        self.request_number_of_processors = self.number_of_allocated_processors
        
        self.request_number_of_nodes = self.number_of_allocated_processors 
        
        # Carbon consideration index from the SWF file (if available)
        # If the file has 19 fields, the last one is the carbon consideration index
        if len(s_array) >= 19:
            self.carbon_consideration = float(s_array[18])
        else:
            # Fallback: all jobs have carbon consideration 0 (very low concern)
            # This means no carbon optimization when using original SWF files
            self.carbon_consideration = 1 
    
        self.scheduled_time = -1
        self.allocated_machines = None
        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_job_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0
        self.power_usage = -1.0

    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __hash__(self):
        return hash(self.job_id)

    
    def encode_vector(self,current_timestamp):
        power_std = math.sqrt(
                    (self.config_dict["constant_power_per_processor"]**2) * (
                            self.request_time_std**2 * self.processor_std**2 + 
                            self.processor_mean**2 * self.request_time_std**2 + 
                            self.processor_std**2 * self.request_time_mean**2
                        )
                    )

        ## source: https://stats.stackexchange.com/questions/52646/variance-of-product-of-multiple-independent-random-variables

        enconding = np.array([
            (current_timestamp - self.submit_time) / self.config_dict['max_wait_time'],
            (self.request_time - self.request_time_mean ) / self.request_time_std,
            (self.request_number_of_processors - self.processor_mean ) / self.processor_std ,  
            (self.power_usage - self.request_time_mean * self.processor_mean*self.config_dict["constant_power_per_processor"])/  power_std ,
            self.carbon_consideration
        ])

        assert not np.any(np.isnan(enconding))

        print(enconding)

        return enconding
    def __str__(self):
        return "J[" + str(self.job_id) + "]-[" + str(self.request_number_of_processors) + "]-[" + str(
            self.submit_time) + "]-[" + str(self.request_time) + "]" + str(self.scheduled_time)

