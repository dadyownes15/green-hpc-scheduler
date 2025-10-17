#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=eta_sweep
#SBATCH --array=1-3%3 
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=16
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-24:00:00


set -e

# Activate the virtual environment
source venv/bin/activate


# Define eta values
etas=(0.001 0.0001 0.00001)

# Select eta based on the SLURM array ID
eta=${etas[$SLURM_ARRAY_TASK_ID-1]}

export WANDB_API_KEY="418d10fc7ab5763a7e2ec89f2dc5aed81c38bd8e"


# Define log file (timestamped)
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"

# Run your Python sweep script and save output to log
python sweep_multi_env.py --sweep sweep_config.yaml --count 50 --eta "$eta" | tee "$LOGFILE"

# Deactivate virtual environment
deactivate