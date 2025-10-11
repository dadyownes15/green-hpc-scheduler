#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu
#SBATCH --job-name=rl_train
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=16
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=02:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

set -e

# Activate the virtual environment
source venv/bin/activate

export WANDB_API_KEY="418d10fc7ab5763a7e2ec89f2dc5aed81c38bd8e"


# Define log file (timestamped)
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"

# Run your Python sweep script and save output to log
python multi_env.py | tee "$LOGFILE"

# Deactivate virtual environment
deactivate