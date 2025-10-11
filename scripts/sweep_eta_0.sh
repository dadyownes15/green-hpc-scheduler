#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu
#SBATCH --job-name=MyJob
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

set -e

# Activate the virtual environment
source venv/bin/activate

# Define log file (timestamped)
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"

# Run your Python sweep script and save output to log
python sweep_multi_env.py --sweep multi_env_sweep.yaml --count 100 --eta 0 | tee "$LOGFILE"

# Deactivate virtual environment
deactivate

echo "Run completed. Logs saved to $LOGFILE"