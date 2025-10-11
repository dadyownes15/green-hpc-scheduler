#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=1-5%5           # 5 tasks total, 3 can run in parallel
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00

# Define eta values
etas=(1 0 0.5 0.1 0.01)

# Select eta based on the SLURM array ID
eta=${etas[$SLURM_ARRAY_TASK_ID-1]}

echo "Running with eta=$eta"

# Activate your virtual environment if needed
source venv/bin/activate

# Run the Python script with the chosen eta
python multi_env_train.py --seeds 5 --eta "$eta"

# Deactivate environment
deactivate
