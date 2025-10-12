#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=1-5%5           
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

export WANDB_API_KEY="418d10fc7ab5763a7e2ec89f2dc5aed81c38bd8e"
# Define log file (timestamped)
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"


# Define eta values
etas=(1 0 0.75 0.5 0.25 0.9 0.95 0.99)

# Select eta based on the SLURM array ID
eta=${etas[$SLURM_ARRAY_TASK_ID-1]}

echo "Running with eta=$eta"

# Activate your virtual environment if needed
source venv/bin/activate

# Run the Python script with the chosen eta
python train_and_eval.py --eta "$eta" --seeds 5 --total-timesteps 3000000  | tee "$LOGFILE"

# Deactivate environment
deactivate
