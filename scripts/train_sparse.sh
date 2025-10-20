#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=1-6%6           
#SBATCH --cpus-per-task=16
#SBATCH --time=7:00:00

export WANDB_API_KEY="418d10fc7ab5763a7e2ec89f2dc5aed81c38bd8e"
# Define log file (timestamped)
LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"


# Define eta values
etas=(0.50)

# Select eta based on the SLURM array ID
eta=${etas[$SLURM_ARRAY_TASK_ID-1]}
config="config_file/s_opt_eta_$eta.ini"

echo "Running with eta=$eta"

# Activate your virtual environment if needed
source venv/bin/activate

# Run the Python script with the chosen eta
python train_and_eval.py --config $config --seeds 5 --total-timesteps 3000000  | tee "$LOGFILE"

# Deactivate environment
deactivate
