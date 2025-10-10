#!/bin/bash
#SBATCH --job-name=multi_env_eta_sweep
# We start 5 tasks numbered 1-5 but only 3 can run in parallel
#SBATCH --array=1-5%3
# Number of CPUs to allocate for each task
#SBATCH --cpus-per-task=4
# Max run time is 24 hours
#SBATCH --time=5:00:00

set -euo pipefail

# Eta values to sweep over (index matches SLURM_ARRAY_TASK_ID - 1)
eta_values=(1 0.5 0.1 0.01 0)

index=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$index" -lt 0 ] || [ "$index" -ge "${#eta_values[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

eta="${eta_values[$index]}"
echo "Launching sweep_multi_env.py for eta=${eta}"

python sweep_multi_env.py --sweep multi_env_sweep.yaml --eta "${eta}"
