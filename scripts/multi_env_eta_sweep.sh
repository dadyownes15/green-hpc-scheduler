#!/bin/bash
#SBATCH --job-name=multi_env_eta_sweep
# Number of CPUs to allocate for each task
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/multi_env_eta_sweep_%A_%a.out
#SBATCH --error=logs/multi_env_eta_sweep_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "python executable not found; load your environment before submitting." >&2
  exit 127
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date --iso-8601=seconds)] SLURM job ${SLURM_JOB_ID:-N/A} (array index ${SLURM_ARRAY_TASK_ID:-N/A}) running on ${HOSTNAME:-unknown}"
echo "Working directory: ${PROJECT_ROOT}"
echo "Python binary: ${PYTHON_BIN}"
echo "Git commit: $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"

# Eta values to sweep over (index matches SLURM_ARRAY_TASK_ID - 1)
eta_values=(1 0.5 0.1 0.01 0)

index=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$index" -lt 0 ] || [ "$index" -ge "${#eta_values[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

eta="${eta_values[$index]}"
echo "Launching sweep_multi_env.py for eta=${eta}"

"${PYTHON_BIN}" sweep_multi_env.py --sweep multi_env_sweep.yaml --eta "${eta}"

echo "[$(date --iso-8601=seconds)] Finished eta=${eta}"
