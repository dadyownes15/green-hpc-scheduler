#!/bin/bash
#SBATCH --job-name=multi_env_eta_debug
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/multi_env_eta_debug_%j.out
#SBATCH --error=logs/multi_env_eta_debug_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "python executable not found; load your environment before submitting." >&2
  exit 127
fi

ETA="${1:-0.5}"

echo "Running sweep_multi_env.py with eta=${ETA} (count=1)"
"${PYTHON_BIN}" sweep_multi_env.py --sweep multi_env_sweep.yaml --eta "${ETA}" --count 1
