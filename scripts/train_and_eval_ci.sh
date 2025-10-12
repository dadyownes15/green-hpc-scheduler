#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=1-5%5           
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
set -euo pipefail

USER_CIS=( "uniform" "two_bins" "exp_drop")
SEED_COUNT=5
TOTAL_TIMESTEPS=3000000

echo "Starting train_and_eval sweep over user_ci settings: ${USER_CIS[*]}"

source venv/bin/activate

for CI in "${USER_CIS[@]}"; do
    for SEED in $(seq 1 "${SEED_COUNT}"); do
        echo "Running user_ci=${CI}, seed=${SEED}"
        python train_and_eval.py \
            --user-ci "${CI}" \
            --seed "${SEED}" \
            --seeds 1 \
            --total-timesteps "${TOTAL_TIMESTEPS}" \
            --run-name "train_ci_${CI}_seed${SEED}"
    done
done

deactivate

echo "Completed train_and_eval_ci sweep."
