#!/bin/bash
# Run train_and_eval.py across multiple user_ci settings and seeds.

set -euo pipefail

USER_CIS=( "uniform" "two_bins" "exp_drop")
SEED_COUNT=5
TOTAL_TIMESTEPS=3000000

echo "Starting train_and_eval sweep over user_ci settings: ${USER_CIS[*]}"

source venv/bin/activate

for CI in "${USER_CIS[@]}"; do
    echo "Running user_ci=${CI} across ${SEED_COUNT} seeds"
    python train_and_eval.py \
        --user-ci "${CI}" \
        --seeds "${SEED_COUNT}" \
        --total-timesteps "${TOTAL_TIMESTEPS}" \
        --run-name "train_ci_${CI}"
done

deactivate

echo "Completed train_and_eval_ci sweep."
