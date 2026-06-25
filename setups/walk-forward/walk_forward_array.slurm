#!/bin/bash
#
# Slurm array: one job index = one (crop, country, model) walk-forward run.
# Same pattern as run_benchmark_agml_lstm_us.sh (crop_country_list + --array).
#
#   ./generate_walk_forward_tasks.sh   # US/BR → one task per fold; others → all folds
#   sbatch --array=0-<N> walk_forward_array.slurm
#   FORECAST_TYPE=middle-of-season sbatch --array=0-<N> walk_forward_array.slurm
#   SPLIT_FOLDS_COUNTRIES="" ./generate_walk_forward_tasks.sh   # disable per-fold split
#
#SBATCH --job-name=wf_array
#SBATCH --output=log/wf_%A_%a.out
#SBATCH --error=log/wf_%A_%a.err
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
##SBATCH --mail-user=michiel.kallenberg@wur.nl
##SBATCH --mail-type=END,FAIL
##SBATCH --array=0-0

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/walk_forward_env.sh"

TASK_LIST="${TASK_LIST:-$SCRIPT_DIR/walk_forward_tasks.txt}"
WF_RUN_LABEL="${WF_RUN_LABEL:-final}"
WANDB_PROJECT="${WANDB_PROJECT:-AAAI2027-CYP-WF}"
EPOCHS="${EPOCHS:-100}"
TEST_YEARS="${TEST_YEARS:-5}"

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: $TASK_LIST not found. Run: ./generate_walk_forward_tasks.sh"
    exit 1
fi

# Array task 0 = first data line (same as run_benchmark_agml_lstm_us.sh)
line=$(grep -v '^#' "$TASK_LIST" | awk "NR == ${SLURM_ARRAY_TASK_ID:-0} + 1")

if [ -z "${line:-}" ]; then
    echo "ERROR: No task for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0} in $TASK_LIST"
    exit 1
fi

read -r WF_CROP WF_COUNTRY WF_MODEL WF_FOLD_EXTRA <<< "$line"
unset WF_FOLD_IDX
FOLD_ARGS=()
if [ -n "${WF_FOLD_EXTRA:-}" ] && [[ "$WF_FOLD_EXTRA" =~ ^[0-9]+$ ]]; then
    WF_FOLD_IDX="$WF_FOLD_EXTRA"
    export WF_FOLD_IDX
    RUN_ID="${WF_MODEL}-${WF_CROP}-${WF_COUNTRY}-${WF_FORECAST_SLUG}-wf-f${WF_FOLD_IDX}"
    FOLD_ARGS=(--wf_fold_idx "$WF_FOLD_IDX")
else
    RUN_ID="${WF_MODEL}-${WF_CROP}-${WF_COUNTRY}-${WF_FORECAST_SLUG}-wf"
fi

echo "Array task ${SLURM_ARRAY_TASK_ID:-0}: ${WF_CROP} ${WF_COUNTRY} ${WF_MODEL} ${WF_FOLD_IDX:-all-folds} forecast=${FORECAST_TYPE}"
echo "Node: $(hostname)  Start: $(date)"

if ! has_crop_country_data "$WF_CROP" "$WF_COUNTRY"; then
    echo "ERROR: No data for ${WF_CROP}-${WF_COUNTRY}"
    exit 1
fi

walk_forward_output_dirs "$WF_RUN_LABEL" "$WF_MODEL" "$WF_COUNTRY" "$WF_CROP"

hps_args=()
if ! get_hyperparameters "$WF_MODEL" "$WF_CROP" "$WF_COUNTRY" hps_args; then
    echo "ERROR: No HPS for ${WF_MODEL} ${WF_COUNTRY} ${WF_CROP}"
    exit 1
fi

if [ "$WF_MODEL" = "patchtst" ]; then
    script="$SCRIPT_DIR/tstBaselines.py"
else
    script="$SCRIPT_DIR/linearBaselines.py"
fi
read -ra model_flags <<< "$(get_model_flags "$WF_MODEL" "$WF_CROP")"

run_python "$script" \
    --crop "$WF_CROP" \
    --country "$WF_COUNTRY" \
    --model_type "$WF_MODEL" \
    --aggregation daily \
    --epochs "$EPOCHS" \
    --drop_tavg \
    --test_years "$TEST_YEARS" \
    "${FOLD_ARGS[@]}" \
    "${model_flags[@]}" \
    "${hps_args[@]}" \
    --save_checkpoint_dir "$CHECKPOINT_SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id "$RUN_ID" \
    --forecast_type "$FORECAST_TYPE"

echo "Done ${WF_CROP}-${WF_COUNTRY} ${WF_MODEL} (${FORECAST_TYPE}): $(date)"
echo "CSVs: $RESULTS_DIR"
echo "Checkpoints: $CHECKPOINT_SAVE_DIR"
