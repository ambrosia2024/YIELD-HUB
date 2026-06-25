#!/bin/bash
# One Slurm job = one (crop, country, model) walk-forward run.
#
# Submit:
#   sbatch --export=ALL,WF_CROP=wheat,WF_COUNTRY=NL,WF_MODEL=patchtst walk_forward_one.slurm
#
# Or use walk_forward_submit.sh to submit many jobs.
#
#SBATCH --job-name=wf_one
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michiel.kallenberg@wur.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

set -euo pipefail

: "${WF_CROP:?Set WF_CROP (e.g. wheat)}"
: "${WF_COUNTRY:?Set WF_COUNTRY (e.g. NL)}"
: "${WF_MODEL:?Set WF_MODEL (patchtst or xlinear)}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/walk_forward_env.sh"

WF_RUN_LABEL="${WF_RUN_LABEL:-final}"
WANDB_PROJECT="${WANDB_PROJECT:-AAAI2027-CYP-WF}"
EPOCHS="${EPOCHS:-100}"
TEST_YEARS="${TEST_YEARS:-5}"
RUN_ID="${RUN_ID:-${WF_MODEL}-${WF_CROP}-${WF_COUNTRY}-${WF_FORECAST_SLUG}-wf}"

mkdir -p log

if ! has_crop_country_data "$WF_CROP" "$WF_COUNTRY"; then
    echo "ERROR: No data for ${WF_CROP}-${WF_COUNTRY}"
    exit 1
fi

walk_forward_output_dirs "$WF_RUN_LABEL" "$WF_MODEL" "$WF_COUNTRY" "$WF_CROP"

hps_args=()
if ! get_hyperparameters "$WF_MODEL" "$WF_CROP" "$WF_COUNTRY" hps_args; then
    echo "ERROR: No HPS in $(basename "$HYPERPARAMS_JSON") for ${WF_MODEL} ${WF_COUNTRY} ${WF_CROP}"
    exit 1
fi

if [ "$WF_MODEL" = "patchtst" ]; then
    script="$SCRIPT_DIR/tstBaselines.py"
    read -ra model_flags <<< "$(get_model_flags "$WF_MODEL" "$WF_CROP")"
else
    script="$SCRIPT_DIR/linearBaselines.py"
    read -ra model_flags <<< "$(get_model_flags "$WF_MODEL" "$WF_CROP")"
fi

echo "Job: ${WF_CROP}-${WF_COUNTRY} ${WF_MODEL} forecast=${FORECAST_TYPE}"
echo "Checkpoints: $CHECKPOINT_SAVE_DIR"
echo "Results: $RESULTS_DIR"

run_python "$script" \
    --crop "$WF_CROP" \
    --country "$WF_COUNTRY" \
    --model_type "$WF_MODEL" \
    --aggregation daily \
    --epochs "$EPOCHS" \
    --drop_tavg \
    --test_years "$TEST_YEARS" \
    "${model_flags[@]}" \
    "${hps_args[@]}" \
    --save_checkpoint_dir "$CHECKPOINT_SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id "$RUN_ID" \
    --forecast_type "$FORECAST_TYPE"

echo "Done: $(date)"
