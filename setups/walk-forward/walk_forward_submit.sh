#!/bin/bash
# Submit one Slurm job per (crop, country, model) — typical HPC workflow.
#
#   cd setups/walk-forward
#   DRY_RUN=1 ./walk_forward_submit.sh    # print sbatch commands only
#   ./walk_forward_submit.sh
#
# Env overrides: MIN_YEARS, EXCLUDED_COUNTRIES, WF_RUN_LABEL, SLURM_PARTITION, EPOCHS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/walk_forward_env.sh"

MIN_YEARS="${MIN_YEARS:-8}"
EXCLUDED_COUNTRIES="${EXCLUDED_COUNTRIES:-}"
WF_RUN_LABEL="${WF_RUN_LABEL:-final}"
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-100}"
TEST_YEARS="${TEST_YEARS:-5}"

YEARS_JSON="${CONFIG_DIR}/years_dict.json"
mkdir -p log

while IFS='|' read -r crop country; do
    for model in patchtst xlinear; do
        if ! has_crop_country_data "$crop" "$country"; then
            echo "Skip (no data): ${crop} ${country} ${model}"
            continue
        fi

        walk_forward_output_dirs "$WF_RUN_LABEL" "$model" "$country" "$crop"
        if [ "$FORCE_RERUN" != "1" ] && find "$CHECKPOINT_SAVE_DIR" -name '*.ckpt' -print -quit 2>/dev/null | grep -q .; then
            echo "Skip (done): ${crop} ${country} ${model}"
            continue
        fi

        hps_check=()
        if ! get_hyperparameters "$model" "$crop" "$country" hps_check; then
            echo "Skip (no HPS): ${crop} ${country} ${model}"
            continue
        fi

        export WF_CROP="$crop" WF_COUNTRY="$country" WF_MODEL="$model"
        export WF_RUN_LABEL EPOCHS TEST_YEARS WANDB_PROJECT="${WANDB_PROJECT:-AAAI2027-CYP-WF}"

        cmd=(
            sbatch
            -p "$SLURM_PARTITION"
            --gpus=1
            --job-name="wf_${crop}_${country}_${model}"
            --export=ALL
            "$SCRIPT_DIR/walk_forward_one.slurm"
        )

        if [ "$DRY_RUN" = "1" ]; then
            echo "${cmd[*]}"
        else
            "${cmd[@]}"
        fi
    done
done < <(run_python -c "
import json
with open('$YEARS_JSON') as f:
    data = json.load(f)
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()
for crop in data.keys():
    for country, years in data[crop].items():
        if len(years) >= MIN_YEARS and country not in EXCLUDED:
            print(f'{crop}|{country}')
")

echo "Submitted. Monitor: squeue -u \$USER"
