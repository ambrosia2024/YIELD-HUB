#!/bin/bash
# Full walk-forward sweep in ONE Slurm job (all countries × patchtst + xlinear).
#
# Prefer Slurm array (like run_benchmark_agml_lstm_us.sh)? Use:
#   ./generate_walk_forward_tasks.sh && sbatch --array=0-N walk_forward_array.slurm
# Or one sbatch per task: ./walk_forward_submit.sh
#
# Smoke test: sbatch walk_forward_test.slurm
# MOS sweep:  FORECAST_TYPE=middle-of-season sbatch walk_forward.sh
# EOS sweep:  FORECAST_TYPE=end-of-season sbatch walk_forward.sh  (default)
#
# This script: one sbatch → bash loops countries; up to MAX_PARALLEL Python runs
# in the background on the same GPU node. Walk-forward folds are still serial inside Python.
#
#SBATCH --job-name=walkforward_training
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michiel.kallenberg@wur.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/walk_forward_env.sh"

WANDB_PROJECT="${WANDB_PROJECT:-AAAI2027-CYP-WF}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
MIN_YEARS=8
EXCLUDED_COUNTRIES="${EXCLUDED_COUNTRIES:-}"
YEARS_JSON="${CONFIG_DIR}/years_dict.json"

WF_RUN_LABEL="${WF_RUN_LABEL:-final}"
mkdir -p "$SCRIPT_DIR/log/final"

if [ ! -f "$YEARS_JSON" ]; then
    echo "ERROR: YEARS_JSON not found at $YEARS_JSON"
    exit 1
fi
if [ ! -f "$HYPERPARAMS_JSON" ]; then
    echo "ERROR: Hyperparameters file not found at $HYPERPARAMS_JSON (FORECAST_TYPE=$FORECAST_TYPE)"
    exit 1
fi

EPOCHS="${EPOCHS:-100}"

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
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

echo "Configuration:"
echo "  MIN_YEARS: $MIN_YEARS"
echo "  EXCLUDED_COUNTRIES: ${EXCLUDED_COUNTRIES:-<none>}"
echo "  WANDB_PROJECT: $WANDB_PROJECT"
echo "  MAX_PARALLEL: $MAX_PARALLEL"
echo "  FORCE_RERUN: $FORCE_RERUN"
echo "  FORECAST_TYPE: $FORECAST_TYPE"
echo "  HYPERPARAMS_JSON: $HYPERPARAMS_JSON"
echo "  EPOCHS: $EPOCHS"
echo "  Output root: $YIELD_HUB_OUTPUT_ROOT (run label: $WF_RUN_LABEL, slug: $WF_FORECAST_SLUG)"
echo "  Maize countries: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"

get_countries_for_crop() {
    local crop=$1
    if   [ "$crop" = "maize" ]; then echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() { echo "$1" | tr ' ' '\n' | sort | tr '\n' ' '; }

already_trained() {
    local checkpoint_dir=$1
    [ "$FORCE_RERUN" = "1" ] && return 1
    find "$checkpoint_dir" -name '*.ckpt' -print -quit 2>/dev/null | grep -q .
}

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"
for _ in $(seq 1 "$MAX_PARALLEL"); do echo >&3; done

run_model() {
    local log_file=$1
    shift
    local cmd=("$@")
    read -u 3
    {
        "${cmd[@]}" > "$log_file" 2>&1
        echo >&3
    } &
}

crops=("wheat" "maize")

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop "$crop")")
    for country in $countries; do
        for model in patchtst xlinear; do

            if ! has_crop_country_data "$crop" "$country"; then
                echo "Skipping ${model} ${country} ${crop} (no data)"
                continue
            fi

            RUN_ID="${model}-${crop}-${country}-${WF_FORECAST_SLUG}-wf"
            walk_forward_output_dirs "$WF_RUN_LABEL" "$model" "$country" "$crop"

            if already_trained "$CHECKPOINT_SAVE_DIR"; then
                echo "Skipping ${model} ${country} ${crop} (checkpoints exist; FORCE_RERUN=1 to redo)"
                continue
            fi

            hps_args=()
            if ! get_hyperparameters "$model" "$crop" "$country" hps_args; then
                echo "ERROR: No HPS for ${model} ${country} ${crop} — skipping"
                continue
            fi

            if [ "$model" = "patchtst" ]; then
                script="$SCRIPT_DIR/tstBaselines.py"
            else
                script="$SCRIPT_DIR/linearBaselines.py"
            fi

            read -ra model_flags <<< "$(get_model_flags "$model" "$crop")"

            cmd=(
                run_python "$script"
                --crop "$crop"
                --country "$country"
                --model_type "$model"
                --aggregation daily
                --epochs "$EPOCHS"
                --drop_tavg
                --test_years 5
                "${model_flags[@]}"
                "${hps_args[@]}"
                --save_checkpoint_dir "$CHECKPOINT_SAVE_DIR"
                --results_dir "$RESULTS_DIR"
                --wandb_project "$WANDB_PROJECT"
                --run_id "$RUN_ID"
                --forecast_type "$FORECAST_TYPE"
            )

            log_file="log/final_${RUN_ID}.txt"
            echo "Starting walk-forward: ${model} ${country} ${crop} (run_id: ${RUN_ID})"
            if [ "$DRY_RUN" = "1" ]; then
                echo "DRY_RUN: ${cmd[*]}"
                continue
            fi

            run_model "$log_file" "${cmd[@]}"

        done
    done
done

wait
echo "Walk-forward training completed: $(date)"
