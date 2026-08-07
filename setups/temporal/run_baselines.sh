#!/bin/bash
#SBATCH --job-name=temporal_training
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=log-without/%x_%j.out
#SBATCH --error=log-without/%x_%j.err

#–––––––––––––––––––––––––––––––––– Parse Arguments –––––––––––––––––––––––––––––––––#
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
# Usage: sbatch run_baselines.sh <mos|eos> <output_dir>
#
# Arguments:
#   $1 - Configuration type: "mos" or "eos"
#   $2 - Output directory for checkpoints and results
#
# Example:
#   sbatch run_baselines.sh mos /path/to/output
#   sbatch run_baselines.sh eos /path/to/output

# Check arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <mos|eos> <output_dir>"
    echo "Example: $0 mos /path/to/output"
    exit 1
fi

CONFIG_TYPE=$1
OUTPUT_DIR=$2

# Validate CONFIG_TYPE
if [ "$CONFIG_TYPE" != "mos" ] && [ "$CONFIG_TYPE" != "eos" ]; then
    echo "Error: CONFIG_TYPE must be 'mos' or 'eos', got '$CONFIG_TYPE'"
    exit 1
fi

echo "Configuration: $CONFIG_TYPE"
echo "Output directory: $OUTPUT_DIR"

#–––––––––––––––––––––––––––––––––– Setup ––––––––––––––––––––––––––––––––––#
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cybench
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

MIN_YEARS=8
EXCLUDED_COUNTRIES=""

# Path to years_dict.json config file
YEARS_DICT_FILE="../configurations/years_dict.json"

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

# Filtering countries that have at least 8 or more years of data
while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import sys
import os
import json

# Suppress module-level debug output
sys.stdout = open(os.devnull, 'w')

# Restore stdout for actual output
sys.stdout.close()
sys.stdout = sys.__stdout__

with open('$YEARS_DICT_FILE', 'r') as f:
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
echo "  EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "  Maize countries: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"

#–––––––––––––––––––––––––––––––––– Set config-specific variables ––––––––––––––––––#
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#

if [ "$CONFIG_TYPE" = "eos" ]; then
    HPS_FILE="../configurations/eosHyperparameters.json"
    LOG_DIR="log-${CONFIG_TYPE}"
    WANDB_PROJECT="AAAI2027-redefiningCYP-HPO-eos-final"
    FORECAST_TYPE="end-of-season"
else
    HPS_FILE="../configurations/mosHyperparameters.json"
    LOG_DIR="log-${CONFIG_TYPE}"
    WANDB_PROJECT="AAAI2027-redefiningCYP-HPO-mos-final"
    FORECAST_TYPE="middle-of-season"
fi

CHECKPOINT_DIR="$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

#–––––––––––––––––––––––––––––––––– Helper functions ––––––––––––––––––––––––––––––––#
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#

get_patchtst_flags() {
    local crop=$1
    if [ "$crop" = "wheat" ]; then
        echo "--use_gdd --use_rue --use_farquhar --use_sota_features"
    elif [ "$crop" = "maize" ]; then
        echo "--use_sota_features"
    fi
}

get_xlinear_flags() {
    echo "--use_gdd --use_rue --use_farquhar --use_sota_features --include_spatial_features --use_heat_stress_days --lag_years 2 --use_revin"
}

# Load hyperparameter overrides from config file
get_override_hps() {
    local country=$1
    local model=$2
    local crop=$3
    local -n out_array=$4

    local raw_output
    raw_output=$(python3 -c "
import sys
import os
import json

# Suppress module-level debug output
sys.stdout = open(os.devnull, 'w')

# Restore stdout for actual output
sys.stdout.close()
sys.stdout = sys.__stdout__

with open('$HPS_FILE', 'r') as f:
    data = json.load(f)

try:
    hps = data['$model']['$country']['$crop']['hyperparameters']
    for k, v in hps.items():
        if isinstance(v, bool):
            if v:  # Only output flag if True
                print(f'--{k}')
        elif isinstance(v, list):
            print(f'--{k}')
            print(' '.join(map(str, v)))
        else:
            print(f'--{k}')
            print(f'{v}')
except KeyError:
    sys.exit(1)
") || return 1
    readarray -t out_array <<< "$raw_output"
    return 0
}

get_countries_for_crop() {
    local crop=$1
    if   [ "$crop" = "maize" ]; then echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() { echo "$1" | tr ' ' '\n' | sort | tr '\n' ' '; }

# Train 2 scripts in parallel
MAX_PARALLEL=5
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"
for i in $(seq 1 $MAX_PARALLEL); do echo >&3; done

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

# Helper to generate random 6-character alphanumeric string
generate_random_suffix() {
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1
}

#–––––––––––––––––––––––––––––––––– Training ––––––––––––––––––––––––––––––––––#
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#

crops=("wheat" "maize")

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop "$crop")")
    for country in $countries; do
        for model in patchtst xlinear; do

            # Generate unique run ID for this specific combination
            RANDOM_SUFFIX=$(generate_random_suffix)
            RUN_ID="${model}-${crop}-${country}-${RANDOM_SUFFIX}"

            # Check if checkpoint with this run_id already exists
            checkpoint_save_dir="${CHECKPOINT_DIR}/yield-${model}/${country}/${crop}/"
            existing_checkpoint=$(find "$checkpoint_save_dir" -name "*runid:${RUN_ID}.ckpt" 2>/dev/null | head -n 1)
            if [ -n "$existing_checkpoint" ]; then
                echo "Skipping ${model} ${country} ${crop} (run_id ${RUN_ID} already done)"
                continue
            fi

            hps_args=()
            # Load hyperparameters from config file
            if ! get_override_hps "$country" "$model" "$crop" hps_args; then
                echo "ERROR: Could not load HPS for ${model} ${country} ${crop} from ${HPS_FILE} — skipping"
                continue
            fi

            echo "Starting final training: ${model} ${country} ${crop} (run_id: ${RUN_ID})"

            if [ "$model" = "patchtst" ]; then
                script="tstBaselines.py"
                read -ra model_flags <<< "$(get_patchtst_flags "$crop")"
            else
                script="linearBaselines.py"
                read -ra model_flags <<< "$(get_xlinear_flags)"
            fi

            cmd=(
                python "$script"
                --crop "$crop"
                --country "$country"
                --model_type "$model"
                --aggregation daily
                --epochs 50
                --test_years 5
                "${model_flags[@]}"
                "${hps_args[@]}"
                --save_checkpoint_dir "$checkpoint_save_dir"
                --results_dir "$checkpoint_save_dir"
                --wandb_project "$WANDB_PROJECT"
                --run_id "$RUN_ID"
                --forecast_type "$FORECAST_TYPE"
                --loss_type mql
                --use_wfan
                --load_and_test_trained_model
            )

            run_model \
                "${LOG_DIR}/final_${RUN_ID}_${model}_${country}_${crop}.txt" \
                "${cmd[@]}"

        done
    done
done

wait
echo "Final training completed: $(date)"