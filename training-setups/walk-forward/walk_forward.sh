#!/bin/bash
#SBATCH --job-name=walkforward_training
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cybench
export CUDA_VISIBLE_DEVICES=0

# Configurations saved under CONFIG_DIR
CONFIG_DIR="../configurations"

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Script directory: $CONFIG_DIR"

# Configuration
MIN_YEARS=8
EXCLUDED_COUNTRIES=""
YEARS_JSON="${CONFIG_DIR}/years_dict.json"
HYPERPARAMS_JSON="${CONFIG_DIR}/eosHyperparameters.json"

# Verify config files exist
if [ ! -f "$YEARS_JSON" ]; then
    echo "ERROR: YEARS_JSON not found at $YEARS_JSON"
    exit 1
fi
if [ ! -f "$HYPERPARAMS_JSON" ]; then
    echo "ERROR: eosHyperparameters.json not found at $HYPERPARAMS_JSON"
    exit 1
fi

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

# Load countries from JSON
while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
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
echo "  EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "  Maize countries: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"

#–––––––––––––––––––––––––––––––––– Helper functions –––––––––––––––––––––––––––––––––– #
#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– #

get_model_flags() {
    local model=$1
    local crop=$2
    python3 -c "
import json
with open('$HYPERPARAMS_JSON') as f:
    config = json.load(f)
flags = config['model_flags']['$model'].get('$crop', config['model_flags']['$model'].get('default', []))
print(' '.join(flags))
"
}

get_hyperparameters() {
    local model=$1
    local crop=$2
    local country=$3
    local -n out_array=$4

    local output
    output=$(python3 -c "
import json
with open('$HYPERPARAMS_JSON') as f:
    config = json.load(f)

try:
    hps = config['overrides']['$country']['$model']['$crop']
    for k, v in hps.items():
        # Convert snake_case to --flag format
        print(f'--{k}')
        print(f'{v}')
except KeyError:
    exit(1)
") || return 1

    readarray -t out_array <<< "$output"
    return 0
}

get_countries_for_crop() {
    local crop=$1
    if   [ "$crop" = "maize" ]; then echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() { echo "$1" | tr ' ' '\n' | sort | tr '\n' ' '; }

# Train models parallely.
MAX_PARALLEL=2
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

#–––––––––––––––––––––––––––––––––– Training –––––––––––––––––––––––––––––––––– #
#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#
# Use same directory for checks and saves
CHECKPOINT_DIR="modelCheckpoints/final"
mkdir -p "$CHECKPOINT_DIR" log/final

crops=("wheat" "maize")

# Helper to generate random 6-character alphanumeric string
generate_random_suffix() {
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1
}

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
            # Get hyperparameters from eosHyperparameters.json
            if get_hyperparameters "$model" "$crop" "$country" hps_args; then
                echo "Using HPS from eosHyperparameters.json for ${model} ${country} ${crop}"
            else
                echo "ERROR: No hyperparameters found in eosHyperparameters.json for ${model} ${country} ${crop} — skipping"
                continue
            fi

            echo "Starting final training: ${model} ${country} ${crop} (run_id: ${RUN_ID})"

            if [ "$model" = "patchtst" ]; then
                script="tstBaselines.py"
            else
                script="linearBaselines.py"
            fi

            # Read model flags from JSON
            read -ra model_flags <<< "$(get_model_flags "$model" "$crop")"

            cmd=(
                python "$script"
                --crop "$crop"
                --country "$country"
                --model_type "$model"
                --aggregation daily
                --epochs 2
                --drop_tavg
                --test_years 5
                "${model_flags[@]}"
                "${hps_args[@]}"
                --save_checkpoint_dir "$checkpoint_save_dir"
                --results_dir "$checkpoint_save_dir"
                --wandb_project "AAAI2027-CYP-WF"
                --run_id "$RUN_ID"
                --forecast_type "end-of-season"
            )

            run_model \
                "log/final_${RUN_ID}_${model}_${country}_${crop}.txt" \
                "${cmd[@]}"

        done
    done
done

wait
echo "Final training completed: $(date)"
