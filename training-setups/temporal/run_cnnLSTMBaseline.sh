#!/bin/bash
#SBATCH --job-name=cybench
#SBATCH --time=48:00:00
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cybench
export CUDA_VISIBLE_DEVICES=0

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Minimum number of years required for training
MIN_YEARS=8

# Countries to exclude (already trained)
EXCLUDED_COUNTRIES=""

# Path to years dictionary JSON file
YEARS_DICT_JSON="../configurations/years_dict.json"

# Dynamically extract crops and filtered countries
crops=($(python3 -c "
import json

with open('$YEARS_DICT_JSON', 'r') as f:
    data = json.load(f)
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

# Get all crops
for crop in data.keys():
    print(crop)
"))

# Get countries for each crop with >= MIN_YEARS
declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if [ "$crop" = "maize" ]; then
        MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then
        WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import json

with open('$YEARS_DICT_JSON', 'r') as f:
    data = json.load(f)
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

for crop in data.keys():
    for country, years in data[crop].items():
        if len(years) >= MIN_YEARS and country not in EXCLUDED:
            print(f'{crop}|{country}')
")

echo "Configuration:"
echo "MIN_YEARS: $MIN_YEARS"
echo "EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "Crops: ${crops[@]}"
echo "Maize countries with >=$MIN_YEARS years: ${!MAIZE_COUNTRIES[@]}"
echo "Wheat countries with >=$MIN_YEARS years: ${!WHEAT_COUNTRIES[@]}"

# CNN-LSTM hyperparameters
CNNLSTM_HIDDEN_SIZE=64
CNNLSTM_NUM_LAYERS=2
CNNLSTM_DROPOUT=0.1

# MAX_PARALLEL based on GPU
MAX_PARALLEL=8

# Create semaphore pipe
PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"

# Fill pipe with tokens
for i in $(seq 1 $MAX_PARALLEL); do
    echo >&3
done

mkdir -p modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results log

# Acquires a token, launches process, releases token when done
run_model() {
    local log_file=$1
    shift
    local cmd=("$@")

    # Acquire token (blocks if MAX_PARALLEL already running)
    read -u 3

    {
        "${cmd[@]}" > "$log_file" 2>&1
        # Release token when process finishes
        echo >&3
    } &
}

# Merge helper
merge_results() {
    echo "Merging results..."
    for metric in nrmse mape r2 rmse mae mse smape; do
        final_csv="modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/${metric}.csv"
        first=1
        for tmp_dir in modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/tmp_*/; do
            src="${tmp_dir}${metric}.csv"
            if [ -f "$src" ]; then
                if [ $first -eq 1 ]; then
                    cp "$src" "$final_csv"
                    first=0
                else
                    tail -n +2 "$src" >> "$final_csv"
                fi
            fi
        done
        echo "Merged $metric.csv"
    done
    rm -rf modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/tmp_*/
}

# Function to get countries for a crop
get_countries_for_crop() {
    local crop=$1
    if [ "$crop" = "maize" ]; then
        echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then
        echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

# Sort countries alphabetically for consistent ordering
sort_countries() {
    echo $1 | tr ' ' '\n' | sort | tr '\n' ' '
}

# Check if result file exists and is non-empty
is_completed() {
    local result_file="modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/${1}_${2}_${3}.txt"
    if [ -f "$result_file" ] && [ -s "$result_file" ]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# CNN-LSTM models
echo "--------------------------------------"
echo "Running CNN-LSTM models"
echo "--------------------------------------"

for crop in "${crops[@]}"; do
   countries=$(sort_countries "$(get_countries_for_crop $crop)")
   for country in $countries; do
       # Check if already completed
       if is_completed "cnnlstm" "$country" "$crop"; then
           echo "Skipping cnnlstm $country $crop (already completed)"
           continue
       fi

       tmp_dir="modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/tmp_cnnlstm_${country}_${crop}"
       mkdir -p "$tmp_dir"

       echo "Starting cnnlstm $country $crop"

       cmd=(
           python cnnLSTMBaseline.py
           --crop $crop
           --country $country
           --aggregation daily
           --batch_size 64
           --epochs 50
           --lag_years 0
           --test_years 5
           --use_cwb_feature
           --cnnlstm_hidden_size $CNNLSTM_HIDDEN_SIZE
           --cnnlstm_num_layers $CNNLSTM_NUM_LAYERS
           --cnnlstm_dropout $CNNLSTM_DROPOUT
           --wandb_project AAAI2027-redefiningCYP-mos-weekly-cnnlstm
           --save_checkpoint_dir modelCheckpoints-redefiningCYP-mos-daily/yield-cnnlstm-cybench/$country/$crop/
           --results_dir "$tmp_dir"
           --forecast_type "middle-of-season"
       )

       run_model \
           "modelCheckpoints-redefiningCYP-mos-daily/cnnlstm-results/cnnlstm_${country}_${crop}.txt" \
           "${cmd[@]}"

   done
done

# Wait for all remaining jobs
wait

# Merge all CSVs into final files
merge_results

echo "End time: $(date)"
echo "All jobs finished."
