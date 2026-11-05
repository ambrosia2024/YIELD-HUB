#!/bin/bash
#SBATCH --job-name=arch_benchmark
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/setups/temporal/slurm-logs/arch_bench_%j.out
#SBATCH --error=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/setups/temporal/slurm-logs/arch_bench_%j.err

#====================================================================================#
#  BASIC ARCHITECTURE BENCHMARK (B0 — raw features only)
#  ------------------------------------------------------------------------------
#  Goal: under the most basic setup, compare which architecture family is best.
#  For wheat and maize, every country with MORE THAN 10 YEARS of data Okay, d(total years
#  in years_dict.json), train:
#     - cnnLSTMBaseline.py                 (1 model)
#     - tstBaselines.py   x 8 model_types  (autoformer patchtst tsmixer informer
#                                           tst itransformer timexer timesnet)
#     - linearBaselines.py x 5 model_types (nlinear dlinear xlinear rlinear olinear)
#  => 14 runs per (crop, country).
#
#  Fixed settings (NO engineered features — raw dataset + CWB only):
#     aggregation=daily, use_cwb_feature ON, loss_type=mql, epochs=20,
#     test_years=5, lag_years=0, evaluation_tier=accuracy, forecast=end-of-season.
#  Explicitly OFF: use_sota_features, include_spatial_features, use_recursive_lags,
#     use_revin, use_gdd, use_heat_stress_days, use_rue, use_rtai,
#     use_residual_trend, use_exponential_weighting, multi_year_summaries,
#     if_tokenize, use_wfan. (All are store_true defaults-off, so simply not passed;
#     lag_years is forced to 0 because its script default is 1.)
#
#  Usage:   sbatch benchmark_architectures.sh <output_dir>
#  Example: sbatch benchmark_architectures.sh checkpoints-benchmark
#====================================================================================#

if [ $# -lt 1 ]; then
    echo "Error: missing output directory"
    echo "Usage: $0 <output_dir>"
    exit 1
fi

OUTPUT_DIR=$1

#–––––––––––––––––––––––––––––––––– Setup ––––––––––––––––––––––––––––––––––#
# Hardcoded absolute script dir (NEVER use BASH_SOURCE in a SLURM job — the script
# is copied to the spool dir before execution). cd into it so the scripts'
# relative imports (../../process, ../../architectures) resolve.
SCRIPT_DIR="/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/setups/temporal"
cd "$SCRIPT_DIR" || { echo "Cannot cd to $SCRIPT_DIR"; exit 1; }

# conda activate is unreliable on these nodes — call the env interpreter directly.
PYTHON="/home/vsaxena1/miniconda3/envs/cybench/bin/python"
export CUDA_VISIBLE_DEVICES=0

YEARS_DICT_FILE="$SCRIPT_DIR/../configurations/years_dict.json"
WANDB_PROJECT="CYBENCH-resolution-benchmark"
FORECAST_TYPE="end-of-season"
EPOCHS=20
TEST_YEARS=5
MIN_TOTAL_YEARS=10        # "more than 9 years of data" => total years >= 10

LOG_DIR="$SCRIPT_DIR/log-benchmark"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$SCRIPT_DIR/slurm-logs"

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR | wandb: $WANDB_PROJECT | forecast: $FORECAST_TYPE"

# Resolutions to sweep
# RESOLUTIONS=(daily weekly monthly dekad)
RESOLUTIONS=(weekly monthly dekad)

# Models to sweep per script
TST_MODELS=(tsmixer)
# LINEAR_MODELS=(xlinear)

#–––––––––––––––––––––– Country selection (total years > 10) ––––––––––––––––––––#
declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
done < <("$PYTHON" -c "
import json
with open('$YEARS_DICT_FILE') as f:
    data = json.load(f)
MIN_TOTAL = $MIN_TOTAL_YEARS
for crop in ('maize', 'wheat'):
    for country, years in data.get(crop, {}).items():
        if len(years) >= MIN_TOTAL:   # total years > 10
            print(f'{crop}|{country}')
")

echo "  Maize countries ($(echo ${#MAIZE_COUNTRIES[@]})): $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries ($(echo ${#WHEAT_COUNTRIES[@]})): $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"

get_countries_for_crop() {
    local crop=$1
    if   [ "$crop" = "maize" ]; then echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then echo "${!WHEAT_COUNTRIES[@]}"
    fi
}
sort_countries() { echo "$1" | tr ' ' '\n' | sort | tr '\n' ' '; }
generate_random_suffix() {
    printf '%06x' "$(( $(od -An -N3 -tu4 /dev/urandom | tr -d ' ') & 16777215 ))"
}
#–––––––––––––––––––––––––– Parallel execution pool ––––––––––––––––––––––––––#
MAX_PARALLEL=5
PIPE=$(mktemp -u); mkfifo "$PIPE"; exec 3<>"$PIPE"; rm "$PIPE"
for i in $(seq 1 $MAX_PARALLEL); do echo >&3; done

run_model() {
    local log_file=$1; shift
    local cmd=("$@")
    read -u 3
    {
        "${cmd[@]}" > "$log_file" 2>&1
        echo >&3
    } &
}

#–––––––––––––––––––––––––––– One training run ––––––––––––––––––––––––––––––#
# run_one <script> <crop> <country> <model_type_or_empty> <label> <resolution>
run_one() {
    local script=$1 crop=$2 country=$3 model_type=$4 label=$5 resolution=$6

    # Per-run dir includes resolution so daily/weekly/monthly never collide.
    local out_dir="${OUTPUT_DIR}/${resolution}/${label}/${crop}/${country}"

    # Skip if a checkpoint already exists for this exact cell.
    if find "$out_dir/checkpoints" -name "*.ckpt" 2>/dev/null | grep -q .; then
        echo "Skipping ${resolution} ${label} ${crop} ${country} (checkpoint exists)"
        return
    fi
    mkdir -p "$out_dir/checkpoints" "$out_dir/results"

    local suffix; suffix=$(generate_random_suffix)
    local run_id="${resolution}-${label}-${crop}-${country}-${suffix}"

    local cmd=(
        "$PYTHON" "$SCRIPT_DIR/${script}"
        --crop "$crop"
        --country "$country"
        --aggregation "$resolution"
        --epochs "$EPOCHS"
        --test_years "$TEST_YEARS"
        --lag_years 0
        --use_cwb_feature
        --loss_type mql
        --evaluation_tier accuracy
        --forecast_type "$FORECAST_TYPE"
        --save_checkpoint_dir "$out_dir/checkpoints"
        --results_dir "$out_dir/results"
        --wandb_project "$WANDB_PROJECT"
        --run_id "$run_id"
    )
    # cnnLSTM has no --model_type (it is a fixed architecture).
    if [ -n "$model_type" ]; then
        cmd+=(--model_type "$model_type")
    fi

    echo "Starting: ${resolution} ${label} ${crop} ${country} (run_id: ${run_id})"
    run_model "${LOG_DIR}/${run_id}.txt" "${cmd[@]}"
}

#–––––––––––––––––––––––––––––– Training sweep ––––––––––––––––––––––––––––––#
crops=("wheat" "maize")

for resolution in "${RESOLUTIONS[@]}"; do
    echo "=== Resolution: ${resolution} ==="
    for crop in "${crops[@]}"; do
        countries=$(sort_countries "$(get_countries_for_crop "$crop")")
        for country in $countries; do

            # Transformer family
            for m in "${TST_MODELS[@]}"; do
                run_one "tstBaselines.py" "$crop" "$country" "$m" "$m" "$resolution"
            done

            # Linear family
            # for m in "${LINEAR_MODELS[@]}"; do
            #    run_one "linearBaselines.py" "$crop" "$country" "$m" "$m" "$resolution"
            # done

        done
    done
done

wait
echo "Benchmark completed: $(date)"
