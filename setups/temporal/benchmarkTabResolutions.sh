#!/bin/bash
#SBATCH --job-name=tab_benchmark
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/setups/temporal/slurm-logs/tab_res_bench_%j.out
#SBATCH --error=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/setups/temporal/slurm-logs/tab_res_bench_%j.err

#====================================================================================#
#  TAB MODEL RESOLUTION BENCHMARK (B0 — raw features only)
#  ------------------------------------------------------------------------------
#  Goal: compare tabular model accuracy across temporal resolutions.
#  For wheat and maize, every country with >= 10 years of data, train:
#     - tabBaselines.py x 5 model_types (tabicl tabpfn tabdpt mitra limix)
#  => 5 runs per (crop, country, resolution).
#
#  tabBaselines.py always loads data at daily resolution internally and bins it
#  down to the target --resolution, so --aggregation is NOT passed.
#
#  Fixed settings (NO engineered features — raw dataset + CWB only):
#     resolution=<swept>, use_cwb_feature ON, loss_type=mql, epochs=30,
#     test_years=5, lag_years=0, evaluation_tier=accuracy, forecast=end-of-season.
#  Explicitly OFF: use_sota_features, include_spatial_features, use_recursive_lags,
#     use_gdd, use_heat_stress_days, use_rue, use_rtai, use_residual_trend,
#     use_exponential_weighting, multi_year_summaries.
#
#  Checkpoints are saved as *.tabtune files (not *.ckpt); skip logic checks for those.
#
#  Usage:   sbatch benchmarkTabResolutions.sh <output_dir>
#  Example: sbatch benchmarkTabResolutions.sh tabResolutionBenchmarkOutputDir
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

YEARS_DICT_FILE="$SCRIPT_DIR/../configurations/years_dict.json"
WANDB_PROJECT="CYBENCH-resolution-benchmark"
FORECAST_TYPE="end-of-season"
EPOCHS=30
TEST_YEARS=5
MIN_TOTAL_YEARS=10

LOG_DIR="$SCRIPT_DIR/log-benchmark"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$SCRIPT_DIR/slurm-logs"

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Output dir: $OUTPUT_DIR | wandb: $WANDB_PROJECT | forecast: $FORECAST_TYPE"

# Resolutions to sweep (tabBaselines bins daily data down to each)
RESOLUTIONS=(weekly dekad monthly)

# Tab model types
TAB_MODELS=(tabicl tabpfn tabdpt mitra)

#–––––––––––––––––––––– Country selection (total years >= 10) ––––––––––––––––––––#
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
        if len(years) >= MIN_TOTAL:
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
# run_one <crop> <country> <model_type> <resolution>
run_one() {
    local crop=$1 country=$2 model_type=$3 resolution=$4

    local out_dir="${OUTPUT_DIR}/${resolution}/${model_type}/${crop}/${country}"

    # Skip if a .tabtune checkpoint already exists for this cell.
    if find "$out_dir/checkpoints" -name "*.tabtune" 2>/dev/null | grep -q .; then
        echo "Skipping ${resolution} ${model_type} ${crop} ${country} (checkpoint exists)"
        return
    fi
    mkdir -p "$out_dir/checkpoints" "$out_dir/results"

    local suffix; suffix=$(generate_random_suffix)
    local run_id="${resolution}-${model_type}-${crop}-${country}-${suffix}"

    local cmd=(
        "$PYTHON" "$SCRIPT_DIR/tabFMBaselines.py"
        --crop          "$crop"
        --country       "$country"
        --model_type    "$model_type"
        --resolution    "$resolution"
        --epochs        "$EPOCHS"
        --test_years    "$TEST_YEARS"
        --lag_years     0
        --use_cwb_feature
        --loss_type     mql
        --evaluation_tier accuracy
        --forecast_type "$FORECAST_TYPE"
        --save_checkpoint_dir "$out_dir/checkpoints"
        --results_dir         "$out_dir/results"
        --wandb_project "$WANDB_PROJECT"
        --run_id        "$run_id"
    )

    echo "Starting: ${resolution} ${model_type} ${crop} ${country} (run_id: ${run_id})"
    run_model "${LOG_DIR}/${run_id}.txt" "${cmd[@]}"
}

#–––––––––––––––––––––––––––––– Training sweep ––––––––––––––––––––––––––––––#
crops=("wheat" "maize")

for resolution in "${RESOLUTIONS[@]}"; do
    echo "=== Resolution: ${resolution} ==="
    for crop in "${crops[@]}"; do
        countries=$(sort_countries "$(get_countries_for_crop "$crop")")
        for country in $countries; do
            for m in "${TAB_MODELS[@]}"; do
                run_one "$crop" "$country" "$m" "$resolution"
            done
        done
    done
done

wait
echo "Benchmark completed: $(date)"
