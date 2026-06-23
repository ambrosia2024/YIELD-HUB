#!/bin/bash
#SBATCH --job-name=cybench_hpo_new
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

# Configurations – set yourself
# Forecast type: end-of-season, three-quarter-of-season, middle-of-season, quarter-of-season
FORECAST_TYPE="${1:-end-of-season}"

# Model checkpoint directory (override externally via 2nd argument)
CHECKPOINT_DIR="${2:-modelCheckpoints}"

# Minimum years required for a country to be included
MIN_YEARS=8
EXCLUDED_COUNTRIES=""

YEARS_DICT=$(cat <<'EOF'
{
  "maize": {
    "BR": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "US": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
  },
  "wheat": {
    "BR": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "US": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
  }
}
EOF
)

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if [ "$crop" = "maize" ]; then
        MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then
        WHEAT_COUNTRIES[$country]=1
    fi
done < <(python3 -c "
import ast

data = ast.literal_eval('''$YEARS_DICT''')
MIN_YEARS = $MIN_YEARS
EXCLUDED = '$EXCLUDED_COUNTRIES'.split()

for crop in data.keys():
    for country, years in data[crop].items():
        if len(years) >= MIN_YEARS and country not in EXCLUDED:
            print(f'{crop}|{country}')
")

echo "========================================"
echo "HPO Configuration"
echo "========================================"
echo "  Forecast Type: $FORECAST_TYPE"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  MIN_YEARS: $MIN_YEARS"
echo "  EXCLUDED_COUNTRIES: $EXCLUDED_COUNTRIES"
echo "  Maize countries with >=$MIN_YEARS years: $(echo ${!MAIZE_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "  Wheat countries with >=$MIN_YEARS years: $(echo ${!WHEAT_COUNTRIES[@]} | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "========================================"


for i in $(seq 1 $MAX_PARALLEL); do
    echo >&3
done

mkdir -p "$CHECKPOINT_DIR/in-season/results" "$CHECKPOINT_DIR/in-season/hpo" log

# Acquires a token, launches process, releases token when done
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


is_completed() {
    local model=$1
    local country=$2
    local crop=$3
    local hpo_log_file="$CHECKPOINT_DIR/in-season/hpo/hpo_${model}_${country}_${crop}.txt"

    # Check if HPO log file exists and contains "HPO Experiment complete:"
    if [ -f "$hpo_log_file" ] && grep -q "HPO Experiment complete:" "$hpo_log_file" 2>/dev/null; then
        return 0  # Completed - HPO finished successfully
    fi

    return 1  # Not completed - file not found or completion marker not present
}

get_countries_for_crop() {
    local crop=$1
    if [ "$crop" = "maize" ]; then
        echo "${!MAIZE_COUNTRIES[@]}"
    elif [ "$crop" = "wheat" ]; then
        echo "${!WHEAT_COUNTRIES[@]}"
    fi
}

sort_countries() {
    echo $1 | tr ' ' '\n' | sort | tr '\n' ' '
}

crops=("wheat" "maize")

# ========================================
# HPO: PatchTST (Transformers)
# ========================================
echo "========================================"
echo "Running HPO for patchtst"
echo "========================================"

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop $crop)")
    for country in $countries; do

        if is_completed "patchtst" "$country" "$crop"; then
            echo "Skipping patchtst $country $crop (already completed)"
            continue
        fi

        results_dir="$CHECKPOINT_DIR/results/hpo_patchtst_${country}_${crop}"
        echo "Starting HPO: patchtst $country $crop (forecast=$FORECAST_TYPE)"

        cmd=(
            python tstBaselines.py
            --crop "$crop"
            --country "$country"
            --model_type patchtst
            --aggregation daily
            --epochs 10
            --test_years 5
            --n_trials 50
            --hpo_objective nrmse
            --hpo_study_name "AAAI2027-HPO"
            --forecast_type "$FORECAST_TYPE"
            --save_checkpoint_dir "$CHECKPOINT_DIR/in-season/yield-patchtst-hpo/$country/$crop/"
            --results_dir "$results_dir"
        )

        run_model \
            "$CHECKPOINT_DIR/in-season/hpo/hpo_patchtst_${country}_${crop}.txt" \
            "${cmd[@]}"

    done
done

MAX_PARALLEL=6

PIPE=$(mktemp -u)
mkfifo "$PIPE"
exec 3<>"$PIPE"
rm "$PIPE"

for i in $(seq 1 $MAX_PARALLEL); do
    echo >&3
done

# ========================================
# HPO: XLinear (Linear Baselines)
# ========================================
echo "========================================"
echo "Running HPO for xlinear"
echo "========================================"

for crop in "${crops[@]}"; do
    countries=$(sort_countries "$(get_countries_for_crop $crop)")
    for country in $countries; do

        if is_completed "xlinear" "$country" "$crop"; then
            echo "Skipping xlinear $country $crop (already completed)"
            continue
        fi

        results_dir="$CHECKPOINT_DIR/results/hpo_xlinear_${country}_${crop}"
        echo "Starting HPO: xlinear $country $crop (forecast=$FORECAST_TYPE)"

        cmd=(
            python linearBaselines.py
            --crop "$crop"
            --country "$country"
            --model_type xlinear
            --aggregation daily
            --epochs 10
            --test_years 5
            --n_trials 50
            --hpo_objective nrmse
            --hpo_study_name "AAAI2027-HPO"
            --forecast_type "$FORECAST_TYPE"
            --save_checkpoint_dir "$CHECKPOINT_DIR/in-season/yield-xlinear-hpo/$country/$crop/"
            --results_dir "$results_dir"
        )

        run_model \
            "$CHECKPOINT_DIR/in-season/hpo/hpo_xlinear_${country}_${crop}.txt" \
            "${cmd[@]}"

    done
done

wait

echo "End time: $(date)"
echo "All HPO jobs finished."
