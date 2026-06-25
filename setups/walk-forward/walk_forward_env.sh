# Shared environment for walk-forward Slurm scripts.
# Source from walk_forward.sh / walk_forward_test.slurm after setting SCRIPT_DIR.

set -euo pipefail

cd "$SCRIPT_DIR"

YIELD_HUB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CYBENCH_PKG_ROOT="$(cd "$YIELD_HUB_ROOT/.." && pwd)"
REPO_ROOT="$(cd "$CYBENCH_PKG_ROOT/.." && pwd)"

CONFIG_DIR="${CONFIG_DIR:-$SCRIPT_DIR/../configurations}"
CYBENCH_DATA_DIR="${CYBENCH_DATA_DIR:-$CYBENCH_PKG_ROOT/data}"
# Forecast horizon: end-of-season (EOS) or middle-of-season (MOS).
# Selects hyperparameter JSON and is passed to Python as --forecast_type.
FORECAST_TYPE="${FORECAST_TYPE:-end-of-season}"
# All walk-forward artifacts (CSVs, .ckpt) — absolute paths, safe with poetry run
YIELD_HUB_OUTPUT_ROOT="${YIELD_HUB_OUTPUT_ROOT:-$YIELD_HUB_ROOT/output/walk-forward}"

# Map FORECAST_TYPE → hyperparameter file (patchtst / xlinear HPO results).
resolve_walk_forward_config() {
    case "$FORECAST_TYPE" in
        end-of-season)
            HYPERPARAMS_JSON="${HYPERPARAMS_JSON:-$CONFIG_DIR/eosHyperparameters.json}"
            WF_FORECAST_SLUG="eos"
            ;;
        middle-of-season)
            HYPERPARAMS_JSON="${HYPERPARAMS_JSON:-$CONFIG_DIR/mosHyperparameters.json}"
            WF_FORECAST_SLUG="mos"
            ;;
        *)
            echo "ERROR: FORECAST_TYPE must be 'end-of-season' or 'middle-of-season' (got: $FORECAST_TYPE)"
            exit 1
            ;;
    esac
    export HYPERPARAMS_JSON WF_FORECAST_SLUG FORECAST_TYPE
}

resolve_walk_forward_config

if [ -f "$YIELD_HUB_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$YIELD_HUB_ROOT/.env"
    set +a
    CYBENCH_DATA_DIR="${CYBENCH_DATA_DIR:-$CYBENCH_PKG_ROOT/data}"
fi

mkdir -p log

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${YIELD_HUB_ROOT}/process:${YIELD_HUB_ROOT}/architectures:${PYTHONPATH:-}"
export PATH="${HOME}/.local/bin:${PATH}"

run_python() {
    if command -v poetry >/dev/null 2>&1 && [ -f "$REPO_ROOT/pyproject.toml" ]; then
        (cd "$REPO_ROOT" && poetry run python "$@")
    elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        "$REPO_ROOT/.venv/bin/python" "$@"
    elif command -v python3 >/dev/null 2>&1; then
        python3 "$@"
    else
        python "$@"
    fi
}

echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"
echo "Python: $(run_python -c 'import sys; print(sys.executable)')"
echo "Version: $(run_python --version 2>&1)"
echo "CY-Bench data dir: $CYBENCH_DATA_DIR"
echo "Walk-forward output root: $YIELD_HUB_OUTPUT_ROOT"
echo "Forecast type: $FORECAST_TYPE (HPS: $(basename "$HYPERPARAMS_JSON"))"

if [ ! -d "$CYBENCH_DATA_DIR" ]; then
    echo "ERROR: CYBENCH_DATA_DIR does not exist: $CYBENCH_DATA_DIR"
    exit 1
fi
if [ ! -e "$CYBENCH_PKG_ROOT/data" ] || [ -L "$CYBENCH_PKG_ROOT/data" ]; then
    ln -sfn "$CYBENCH_DATA_DIR" "$CYBENCH_PKG_ROOT/data"
    echo "Linked $CYBENCH_PKG_ROOT/data -> $CYBENCH_DATA_DIR"
fi

run_python -c "import cybench; from cybench.datasets.configured import load_dfs_crop" || {
    echo "ERROR: Cannot import cybench. From $REPO_ROOT run: poetry install"
    exit 1
}

get_model_flags() {
    local model=$1 crop=$2
    run_python -c "
import json
with open('$HYPERPARAMS_JSON') as f:
    config = json.load(f)
flags = config['model_flags']['$model'].get('$crop', config['model_flags']['$model'].get('default', []))
print(' '.join(flags))
"
}

get_hyperparameters() {
    local model=$1 crop=$2 country=$3
    local -n out_array=$4
    local output
    output=$(run_python -c "
import json
with open('$HYPERPARAMS_JSON') as f:
    config = json.load(f)
try:
    hps = config['overrides']['$country']['$model']['$crop']
    for k, v in hps.items():
        print(f'--{k}')
        print(f'{v}')
except KeyError:
    exit(1)
") || return 1
    readarray -t out_array <<< "$output"
    return 0
}

has_crop_country_data() {
    local crop=$1 country=$2
    [ -f "$CYBENCH_DATA_DIR/$crop/$country/yield_${crop}_${country}.csv" ]
}

# Set CHECKPOINT_SAVE_DIR and RESULTS_DIR (exported) for one run.
# Layout: output/walk-forward/<run-label>/yield-<model>/<country>/<crop>/{checkpoints,results}/
walk_forward_output_dirs() {
    local run_label=$1 model=$2 country=$3 crop=$4
    local run_base="$YIELD_HUB_OUTPUT_ROOT/$run_label/${WF_FORECAST_SLUG}/yield-${model}/${country}/${crop}"
    if [ -n "${WF_FOLD_IDX:-}" ]; then
        run_base="$run_base/fold_$(printf '%02d' "$WF_FOLD_IDX")"
    fi
    CHECKPOINT_SAVE_DIR="$run_base/checkpoints"
    RESULTS_DIR="$run_base/results"
    export CHECKPOINT_SAVE_DIR RESULTS_DIR
    mkdir -p "$CHECKPOINT_SAVE_DIR" "$RESULTS_DIR"
}
