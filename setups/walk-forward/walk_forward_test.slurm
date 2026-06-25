#!/bin/bash
# Smoke test: one crop/country, few epochs, serial execution.
# Submit (CPU, default — faster queue):  sbatch walk_forward_test.slurm
# Submit (GPU):  sbatch -p gpu --gpus=1 walk_forward_test.slurm
# Partitions: sinfo -s
#
#SBATCH --job-name=wf_smoke_test
#SBATCH --time=04:00:00
#SBATCH -p main
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=michiel.kallenberg@wur.nl
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

# sbatch copies the script to a spool dir; use the directory where you ran sbatch
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export FORECAST_TYPE="${FORECAST_TYPE:-middle-of-season}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/walk_forward_env.sh"

# CPU partition: do not bind to a GPU (Lightning will use accelerator=cpu)
unset CUDA_VISIBLE_DEVICES

#–––––––––––––––––––––––––––––––––– Smoke-test knobs ––––––––––––––––––––––––––– #
TEST_CROP="${TEST_CROP:-wheat}"
TEST_COUNTRY="${TEST_COUNTRY:-NL}"
# One model only (informer = quick manual test; patchtst = production transformer)
TEST_MODEL="${TEST_MODEL:-informer}"
TEST_EPOCHS="${TEST_EPOCHS:-2}"
TEST_YEARS="${TEST_YEARS:-2}"
TEST_AGGREGATION="${TEST_AGGREGATION:-daily}"
WANDB_PROJECT="${WANDB_PROJECT:-test-cluster-wf}"
WF_RUN_LABEL="${WF_RUN_LABEL:-smoke-test}"
RUN_ID="${TEST_MODEL}-${TEST_CROP}-${TEST_COUNTRY}-${WF_FORECAST_SLUG}-smoke"

walk_forward_output_dirs "$WF_RUN_LABEL" "$TEST_MODEL" "$TEST_COUNTRY" "$TEST_CROP"

echo ""
echo "========== SMOKE TEST =========="
echo "  ${TEST_CROP}-${TEST_COUNTRY}  model=${TEST_MODEL}  epochs=${TEST_EPOCHS}"
echo "  test_years=${TEST_YEARS}  aggregation=${TEST_AGGREGATION}  forecast_type=${FORECAST_TYPE}"
echo "  hyperparameters: $(basename "$HYPERPARAMS_JSON")"
echo "  checkpoints: ${CHECKPOINT_SAVE_DIR}"
echo "  results (CSV): ${RESULTS_DIR}"
echo "  wandb: ${WANDB_PROJECT}"
echo "================================"
echo ""

if ! has_crop_country_data "$TEST_CROP" "$TEST_COUNTRY"; then
    echo "ERROR: No data for ${TEST_CROP}-${TEST_COUNTRY} under $CYBENCH_DATA_DIR"
    exit 1
fi

mkdir -p log

# Map model -> script (informer/patchtst/etc. -> tstBaselines; xlinear -> linearBaselines)
case "$TEST_MODEL" in
    patchtst|autoformer|informer|tsmixer|tst|itransformer|timexer|timesnet)
        script="$SCRIPT_DIR/tstBaselines.py"
        ;;
    xlinear|nlinear|dlinear|rlinear|olinear)
        script="$SCRIPT_DIR/linearBaselines.py"
        ;;
    *)
        echo "ERROR: Unknown TEST_MODEL=$TEST_MODEL"
        exit 1
        ;;
esac

hps_args=()
if get_hyperparameters "$TEST_MODEL" "$TEST_CROP" "$TEST_COUNTRY" hps_args; then
    echo "Using HPS from $(basename "$HYPERPARAMS_JSON")"
else
    echo "No HPS override for ${TEST_MODEL}/${TEST_COUNTRY}/${TEST_CROP}; using script defaults"
    hps_args=()
fi

model_flags=()
if [ "$TEST_MODEL" = "patchtst" ] || [ "$TEST_MODEL" = "xlinear" ]; then
    read -ra model_flags <<< "$(get_model_flags "$TEST_MODEL" "$TEST_CROP")"
fi

# Serial run (foreground) — full walk_forward.sh uses MAX_PARALLEL for many jobs
run_python "$script" \
    --crop "$TEST_CROP" \
    --country "$TEST_COUNTRY" \
    --model_type "$TEST_MODEL" \
    --aggregation "$TEST_AGGREGATION" \
    --forecast_type "$FORECAST_TYPE" \
    --epochs "$TEST_EPOCHS" \
    --test_years "$TEST_YEARS" \
    --lag_years 0 \
    --drop_tavg \
    "${model_flags[@]}" \
    "${hps_args[@]}" \
    --save_checkpoint_dir "$CHECKPOINT_SAVE_DIR" \
    --results_dir "$RESULTS_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id "$RUN_ID" \
    2>&1 | tee "log/smoke_${RUN_ID}.txt"

echo ""
echo "Smoke test finished: $(date)"
echo "CSVs: ${RESULTS_DIR}"
echo "Checkpoints: ${CHECKPOINT_SAVE_DIR}"
echo "Log: log/smoke_${RUN_ID}.txt"
