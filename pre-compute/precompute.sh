#!/bin/bash
#SBATCH --job-name=precompute
#SBATCH --time=16:00:00
#SBATCH -p gpu_a100
#SBATCH -n 1
#SBATCH --gpus=1
# GPU partition requires --gpus; the GPU itself is idle throughout — all steps
# are CPU-only (pandas / scipy / sklearn / raster I/O).
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=v.saxena@maastrichtuniversity.nl
#SBATCH --output=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/pre-compute/slurm-logs/precompute_%j.out
#SBATCH --error=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/pre-compute/slurm-logs/precompute_%j.err

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Usage: sbatch precompute.sh
#
# Runs the full pre-compute pipeline in order:
#   Step 1  fetch_koppen.py            — Koppen-Geiger climate zone labels
#                                        run ONCE, both crops, all countries
#   Step 2  fetch_aez.py              — GAEZ v4 agro-climatic zone labels
#                                        (COMMENTED — run separately if needed)
#   Step 3  compute_shift_distance.py — domain-classifier shift (country vs pool)
#                                        run PER CROP, auto-discovers countries
#   Step 4  fetch_gaez_predictors.py  — GAEZ v5 multi-predictor table (55 vars)
#                                        run ONCE, both crops, both periods
#   Step 5  compute_rawdata_shift.py  — per-variable train/test shift metrics
#                                        run PER CROP, all eligible countries
#
# Country lists are derived from years_dict.json with a MIN_YEARS=8 filter.
# Time budget: 16h to accommodate GAEZ v5 raster downloads (~10 GB).
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

PRECOMPUTE_DIR="/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/pre-compute"
PYTHON="/home/vsaxena1/miniconda3/envs/cybench/bin/python"
cd "$PRECOMPUTE_DIR"

echo "Running on node: $(hostname)"
echo "Start time:      $(date)"
echo ""

#–––––––––––––––––––––––––––– Build country lists ––––––––––––––––––––––––––––#

MIN_YEARS=8

declare -A MAIZE_COUNTRIES
declare -A WHEAT_COUNTRIES

while IFS='|' read -r crop country; do
    if   [ "$crop" = "maize" ]; then MAIZE_COUNTRIES[$country]=1
    elif [ "$crop" = "wheat" ]; then WHEAT_COUNTRIES[$country]=1
    fi
done < <("$PYTHON" -c "
import json
from pathlib import Path
CYBENCH_ROOT = Path('/gpfs/home4/vsaxena1/AgML-CY-Bench')
years_dict = CYBENCH_ROOT / 'YIELD-HUB' / 'setups' / 'configurations' / 'years_dict.json'
with open(years_dict) as f:
    data = json.load(f)
for crop in data:
    for country, years in data[crop].items():
        if len(years) >= $MIN_YEARS:
            print(f'{crop}|{country}')
")

MAIZE_LIST=$(echo "${!MAIZE_COUNTRIES[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')
WHEAT_LIST=$(echo "${!WHEAT_COUNTRIES[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')

# Union of both crops — for crop-agnostic scripts (fetch_koppen.py)
ALL_COUNTRIES=$(echo "${MAIZE_LIST} ${WHEAT_LIST}" | tr ' ' '\n' | sort -u | tr '\n' ' ')

echo "MIN_YEARS filter : $MIN_YEARS"
echo "Maize countries  : $MAIZE_LIST"
echo "Wheat countries  : $WHEAT_LIST"
echo "All (union)      : $ALL_COUNTRIES"
echo ""

#–––––––––––––––––––– Step 1: Koppen-Geiger labels ––––––––––––––––––––––––––#
# Koppen zones are location-based and crop-agnostic.  Run once with --crop
# maize wheat and all countries from both crops.  --resume makes re-runs safe.

# echo "=== Step 1: fetch_koppen.py — both crops, all countries ==="
# "$PYTHON" "$PRECOMPUTE_DIR/fetch_koppen.py" \
#     --crop maize wheat \
#     --countries $ALL_COUNTRIES \
#     --resume
# echo "fetch_koppen.py done: $(date)"
# echo ""

#–––––––––––––––––––– Step 2: GAEZ v4 agro-climatic zone labels ––––––––––––––#
# GAEZ v4 suitability is crop-specific.  Commented out — run separately if needed.

# echo "=== Step 2a: fetch_aez.py — maize ==="
# "$PYTHON" "$PRECOMPUTE_DIR/fetch_aez.py" \
#     --crop maize \
#     --countries $MAIZE_LIST \
#     --resume
# echo "fetch_aez.py maize done: $(date)"
# echo ""

# echo "=== Step 2b: fetch_aez.py — wheat ==="
# "$PYTHON" "$PRECOMPUTE_DIR/fetch_aez.py" \
#     --crop wheat \
#     --countries $WHEAT_LIST \
#     --resume
# echo "fetch_aez.py wheat done: $(date)"
# echo ""

#–––––––––––––––––––– Step 3: Domain shift characterisation ––––––––––––––––#
# --countries omitted: script auto-discovers all countries with >= 8 years
# from years_dict.json (correct for a full-pool shift characterisation).

# echo "=== Step 3a: compute_shift_distance.py — maize ==="
# "$PYTHON" "$PRECOMPUTE_DIR/compute_shift_distance.py" --crop maize
# echo "compute_shift_distance.py maize done: $(date)"
# echo ""

# echo "=== Step 3b: compute_shift_distance.py — wheat ==="
# "$PYTHON" "$PRECOMPUTE_DIR/compute_shift_distance.py" --crop wheat
# echo "compute_shift_distance.py wheat done: $(date)"
# echo ""

#–––––––––––––––––––– Step 4: GAEZ v5 multi-predictor table ––––––––––––––––#
# 55 predictors (agro-climatic potential, yield-gap, soil quality, land use).
# Crop-agnostic raster download (~10 GB); --resume makes re-runs safe.
# Both periods 2001-2020 and 1981-2000 to match the ACZ baseline convention.

# echo "=== Step 4: fetch_gaez_predictors.py — maize + wheat, both periods ==="
# "$PYTHON" "$PRECOMPUTE_DIR/fetch_gaez_predictors.py" \
#     --crop maize wheat \
#     --periods 2001-2020 1981-2000 \
#     --resume
# echo "fetch_gaez_predictors.py done: $(date)"
# echo ""

#–––––––––––––––––––– Step 5: Per-variable train/test shift ––––––––––––––––#
# Reads raw CSVs directly (no model pipeline).  Fast: minutes per crop.
# test_years=5 matches the fixed split used in all model training runs.

echo "=== Step 5a: compute_rawdata_shift.py — maize ==="
"$PYTHON" "$PRECOMPUTE_DIR/compute_rawdata_shift.py" \
    --crop maize \
    --test_years 5
echo "compute_rawdata_shift.py maize done: $(date)"
echo ""

echo "=== Step 5b: compute_rawdata_shift.py — wheat ==="
"$PYTHON" "$PRECOMPUTE_DIR/compute_rawdata_shift.py" \
    --crop wheat \
    --test_years 5
echo "compute_rawdata_shift.py wheat done: $(date)"
echo ""

echo "=== Pre-compute pipeline complete: $(date) ==="
