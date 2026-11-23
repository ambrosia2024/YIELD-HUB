#!/bin/bash
# generate_data.sh — runs pre-compute scripts and builds dashboard/data/*.json
# Run from: dashboard/   e.g.  bash scripts/generate_data.sh [maize|wheat|both]
set -e
CROP="${1:-both}"
PY="/home/vsaxena1/miniconda3/envs/cybench/bin/python3"
REPO="/gpfs/home4/vsaxena1/AgML-CY-Bench"
PRECOMPUTE="$REPO/YIELD-HUB/pre-compute"
SRC="$REPO/cybench/data/variable_shift"
DASH="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== generate_data.sh (crop=$CROP) ==="

JSON_OUT="$DASH/data/eda_variable_shift.json"

echo "[1/3] compute_rawdata_shift.py ..."
if [ "$CROP" = "both" ]; then
  $PY "$PRECOMPUTE/compute_rawdata_shift.py" --crop maize --json_output "$JSON_OUT"
  $PY "$PRECOMPUTE/compute_rawdata_shift.py" --crop wheat --json_output "$JSON_OUT"
else
  $PY "$PRECOMPUTE/compute_rawdata_shift.py" --crop "$CROP" --json_output "$JSON_OUT"
fi

echo "[2/3] compute_eda_peryear.py ..."
$PY "$PRECOMPUTE/compute_eda_peryear.py" --crop "$CROP"

echo "[3/3] Copying eda_peryear.json to dashboard/data/ ..."
cp "$SRC/eda_peryear.json" "$DASH/data/eda_peryear.json"
echo "  Done."

echo "Done. Run: bash scripts/populate_dashboard.sh"
