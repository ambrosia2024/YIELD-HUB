#!/bin/bash
# populate_dashboard.sh — injects data into all HTML pages
# Run from: dashboard/   e.g.  bash scripts/populate_dashboard.sh
set -e
PY="/home/vsaxena1/miniconda3/envs/cybench/bin/python3"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "=== populate_dashboard.sh ==="

echo "[1] pages/tab_EDA.html"
$PY "$SCRIPTS/generate_eda.py"

# Add more tabs here as generators are written:
# echo "[2] pages/tab_tst.html"
# $PY "$SCRIPTS/generate_tst.py"

echo "All pages refreshed."
