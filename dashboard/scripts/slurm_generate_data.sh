#!/bin/bash
#SBATCH --job-name=eda_full
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/dashboard/logs/eda_full_%j.out
#SBATCH --error=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/dashboard/logs/eda_full_%j.err

PYTHON=/home/vsaxena1/miniconda3/envs/cybench/bin/python3
PRECOMPUTE=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/pre-compute
DASHBOARD=/gpfs/home4/vsaxena1/AgML-CY-Bench/YIELD-HUB/dashboard

echo "=== EDA full pipeline ==="
echo "Start: $(date)"

echo "[1/5] compute_rawdata_shift.py maize ..."
$PYTHON $PRECOMPUTE/compute_rawdata_shift.py --crop maize --n_jobs 8

echo "[2/5] compute_rawdata_shift.py wheat ..."
$PYTHON $PRECOMPUTE/compute_rawdata_shift.py --crop wheat --n_jobs 8

echo "[3/5] compute_eda_peryear.py ..."
$PYTHON $PRECOMPUTE/compute_eda_peryear.py --crop both --n_jobs 8

echo "[4/5] build_eda_data.py ..."
$PYTHON $DASHBOARD/scripts/build_eda_data.py /gpfs/home4/vsaxena1/AgML-CY-Bench/cybench/data/variable_shift $DASHBOARD/data

echo "[5/5] generate_eda.py ..."
$PYTHON $DASHBOARD/scripts/generate_eda.py

echo "Done: $(date)"
