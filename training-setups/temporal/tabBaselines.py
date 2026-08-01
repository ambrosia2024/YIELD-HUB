# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: A tabular foundation model based in-season and end-of-season crop yield prediction script.
             Implements TabPFN, TabICL, and TabDPT with full coherence to TST baselines.

Python version: 3.12.0

--------------------
Architecture overview: This script implements tabular foundation models that treat time series
as flattened tabular features while preserving temporal ordering through feature naming:

    • TabPFN: Tabular Prior-data Fitted Network (https://arxiv.org/abs/2206.13494)
    • TabICL: In-Context Learning for Tabular Data (https://arxiv.org/abs/2408.11586)
    • TabDPT: Diffusion-based Tabular Pre-trained Transformer (https://arxiv.org/abs/2406.03991)

All models share the same data pipeline and evaluation as TST baselines for direct comparison.

--------------------
Data pipeline:
The script processes agricultural data through multiple stages:

1. INPUT FEATURES
   - Weather: tmin, tmax, tavg, precipitation, radiation, (optional: cwb)
   - Remote Sensing: NDVI, FPAR, SSM, RSM
   - Soil Properties: Available water capacity, organic carbon, pH, texture
   - Location: Country, state, latitude, longitude
   - Crop Calendar: Start/end of season with cyclic encoding (sin/cos)
   - Historical Lags: Yield from previous years (1-2 years, configurable)

2. TEMPORAL AGGREGATION
   - daily:   365 time steps → flattened to 365 features per variable
   - weekly:  52 time steps  → flattened to 52 features per variable
   - dekad:   36 time steps  → flattened to 36 features per variable

3. FEATURE FLATTENING (for tabular models)
   Time series features are flattened while preserving temporal order:
   [dek1_tmin, dek1_tmax, ..., dek1_ndvi, dek2_tmin, ..., dek36_ndvi]

4. NORMALIZATION
   - Features: z-score normalization (handled internally by tabular models)
   - Targets: Normalized to zero mean, unit variance

5. IN-SEASON vs END-SEASON predictions
   – --forecast_type: When to make the prediction (end-of-season, three-quarter-of-season,
                      middle-of-season, quarter-of-season)

--------------------
Other optional/advanced features:

1. RESIDUAL TREND MODELING (--use_residual_trend)
   Uses Mann-Kendall trend detection to identify significant linear trends in
   training yields, then models residuals (yield - trend) to improve forecasting.

2. RECURSIVE LAG PREDICTION (--use_recursive_lags)
   For true out-of-sample testing: uses predicted yields as lag features during
   test set evaluation instead of ground truth.

3. SPATIAL FEATURES (--include_spatial_features)
   Adds explicit latitude/longitude as static features.

4. FEATURE ABLATION TOGGLES
   --use_cwb_feature: Include crop water balance
   --drop_tavg: Drop average temperature
   --use_gdd: Adds cumulative GDD as a time series feature
   --use_heat_stress_days: Adds heat/frost/dry stress day counts
   --use_rue: Adds RUE index as a time series feature
   --use_farquhar: Adds Farquhar photosynthesis proxy

-------------
Training workflow:
1. Data module handles train/val/test splits
2. Lightning trainer manages GPU distribution
3. Tabular models fit once in on_train_start() (no iterative training)
4. Evaluation metrics computed on test set

Evaluation metrics:
    • MSE, MAE, RMSE: Standard error metrics
    • R²: Coefficient of determination
    • MAPE, SMAPE: Percentage-based error metrics
    • NRMSE: Normalized RMSE

------
Output generated:
    Results CSV: Detailed predictions with actuals, errors, metadata
    WandB: Full experiment tracking with metrics, parameters

--------------
Usage:
# Basic training with TabPFN
    python tabBaselines.py --crop maize --country NL --model_type tabpfn --aggregation dekad --epochs 5 --results_dir checkpoints-test/results --save_checkpoint_dir checkpoints-test/results --wandb_project test-and-delete-later --forecast_type end-of-season

# Use all domain features
    python tabBaselines.py --crop maize --country NL --model_type tabicl --use_gdd --use_rue --use_farquhar --aggregation dekad

# Quick test run (2 epochs for validation loop testing)
    python tabBaselines.py --crop wheat --country NL --model_type tabdpt --epochs 2 --aggregation dekad --test_years 3

--------------------
Hyperparameters:
    --max_train_samples: Subsample training data for TabPFN context limits (default: None)
    --subsample: Subsampling method ('random' or 'quantile', default: 'random')
    --preprocess: Preprocessing mode ('none' or 'sklearn', default: 'none')
    --allow_cpu_fallback: Allow CPU fallback if GPU fails (default: False)

------------
Core dependencies:
    - torch>=2.0: PyTorch for Lightning integration
    - lightning: PyTorch Lightning for training framework
    - torchmetrics: Evaluation metrics
    - wandb: Experiment tracking
    - tabpfn/tabicl/tabdpt: Tabular foundation models (install as needed)
    - pymannkendall: Trend detection for residual modeling
"""

import os
import sys
import random
import argparse
import logging
import uuid
import csv

from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

# CY-BENCH Dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_TYPE, set_forecast_type, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

cybench.config.FORECAST_LEAD_TIME = "0-days"

from cybench.process.alignment_patch import patch_alignment
patch_alignment()

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

# Custom functions and classes
sys.path.append('../../process/')
from helpers import generate_checkpoint_name, save_test_results_to_csv
from validateModel import print_metrics_table
from loadData import calculate_fixed_split, TabularCYBenchDataModule
from alignment_patch import verify_forecast_horizon_config
from spatiotemporal_metrics import (
    compute_all_spatiotemporal_metrics,
    save_spatiotemporal_metrics
)

sys.path.append('../../architectures/')
from modelconfig import TabModelConfig
from tabLayer import create_model

# Set matmul precision
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        torch.set_float32_matmul_precision('high')
        logger.info(f"Enabled high matmul precision (GPU capability {capability})")
    else:
        logger.info(f"Keeping default matmul precision (GPU capability {capability} < 8)")
else:
    logger.info("Running on CPU, matmul precision setting has no effect")

# Main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-BENCH Tabular Foundation Models for Yield Forecasting")
    parser.add_argument('--crop', default="maize")
    parser.add_argument('--country', default="NL")
    parser.add_argument('--model_type', default="tabpfn",
                        choices=['tabpfn', 'tabicl', 'tabdpt'])
    parser.add_argument('--aggregation', default="dekad",
                        choices=['daily', 'weekly', 'dekad'])
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--lag_years', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-tab',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', default='checkpoints/results',
                        help='Directory to save CSV results')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Validation epochs (tabular models fit once, this controls validation passes)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_residual_trend', action='store_true')
    parser.add_argument('--use_recursive_lags', action='store_true',
                        help='Use predicted yields as lags during testing')
    # Domain feature flags
    parser.add_argument('--use_gdd', action='store_true')
    parser.add_argument('--use_heat_stress_days', action='store_true')
    parser.add_argument('--use_rue', action='store_true')
    parser.add_argument('--use_farquhar', action='store_true')
    # Feature config flags
    parser.add_argument('--use_cwb_feature', action='store_true')
    parser.add_argument('--drop_tavg', action='store_true')
    # Tabular model specific parameters
    parser.add_argument('--preprocess', default='none', choices=['none', 'sklearn'])
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Subsample training data to this many samples')
    parser.add_argument('--subsample', default='random', choices=['random', 'quantile'])
    parser.add_argument('--subsample_bins', type=int, default=10)
    parser.add_argument('--allow_cpu_fallback', action='store_true')
    parser.add_argument('--predict_batch_size', type=int, default=256)
    parser.add_argument('--min_train_samples', type=int, default=100)
    # WandB and experiment tracking
    parser.add_argument('--wandb_project', default=None,
                        help='Custom WandB project name')
    parser.add_argument('--wandb_run_name', default=None)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--forecast_type', default="end-of-season",
                        choices=['end-of-season', 'three-quarter-of-season', 'middle-of-season',
                                 'quarter-of-season'])
    # Exponential weighting
    parser.add_argument('--use_exponential_weighting', action='store_true')
    parser.add_argument('--exponential_tau', type=float, default=10.0)
    # Multi-year summaries
    parser.add_argument('--multi_year_summaries', action='store_true')
    parser.add_argument('--multi_year_window', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--multi_year_features', nargs='+', default=['weather'],
                        choices=['weather', 'remote_sensing', 'phenology', 'all'])
    # Test years
    parser.add_argument('--test_years', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=None)
    # Loss function (for coherence with TST baselines - tabular models use their internal loss)
    parser.add_argument('--loss', default='mse', choices=['mse', 'pinball'],
                        help='Loss function: tabular models use their internal loss, this is for tracking only')

    args = parser.parse_args()

    # Set forecast type
    set_forecast_type("0-days")
    print(f"[Forecast Type] {args.forecast_type}")

    forecast_to_fraction = {
        'end-of-season': 1.0,
        'three-quarter-of-season': 0.75,
        'middle-of-season': 0.5,
        'quarter-of-season': 0.25,
    }
    data_fraction = forecast_to_fraction[args.forecast_type]
    print(f"[Data Fraction] Using {data_fraction:.0%} of season data")

    # Set num_workers
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 1
        args.num_workers = min(cpu_count // 4, 8)
        print(f"[Auto-config] Setting num_workers={args.num_workers}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())[:8]

    print(f"\n{'=' * 70}")
    print(f"CY-BENCH | {args.model_type.upper()} | {args.crop}-{args.country} "
          f"|  {args.aggregation.upper()}")
    print(f"SOTA={args.use_sota_features}  Spatial={args.include_spatial_features}  "
          f"Lag={args.lag_years}")
    print(f"ResidualTrend={args.use_residual_trend}  RecursiveLags={args.use_recursive_lags}")
    print(f"Domain features: GDD={args.use_gdd}  HeatStress={args.use_heat_stress_days}  "
          f"RUE={args.use_rue}  Farquhar={args.use_farquhar}")
    print(f"TestYears={args.test_years}")
    print(f"Exponential Weighting: {args.use_exponential_weighting} (tau={args.exponential_tau})")
    print(f"Multi-Year Summaries: {args.multi_year_summaries} (window={args.multi_year_window})")
    print(f"Flattened features: {args.aggregation} → tabular")
    print(f"{'=' * 70}\n")

    # Create config
    config = TabModelConfig(
        crop=args.crop,
        country=args.country,
        model_type=args.model_type,
        aggregation=args.aggregation,
        data_fraction=data_fraction,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        use_residual_trend=args.use_residual_trend,
        lag_years=args.lag_years,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        test_years=args.test_years,
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        use_recursive_lags=args.use_recursive_lags,
        use_gdd=args.use_gdd,
        use_heat_stress_days=args.use_heat_stress_days,
        use_rue=args.use_rue,
        use_farquhar=args.use_farquhar,
        use_exponential_weighting=args.use_exponential_weighting,
        exponential_tau=args.exponential_tau,
        multi_year_summaries=args.multi_year_summaries,
        multi_year_window=args.multi_year_window,
        multi_year_features=args.multi_year_features,
        results_dir=args.results_dir,
        preprocess=args.preprocess,
        max_train_samples=args.max_train_samples,
        subsample=args.subsample,
        subsample_bins=args.subsample_bins,
        allow_cpu_fallback=args.allow_cpu_fallback,
        predict_batch_size=args.predict_batch_size,
        min_train_samples=args.min_train_samples,
    )

    verify_forecast_horizon_config(config)

    print(f"[Feature Config] Weather features: {config.weather_features}")
    print(f"[Feature Config] Total flattened features: ~{config.flattened_feature_size}")

    if config.use_recursive_lags and config.lag_years > 0:
        print(f"\n{'=' * 70}")
        print(f"[RECURSIVE LAGS ENABLED]")
        print(f"During testing, model predictions will be used as lag features")
        print(f"instead of observed historical yields.")
        print(f"{'=' * 70}\n")

    # Create checkpoint directory
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    # Get available years
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)

    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # Calculate fixed split
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=args.test_years,
        val_years=2
    )

    print(f"\n[Split Config - Fixed]")
    print(f"Total years: {fixed_splits['total_years']}")
    print(f"Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    print(f"\n{'=' * 70}")
    print(f"PHASE: Final Model Training and Evaluation")
    print(f"{'=' * 70}\n")

    # Create datamodule
    dm = TabularCYBenchDataModule(config)
    dm.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    # Create model
    model = create_model(config)

    # WandB logger
    try:
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-TABULAR"
        base_run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model_type}-{args.crop}-{args.country}"
        wandb_run_name = args.run_id and f"{base_run_name}-{run_id}" or base_run_name
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args),
            group=f"{args.crop}-{args.country}"
        )
        loggers = [wandb_logger]
    except Exception as e:
        print(f"[WandB Warning] Could not initialise WandB logger: {e}")
        loggers = [CSVLogger("logs/", name="cybench-tabular")]

    # Setup callbacks
    callbacks = [LearningRateMonitor(logging_interval='epoch')]

    # Create trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nFitting tabular model...")
    trainer.fit(model, dm)

    print("\nEvaluating model...")
    test_results = trainer.test(model, dm)

    if test_results:
        r = test_results[0]
        final_metrics = {
            'mse': r.get('test/mse'),
            'mae': r.get('test/mae'),
            'rmse': r.get('test/rmse'),
            'r2': r.get('test/r2'),
            'mape': r.get('test/mape'),
            'smape': r.get('test/smape'),
            'nrmse': r.get('test/nrmse'),
        }
    else:
        final_metrics = {}

    # Get per-year metrics
    print(f"\n[CSV Results] Retrieving per-year metrics...")
    if hasattr(model, '_test_results_per_year') and model._test_results_per_year:
        per_year_metrics = model._test_results_per_year
    else:
        per_year_metrics = {}

    # Log per-year metrics
    print(f"\n[CSV Results] Per-Year Test Metrics:")
    for year in sorted(fixed_splits['test_years']):
        print(f"Year {year}:")
        for metric in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape']:
            key = f'{metric}_{year}'
            if key in per_year_metrics:
                print(f"  {metric.upper()}: {per_year_metrics[key]:.4f}")

    if 'mse_overall' in per_year_metrics:
        print(f"\n  Overall:")
        for metric in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape']:
            key = f'{metric}_overall'
            if key in per_year_metrics:
                print(f"  {metric.upper()}: {per_year_metrics[key]:.4f}")

    # Save results
    actual_test_years = set()
    for key in per_year_metrics.keys():
        if key.endswith('_overall'):
            continue
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            actual_test_years.add(int(parts[1]))

    metrics_save_dir = os.path.join(args.save_checkpoint_dir,
                                     f'{args.model_type}_{args.country}_{args.crop}_{run_id}_metrics')
    config.results_dir = metrics_save_dir

    save_test_results_to_csv(
        config=config,
        test_results=per_year_metrics,
        test_years=sorted(actual_test_years),
        run_id=run_id,
        timestamp=timestamp,
        uncertainty_metrics=None,
    )

    # Compute spatiotemporal metrics
    print(f"\n[Spatiotemporal Metrics] Computing spatial, temporal, and anomaly correlations...")
    try:
        test_preds = []
        test_targets = []
        test_years_list = []
        test_regions = []

        for batch_idx, batch in enumerate(dm.test_dataloader()):
            X, y, years, adm_ids, lats, lons, _ = batch
            pred_dict = model.predict(batch)
            preds = pred_dict['predictions'].detach().cpu().numpy().flatten()
            targets = pred_dict['targets'].detach().cpu().numpy().flatten()

            batch_years = years.numpy().flatten() if hasattr(years, 'numpy') else years
            batch_regions = adm_ids.numpy().flatten().tolist() if hasattr(adm_ids, 'numpy') else list(adm_ids)

            test_preds.extend(preds)
            test_targets.extend(targets)
            test_years_list.extend(batch_years)
            test_regions.extend(batch_regions)

        test_preds = np.array(test_preds)
        test_targets = np.array(test_targets)
        test_years_list = np.array(test_years_list)
        test_regions = np.array(test_regions)

        spatiotemporal_results = compute_all_spatiotemporal_metrics(
            y_true=test_targets,
            y_pred=test_preds,
            years=test_years_list,
            regions=test_regions
        )

        print(f"\n[Spatiotemporal Metrics] Summary:")
        print(f"  Spatial (r_sp): {spatiotemporal_results['spatial']['r_sp_overall']:.4f}")
        print(f"  Temporal (r_tm): {spatiotemporal_results['temporal']['r_tm_overall']:.4f}")
        print(f"  Anomaly (r_an): {spatiotemporal_results['anomaly']['r_an_overall']:.4f}")

        save_spatiotemporal_metrics(
            metrics=spatiotemporal_results,
            save_dir=metrics_save_dir,
            run_id=run_id,
            timestamp=timestamp
        )

    except Exception as e:
        print(f"[Spatiotemporal Metrics] Warning: Could not compute: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SPLIT SUMMARY: {args.crop}-{args.country}")
    print(f"{'=' * 70}")
    print(f"Available years ({len(all_years)}): {all_years}")
    print(f"Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    print_metrics_table(
        f"FINAL RESULTS: {args.crop}-{args.country}",
        final_metrics
    )

    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country}")
    print(f"Model: {args.model_type}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Test years: {fixed_splits['test_years']}")
    print(f"{'=' * 70}\n")
