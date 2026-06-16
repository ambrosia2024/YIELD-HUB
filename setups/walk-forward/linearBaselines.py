# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: A linear architecture based in-season and end-of-season crop yield prediction script that trains state-of-the-art time architectures 
            with agricultural domain knowledge. The training-works based on walk forward method. 
Python version: 3.12.0
--------------------
Architecture overview: This script implements unified linear baseline architectures that serve as
strong references for evaluating transformer complexity:

    • NLinear: Simple linear layer with last-value normalization (https://arxiv.org/abs/2205.13504)
    • DLinear: Decomposed linear (trend + remainder) (https://arxiv.org/abs/2205.13504)
    • XLinear: Linear with exogenous variable handling (https://arxiv.org/pdf/2305.10721)
    • RLinear: NLinear with RevIN (Reversible Instance Normalization) (https://arxiv.org/abs/2403.14587)

------------
Pipeline: The script processes agricultural data through multiple stages:

1. INPUT FEATURES
   - Weather: tmin, tmax, tavg, precipitation, radiation, (optional: cwb)
   - Remote Sensing: NDVI, FPAR, SSM, RSM
   - Soil Properties: Available water capacity, organic carbon, pH, texture
   - Location: Country, state, latitude, longitude
   - Crop Calendar: Start/end of season with cyclic encoding (sin/cos)
   - Temporal Encoding: Fourier features (sin/cos of day-of-year, month)
   - Historical Lags: Yield from previous years (1-2 years, configurable)

2. TEMPORAL AGGREGATION
   - daily:   365 time steps (raw daily data)
   - weekly:  52 time steps  (weekly averages, Monday-start)
   - dekad:   36 time steps  (10-day periods, standard in ag monitoring)

3. GROWTH-STAGE PROCESSING
   Weather data is masked to only include observations between crop start-of-season
   (SOS) and end-of-season (EOS) dates, ensuring the model focuses on the growth
   period.

4. NORMALIZATION
   - Time series: Per-feature min-max scaling
   - Static features: Mean-centering and scaling
   - Targets: Normalized to zero mean, unit variance

5. IN-SEASON vs END-SEASON predictions
   – --forecast_type: When to make the prediction (end-of-season, three-quarter-of-season,
                      middle-of-season, quarter-of-season).
   Controls what portion of the season is observed before forecasting.
----------------
Other optional/advanced features:

1. RESIDUAL TREND MODELING (--use_residual_trend)
   Uses Mann-Kendall trend detection to identify significant linear trends in
   training yields, then models residuals (yield - trend) to improve forecasting
   for datasets with strong yield progression over time.

2. RECURSIVE LAG PREDICTION (--use_recursive_lags)
   For true out-of-sample testing: uses predicted yields as lag features during
   test set evaluation instead of ground truth, preventing data leakage.

3. SPATIAL FEATURES (--include_spatial_features)
   Adds explicit latitude/longitude as static features (beyond location embeddings).

4. FEATURE ABLATION TOGGLES
   --use_cwb_feature: Include crop water balance (redundant with prec+temp)
   --drop_tavg: Drop average temperature if dataset computes it as (tmin+tmax)/2
   --use_gdd : Adds cumulative GDD as a time series channel
   --use_heat_stress_days: Adds heat/frost/dry stress day counts as static features
   --use_rue: Adds RUE (Radiation Use Efficiency) index as a time series channel
   --use_farquhar: Adds Farquhar photosynthesis proxy as a time series channel

-------------
Training workflow:
1. Data module handles train/val/test splits and normalization
2. Lightning trainer manages GPU distribution, mixed precision, checkpoints
3. Early stopping on validation loss with patience monitoring
4. Walk-forward validation with expanding window
5. WandB logging tracks metrics, hyperparameters, and artifacts

Evaluation metrics:
    • MSE, MAE, RMSE: Standard error metrics
    • R²: Coefficient of determination
    • MAPE, SMAPE: Percentage-based error metrics
    • NRMSE: Normalized RMSE (test set only)

------
Output generated:
    Checkpoints: Saved to checkpoints/ with descriptive filenames
    Results CSV: Detailed predictions with actuals, errors, metadata
    WandB: Full experiment tracking with metrics, parameters, artifacts

--------------
Usage:
# Basic walk-forward with NLinear
    python linearBaselines.py --crop maize --country NL --model_type nlinear --epochs 50 --aggregation daily

# Use all SOTA features (Fourier encoding + residual trend + recursive lags)
    python linearBaselines.py --crop maize --country NL --model_type xlinear --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily

# Quick test run (5 epochs)
    python linearBaselines.py --crop wheat --country NL --model_type nlinear --epochs 2 --aggregation daily --test_years 2 --lag_years 0 --results_dir checkpoints-test/results --save_checkpoint_dir checkpoints-test/results --wandb_project test-and-delete-later --use_exponential_weighting --exponential_tau 3 --multi_year_summaries --multi_year_window 1 --multi_year_features all

--------------------
Hyperparameters:
    --lr:              Learning rate (default: 1e-4)
    --weight_decay:    L2 regularization (default: 1e-5)
    --batch_size:      Training batch size (default: 16)
    --lag_years:       Historical yield lags (1 or 2, default: 1)
    --aggregation:     Temporal resolution (daily/weekly/dekad, default: dekad)
    --seed:            Random seed for reproducibility (default: 42)
    --use_revin:       Enable RevIN normalization for XLinear (default: False)
    --forecast_type:   When to make the prediction (default: end-of-season)

------------
Core dependencies:
    - torch>=2.0: PyTorch for model implementation
    - lightning: PyTorch Lightning for training framework
    - torchmetrics: Evaluation metrics
    - wandb: Experiment tracking
    - pymannkendall: Trend detection for residual modeling
    - pandas, numpy: Data manipulation
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

# Configure module-level logger BEFORE any imports that might use it
logger = logging.getLogger(__name__)

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# CY-BENCH Dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_TYPE, set_forecast_type, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Important: The original cybench alignment file doesn't handle for ex:- "end-of-season" lead_time. 
# Since I wanted the forecast_type to be a categorical value between 'end-of-season', 'three-quarter-of-season', 'middle-of-season', and 'quarter-of-season'
# It is important to set FORECAST_LEAD_TIME to 0-days to load full season data, and then trim it after.
cybench.config.FORECAST_LEAD_TIME = "0-days"

# Apply the alignment patch beofre importing datasets 
from cybench.process.alignment_patch import patch_alignment
patch_alignment()

# Import the datasets (after patching is in place)
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

# Loading custom functions and classes
sys.path.append('../../process/')
from helpers import generate_checkpoint_name, save_test_results_to_csv
from validateModel import print_metrics_table, run_walk_forward_validation
from loadData import calculate_fixed_split, DailyCYBenchSeqDataModule
from alignment_patch import verify_forecast_horizon_config

sys.path.append('../../architectures/')
from modelconfig import LinearModelConfig
from linearLayer import create_model

# Set matmul precision conditionally based on GPU capability
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision('high')
        logger.info(f"Enabled high matmul precision (GPU capability {capability})")
    else:
        logger.info(f"Keeping default matmul precision (GPU capability {capability} < 8.0)")
else:
    logger.info("Running on CPU, matmul precision setting has no effect")

# Main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-BENCH Time Series Yield Forecasting with Linear Baseline Models and Walk-Forward Validation")
    parser.add_argument('--crop', default="maize")
    parser.add_argument('--country', default="NL")
    parser.add_argument('--model_type', default="nlinear",
                        choices=['nlinear', 'dlinear', 'xlinear', 'rlinear', 'olinear'])
    parser.add_argument('--aggregation', default="dekad",
                        choices=['daily', 'weekly', 'dekad'])
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--lag_years', type=int, default=1, choices=[0, 1, 2],
                        help='Number of lagged yield years (max 2, default: 1)')
    parser.add_argument('--load_checkpoint', default=None,
                        help='Path to checkpoint to load for fine-tuning')
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-linear',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', default='checkpoints/results',
                        help='Directory to save CSV results (default: checkpoints/results/)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_residual_trend', action='store_true')
    parser.add_argument('--use_recursive_lags', action='store_true',
                        help='Use predicted yields as lags during testing (true out-of-sample)')
    # Domain feature engineering flags
    parser.add_argument('--use_gdd', action='store_true',
                        help='Add cumulative GDD as a time series channel. '
                             'Uses crop-specific base/upper thresholds from cybench.config. '
                             'GDD = max(min(Tavg, Tupper) - Tbase, 0), then cumsum.')
    parser.add_argument('--use_heat_stress_days', action='store_true',
                        help='Add heat/frost/dry stress day counts as static features. '
                             'Captures threshold exceedance events missed by averages.')
    parser.add_argument('--use_rue', action='store_true',
                        help='Add RUE (Radiation Use Efficiency) index as a time series channel. '
                             'RUE = cumPAR * T_stress * W_stress. Experimental.')
    parser.add_argument('--use_farquhar', action='store_true',
                        help='Add Farquhar photosynthesis proxy as a time series channel. '
                             'Based on FvCB C3 model. Seasonal-scale approximation only.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Default: auto-calculated as min(cpu_count//4, 8). '
                             'For 3 concurrent scripts, this balances CPU usage. Set manually to override.')
    parser.add_argument('--test_years', type=int, default=3,
                        help='Number of years for final test set (default: 3)')
    # Feature configuration flags
    parser.add_argument('--use_cwb_feature', action='store_true',
                        help='Include crop water balance (cwb) as a feature')
    parser.add_argument('--drop_tavg', action='store_true',
                        help='Drop tavg feature')
    parser.add_argument('--use_revin', action='store_true',
                        help='Use RevIN normalization for XLinear endogenous series')
    parser.add_argument('--lr_decay_every', type=int, default=None,
                        help='Decay learning rate by half every N epochs (default: None, no decay)')
    parser.add_argument('--wandb_project', default=None,
                        help='Custom WandB project name (default: CYBENCH-LSTF-AAAI2027-new)')
    parser.add_argument('--wandb_run_name', default=None,
                        help='Custom WandB run name (default: model_type-crop-country)')
    parser.add_argument('--run_id', default=None,
                        help='Custom run ID for checkpoint naming and results tracking (default: auto-generated UUID)')
    parser.add_argument('--forecast_type', default="end-of-season",
                        choices=['end-of-season', 'three-quarter-of-season', 'middle-of-season',
                                 'quarter-of-season'],
                        help='When to make the prediction (default: end-of-season). '
                             'Controls what portion of the season is observed before forecasting: '
                             'end-of-season (100%%), three-quarter-of-season (75%%), '
                             'middle-of-season (50%%), quarter-of-season (25%%).')
    # Exponential sample weighting for non-stationarity
    parser.add_argument('--use_exponential_weighting', action='store_true',
                        help='Enable exponential sample weighting based on year distance. '
                             'Recent samples get higher weight: weight = exp(-(current_year - sample_year) / tau). '
                             'Helps model focus on recent patterns when feature-yield relationships shift over time.')
    parser.add_argument('--exponential_tau', type=float, default=10.0,
                        help='Decay constant (tau) for exponential weighting (default: 10.0). '
                             'Higher values = slower decay (more uniform weighting). '
                             'Examples: tau=5 gives 2023=1.0, 2020=0.55, 2015=0.25; '
                             'tau=10 gives 2023=1.0, 2020=0.74, 2015=0.55.')
    # Multi-year context features
    parser.add_argument('--multi_year_summaries', action='store_true',
        help='Enable multi-year summary features from previous growing seasons')
    parser.add_argument('--multi_year_window', type=int, default=1, choices=[1, 2, 3],
        help='Years of historical context (1=T-1, 2=T-1,T-2, 3=T-1,T-2,T-3). Default: 1')
    parser.add_argument('--multi_year_features', nargs='+', default=['weather'],
        choices=['weather', 'remote_sensing', 'phenology', 'all'],
        help='Which features to summarize from previous years. '
             'weather=temp/precip/gdd, remote_sensing=NDVI/FPAR/SSM, phenology=crop calendar changes')
    # XLinear-specific hyperparameters (only used when model_type='xlinear')
    parser.add_argument('--xlinear_hidden_size', type=int, default=64,
                        help='XLinear: dimension of hidden embeddings for all linear layers (default: 64)')
    parser.add_argument('--xlinear_temporal_ff', type=int, default=128,
                        help='XLinear: feed-forward dimension in the Time-wise Gating Module (default: 128)')
    parser.add_argument('--xlinear_channel_ff', type=int, default=16,
                        help='XLinear: feed-forward dimension in the Variate-wise Gating Module (default: 16)')
    parser.add_argument('--xlinear_dropout', type=float, default=0.1,
                        help='XLinear: dropout probability for regularization (default: 0.1)')
    args = parser.parse_args()

    # The original alignment.py in cybench repo only supports "middle-of-season", "quarter-of-season", and "N-days" predictions. Since, we wanted to have "middle-of-season", "quarter-of-season", "end-of-season" and "three-quarter-of-season", we set lead_time to "0-days" which makes alignment.py load
    # the full season (SOS to EOS). The actual forecast timing is then controlled via data_fraction parameter below during feature building.
    set_forecast_type("0-days")
    print(f"[Forecast Type] {args.forecast_type}")

    # Map forecast_type to data_fraction (portion of season data to use)
    forecast_to_fraction = {
        'end-of-season': 1.0,           # 100% of season observed
        'three-quarter-of-season': 0.75, # 75% of season observed
        'middle-of-season': 0.5,        # 50% of season observed
        'quarter-of-season': 0.25,      # 25% of season observed
    }
    data_fraction = forecast_to_fraction[args.forecast_type]
    print(f"[Data Fraction] Using {data_fraction:.0%} of season data (from SOS to EOS)")

    # Set num_workers if not specified
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 1
        # For 3 concurrent scripts: divide by 4, cap at 8 for balance – gives good parallelism without overflooding the memory of the system
        args.num_workers = min(cpu_count // 4, 8)
        print(f"[Auto-config] Setting num_workers={args.num_workers} based on {cpu_count} CPU cores")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate unique run identifier and timestamp for CSV tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())[:8]  # Use provided run_id or generate short UUID

    print(f"\n{'=' * 70}")
    print(f"CY-BENCH WALK-FORWARD  |  {args.model_type.upper()}  |  {args.crop}-{args.country}  "
          f"|  {args.aggregation.upper()}")
    print(f"SOTA={args.use_sota_features}  Spatial={args.include_spatial_features}  "
          f"Lag={args.lag_years}  RevIN={args.use_revin}")
    print(f"RecursiveLags={args.use_recursive_lags}  ResidualTrend={args.use_residual_trend}")
    print(f"Domain features: GDD={args.use_gdd}  HeatStress={args.use_heat_stress_days}  "
          f"RUE={args.use_rue}  Farquhar={args.use_farquhar}")
    print(f"TestYears={args.test_years}")
    print(f"Exponential Weighting: {args.use_exponential_weighting} (tau={args.exponential_tau})")
    print(f"Multi-Year Summaries: {args.multi_year_summaries} (window={args.multi_year_window}, features={args.multi_year_features})")
    print(f"lr={args.lr}  wd={args.weight_decay}  epochs={args.epochs}  "
          f"batch={args.batch_size}  seed={args.seed}")
    print(f"{'=' * 70}\n")

    # Create LR scheduler lambda if requested
    lr_scheduler_lambda = None
    if args.lr_decay_every is not None:
        def lr_scheduler_lambda(epoch):
            decay_factor = args.lr_decay_every
            decay_steps = epoch // decay_factor
            return 0.5 ** decay_steps

    config = LinearModelConfig(
        crop=args.crop, country=args.country,
        model_type=args.model_type, aggregation=args.aggregation,
        data_fraction=data_fraction,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        lag_years=args.lag_years,
        load_checkpoint=args.load_checkpoint,
        seed=args.seed, batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        test_years=args.test_years,
        use_residual_trend=args.use_residual_trend,
        use_recursive_lags=args.use_recursive_lags,
        use_gdd=args.use_gdd,
        use_heat_stress_days=args.use_heat_stress_days,
        use_rue=args.use_rue,
        use_farquhar=args.use_farquhar,
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        use_revin=args.use_revin,
        use_exponential_weighting=args.use_exponential_weighting,
        exponential_tau=args.exponential_tau,
        multi_year_summaries=args.multi_year_summaries,
        multi_year_window=args.multi_year_window,
        multi_year_features=args.multi_year_features,
        results_dir=args.results_dir,
        lr_scheduler_lambda=lr_scheduler_lambda,
        xlinear_hidden_size=args.xlinear_hidden_size,
        xlinear_temporal_ff=args.xlinear_temporal_ff,
        xlinear_channel_ff=args.xlinear_channel_ff,
        xlinear_dropout=args.xlinear_dropout,
    )

    # Show forecast horizon configuration
    verify_forecast_horizon_config(config)

    print(f"[Feature Config] Weather features: {config.weather_features}")
    print(f"[Feature Config] Total time series vars ({len(config.time_series_vars)}): {config.time_series_vars}")

    if config.use_recursive_lags and config.lag_years > 0:
        print(f"\n{'!' * 70}")
        print(f"[RECURSIVE LAGS ENABLED]")
        print(f"During testing, model predictions will be used as lag features")
        print(f"instead of observed (actual) historical yields.")
        print(f"This provides true out-of-sample evaluation with error accumulation.")
        print(f"{'!' * 70}\n")

    # Create checkpoint directory
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    print(f"\n[Checkpoint Config]")
    print(f"Save directory: {args.save_checkpoint_dir}")
    if args.load_checkpoint:
        print(f"Load checkpoint: {args.load_checkpoint}")

    # Get available years
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)

    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # WandB logger setup for walk-forward
    try:
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-LSTF-AAAI2027-new"
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model_type}-{args.crop}-{args.country}-wf"
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args),
            group=f"{args.crop}-{args.country}-wf"
        )
        loggers = [wandb_logger]
    except Exception as e:
        print(f"[WandB Warning] Could not initialise WandB logger: {e}")
        loggers = [CSVLogger("logs/", name="cybench-linear-wf")]

    # Run walk-forward validation with all future years testing
    wf_results = run_walk_forward_validation(
        all_years=all_years,
        test_years=args.test_years,
        config=config,
        create_model_fn=create_model,
        datamodule_class=DailyCYBenchSeqDataModule,
        trainer_class=Trainer,
        max_epochs=args.epochs,
        loggers=loggers,
        run_id=run_id,
        timestamp=timestamp,
        save_checkpoint_dir=args.save_checkpoint_dir,
        save_top_k=1,
        save_last=False
    )

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {wf_results['csv_path']}")

    # DEBUG: Print per-year breakdown from aggregated results
    if 'per_year_metrics' in wf_results:
        print(f"\n[DEBUG] Per-year metrics (from all folds):")
        for year in sorted(wf_results['per_year_metrics'].keys()):
            r2 = wf_results['per_year_metrics'][year].get('r2')
            nrmse = wf_results['per_year_metrics'][year].get('nrmse')
            print(f"Year {year}: R2={r2:.4f}, NRMSE={nrmse:.4f}")

    # Print average performance of 1st test year only (using pre-calculated metrics)
    if wf_results.get('first_year_metrics'):
        print(f"\n{'=' * 70}")
        print(f"AVERAGE PERFORMANCE - 1ST TEST YEAR ONLY")
        print(f"{'=' * 70}")
        first_year_metrics = wf_results['first_year_metrics']
        for metric_name in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']:
            value = first_year_metrics.get(metric_name)
            std_val = first_year_metrics.get(f'{metric_name}_std', 0.0)
            if value is not None:
                if metric_name in ['mape', 'smape']:
                    print(f"{metric_name.upper()}: {value:.2f}% (+/- {std_val:.2f}%)")
                else:
                    print(f"{metric_name.upper()}: {value:.4f} (+/- {std_val:.4f})")
        print(f"{'=' * 70}\n")

    # Print experiment completion message
    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country}")
    print(f"Model: {args.model_type}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Test years: {args.test_years}")
    print(f"{'=' * 70}\n")