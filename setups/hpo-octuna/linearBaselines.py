# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Optuna-based Hyperparameter Optimization applied on time-series transformers architectures.
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
                      middle-of-season, quarter-of-season, 60-days, 90-days, 120-days).

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
4. Model checkpointing saves best model based on validation loss
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
    HPO Results: Text file and CSV with best hyperparameters

--------------
Usage:
# Basic HPO with NLinear
    python linearBaselines.py --crop maize --country NL --model_type nlinear --n_trials 10 --epochs 3 --hpo_objective multi

# HPO with XLinear (includes model-specific hyperparameters)
    python linearBaselines.py --crop maize --country NL --model_type xlinear --n_trials 50 --epochs 5 --hpo_objective nrmse --hpo_results_file checkpoints-test/results/HPO/octuna_file.txt --hpo_study_name test-and-delete-later

# Quick test run (2 trials)
    python linearBaselines.py --crop wheat --country NL --model_type nlinear --n_trials 4 --epochs 2 --results_dir checkpoints-test/results --forecast_type middle-of-season --aggregation daily

--------------------
Hyperparameters:
    --lr:              Learning rate (default: 1e-4)
    --weight_decay:    L2 regularization (default: 1e-5)
    --batch_size:      Training batch size (default: 16)
    --lag_years:       Historical yield lags (1 or 2, default: 1)
    --aggregation:     Temporal resolution (daily/weekly/dekad, default: dekad)
    --seed:            Random seed for reproducibility (default: 42)
    --data_fraction:   Season length fed to the model (default: 1 – full season)
    --use_revin:       Enable RevIN normalization for XLinear (default: False)


------------
Core dependencies:
    - torch>=2.0: PyTorch for model implementation
    - lightning: PyTorch Lightning for training framework
    - torchmetrics: Evaluation metrics
    - optuna: Hyperparameter optimization framework
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
import optuna

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
from validateModel import print_metrics_table
from loadData import calculate_fixed_split, DailyCYBenchSeqDataModule
from hpoOptuna import print_best_results, save_results_to_file, save_best_params_to_csv, run_hpo
from alignment_patch import verify_forecast_horizon_config

# Feature caching for HPO (on-demand only)
from featureCache import (
    get_global_cache, reset_global_cache
)

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

# Boilerplate code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-BENCH Time Series Yield Forecasting with Linear Baseline Models and HPO")
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
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-linear',
                        help='Directory to save model checkpoints')
    # XLinear-specific hyperparameters (only used when model_type='xlinear')
    parser.add_argument('--xlinear_hidden_size', type=int, default=64,
                        help='XLinear: dimension of hidden embeddings for all linear layers (default: 64)')
    parser.add_argument('--xlinear_temporal_ff', type=int, default=128,
                        help='XLinear: feed-forward dimension in the Time-wise Gating Module (default: 128)')
    parser.add_argument('--xlinear_channel_ff', type=int, default=16,
                        help='XLinear: feed-forward dimension in the Variate-wise Gating Module (default: 16)')
    parser.add_argument('--xlinear_dropout', type=float, default=0.1,
                        help='XLinear: dropout probability for regularization (default: 0.1)')
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
    # Optuna HPO arguments
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--hpo_storage', type=str, default=None,
                        help='Optuna storage URL for distributed optimization (e.g., sqlite:///optuna.db)')
    parser.add_argument('--hpo_study_name', type=str, default=None,
                        help='Optuna study name (default: auto-generated)')
    parser.add_argument('--hpo_results_file', type=str, default=None,
                        help='Path to save HPO results text file (default: auto-generated in results_dir)')
    parser.add_argument('--hpo_objective', type=str, default='nrmse',
                        choices=['nrmse', 'r2', 'multi'],
                        help='Optimization objective: nrmse (minimize), r2 (maximize), or multi (both)')
    # Removed: --optimize parameter (everything is optimized together)
    args = parser.parse_args()

    # Auto-enable use_recursive_lags when lag_years > 0
    # This ensures true out-of-sample testing when using historical lag features
    if args.lag_years > 0 and not args.use_recursive_lags:
        args.use_recursive_lags = True
        print(f"[Auto-config] Enabled use_recursive_lags=True (lag_years={args.lag_years} > 0)")

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
    run_id = str(uuid.uuid4())[:8]  # Short unique identifier

    print(f"\n{'=' * 70}")
    print(f"CY-BENCH HPO  |  {args.model_type.upper()}  |  {args.crop}-{args.country}  "
          f"|  {args.aggregation.upper()}")
    print(f"  DataFraction={data_fraction}  SOTA={args.use_sota_features}  "
          f"Spatial={args.include_spatial_features}  Lag={args.lag_years}  RevIN={args.use_revin}")
    print(f"  RecursiveLags={args.use_recursive_lags}  ResidualTrend={args.use_residual_trend}")
    print(f"  Domain features: GDD={args.use_gdd}  HeatStress={args.use_heat_stress_days}  "
          f"RUE={args.use_rue}  Farquhar={args.use_farquhar}")
    print(f"  TestYears={args.test_years}")
    print(f"  Exponential Weighting: {args.use_exponential_weighting} (tau={args.exponential_tau})")
    print(f"  Multi-Year Summaries: {args.multi_year_summaries} (window={args.multi_year_window}, features={args.multi_year_features})")
    print(f"  Trials={args.n_trials}  Objective={args.hpo_objective}")
    print(f"{'=' * 70}\n")

    # Create LR scheduler lambda if requested
    lr_scheduler_lambda = None
    if args.lr_decay_every is not None:
        def lr_scheduler_lambda(epoch):
            decay_factor = args.lr_decay_every
            decay_steps = epoch // decay_factor
            return 0.5 ** decay_steps

    # Get available years
    df_y, dfs_x = load_dfs_crop(args.crop, [args.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {args.crop}-{args.country}")
        sys.exit(1)

    ds = CYDataset(args.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # Calculate fixed train/val/test split
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=args.test_years,
        val_years=2
    )

    print(f"\n[Split Config - Fixed]")
    print(f"  Total years: {fixed_splits['total_years']}")
    print(f"  Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"  Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"  Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # ==================== CREATE TEMP CONFIG ====================
    # Create temp config for forecast horizon verification
    temp_config = LinearModelConfig(
        crop=args.crop, country=args.country,
        model_type=args.model_type, aggregation=args.aggregation,
        data_fraction=data_fraction,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        lag_years=args.lag_years,
    )
    verify_forecast_horizon_config(temp_config)

    # ==================== OPTUNA HPO INTEGRATION ====================
    print(f"\n{'=' * 70}")
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'=' * 70}\n")

    # Auto-generate HPO results file path if not provided
    if args.hpo_results_file is None:
        hpo_results_dir = os.path.join(args.results_dir, "HPO")
        os.makedirs(hpo_results_dir, exist_ok=True)
        hpo_results_file = os.path.join(hpo_results_dir,
            f"{args.model_type}_{args.crop}_{args.country}_HPO_results_{timestamp}.txt")
    else:
        hpo_results_file = args.hpo_results_file
        # Create directory if it doesn't exist
        hpo_results_dir = os.path.dirname(hpo_results_file)
        if hpo_results_dir:  # Only create if there's a directory component
            os.makedirs(hpo_results_dir, exist_ok=True)

    # Auto-generate study name if not provided
    if args.hpo_study_name is None:
        study_name = f"{args.model_type}_{args.crop}_{args.country}_{args.aggregation}"
    else:
        study_name = args.hpo_study_name

    print(f"[HPO Config]")
    print(f"  Objective: {args.hpo_objective}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Study name: {study_name}")
    print(f"  Results file: {hpo_results_file}")
    print(f"  Storage: {args.hpo_storage if args.hpo_storage else 'In-memory'}")

    def optuna_objective(trial):
        """Optuna objective function for hyperparameter optimization"""

        # === FEATURE FLAGS ===
        use_gdd_hp = trial.suggest_categorical('use_gdd', [False, True])
        use_heat_stress_days_hp = trial.suggest_categorical('use_heat_stress_days', [False, True])
        use_rue_hp = trial.suggest_categorical('use_rue', [False, True])
        use_farquhar_hp = trial.suggest_categorical('use_farquhar', [False, True])
        use_sota_features_hp = trial.suggest_categorical('use_sota_features', [False, True])
        include_spatial_features_hp = trial.suggest_categorical('include_spatial_features', [False, True])
        use_residual_trend_hp = trial.suggest_categorical('use_residual_trend', [False, True])
        use_cwb_feature_hp = trial.suggest_categorical('use_cwb_feature', [False, True])
        drop_tavg_hp = trial.suggest_categorical('drop_tavg', [False, True])
        lag_years_hp = trial.suggest_categorical('lag_years', [0, 1, 2, 3])

        # use_recursive_lags is only applicable when lag_years > 0
        # We'll handle this after suggesting lag_years

        use_exponential_weighting_hp = trial.suggest_categorical('use_exponential_weighting', [False, True])
        exponential_tau_hp = trial.suggest_categorical('exponential_tau', [0, 10, 25, 50, 75, 100])
        multi_year_summaries_hp = trial.suggest_categorical('multi_year_summaries', [False, True])
        multi_year_window_hp = trial.suggest_categorical('multi_year_window', [1, 2, 3])
        multi_year_features_hp = trial.suggest_categorical('multi_year_features', ['weather', 'remote_sensing', 'phenology', 'all'])

        # For xlinear models, also tune use_revin
        if args.model_type == 'xlinear':
            use_revin_hp = trial.suggest_categorical('use_revin', [False, True])
        else:
            use_revin_hp = args.use_revin

        # === MODEL HYPERPARAMETERS ===
        if args.model_type == 'xlinear':
            xlinear_hidden_size = trial.suggest_categorical('xlinear_hidden_size', [32, 64, 128, 256])
            xlinear_temporal_ff = trial.suggest_categorical('xlinear_temporal_ff', [64, 128, 256, 512])
            xlinear_channel_ff = trial.suggest_categorical('xlinear_channel_ff', [8, 16, 32, 64])
            xlinear_dropout = trial.suggest_float('xlinear_dropout', 0.0, 0.5)

            # Shared hyperparameters
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        else:
            # For other linear models
            xlinear_hidden_size = args.xlinear_hidden_size
            xlinear_temporal_ff = args.xlinear_temporal_ff
            xlinear_channel_ff = args.xlinear_channel_ff
            xlinear_dropout = args.xlinear_dropout

            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

        # FIXED: seed is never tuned - always use the command-line arg
        seed = args.seed

        # Handle use_recursive_lags based on lag_years_hp
        # Only enable when lag_years > 0, otherwise False
        if lag_years_hp > 0:
            use_recursive_lags_hp = trial.suggest_categorical('use_recursive_lags', [False, True])
        else:
            use_recursive_lags_hp = False

        # Create config with suggested parameters
        if args.model_type == 'xlinear':
            hpo_config = LinearModelConfig(
                crop=args.crop, country=args.country,
                model_type=args.model_type, aggregation=args.aggregation,
                data_fraction=data_fraction,
                use_sota_features=use_sota_features_hp,
                include_spatial_features=include_spatial_features_hp,
                lag_years=lag_years_hp,
                load_checkpoint=None,  # Never load checkpoint during HPO
                seed=seed, batch_size=batch_size,
                num_workers=args.num_workers,
                max_epochs=args.epochs, lr=lr, weight_decay=weight_decay,
                test_years=args.test_years,
                use_residual_trend=use_residual_trend_hp,
                use_recursive_lags=use_recursive_lags_hp,
                use_gdd=use_gdd_hp,
                use_heat_stress_days=use_heat_stress_days_hp,
                use_rue=use_rue_hp,
                use_farquhar=use_farquhar_hp,
                use_cwb_feature=use_cwb_feature_hp,
                drop_tavg=drop_tavg_hp,
                use_revin=use_revin_hp,
                use_exponential_weighting=use_exponential_weighting_hp,
                exponential_tau=exponential_tau_hp,
                multi_year_summaries=multi_year_summaries_hp,
                multi_year_window=multi_year_window_hp,
                multi_year_features=[multi_year_features_hp],
                results_dir=args.results_dir,
                lr_scheduler_lambda=lr_scheduler_lambda,
                xlinear_hidden_size=xlinear_hidden_size,
                xlinear_temporal_ff=xlinear_temporal_ff,
                xlinear_channel_ff=xlinear_channel_ff,
                xlinear_dropout=xlinear_dropout,
                use_parallel=True,  # Enable parallel feature building for HPO speedup
            )
        else:
            # For other linear models
            hpo_config = LinearModelConfig(
                crop=args.crop, country=args.country,
                model_type=args.model_type, aggregation=args.aggregation,
                data_fraction=data_fraction,
                use_sota_features=use_sota_features_hp,
                include_spatial_features=include_spatial_features_hp,
                lag_years=lag_years_hp,
                load_checkpoint=None,
                seed=seed, batch_size=batch_size,
                num_workers=args.num_workers,
                max_epochs=args.epochs, lr=lr, weight_decay=weight_decay,
                test_years=args.test_years,
                use_residual_trend=use_residual_trend_hp,
                use_recursive_lags=use_recursive_lags_hp,
                use_gdd=use_gdd_hp,
                use_heat_stress_days=use_heat_stress_days_hp,
                use_rue=use_rue_hp,
                use_farquhar=use_farquhar_hp,
                use_cwb_feature=use_cwb_feature_hp,
                drop_tavg=drop_tavg_hp,
                use_revin=use_revin_hp,
                use_exponential_weighting=use_exponential_weighting_hp,
                exponential_tau=exponential_tau_hp,
                multi_year_summaries=multi_year_summaries_hp,
                multi_year_window=multi_year_window_hp,
                multi_year_features=[multi_year_features_hp],
                results_dir=args.results_dir,
                lr_scheduler_lambda=lr_scheduler_lambda,
                xlinear_hidden_size=xlinear_hidden_size,
                xlinear_temporal_ff=xlinear_temporal_ff,
                xlinear_channel_ff=xlinear_channel_ff,
                xlinear_dropout=xlinear_dropout,
                use_parallel=True,  # Enable parallel feature building for HPO speedup
            )

        # Create datamodule
        dm = DailyCYBenchSeqDataModule(hpo_config)
        dm.setup(
            train_years=fixed_splits['train_years'],
            val_years=fixed_splits['val_years'],
            test_years=fixed_splits['test_years']
        )

        # Create model
        model = create_model(hpo_config)

        # Setup callbacks for HPO (faster training)
        hpo_callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=False),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        # CSV logger for each trial
        trial_logger = CSVLogger("logs/", name=f"cybench-hpo-{trial.number}")

        # Create trainer
        trainer = Trainer(
            max_epochs=hpo_config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=hpo_callbacks,
            logger=[trial_logger],
            log_every_n_steps=50,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )

        # Train
        print(f"\n[Trial {trial.number}] Training with hyperparameters...")
        try:
            trainer.fit(model, dm)

            # Get validation metrics
            val_metrics = trainer.validate(model, dm)
            if val_metrics:
                val_results = val_metrics[0]

                # Handle different objectives
                if args.hpo_objective == 'nrmse':
                    return val_results.get('val/nrmse', float('inf'))
                elif args.hpo_objective == 'r2':
                    return val_results.get('val/r2', -float('inf'))  # Return r2 directly for maximization
                elif args.hpo_objective == 'multi':
                    # Multi-objective: return tuple (nrmse, r2)
                    nrmse = val_results.get('val/nrmse', float('inf'))
                    r2 = val_results.get('val/r2', -float('inf'))
                    return nrmse, r2  # Return r2 directly for maximization
            else:
                if args.hpo_objective == 'multi':
                    return float('inf'), float('inf')
                else:
                    return float('inf')
        except Exception as e:
            print(f"[Trial {trial.number}] Failed: {e}")
            if args.hpo_objective == 'multi':
                return float('inf'), float('inf')
            else:
                return float('inf')

    # Enqueue default baseline configuration for comparison
    baseline_params = {
        'use_gdd': args.use_gdd,
        'use_heat_stress_days': args.use_heat_stress_days,
        'use_rue': args.use_rue,
        'use_farquhar': args.use_farquhar,
        'use_sota_features': args.use_sota_features,
        'include_spatial_features': args.include_spatial_features,
        'use_residual_trend': args.use_residual_trend,
        'use_cwb_feature': args.use_cwb_feature,
        'drop_tavg': args.drop_tavg,
        'lag_years': args.lag_years,
        'use_exponential_weighting': args.use_exponential_weighting,
        'exponential_tau': int(args.exponential_tau),
        'multi_year_summaries': args.multi_year_summaries,
        'multi_year_window': args.multi_year_window,
        'multi_year_features': args.multi_year_features[0] if args.multi_year_features else 'weather',
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 1e-5,
    }

    # Add model-specific hyperparameters
    if args.model_type == 'xlinear':
        baseline_params.update({
            'xlinear_hidden_size': 64,
            'xlinear_temporal_ff': 128,
            'xlinear_channel_ff': 16,
            'xlinear_dropout': 0.1,
            'use_revin': args.use_revin,
        })

    # Add use_recursive_lags only if lag_years > 0
    if args.lag_years > 0:
        baseline_params['use_recursive_lags'] = args.use_recursive_lags

    # Run optimization
    study = run_hpo(
        objective=optuna_objective,
        study_name=study_name,
        hpo_objective=args.hpo_objective,
        n_trials=args.n_trials,
        storage=args.hpo_storage,
        baseline_params=baseline_params,
    )

    # Print and save results
    print_best_results(study, args.hpo_objective)

    # Log baseline NRMSE for comparison
    baseline_trial = [t for t in study.trials if t.number == 0]
    if baseline_trial:
        baseline_nrmse = baseline_trial[0].value
        if baseline_nrmse and np.isfinite(baseline_nrmse):
            print(f"\n[Baseline] Baseline NRMSE: {baseline_nrmse:.6f}")
            # Also append to results file for permanent record
            with open(hpo_results_file, 'a') as f:
                f.write(f"\n[Baseline NRMSE for comparison] {baseline_nrmse:.6f}\n")

    # Create checkpoint directory for saving results
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    # ==================== PARAMETER EFFECT ANALYSIS ====================
    print(f"\n{'=' * 80}")
    print(f"PARAMETER EFFECT ANALYSIS (ALL PARAMETERS)")
    print(f"{'=' * 80}\n")

    try:
        # Collect ALL parameters to analyze (both features and hyperparameters)
        feature_params = {
            'use_gdd', 'use_rue', 'use_farquhar', 'use_heat_stress_days',
            'use_sota_features', 'include_spatial_features', 'use_residual_trend',
            'use_cwb_feature', 'drop_tavg',
            'lag_years', 'use_recursive_lags', 'use_exponential_weighting', 'exponential_tau',
            'multi_year_summaries', 'multi_year_window', 'multi_year_features'
        }

        # Add use_revin for xlinear models
        if args.model_type == 'xlinear':
            feature_params.add('use_revin')

        # Model hyperparameters to analyze
        model_params = {'batch_size', 'lr', 'weight_decay'}
        if args.model_type == 'xlinear':
            model_params.update({
                'xlinear_hidden_size', 'xlinear_temporal_ff',
                'xlinear_channel_ff', 'xlinear_dropout'
            })

        # Combine all parameters
        params_to_analyze = feature_params | model_params

        # Analyze each parameter
        param_results = []

        for param_name in params_to_analyze:
            # Use all trials for analysis (no dependency filtering)
            # This ensures all parameters appear in the output
            trials_to_analyze = list(study.trials)

            # Group trials by parameter value
            value_groups = {}
            for trial in trials_to_analyze:
                if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
                    value = trial.params[param_name]
                    if value not in value_groups:
                        value_groups[value] = []
                    obj_value = trial.values[0] if args.hpo_objective == 'multi' else trial.value
                    # Skip failed trials (inf or nan values)
                    if not np.isfinite(obj_value):
                        continue
                    value_groups[value].append(obj_value)

            # Note: We still analyze parameters with only 1 value to report all tuned params
            # For parameters with <2 values, we'll report them but note limited variation
            if not value_groups:
                continue  # Skip only if parameter never appeared

            # Calculate statistics for each value
            value_stats = {}
            for value, vals in value_groups.items():
                value_stats[value] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'count': len(vals)
                }

            # Find best and worst values
            best_value = min(value_stats, key=lambda v: value_stats[v]['mean'])
            worst_value = max(value_stats, key=lambda v: value_stats[v]['mean'])

            best_mean = value_stats[best_value]['mean']
            worst_mean = value_stats[worst_value]['mean']

            # Calculate effect size (negative = improvement for NRMSE)
            effect_size = best_mean - worst_mean
            baseline_nrmse = worst_mean

            # Calculate percent improvement
            pct_improvement = (effect_size / baseline_nrmse * 100) if baseline_nrmse > 0 else 0

            # Store all value statistics for detailed reporting
            all_values_stats = []
            for value, stats in value_stats.items():
                is_best = (value == best_value)
                is_worst = (value == worst_value)
                all_values_stats.append({
                    'value': value,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'count': stats['count'],
                    'is_best': is_best,
                    'is_worst': is_worst,
                })

            # Sort by mean NRMSE (ascending)
            all_values_stats.sort(key=lambda x: x['mean'])

            param_results.append({
                'model_type': args.model_type,
                'country': args.country,
                'crop': args.crop,
                'parameter': param_name,
                'best_value': best_value,
                'worst_value': worst_value,
                'best_mean': best_mean,
                'worst_mean': worst_mean,
                'effect_size': effect_size,
                'pct_improvement': pct_improvement,
                'all_values': all_values_stats,
            })

        # Sort by absolute effect size
        param_results.sort(key=lambda x: abs(x['effect_size']), reverse=True)

        # Print table
        if param_results:
            print(f"\n{'='*80}")
            print(f"PARAMETER EFFECT ANALYSIS - Absolute NRMSE Values")
            print(f"{'='*80}")
            print(f"Model: {args.model_type} | Crop: {args.crop} | Country: {args.country}")
            print(f"Objective: {args.hpo_objective} (lower NRMSE is better)")
            print(f"Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")

            for r in param_results:
                print(f"\n{r['parameter']}:")
                print(f"  {'Value':<20} {'NRMSE (mean±std)':<20} {'Trials':<10} {'Status':<10}")
                print(f"  {'-'*60}")

                for val_stat in r['all_values']:
                    # Format value display
                    if isinstance(val_stat['value'], bool):
                        val_str = f"{val_stat['value']} ({'ON' if val_stat['value'] else 'OFF'})"
                    elif isinstance(val_stat['value'], str):
                        val_str = f"'{val_stat['value']}'"
                    else:
                        val_str = str(val_stat['value'])

                    # Status indicator
                    status = ""
                    if val_stat['is_best']:
                        status = "★ BEST"
                    elif val_stat['is_worst']:
                        status = "✗ WORST"

                    print(f"  {val_str:<20} {val_stat['mean']:.4f}±{val_stat['std']:.4f}  {val_stat['count']:<10} {status:<10}")

                # Show range
                if len(r['all_values']) > 1:
                    nrmse_range = r['worst_mean'] - r['best_mean']
                    print(f"  Range: {nrmse_range:.4f} (worst {r['worst_mean']:.4f} - best {r['best_mean']:.4f})")
                else:
                    print(f"  Note: Only 1 value tested (no variation)")

            # Save to CSV for aggregation (expanded format with all values)
            csv_file = os.path.join(
                args.results_dir,
                f'{args.model_type}_{args.crop}_{args.country}_parameter_effects.csv'
            )
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'model_type', 'country', 'crop', 'parameter',
                    'value', 'value_type', 'nrmse_mean', 'nrmse_std', 'trial_count', 'is_best', 'is_worst'
                ])

                # Add baseline NRMSE from trial 0
                baseline_trial = [t for t in study.trials if t.number == 0]
                if baseline_trial:
                    baseline_nrmse = baseline_trial[0].value if args.hpo_objective != 'multi' else baseline_trial[0].values[0]
                    if baseline_nrmse and baseline_nrmse != float('inf'):
                        writer.writerow([
                            args.model_type, args.country, args.crop, 'baseline',
                            'Trial 0', 'baseline',
                            f"{baseline_nrmse:.6f}", '0', 1, False, False
                        ])

                for r in param_results:
                    for val_stat in r['all_values']:
                        # Format value for CSV
                        if isinstance(val_stat['value'], bool):
                            val_str = 'True' if val_stat['value'] else 'False'
                        else:
                            val_str = str(val_stat['value'])

                        writer.writerow([
                            r['model_type'], r['country'], r['crop'], r['parameter'],
                            val_str, type(val_stat['value']).__name__,
                            f"{val_stat['mean']:.6f}", f"{val_stat['std']:.6f}",
                            val_stat['count'], val_stat['is_best'], val_stat['is_worst']
                        ])

            print(f"\n[Parameter Effects] CSV saved to: {csv_file}")

            # Also save human-readable TXT
            txt_file = os.path.join(
                args.results_dir,
                f'{args.model_type}_{args.crop}_{args.country}_parameter_effects.txt'
            )
            with open(txt_file, 'w') as f:
                f.write(f"Parameter Effect Analysis (ALL PARAMETERS)\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"Model: {args.model_type} | Crop: {args.crop} | Country: {args.country}\n")
                f.write(f"Objective: {args.hpo_objective} (lower NRMSE is better)\n")
                f.write(f"Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n\n")

                # Add baseline NRMSE from trial 0
                baseline_trial = [t for t in study.trials if t.number == 0]
                if baseline_trial:
                    baseline_nrmse = baseline_trial[0].value if args.hpo_objective != 'multi' else baseline_trial[0].values[0]
                    if baseline_nrmse and baseline_nrmse != float('inf'):
                        f.write(f"BASELINE NRMSE (Trial 0): {baseline_nrmse:.6f}\n\n")

                for r in param_results:
                    f.write(f"\n{r['parameter']}:\n")
                    f.write(f"  {'Value':<20} {'NRMSE (mean±std)':<20} {'Trials':<10} {'Status':<10}\n")
                    f.write(f"  {'-'*60}\n")

                    for val_stat in r['all_values']:
                        # Format value display
                        if isinstance(val_stat['value'], bool):
                            val_str = f"{val_stat['value']} ({'ON' if val_stat['value'] else 'OFF'})"
                        elif isinstance(val_stat['value'], str):
                            val_str = f"'{val_stat['value']}'"
                        else:
                            val_str = str(val_stat['value'])

                        # Status indicator
                        status = ""
                        if val_stat['is_best']:
                            status = "★ BEST"
                        elif val_stat['is_worst']:
                            status = "✗ WORST"

                        f.write(f"  {val_str:<20} {val_stat['mean']:.4f}±{val_stat['std']:.4f}  {val_stat['count']:<10} {status:<10}\n")

                    # Show range
                    if len(r['all_values']) > 1:
                        nrmse_range = r['worst_mean'] - r['best_mean']
                        f.write(f"  Range: {nrmse_range:.4f} (worst {r['worst_mean']:.4f} - best {r['best_mean']:.4f})\n")
                    else:
                        f.write(f"  Note: Only 1 value tested (no variation)\n")

            print(f"[Parameter Effects] TXT saved to: {txt_file}")

        else:
            print("[Parameter Effects] No parameters were varied in this study.")

    except Exception as e:
        print(f"[Parameter Effects] Could not compute: {e}")
        import traceback
        traceback.print_exc()

    # Save results to text file
    save_results_to_file(
        study, hpo_results_file, args.model_type, args.crop, args.country,
        study_name, args.hpo_objective, timestamp
    )

    print(f"\n[HPO] Results saved to: {hpo_results_file}")

    # Save best hyperparameters to CSV files
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    save_best_params_to_csv(study, args.save_checkpoint_dir, args.hpo_objective)
    if args.hpo_objective == 'multi':
        csv_rmse_path = os.path.join(args.save_checkpoint_dir, 'optuna_rmse.csv')
        csv_r2_path = os.path.join(args.save_checkpoint_dir, 'optuna_r2.csv')
        print(f"[HPO] Best RMSE hyperparameters saved to: {csv_rmse_path}")
        print(f"[HPO] Best R² hyperparameters saved to: {csv_r2_path}")

    # Print cache statistics (if cache was used during trials)
    print(f"\n{'=' * 70}")
    print(f"FEATURE CACHE STATISTICS")
    print(f"{'=' * 70}")
    cache = get_global_cache()
    cache_stats = cache.stats()
    print(f"  Cache entries: {cache_stats['size']}")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']}")
    print(f"{'=' * 70}\n")

    # Print completion message
    print(f"\n{'=' * 70}")
    print(f"HPO Experiment complete: {args.crop}-{args.country}")
    print(f"  Model: {args.model_type}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Trials: {args.n_trials}")
    print(f"{'=' * 70}\n")