# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: A time series transformers based in-season and end-of-season crop yield prediction script that trains state-of-the-art time architectures with 
            agricultural domain knowledge. The training-works on temporal split. 
Python version: 3.12.0

--------------------
Architecture overview: This script implements a unified interface to seven transformer-based time series architectures from the HuggingFace Transformers library and nixtla implementations:
    • Autoformer: Auto-correlation mechanism for long-term dependency discovery (https://arxiv.org/abs/2106.13008)
    • PatchTST: Patch time series into sub-sequences for efficient processing (https://arxiv.org/pdf/2211.14730)
    • TSMixer: All-MLP architecture with mixing layers (via PatchTSMixer) (https://arxiv.org/pdf/2303.06053)
    • Informer: ProbSparse self-attention for long sequences (https://arxiv.org/pdf/2012.07436)
    • TST: Vanilla time series transformer with canonical attention (https://arxiv.org/pdf/2010.02803)
    • iTransformer: Inverted transformer treating channels as tokens for cross-variable dependencies (https://arxiv.org/abs/2310.06625)
    • TimeXer: Cross-attention transformer with endogenous patching and exogenous inverted embedding (https://arxiv.org/abs/2402.19072)

All models share a common base class (BaseTimeSeriesModel) ensuring consistent training, evaluation, and inference interfaces.

--------------------
Data pipeline:
The script processes agricultural data through multiple stages:

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
   Weather data is masked to only include observations between crop start-of-season (SOS) and end-of-season (EOS) dates, 
   ensuring the model focuses on the growth period.

4. NORMALIZATION
   - Time series: Per-feature min-max scaling
   - Static features: Mean-centering and scaling
   - Targets: Normalized to zero mean, unit variance

5. IN-SEASON vs END-SEASON predictions
   – --forecast_type: When to make the prediction (end-of-season, three-quarter-of-season,
                      middle-of-season, quarter-of-season, 60-days, 90-days, 120-days).

--------------------
Other optional/advanced features:

1. RESIDUAL TREND MODELING (--use_residual_trend)
   Uses Mann-Kendall trend detection to identify significant linear trends in training yields, then models residuals (yield - trend) to improve forecasting for datasets with strong yield progression over time.

2. RECURSIVE LAG PREDICTION (--use_recursive_lags)
   For true out-of-sample testing: uses predicted yields as lag features during test set evaluation instead of ground truth, preventing data leakage.

3. SPATIAL FEATURES (--include_spatial_features)
    Adds explicit latitude/longitude as static features (beyond location embeddings).

4. FEATURE ABLATION TOGGLES
   --use_cwb_feature: Include climate water balance
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
    WandB: Full experiment tracking with metrics, parameters, artifacts

--------------------
Hyperparameters:
Key hyperparameters:
    --lr:              Learning rate (default: 1e-4)
    --weight_decay:    L2 regularization (default: 1e-5)
    --batch_size:      Training batch size (default: 16)
    --lag_years:       Historical yield lags (1 or 2, default: 1)
    --aggregation:     Temporal resolution (daily/weekly/dekad, default: dekad)
    --seed:            Random seed for reproducibility (default: 42)
--------------
Usage:
# Basic training with Autoformer: python tstBaselines.py --crop maize --country NL --model_type autoformer --epochs 50 --aggregation daily

# Use all SOTA features (Fourier encoding + residual trend + recursive lags)
    python tstBaselines.py --crop maize --country NL --model_type tst --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily

# Quick test run (5 epochs)
    python tstBaselines.py --crop wheat --country NL --model_type informer --epochs 2 --aggregation daily --lag_years 0 --test_years 5 --results_dir checkpoints-test/results --save_checkpoint_dir checkpoints-test/results --wandb_project test-and-delete-later --forecast_type end-of-season --use_exponential_weighting --exponential_tau 10 --multi_year_summaries --multi_year_window 1 --multi_year_features all
    python tstBaselines.py --crop wheat --country NL --model_type informer --epochs 2 --aggregation daily --lag_years 2 --test_years 5 --results_dir checkpoints-test/results --save_checkpoint_dir checkpoints-test/results --use_recursive_lags --wandb_project test-and-delete-later --forecast_type middle-of-season --if_tokenize
    python tstBaselines.py --crop wheat --country NL --model_type informer --epochs 2 --aggregation daily --lag_years 2 --test_years 5 --results_dir checkpoints-test/results --save_checkpoint_dir checkpoints-test/results --use_recursive_lags --wandb_project test-and-delete-later --forecast_type middle-of-season --use_wfan
    
------------
Core dependencies:
    - torch>=2.0: PyTorch for model implementation
    - lightning: PyTorch Lightning for training framework
    - transformers>=4.30: Time series model architectures
    - torchmetrics: Evaluation metrics
    - wandb: Experiment tracking
    - pymannkendall: Trend detection for residual modeling
    - pandas, numpy: Data manipulation

Internal dependencies:
    - cybench.datasets: Agricultural data loading utilities
    - cybench.config: Domain constants and configurations

"""

# %% Loading libraries
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
from alignment_patch import verify_forecast_horizon_config

sys.path.append('../../architectures/')
from modelconfig import TSTModelConfig
from tstLayer import create_model

# Setting precision
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
        description="CY-BENCH Time Series Yield Forecasting with Temporal (Fixed) Split")
    parser.add_argument('--crop', default="maize")
    parser.add_argument('--country', default="NL")
    parser.add_argument('--model_type', default="autoformer",
                        choices=['autoformer', 'patchtst', 'tsmixer', 'informer', 'tst', 'itransformer', 'timexer', 'timesnet'])
    parser.add_argument('--aggregation', default="dekad",
                        choices=['daily', 'weekly', 'dekad'])
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--lag_years', type=int, default=1, choices=[0, 1, 2],
                        help='Number of lagged yield years (max 2, default: 1)')
    parser.add_argument('--use_recursive_lags', action='store_true',
                        help='Use predicted yields as lags during testing for true out-of-sample evaluation '
                             '(default: False, uses observed test-set yields as lags)')
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
    parser.add_argument('--load_checkpoint', default=None,
                        help='Path to checkpoint to load for fine-tuning')
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-test',
                        help='Directory to save model checkpoints (default: checkpoints/)')
    parser.add_argument('--results_dir', default='checkpoints/results',
                        help='Directory to save CSV results (default: checkpoints/results/)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_residual_trend', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Default: auto-calculated as min(cpu_count//4, 8). '
                             'For 3 concurrent scripts, this balances CPU usage. Set manually to override.')
    parser.add_argument('--test_years', type=int, default=3,
                        help='Number of years for final test set (default: 3)')
    # Feature configuration flags for ablation studies
    parser.add_argument('--use_cwb_feature', action='store_true',
                        help='Include crop water balance (cwb) as a feature. '
                             'Note: cwb is derived from prec and ET0 (depends on temperature), '
                             'so it may be redundant with existing weather features.')
    parser.add_argument('--drop_tavg', action='store_true',
                        help='Drop tavg feature if dataset computes it as (tmin+tmax)/2, '
                             'which carries no additional information beyond tmin/tmax.')
    parser.add_argument('--lr_decay_every', type=int, default=None,
                        help='Decay learning rate by half every N epochs (default: None, no decay)')
    parser.add_argument('--wandb_project', default=None,
                        help='Custom WandB project name (default: CYBENCH-LSTF-AAAI2027)')
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
    # PatchTST-specific hyperparameters (only used when model_type='patchtst')
    parser.add_argument('--patchtst_d_model', type=int, default=64,
                        help='PatchTST: dimension of the transformer hidden states (default: 64)')
    parser.add_argument('--patchtst_num_attention_heads', type=int, default=4,
                        help='PatchTST: number of parallel attention heads (default: 4)')
    parser.add_argument('--patchtst_ffn_dim', type=int, default=256,
                        help='PatchTST: dimension of the feed-forward network (default: 256)')
    parser.add_argument('--patchtst_num_layers', type=int, default=3,
                        help='PatchTST: number of transformer encoder layers (default: 3)')
    parser.add_argument('--patchtst_dropout', type=float, default=0.1,
                        help='PatchTST: dropout probability for regularization (default: 0.1)')
    # Tokenization ablation arguments
    parser.add_argument('--if_tokenize', action='store_true',
                        help='Enable fixed average pooling tokenization (ablation study). '
                             'Reduces sequence length via non-overlapping average pooling before feeding to model.')
    parser.add_argument('--tokenize_kernel', type=int, default=7,
                        help='Kernel size for average pooling tokenization (default: 7). '
                             'Only used when --if_tokenize is enabled.')
    parser.add_argument('--tokenize_stride', type=int, default=7,
                        help='Stride for average pooling tokenization (default: 7). '
                             'Only used when --if_tokenize is enabled.')
    # WFAN (Frequency-Adaptive Normalization) arguments
    parser.add_argument('--use_wfan', action='store_true',
                        help='Enable WFAN frequency-adaptive normalization for distribution shift mitigation.')
    parser.add_argument('--wfan_k', type=int, default=2,
                        help='Number of dominant frequency components to remove in WFAN (default: 2). '
                             'Higher values remove more frequency components as non-stationary.')
    parser.add_argument('--wfan_lambda', type=float, default=1.0,
                        help='WFAN loss balancing coefficient for pattern-adaptive prediction (default: 1.0). '
                             'Controls the weight of non-stationary prediction loss.')
    # Loss type argument
    parser.add_argument('--loss_type', default='mse', type=str, choices=['mse', 'mql'],
                        help='Loss function to use: "mse" for standard MSE loss, "mql" for Multi-Quantile Loss. '
                             'When "mql" is used, the model outputs 3 quantiles (0.1, 0.5, 0.9) and '
                             'uncertainty metrics (Pinball Loss, CRPS, PICP, PINAW, Winkler Score) are logged. '
                             'Default: "mse"')
    # Load and test trained model
    parser.add_argument('--load_and_test_trained_model', action='store_true',
                        help='If enabled, load the best trained model from checkpoint and evaluate on test set again. '
                             'This happens at the end of training. Useful for verifying model loading and getting final test metrics.')
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
    print(f"CY-BENCH | {args.model_type.upper()} | {args.crop}-{args.country} "
          f"|  {args.aggregation.upper()}")
    print(f"SOTA={args.use_sota_features}  Spatial={args.include_spatial_features} "
          f"Lag={args.lag_years}")
    print(f"Domain features: GDD={args.use_gdd}  HeatStress={args.use_heat_stress_days} "
          f"RUE={args.use_rue}  Farquhar={args.use_farquhar}")
    print(f"TestYears={args.test_years}")
    print(f"Exponential Weighting: {args.use_exponential_weighting} (tau={args.exponential_tau})")
    print(f"Multi-Year Summaries: {args.multi_year_summaries} (window={args.multi_year_window}, features={args.multi_year_features})")
    print(f"Tokenization: {args.if_tokenize} (kernel={args.tokenize_kernel}, stride={args.tokenize_stride})")
    print(f"WFAN: {args.use_wfan} (K={args.wfan_k}, λ={args.wfan_lambda})")
    print(f"lr={args.lr}  wd={args.weight_decay}  epochs={args.epochs} "
          f"batch={args.batch_size}  seed={args.seed}")
    print(f"{'=' * 70}\n")

    # Create LR scheduler lambda if enabled
    lr_scheduler_lambda = None
    if args.lr_decay_every is not None:
        def lr_scheduler_lambda(epoch):
            decay_factor = args.lr_decay_every
            decay_steps = epoch // decay_factor
            return 0.5 ** decay_steps

    config = TSTModelConfig(
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
        lr_scheduler_lambda=lr_scheduler_lambda,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_num_attention_heads=args.patchtst_num_attention_heads,
        patchtst_ffn_dim=args.patchtst_ffn_dim,
        patchtst_num_layers=args.patchtst_num_layers,
        patchtst_dropout=args.patchtst_dropout,
        if_tokenize=args.if_tokenize,
        tokenize_kernel=args.tokenize_kernel,
        tokenize_stride=args.tokenize_stride,
        use_wfan=args.use_wfan,
        wfan_k=args.wfan_k,
        wfan_lambda=args.wfan_lambda,
        loss_type=args.loss_type,
    )

    # Show forecast horizon configuration
    verify_forecast_horizon_config(config)

    print(f"[Feature Config] Weather features: {config.weather_features}")
    print(f"[Feature Config] Total time series vars ({len(config.time_series_vars)}): {config.time_series_vars}")

    if config.use_recursive_lags and config.lag_years > 0:
        print(f"\n{'=' * 70}")
        print(f"[RECURSIVE LAGS ENABLED]")
        print(f"During testing, model predictions will be used as lag features")
        print(f"instead of observed (actual) historical yields.")
        print(f"This provides true out-of-sample evaluation with error accumulation.")
        print(f"{'=' * 70}\n")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    print(f"\n[Checkpoint Config]")
    print(f"Save directory: {args.save_checkpoint_dir}")
    if args.load_checkpoint:
        print(f"Load checkpoint: {args.load_checkpoint}")

    # Get available years for this country-crop combination
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)

    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # Calculate fixed train/val/test split
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
    print(f"PHASE 3: Final Model Training and Evaluation")
    print(f"{'=' * 70}\n")

    # Create datamodule for the model
    dm_final = DailyCYBenchSeqDataModule(config)
    dm_final.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    # Create model
    model_final = create_model(config)

    # WandB logger for the model
    try:
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-LSTF-AAAI2027"
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
        loggers = [CSVLogger("logs/", name="cybench-tst")]

    # Setup callbacks
    final_callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.001, verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=args.save_checkpoint_dir,
            filename=f'{generate_checkpoint_name(args)}_{{epoch:02d}}_{{val_loss:.4f}}_runid:{run_id}',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    if args.lr_decay_every is not None:
        print(f"[LR Schedule] Enabled: LR will halve every {args.lr_decay_every} epochs")

    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=final_callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nTraining final model...")
    trainer.fit(model_final, dm_final)

    print("\nEvaluating final model...")
    test_results = trainer.test(model_final, dm_final, ckpt_path="best")
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
        # Add uncertainty metrics if using MQL
        if config.loss_type == "mql":
            final_metrics.update({
                'crps': r.get('test/crps'),
                'picp': r.get('test/picp'),
                'pinaw': r.get('test/pinaw'),
                'pinball_avg': r.get('test/pinball_avg'),
                'winkler_score': r.get('test/winkler_score'),
                'mean_interval_width': r.get('test/mean_interval_width'),
            })

        # Add spatiotemporal metrics if available
        if hasattr(model_final, '_spatiotemporal_metrics') and model_final._spatiotemporal_metrics:
            for key, value in model_final._spatiotemporal_metrics.items():
                final_metrics[key] = value
    else:
        final_metrics = {}

    # Save test results to CSV files with per-year metrics
    print(f"\n[CSV Results] Retrieving per-year metrics from test results...")

    if hasattr(model_final, '_test_results_per_year') and model_final._test_results_per_year:
        per_year_metrics = model_final._test_results_per_year
    else:
        print(f"[CSV Results] Warning: No per-year metrics found on model. Using overall metrics only.")
        per_year_metrics = {}

    # Log per-year metrics to console
    print(f"\n[CSV Results] Per-Year Test Metrics:")
    for year in sorted(fixed_splits['test_years']):
        print(f"Year {year}:")
        for metric in ['nrmse', 'mape', 'r2']:
            if f'{metric}_{year}' in per_year_metrics:
                print(f"{metric.upper()}: {per_year_metrics[f'{metric}_{year}']:.4f}")

    print(f"Overall:")
    for metric in ['nrmse', 'mape', 'r2']:
        if f'{metric}_overall' in per_year_metrics:
            print(f"{metric.upper()}: {per_year_metrics[f'{metric}_overall']:.4f}")

    # Log uncertainty metrics if MQLoss was enabled
    uncertainty_metric_keys = [k for k in per_year_metrics.keys() if k.startswith('pinball_') or k in ['crps', 'picp', 'pinaw', 'winkler_score', 'mean_interval_width']]
    if uncertainty_metric_keys:
        print(f"\n[Uncertainty Metrics] Per-Year Uncertainty Metrics:")
        for year in sorted(fixed_splits['test_years']):
            print(f"Year {year}:")
            year_unc_metrics = {}
            for key in uncertainty_metric_keys:
                if key.endswith(f'_{year}'):
                    metric_name = key.replace(f'_{year}', '')
                    year_unc_metrics[metric_name] = per_year_metrics[key]

            if year_unc_metrics:
                for metric_name in ['pinball_avg', 'crps', 'picp', 'pinaw', 'winkler_score', 'mean_interval_width']:
                    if metric_name in year_unc_metrics:
                        print(f"  {metric_name}: {year_unc_metrics[metric_name]:.4f}")
                # Individual quantile pinball losses
                for q in ['pinball_q10', 'pinball_q50', 'pinball_q90']:
                    if q in year_unc_metrics:
                        print(f"  {q}: {year_unc_metrics[q]:.4f}")

        print(f"\n  Overall:")
        overall_unc_metrics = {}
        for key in uncertainty_metric_keys:
            if key.endswith('_overall'):
                metric_name = key.replace('_overall', '')
                overall_unc_metrics[metric_name] = per_year_metrics[key]

        if overall_unc_metrics:
            for metric_name in ['pinball_avg', 'crps', 'picp', 'pinaw', 'winkler_score', 'mean_interval_width']:
                if metric_name in overall_unc_metrics:
                    print(f"  {metric_name}: {overall_unc_metrics[metric_name]:.4f}")
            for q in ['pinball_q10', 'pinball_q50', 'pinball_q90']:
                if q in overall_unc_metrics:
                    print(f"  {q}: {overall_unc_metrics[q]:.4f}")

    # Save to CSV - extract actual years from test results (not from fixed_splits)
    actual_test_years = set()
    for key in per_year_metrics.keys():
        if key.endswith('_overall'):
            continue
        # Extract year from keys like 'nrmse_2015', 'mape_2017', etc.
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            actual_test_years.add(int(parts[1]))

    save_test_results_to_csv(
        config=config,
        test_results=per_year_metrics,
        test_years=sorted(actual_test_years),
        run_id=run_id,
        timestamp=timestamp
    )

    # Print split summary
    print(f"\n{'=' * 70}")
    print(f"SPLIT SUMMARY: {args.crop}-{args.country}")
    print(f"{'=' * 70}")
    print(f"Available years ({len(all_years)}): {all_years}")
    print(f"Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # Print final results with all metrics
    print_metrics_table(
        f"FINAL RESULTS: {args.crop}-{args.country}",
        final_metrics
    )

    # Load and test trained model if requested
    if args.load_and_test_trained_model:
        print(f"\n{'=' * 70}")
        print("RE-TEST TRAINED MODEL")
        print(f"{'=' * 70}")
        print(f"Checkpoint directory: {args.save_checkpoint_dir}")

        # The model_final is already loaded with best weights from training
        # Just re-run the test to get fresh metrics
        print("\nRe-evaluating trained model on test set...")
        retest_results = trainer.test(model_final, dm_final)

        if retest_results:
            r_retest = retest_results[0]
            retest_metrics = {
                'mse': r_retest.get('test/mse'),
                'mae': r_retest.get('test/mae'),
                'rmse': r_retest.get('test/rmse'),
                'r2': r_retest.get('test/r2'),
                'mape': r_retest.get('test/mape'),
                'smape': r_retest.get('test/smape'),
                'nrmse': r_retest.get('test/nrmse'),
            }

            # Add uncertainty metrics if using MQL
            if config.loss_type == "mql":
                retest_metrics.update({
                    'crps': r_retest.get('test/crps'),
                    'picp': r_retest.get('test/picp'),
                    'pinaw': r_retest.get('test/pinaw'),
                    'pinball_avg': r_retest.get('test/pinball_avg'),
                    'winkler_score': r_retest.get('test/winkler_score'),
                    'mean_interval_width': r_retest.get('test/mean_interval_width'),
                })

            # Add spatiotemporal metrics if available
            if hasattr(model_final, '_spatiotemporal_metrics') and model_final._spatiotemporal_metrics:
                for key, value in model_final._spatiotemporal_metrics.items():
                    retest_metrics[key] = value

            print(f"\n{'=' * 70}")
            print("RE-TEST RESULTS")
            print(f"{'=' * 70}")
            print_metrics_table(
                f"RE-TEST: {args.crop}-{args.country}",
                retest_metrics
            )

    # Log final metrics to wandB
    if wandb_logger is not None:
        import wandb
        if wandb.run is not None:
            wandb.log(final_metrics)
            wandb.finish()

    # Print experiment completion message
    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country}")
    print(f"Model: {args.model_type}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Test years: {fixed_splits['test_years']}")
    print(f"{'=' * 70}\n")
