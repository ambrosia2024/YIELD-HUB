"""
--------------------
Author: XYZ
Description: Script to import different validation helper functions and classes.
Python version: 3.12.0
"""

import os, sys
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Union
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

import torch
import torchmetrics
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Custom library
from loadData import prepare_features_and_targets
from eb_criterion import EBCriterionCallbackV2

# Uncertainty metrics imports
try:
    from neuralforecast.losses.pytorch import MQLoss
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    logging.warning("neuralforecast not available. MQLoss will use fallback implementation.")

try:
    from torchmetrics import CRPS
    TORCHMETRICS_CRPS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_CRPS_AVAILABLE = False
    logging.warning("torchmetrics CRPS not available. Will use fallback implementation.")

# Import forecast horizon verification
from cybench.process.alignment_patch import verify_forecast_horizon_config, test_alignment_patch

def evaluate_predictions_by_year(y_true, y_pred, years, metrics=None, min_samples_per_year=2, epsilon=1e-6):
    """
    Compute overall performance and per year with robustness to small sample sizes and zero/near-zero variance.

    Args:
        y_true: np.ndarray of true targets
        y_pred: np.ndarray of predicted targets
        years: list or np.ndarray of corresponding years
        metrics: list of metrics to compute ['r2', 'rmse', 'mape', 'normalized_rmse']
        min_samples_per_year: minimum number of samples to compute metrics per year
        epsilon: small value to avoid division by zero in normalized RMSE

    Returns:
        dict: nested dictionary {overall: {...}, year1: {...}, ...}
    """

    if metrics is None:
        metrics = ['r2', 'rmse', 'mape', 'normalized_rmse']

    EPSILON = epsilon
    results = {}

    def compute_r2_or_nan(y_t, y_p):
        if len(y_t) < min_samples_per_year or np.std(y_t) < EPSILON:
            return np.nan
        return r2_score(y_t, y_p)

    def compute_rmse(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return np.sqrt(mean_squared_error(y_t, y_p))

    def compute_mape(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return mean_absolute_percentage_error(y_t, y_p)

    def compute_normalized_rmse(y_t, y_p):
        if len(y_t) < min_samples_per_year:
            return compute_rmse(y_t, y_p)
        denom = np.max(y_t) - np.min(y_t)
        return compute_rmse(y_t, y_p) / (denom + EPSILON) * 100

    # Overall metrics 
    results['overall'] = {}
    for metric in metrics:
        if metric == 'r2':
            results['overall']['r2'] = compute_r2_or_nan(y_true, y_pred)
        elif metric == 'rmse':
            results['overall']['rmse'] = compute_rmse(y_true, y_pred)
        elif metric == 'mape':
            results['overall']['mape'] = compute_mape(y_true, y_pred)
        elif metric == 'normalized_rmse':
            results['overall']['normalized_rmse'] = compute_normalized_rmse(y_true, y_pred)

    # Per-year metrics 
    years = np.array(years)
    for year in sorted(set(years)):
        mask = np.where(years == year)[0]
        y_t = y_true[mask]
        y_p = y_pred[mask]

        results[year] = {}
        for metric in metrics:
            if metric == 'r2':
                results[year]['r2'] = compute_r2_or_nan(y_t, y_p)
            elif metric == 'rmse':
                results[year]['rmse'] = compute_rmse(y_t, y_p)
            elif metric == 'mape':
                results[year]['mape'] = compute_mape(y_t, y_p)
            elif metric == 'normalized_rmse':
                results[year]['normalized_rmse'] = compute_normalized_rmse(y_t, y_p)

    return results


def store_model_results(results_dict, model_name, country, crop, file_path="../output/myoutputs/sklearn_model_results.csv"):
    rows = []
    for year, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            rows.append({
                "model": str(model_name),
                "country": str(country[0]),
                "crop": str(crop),
                "year": str(year),
                "metric": str(metric_name),
                "value": value
            })

    new_df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(subset=["model", "country", "crop", "year", "metric"])
    combined_df.to_csv(file_path, index=False)
    return combined_df

def evaluate_OOD_results_from_countries(crop, model_name, pipeline, file_path):
    # Evaluates the trained model on EU countries in CY-BENCH dataset.
    countries_to_evaluate = ["AT", "BE", "BG", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LV", "NL", "PL", "PT", "RO", "SE", "SK"]

    if crop == "maize":
        countries_not_of_interest = ["EE", "FI", "IE", "LV"]
    elif crop == "wheat":
        countries_not_of_interest = ["SK"]
    else:
        raise Exception("Crop can either be maize or wheat.")

    for country in countries_not_of_interest:
        countries_to_evaluate.remove(country)
    
    pbar = tqdm(countries_to_evaluate, desc="Evaluating")
    for country in pbar:
        pbar.set_description(f"Evaluating {model_name} model on {country} country")

        df_y, dfs_x = load_dfs_crop(crop, countries=[country])
        ds = CYDataset(crop=crop, data_target=df_y, data_inputs=dfs_x)

        years_sorted = sorted(list(ds.years))
        train_years = [y for y in years_sorted if y <= 2017]
        test_years  = [y for y in years_sorted if y >= 2018]
        _, test_ds = ds.split_on_years((train_years, test_years))

        X_test, y_test, years_test = prepare_features_and_targets(test_ds)
        y_pred = pipeline.predict(X_test)

        results_by_year = evaluate_predictions_by_year(y_test, y_pred, years_test)
        _ = store_model_results(results_by_year, model_name, country, crop, file_path)


class ModelMetrics(nn.Module):
    """
    Model class with comprehensive metrics for in-season and end-of-season yield forecasting.

    Wraps torchmetrics to provide MSE, MAE, RMSE, R², MAPE, SMAPE, NRMSE.
    Prefix (train/val/test) is used for wandb logging namespacing.

    Inherits from nn.Module so metrics are moved to device automatically when the parent LightningModule is moved to GPU.
    """

    def __init__(self, prefix: str = "test", include_nrmse: bool = True):
        super().__init__()
        self.prefix = prefix
        metrics_dict = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'mape': torchmetrics.MeanAbsolutePercentageError(),
            'smape': torchmetrics.SymmetricMeanAbsolutePercentageError(),
        }
        # Only include NRMSE if requested (exclude for training since targets are normalized)
        if include_nrmse:
            metrics_dict['nrmse'] = torchmetrics.NormalizedRootMeanSquaredError(normalization='mean')
        self.metrics = torchmetrics.MetricCollection(metrics_dict)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.metrics.update(preds, targets)

    def compute(self) -> Dict:
        try:
            return self.metrics.compute()
        except ValueError as e:
            if "Needs at least two samples to calculate r2 score" in str(e):
                # Handle sparse data case (e.g., single test sample) - return NaN for R2
                import torch
                metrics = {}
                # Get other metrics that don't require 2+ samples
                for key in ['mse', 'mae', 'mape', 'smape', 'nrmse']:
                    try:
                        m = self.metrics[key]
                        if hasattr(m, 'compute'):
                            metrics[key] = m.compute()
                    except:
                        pass
                # RMSE from MSE
                if 'mse' in metrics:
                    import math
                    metrics['rmse'] = math.sqrt(metrics['mse'].item()) if isinstance(metrics['mse'], torch.Tensor) else math.sqrt(float(metrics['mse']))
                # R2 as NaN for insufficient samples
                metrics['r2'] = float('nan')
                return metrics
            else:
                raise

    def reset(self):
        self.metrics.reset()

    def log_results(self, step: str = "val"):
        results = self.compute()
        print(f"\n{'-' * 60}")
        print(f"{step.upper()} METRICS ({self.prefix.upper()}):")
        print(f"MSE:   {results['mse']:.4f}")
        print(f"MAE:   {results['mae']:.4f}")
        print(f"RMSE:  {torch.sqrt(results['mse']):.4f}")
        print(f"R²:    {results['r2']:.4f}")
        # MAPE/SMAPE reported as fractions
        print(f"MAPE:  {results['mape']:.4f}")
        print(f"SMAPE: {results['smape']:.4f}")
        # Only print NRMSE if it exists (excluded for training metrics)
        if 'nrmse' in results:
            print(f"NRMSE: {results['nrmse']:.4f}")
        print(f"{'-' * 60}")

def format_metrics_dict(results: Dict) -> Dict[str, float]:
    """
    Convert torchmetrics tensor results to a dict of floats.

    Args:
        results: Dict with tensor metrics from ModelMetrics.compute()

    Returns:
        Dict with all metrics as Python floats
    """
    return {
        'mse': float(results['mse'].item()) if 'mse' in results else None,
        'mae': float(results['mae'].item()) if 'mae' in results else None,
        'rmse': float(torch.sqrt(results['mse']).item()) if 'mse' in results else None,
        'r2': float(results['r2'].item()) if 'r2' in results else None,
        'mape': float(results['mape'].item()) if 'mape' in results else None,
        'smape': float(results['smape'].item()) if 'smape' in results else None,
        'nrmse': float(results['nrmse'].item()) if 'nrmse' in results else None,
    }

def print_metrics_table(title: str, metrics: Dict, step: str = "test"):
    """
    Print a nicely formatted table of all metrics.

    Args:
        title: Section title (ex – "CV Fold 1 Results")
        metrics: Dict from format_metrics_dict()
        step: Step name for context
    """
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    if metrics.get('mse') is not None:
        print(f"MSE:   {metrics['mse']:.4f}")
    if metrics.get('mae') is not None:
        print(f"MAE:   {metrics['mae']:.4f}")
    if metrics.get('rmse') is not None:
        print(f"RMSE:  {metrics['rmse']:.4f}")
    if metrics.get('r2') is not None:
        print(f"R²:    {metrics['r2']:.4f}")
    if metrics.get('mape') is not None:
        print(f"MAPE:  {metrics['mape']:.2f}%")
    if metrics.get('smape') is not None:
        print(f"SMAPE: {metrics['smape']:.2f}%")
    if metrics.get('nrmse') is not None:
        print(f"NRMSE: {metrics['nrmse']:.4f}")

    # Print uncertainty metrics if available
    if metrics.get('crps') is not None or metrics.get('picp') is not None:
        print(f"\n{'=' * 70}")
        print("UNCERTAINTY METRICS")
        print(f"{'=' * 70}")
        if metrics.get('crps') is not None:
            print(f"CRPS:  {metrics['crps']:.4f}")
        if metrics.get('picp') is not None:
            print(f"PICP:  {metrics['picp']:.2%}")
        if metrics.get('pinaw') is not None:
            print(f"PINAW: {metrics['pinaw']:.4f}")
        if metrics.get('pinball_avg') is not None:
            print(f"Pinball Loss (avg): {metrics['pinball_avg']:.4f}")
        if metrics.get('winkler_score') is not None:
            print(f"Winkler Score:     {metrics['winkler_score']:.4f}")
        if metrics.get('mean_interval_width') is not None:
            print(f"Mean Interval Width: {metrics['mean_interval_width']:.4f}")

    # Print spatiotemporal metrics if available
    st_keys = [k for k in metrics.keys() if k.startswith('spatial/') or k.startswith('temporal/') or k.startswith('anomaly/')]
    if st_keys:
        print(f"\n{'=' * 70}")
        print("SPATIOTEMPORAL METRICS")
        print(f"{'=' * 70}")

        # Spatial correlation
        if metrics.get('spatial/r_sp_overall') is not None:
            print(f"Spatial Correlation (r_sp):  {metrics['spatial/r_sp_overall']:.4f}")

        # Temporal correlation
        if metrics.get('temporal/r_tm_overall') is not None:
            print(f"Temporal Correlation (r_tm): {metrics['temporal/r_tm_overall']:.4f}")

        # Anomaly correlation
        if metrics.get('anomaly/r_an_overall') is not None:
            print(f"Anomaly Correlation (r_an):  {metrics['anomaly/r_an_overall']:.4f}")

    print(f"{'-' * 70}")


def save_walk_forward_results(
    results_matrix: List[Dict],
    all_years: List[int],
    test_years: int,
    config,
    run_id: str,
    timestamp: str
) -> str:
    """
    Save walk-forward results to CSV in a structured format.

    Creates a CSV with columns:
    - fold: Fold index (0 to test_years-1)
    - train_end_year: Last year of training data
    - test_year: Year being tested
    - mse, mae, rmse, r2, mape, smape, nrmse: Metric values

    Also saves aggregated metrics per-year and overall.

    Args:
        results_matrix: List of dicts, each containing fold metrics by year
        all_years: All available years in the dataset
        test_years: Number of test years (number of folds)
        config: Model config object with crop, country, etc.
        run_id: Unique run identifier
        timestamp: Timestamp string

    Returns:
        Path to saved CSV file
    """
    os.makedirs(config.results_dir, exist_ok=True)

    # Build flattened DataFrame with config info
    rows = []
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        train_end_year = fold_result['train_end_year']

        for test_year, metrics in fold_result['yearly_metrics'].items():
            row = {
                'run_id': run_id,
                'crop': config.crop,
                'country': config.country,
                'model_type': getattr(config, 'model_type', 'unknown'),
                'aggregation': getattr(config, 'aggregation', 'unknown'),
                'fold': fold_idx + 1,
                'train_end_year': train_end_year,
                'test_year': test_year,
                'mse': metrics.get('mse'),
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'r2': metrics.get('r2'),
                'mape': metrics.get('mape'),
                'smape': metrics.get('smape'),
                'nrmse': metrics.get('nrmse'),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save detailed results
    csv_path = os.path.join(
        config.results_dir,
        f"{config.crop}_{config.country}_walkforward_{timestamp}_{run_id}.csv"
    )
    df.to_csv(csv_path, index=False)

    # Calculate and save per-year aggregates
    metric_cols = ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']
    per_year_agg = df.groupby('test_year').agg({
        **{col: ['mean', 'std'] for col in metric_cols}
    }).reset_index()
    per_year_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in per_year_agg.columns]

    # Add config columns to aggregated
    per_year_agg['run_id'] = run_id
    per_year_agg['crop'] = config.crop
    per_year_agg['country'] = config.country
    per_year_agg['model_type'] = getattr(config, 'model_type', 'unknown')
    per_year_agg['aggregation'] = getattr(config, 'aggregation', 'unknown')

    agg_path = os.path.join(
        config.results_dir,
        f"{config.crop}_{config.country}_walkforward_aggregated_{timestamp}_{run_id}.csv"
    )
    per_year_agg.to_csv(agg_path, index=False)

    # Calculate overall averages - average of per-year averages (not simple mean of all predictions)
    # This ensures each year contributes equally regardless of how many folds predicted it
    overall_avg = {}
    for col in metric_cols:
        mean_col = f'{col}_mean'
        if mean_col in per_year_agg.columns:
            overall_avg[col] = per_year_agg[mean_col].mean()

    print(f"\n[Walk-Forward] CSV Results saved to:")
    print(f"Detailed: {csv_path}")
    print(f"Aggregated: {agg_path}")
    print(f"\n[Walk-Forward] Overall Average (average of per-year means):")
    print(f"MSE:   {overall_avg['mse']:.4f}")
    print(f"MAE:   {overall_avg['mae']:.4f}")
    print(f"RMSE:  {overall_avg['rmse']:.4f}")
    print(f"R2:    {overall_avg['r2']:.4f}")
    print(f"MAPE:  {overall_avg['mape']:.2f}%")
    print(f"SMAPE: {overall_avg['smape']:.2f}%")
    print(f"NRMSE: {overall_avg['nrmse']:.4f}")

    return csv_path


def run_walk_forward_validation(
    all_years: List[int],
    test_years: int,
    config,
    create_model_fn,
    datamodule_class,
    trainer_class,
    max_epochs: int,
    loggers: List,
    run_id: str,
    timestamp: str,
    save_checkpoint_dir: Optional[str] = None,
    save_top_k: int = -1,
    save_last: bool = True,
) -> Dict:
    """
    Run walk-forward validation where each fold is tested on ALL future years.

    For example, with test_years=5 and years 2000-2020:
    - Fold 0 (train up to 2015): test on 2016, 2017, 2018, 2019, 2020
    - Fold 1 (train up to 2016): test on 2017, 2018, 2019, 2020
    - Fold 2 (train up to 2017): test on 2018, 2019, 2020
    - Fold 3 (train up to 2018): test on 2019, 2020
    - Fold 4 (train up to 2019): test on 2020

    Args:
        all_years: Sorted list of all available years
        test_years: Number of years to walk forward (N)
        config: Model configuration object
        create_model_fn: Function that creates a model from config
        datamodule_class: Class for creating datamodules
        trainer_class: PyTorch Lightning Trainer class
        max_epochs: Maximum training epochs per fold
        loggers: List of loggers (WandB, CSV)
        run_id: Unique run identifier
        timestamp: Timestamp string

        save_checkpoint_dir: Directory to save checkpoints (fold-aware filenames)
        save_top_k: Number of checkpoints to save per fold (-1=all, 1=best only)
        save_last: Whether to save last checkpoint (ignored when save_top_k=1)

        save_checkpoint_dir: Directory to save checkpoints (fold-aware filenames)
        save_top_k: Number of checkpoints to save per fold (-1=all, 1=best only)
        save_last: Whether to save last checkpoint (ignored when save_top_k=1)
    Returns:
        Dict containing:
        - results_matrix: List of fold results with yearly metrics
        - overall_metrics: Average metrics across all folds/years
        - per_year_metrics: Aggregated metrics per test year
    """
    from cybench.process.loadData import generate_walk_forward_splits

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION (Test on All Future Years)")
    print(f"{'=' * 70}\n")

    # Verify alignment patch is working before starting validation
    print("\n[Pre-flight check] Verifying alignment patch...")
    test_alignment_patch()

    # Generate walk-forward splits
    wf_splits = generate_walk_forward_splits(all_years, test_years)

    print(f"[Walk-Forward Config]")
    print(f"Number of folds: {len(wf_splits)}")
    print(f"Max epochs per fold: {max_epochs}")
    print(f"\n[Walk-Forward Splits]")
    for s in wf_splits:
        print(f"Fold {s['fold_idx'] + 1}: train_years={len(s['train_years'])} ({min(s['train_years'])}-{max(s['train_years'])})")

    # Store results from all folds
    results_matrix = []

    for fold_idx, split in enumerate(wf_splits):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{len(wf_splits)}")
        print(f"{'=' * 70}")
        print(f"Train years: {split['train_years']}")
        print(f"Train end year: {max(split['train_years'])}")

        # Create config for this fold
        fold_config = _update_config_for_fold(
            config, split['train_years'], max_epochs
        )

        # Verify forecast horizon configuration for this fold
        verify_forecast_horizon_config(fold_config, fold_idx=split['fold_idx'])

        # Create datamodule and model
        dm_fold = datamodule_class(fold_config)
        dm_fold.setup(
            train_years=split['train_years'],
            val_years=[],
            test_years=[]
        )

        model_fold = create_model_fn(fold_config)

        # Use EB-criterion for proper early stopping
        fold_callbacks = [
            EBCriterionCallbackV2(
                batch_size=fold_config.batch_size,
                patience=2,  # Wait 2 epochs after criterion is met before stopping
                smoothing=0.9,
                min_epochs=10,  # Minimum epochs before allowing early stop
                verbose=True,
                stopping_threshold=0.0,  # Paper's threshold
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        # Add ModelCheckpoint callback if directory provided
        if save_checkpoint_dir:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            train_end_year = max(split['train_years'])
            checkpoint_filename = f"{config.crop}_{config.country}_{config.model_type}_fold{fold_idx + 1}_train_end_{train_end_year}_{{epoch}}"
            # Monitor train_loss for checkpoint selection (EB-criterion is for stopping, not selection)
            fold_callbacks.append(
                ModelCheckpoint(
                    dirpath=save_checkpoint_dir,
                    filename=checkpoint_filename,
                    monitor='train_loss',
                    mode='min',
                    save_top_k=save_top_k, 
                    save_last=save_last,
                )
            )

        # Create trainer
        trainer_fold = trainer_class(
            max_epochs=fold_config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=fold_callbacks,
            logger=loggers,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        print(f"\n[Fold {fold_idx + 1}] Training...")
        trainer_fold.fit(model_fold, dm_fold)

        # Test on ALL future years
        train_end_year = max(split['train_years'])
        future_years = [y for y in all_years if y > train_end_year]

        print(f"\n[Fold {fold_idx + 1}] Testing on {len(future_years)} future years: {future_years}")

        fold_yearly_metrics = {}
        first_test_year_logged = False

        for test_year in future_years:
            # Create test datamodule for this single year
            dm_test = datamodule_class(fold_config)
            # Copy normalization statistics from training datamodule to avoid NaN
            dm_test.copy_normalization_from(dm_fold)
            dm_test.setup(
                train_years=[],
                val_years=[],
                test_years=[test_year]
            )

            # Unfortunately, rainer.test() doesn't accept logger argument in this Lightning version
            # But then also, we only want per-fold metrics, not the last fold's metrics overriding everything
            test_results = trainer_fold.test(model_fold, dm_test, verbose=False)

            if test_results:
                r = test_results[0]
                metrics = {
                    'mse': r.get('test/mse'),
                    'mae': r.get('test/mae'),
                    'rmse': r.get('test/rmse'),
                    'r2': r.get('test/r2'),
                    'mape': r.get('test/mape'),
                    'smape': r.get('test/smape'),
                    'nrmse': r.get('test/nrmse'),
                }
                fold_yearly_metrics[test_year] = metrics

                print(f"Year {test_year}: NRMSE={metrics['nrmse']:.4f}, R2={metrics['r2']:.4f}")

                # Log to wandb with fold-specific prefix for first test year only
                if not first_test_year_logged:
                    _log_test_metrics_to_wandb(loggers, metrics, fold_idx)
                    first_test_year_logged = True

        results_matrix.append({
            'fold_idx': fold_idx,
            'train_end_year': train_end_year,
            'yearly_metrics': fold_yearly_metrics,
        })

        # Clear prediction cache between folds to prevent cross-contamination
        # Each fold should start fresh with no cached predictions from previous folds
        if hasattr(model_fold, '_yield_predictions_cache'):
            model_fold._yield_predictions_cache.clear()
            logging.info(f"[Fold {fold_idx + 1}] Cleared prediction cache after testing")
        if hasattr(model_fold, '_prediction_cache'):
            model_fold._prediction_cache.clear()
            logging.info(f"[Fold {fold_idx + 1}] Cleared TST prediction cache after testing")

    # Aggregate results
    aggregated = _aggregate_walk_forward_results(results_matrix, all_years, test_years)

    # Collect per-fold all-year metrics for WandB logging
    fold_all_year_metrics = _collect_fold_all_year_metrics(results_matrix)

    print(f"\n[DEBUG] Walk-forward summary - {len(results_matrix)} folds processed")
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        yearly_metrics = fold_result['yearly_metrics']
        first_year = min(yearly_metrics.keys()) if yearly_metrics else None
        print(f"Fold {fold_idx + 1}: {len(yearly_metrics)} test years, first year={first_year}")

    # Save to CSV
    csv_path = save_walk_forward_results(
        results_matrix, all_years, test_years, config, run_id, timestamp
    )

    # Log to WandB (overall averages only - per-fold first-year metrics already logged)
    _log_walk_forward_to_wandb(loggers, aggregated, run_id, fold_all_year_metrics)

    return {
        'results_matrix': results_matrix,
        'overall_metrics': aggregated['overall'],
        'per_year_metrics': aggregated['per_year'],
        'first_year_metrics': aggregated.get('first_year'),
        'csv_path': csv_path,
    }


def _update_config_for_fold(config, train_years, max_epochs):
    """Update config for a walk-forward fold.

    Creates a new config object for each fold, preserving important settings
    like data_fraction and model-specific hyperparameters.
    """
    # Create a copy of the config with updated test_years and max_epochs
    fold_config = config.__class__(
        crop=config.crop,
        country=config.country,
        model_type=getattr(config, 'model_type', None),
        aggregation=getattr(config, 'aggregation', None),
        data_fraction=config.data_fraction,  # Controls amount of input data fed to the model
        use_sota_features=getattr(config, 'use_sota_features', False),
        include_spatial_features=getattr(config, 'include_spatial_features', False),
        lag_years=getattr(config, 'lag_years', 1),
        load_checkpoint=None,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=max_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        test_years=1,  # Note: walk-forward passes explicit year lists, so this field is not actually used
        use_residual_trend=getattr(config, 'use_residual_trend', False),
        use_recursive_lags=getattr(config, 'use_recursive_lags', False),
        use_gdd=getattr(config, 'use_gdd', False),
        use_heat_stress_days=getattr(config, 'use_heat_stress_days', False),
        use_rue=getattr(config, 'use_rue', False),
        use_farquhar=getattr(config, 'use_farquhar', False),
        use_cwb_feature=getattr(config, 'use_cwb_feature', False),
        drop_tavg=getattr(config, 'drop_tavg', False),
        use_revin=getattr(config, 'use_revin', False),
        results_dir=config.results_dir,
        lr_scheduler_lambda=getattr(config, 'lr_scheduler_lambda', None),
    )

    # Add model-specific hyperparameters if they exist
    for attr in ['xlinear_hidden_size', 'xlinear_temporal_ff', 'xlinear_channel_ff', 'xlinear_dropout',
                 'patchtst_d_model', 'patchtst_num_attention_heads', 'patchtst_ffn_dim',
                 'patchtst_num_layers', 'patchtst_dropout']:
        if hasattr(config, attr):
            setattr(fold_config, attr, getattr(config, attr))

    return fold_config


def _aggregate_walk_forward_results(results_matrix: List[Dict], all_years: List[int], test_years: int) -> Dict:
    """Aggregate walk-forward results into overall and per-year metrics.

    The overall average is calculated as the mean of per-year averages,
    giving equal weight to each year regardless of how many folds predicted it.
    """
    # Collect all metric values
    per_year_values = {}
    first_year_values = {m: [] for m in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']}

    for fold_result in results_matrix:
        train_end_year = fold_result['train_end_year']
        # Find the actual first test year (handles gaps in year sequences)
        tested_years = sorted(fold_result['yearly_metrics'].keys())
        first_test_year = min(tested_years) if tested_years else None

        for test_year, metrics in fold_result['yearly_metrics'].items():
            # Track per-year values
            if test_year not in per_year_values:
                per_year_values[test_year] = {m: [] for m in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']}
            for metric_name, value in metrics.items():
                if value is not None:
                    per_year_values[test_year][metric_name].append(value)

            # Track first test year values (the earliest year tested in this fold)
            if test_year == first_test_year:
                for metric_name, value in metrics.items():
                    if value is not None:
                        first_year_values[metric_name].append(value)

    # Calculate per-year averages first
    per_year = {}
    for year, metrics_dict in sorted(per_year_values.items()):
        per_year[year] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                per_year[year][metric_name] = np.mean(values)
                per_year[year][f'{metric_name}_std'] = np.std(values)
            else:
                per_year[year][metric_name] = None
                per_year[year][f'{metric_name}_std'] = None

    # Calculate overall averages as mean of per-year averages (equal weight per year)
    overall = {}
    for metric_name in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']:
        year_values = [per_year[y][metric_name] for y in per_year if per_year[y][metric_name] is not None]
        if year_values:
            overall[metric_name] = np.mean(year_values)
            overall[f'{metric_name}_std'] = np.std(year_values)
        else:
            overall[metric_name] = None
            overall[f'{metric_name}_std'] = None

    # Calculate first test year averages
    first_year = {}
    for metric_name in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']:
        if first_year_values[metric_name]:
            first_year[metric_name] = np.mean(first_year_values[metric_name])
            first_year[f'{metric_name}_std'] = np.std(first_year_values[metric_name])
        else:
            first_year[metric_name] = None
            first_year[f'{metric_name}_std'] = None

    return {'overall': overall, 'per_year': per_year, 'first_year': first_year}


def _collect_fold_all_year_metrics(results_matrix: List[Dict]) -> Dict:
    """Collect all test year metrics for each fold (for fold-level averaging).

    Returns:
        Dict mapping fold_idx to a dict of test_year -> metrics:
        {
            0: {
                2018: {'nrmse': 0.123, 'r2': 0.456, ...},
                2019: {'nrmse': 0.234, 'r2': 0.567, ...},
                ...
            },
            ...
        }
    """
    fold_metrics = {}
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        fold_metrics[fold_idx] = fold_result['yearly_metrics']

    return fold_metrics


def _log_test_metrics_to_wandb(loggers: List, metrics: Dict, fold_idx: int):
    """Log test metrics to WandB with fold-specific prefix.

    Logs as test/fold_{idx}/{metric} for each fold separately.
    Only called for the first test year of each fold.
    """
    for logger in loggers:
        if hasattr(logger, 'experiment'):  # WandB logger
            metrics_to_log = {}
            for metric_name, value in metrics.items():
                if value is not None:
                    metrics_to_log[f'test/fold_{fold_idx + 1}/{metric_name}'] = value
            if metrics_to_log:
                logger.experiment.log(metrics_to_log)
                print(f"[WandB] Logged fold_{fold_idx + 1} first-year metrics: R2={metrics.get('r2'):.4f}, NRMSE={metrics.get('nrmse'):.4f}")


def _log_walk_forward_to_wandb(loggers: List, aggregated: Dict, run_id: str,
                               fold_all_year_metrics: Dict = None):
    """Log walk-forward metrics to WandB.

    Note: Per-fold first-year metrics are already logged inside the fold loop
    by _log_test_metrics_to_wandb(). This function only logs overall averages.
    """
    for logger in loggers:
        if hasattr(logger, 'experiment'):  # WandB logger
            metrics_to_log = {}

            # Log overall averages across all folds and years
            if aggregated and 'overall' in aggregated:
                for metric_name, value in aggregated['overall'].items():
                    if value is not None and not metric_name.endswith('_std'):
                        metrics_to_log[f'test/{metric_name}'] = value

            # Also log first-year averages (useful for comparison)
            if aggregated and 'first_year' in aggregated:
                for metric_name, value in aggregated['first_year'].items():
                    if value is not None and not metric_name.endswith('_std'):
                        metrics_to_log[f'test/first_year_avg/{metric_name}'] = value

            if metrics_to_log:
                logger.experiment.log(metrics_to_log)
                print(f"[WandB] Logged overall metrics to WandB")


# ============================================================================
# UNCERTAINTY METRICS FOR QUANTILE REGRESSION
# ============================================================================

class UncertaintyMetrics(nn.Module):
    """
    Uncertainty metrics for quantile regression evaluation.

    Computes:
    - Pinball Loss (per quantile and average)
    - CRPS (Continuous Ranked Probability Score)
    - PICP (Prediction Interval Coverage Probability)
    - PINAW (Prediction Interval Normalized Average Width)
    - Winkler Score

    Compatible with PyTorch Lightning for logging during training/validation.
    """

    def __init__(self, quantiles: List[float] = None, prefix: str = ''):
        """
        Args:
            quantiles: List of quantile levels (default: [0.1, 0.5, 0.9])
            prefix: Prefix for metric names (e.g., 'val_', 'test_')
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.quantiles = quantiles
        self.prefix = prefix
        self.reset()

        # Initialize MQLoss from neuralforecast if available
        if NEURALFORECAST_AVAILABLE:
            self.mq_loss = MQLoss(quantiles=quantiles)
        else:
            self.mq_loss = None

    def reset(self):
        """Reset all accumulated values."""
        self.y_true = []
        self.y_pred = []

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Add predictions and targets.

        Args:
            y_true: Ground truth, shape (batch,)
            y_pred: Predictions, shape (batch, n_quantiles)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary of metric names to values
        """
        if len(self.y_true) == 0:
            return {}

        y_true_all = torch.cat(self.y_true, dim=0).numpy()
        y_pred_all = torch.cat(self.y_pred, dim=0).numpy()

        metrics = self._compute_all_uncertainty_metrics(
            y_true_all, y_pred_all, self.quantiles
        )

        # Add prefix to all metric names
        if self.prefix:
            metrics = {f'{self.prefix}{k}': v for k, v in metrics.items()}

        return metrics

    def _compute_all_uncertainty_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]
    ) -> Dict[str, float]:
        """Compute all uncertainty metrics."""
        results = {}

        # 1. Pinball loss for each quantile
        for i, q in enumerate(quantiles):
            q_loss = self._pinball_loss(y_true, y_pred[:, i], q)
            results[f'pinball_q{int(q*100)}'] = float(np.mean(q_loss))

        # Average pinball loss
        pinball_losses = [self._pinball_loss(y_true, y_pred[:, i], q)
                          for i, q in enumerate(quantiles)]
        results['pinball_avg'] = float(np.mean(pinball_losses))

        # 2. CRPS
        results['crps'] = self._compute_crps(y_true, y_pred, quantiles)

        # 3. PICP (using extreme quantiles for interval)
        q_lower_idx = 0
        q_upper_idx = -1
        results['picp'] = self._compute_picp(
            y_true, y_pred[:, q_lower_idx], y_pred[:, q_upper_idx]
        )

        # 4. PINAW
        results['pinaw'] = self._compute_pinaw(
            y_pred[:, q_lower_idx], y_pred[:, q_upper_idx], y_true
        )

        # 5. Winkler Score
        coverage_level = 1 - (quantiles[q_upper_idx] - quantiles[q_lower_idx])
        results['winkler_score'] = self._compute_winkler_score(
            y_true, y_pred[:, q_lower_idx], y_pred[:, q_upper_idx], coverage_level
        )

        # 6. Mean interval width
        results['mean_interval_width'] = float(
            np.mean(y_pred[:, q_upper_idx] - y_pred[:, q_lower_idx])
        )

        return results

    @staticmethod
    def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
        """Compute pinball loss for a single quantile."""
        error = y_true - y_pred
        return np.maximum(quantile * error, (quantile - 1) * error)

    @staticmethod
    def _compute_crps(
        y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]
    ) -> float:
        """
        Compute Continuous Ranked Probability Score (CRPS).

        Uses trapezoidal integration over quantiles.
        """
        n_samples = len(y_true)
        crps_values = np.zeros(n_samples)

        # Ensure quantiles are sorted
        sorted_indices = np.argsort(quantiles)
        sorted_quantiles = np.array(quantiles)[sorted_indices]
        sorted_preds = y_pred[:, sorted_indices]

        for i in range(n_samples):
            y_i = y_true[i]
            preds_i = sorted_preds[i]

            # Compute CRPS using trapezoidal integration
            integral = 0.0
            for j in range(len(sorted_quantiles) - 1):
                # CDF at quantile j: indicator if y_true >= predicted value
                cdf_j = 1.0 if y_i >= preds_i[j] else 0.0
                cdf_j1 = 1.0 if y_i >= preds_i[j + 1] else 0.0

                # CDF difference
                dq = sorted_quantiles[j + 1] - sorted_quantiles[j]

                # Trapezoid area: average height * width
                avg_height = ((cdf_j - sorted_quantiles[j])**2 +
                             (cdf_j1 - sorted_quantiles[j + 1])**2) / 2
                integral += avg_height * dq

            crps_values[i] = integral

        return float(np.mean(crps_values))

    @staticmethod
    def _compute_picp(
        y_true: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray
    ) -> float:
        """Compute Prediction Interval Coverage Probability."""
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        return float(np.mean(in_interval))

    @staticmethod
    def _compute_pinaw(
        y_pred_lower: np.ndarray, y_pred_upper: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> float:
        """Compute Prediction Interval Normalized Average Width."""
        interval_width = y_pred_upper - y_pred_lower
        mean_width = np.mean(interval_width)

        # Compute normalization range
        if y_true is not None:
            range_width = y_true.max() - y_true.min()
        else:
            range_width = y_pred_upper.max() - y_pred_lower.min()

        if range_width == 0:
            return 0.0

        return float(mean_width / range_width)

    @staticmethod
    def _compute_winkler_score(
        y_true: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray,
        alpha: float = 0.1
    ) -> float:
        """Compute Winkler score for prediction intervals."""
        interval_width = y_pred_upper - y_pred_lower
        penalty = 2.0 / alpha

        # Compute score for each sample
        scores = np.where(
            y_true < y_pred_lower,
            interval_width + penalty * (y_pred_lower - y_true),
            np.where(
                y_true > y_pred_upper,
                interval_width + penalty * (y_true - y_pred_upper),
                interval_width
            )
        )

        return float(np.mean(scores))


def compute_uncertainty_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_quantiles: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = None
) -> Dict[str, float]:
    """
    Compute all uncertainty metrics for quantile regression.

    This is a convenience function for one-shot metric computation.

    Args:
        y_true: Ground truth values (in original scale), shape (n_samples,)
        y_pred_quantiles: Predicted values for each quantile (in original scale),
                          shape (n_samples, n_quantiles)
        quantiles: List of quantile levels (default: [0.1, 0.5, 0.9])

    Returns:
        Dictionary with all uncertainty metrics

    Example:
        >>> y_pred = np.random.randn(100, 3)  # 100 samples, 3 quantiles
        >>> y_true = np.random.randn(100)
        >>> metrics = compute_uncertainty_metrics(y_true, y_pred)
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    if isinstance(y_pred_quantiles, torch.Tensor):
        y_pred_quantiles = y_pred_quantiles.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Use the UncertaintyMetrics class for computation
    metrics_obj = UncertaintyMetrics(quantiles=quantiles)
    return metrics_obj._compute_all_uncertainty_metrics(y_true, y_pred_quantiles, quantiles)