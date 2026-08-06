# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Spatial, temporal, and anomaly correlation metrics for crop yield forecasting.

This module provides metrics for evaluating whether models reproduce:
1. Spatial patterns: Geographic distribution of yields within a typical growing season
2. Temporal patterns: Year-to-year yield variability within a typical administrative region
3. Anomaly patterns: Deviations from long-term regional productivity

Python version: 3.12.0
--------------------

Key Metrics:
- Spatial correlation (r_sp): Pearson correlation across regions for each year, then median
- Temporal correlation (r_tm): Pearson correlation across years for each region, then median
- Anomaly correlation (r_an): Pearson correlation on de-meaned region-year observations
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def compute_spatial_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute spatial correlation metrics.

    For each test year t, compute Pearson correlation across administrative regions:
        r_sp(t) = r({y_r,t}_{r∈R}, {ŷ_r,t}_{r∈R})
    Then report median across test years:
        r_sp = median_{t∈T} r_sp(t)

    This metric measures how well the model reproduces relative productivity
    differences between regions during a typical growing season.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample (optional, for logging), shape (n_samples,)

    Returns:
        Dictionary with:
            - r_sp_overall: Median spatial correlation across years
            - r_sp_by_year: Dict of {year: correlation} for each year
            - rmse_by_year: Dict of {year: RMSE} for each year
            - mae_by_year: Dict of {year: MAE} for each year
            - nrmse_by_year: Dict of {year: NRMSE} for each year
            - smape_by_year: Dict of {year: SMAPE} for each year
            - r2_by_year: Dict of {year: R²} for each year
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'year': years
    })
    if regions is not None:
        df['region'] = regions

    results = {}
    r_sp_by_year = {}
    rmse_by_year = {}
    mae_by_year = {}
    nrmse_by_year = {}
    smape_by_year = {}
    r2_by_year = {}

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        y_t = year_df['y_true'].values
        y_p = year_df['y_pred'].values

        # Need at least 2 regions to compute correlation
        if len(y_t) < 2:
            r_sp_by_year[year] = np.nan
            rmse_by_year[year] = np.nan
            mae_by_year[year] = np.nan
            nrmse_by_year[year] = np.nan
            smape_by_year[year] = np.nan
            r2_by_year[year] = np.nan
            continue

        # Pearson correlation
        r_sp_by_year[year] = pearsonr(y_t, y_p)[0]

        # RMSE
        rmse_by_year[year] = np.sqrt(mean_squared_error(y_t, y_p))

        # MAE
        mae_by_year[year] = mean_absolute_error(y_t, y_p)

        # NRMSE (Normalized RMSE): RMSE / mean(y_true)
        mean_y_t = np.mean(y_t)
        nrmse_by_year[year] = rmse_by_year[year] / mean_y_t if mean_y_t != 0 else np.nan

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        # SMAPE = 100/n * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred| + epsilon))
        smape_by_year[year] = 100.0 / len(y_t) * np.sum(
            2 * np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)
        )

        # R² (same as r² for correlation, but computed differently)
        if len(y_t) >= 2:
            r2_by_year[year] = r2_score(y_t, y_p)
        else:
            r2_by_year[year] = np.nan

    # Median spatial correlation across years
    valid_correlations = [v for v in r_sp_by_year.values() if not np.isnan(v)]
    r_sp_overall = np.median(valid_correlations) if valid_correlations else np.nan

    results = {
        'r_sp_overall': r_sp_overall,
        'r_sp_by_year': r_sp_by_year,
        'rmse_by_year': rmse_by_year,
        'mae_by_year': mae_by_year,
        'nrmse_by_year': nrmse_by_year,
        'smape_by_year': smape_by_year,
        'r2_by_year': r2_by_year
    }

    return results


def compute_temporal_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: np.ndarray
) -> Dict[str, float]:
    """
    Compute temporal correlation metrics.

    For each region r, compute Pearson correlation across the available test years:
        r_tm(r) = r({y_r,t}_{t∈T}, {ŷ_r,t}_{t∈T})
    Then report median across regions:
        r_tm = median_{r∈R} r_tm(r)

    This metric characterizes the ability to reproduce inter-annual yield
    variability within a typical region.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)

    Returns:
        Dictionary with:
            - r_tm_overall: Median temporal correlation across regions
            - r_tm_by_region: Dict of {region: correlation} for each region
            - rmse_by_region: Dict of {region: RMSE} for each region
            - mae_by_region: Dict of {region: MAE} for each region
            - nrmse_by_region: Dict of {region: NRMSE} for each region
            - smape_by_region: Dict of {region: SMAPE} for each region
            - r2_by_region: Dict of {region: R²} for each region
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'year': years,
        'region': regions
    })

    results = {}
    r_tm_by_region = {}
    rmse_by_region = {}
    mae_by_region = {}
    nrmse_by_region = {}
    smape_by_region = {}
    r2_by_region = {}

    for region in sorted(df['region'].unique()):
        region_df = df[df['region'] == region]
        y_t = region_df['y_true'].values
        y_p = region_df['y_pred'].values

        # Need at least 2 years to compute correlation
        if len(y_t) < 2:
            r_tm_by_region[region] = np.nan
            rmse_by_region[region] = np.nan
            mae_by_region[region] = np.nan
            nrmse_by_region[region] = np.nan
            smape_by_region[region] = np.nan
            r2_by_region[region] = np.nan
            continue

        # Pearson correlation
        r_tm_by_region[region] = pearsonr(y_t, y_p)[0]

        # RMSE
        rmse_by_region[region] = np.sqrt(mean_squared_error(y_t, y_p))

        # MAE
        mae_by_region[region] = mean_absolute_error(y_t, y_p)

        # NRMSE (Normalized RMSE): RMSE / mean(y_true)
        mean_y_t = np.mean(y_t)
        nrmse_by_region[region] = rmse_by_region[region] / mean_y_t if mean_y_t != 0 else np.nan

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape_by_region[region] = 100.0 / len(y_t) * np.sum(
            2 * np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)
        )

        # R²
        if len(y_t) >= 2:
            r2_by_region[region] = r2_score(y_t, y_p)
        else:
            r2_by_region[region] = np.nan

    # Median temporal correlation across regions
    valid_correlations = [v for v in r_tm_by_region.values() if not np.isnan(v)]
    r_tm_overall = np.median(valid_correlations) if valid_correlations else np.nan

    results = {
        'r_tm_overall': r_tm_overall,
        'r_tm_by_region': r_tm_by_region,
        'rmse_by_region': rmse_by_region,
        'mae_by_region': mae_by_region,
        'nrmse_by_region': nrmse_by_region,
        'smape_by_region': smape_by_region,
        'r2_by_region': r2_by_region
    }

    return results


def compute_anomaly_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: np.ndarray
) -> Dict[str, float]:
    """
    Compute anomaly correlation metrics.

    Observed and predicted yields are centered using the observed regional mean:
        y'_r,t = y_r,t - ȳ_r
        ŷ'_r,t = ŷ_r,t - ȳ_r
    where
        ȳ_r = (1/|T|) Σ_{t∈T} y_r,t

    Anomaly skill is quantified as the Pearson correlation computed over all
    de-meaned region-year observations:
        r_an = r({y'_{r,t}}_{r,t}, {ŷ'_{r,t}}_{r,t})

    This metric evaluates whether the model correctly predicts years with
    above- or below-average yields independently of persistent regional
    productivity differences.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)

    Returns:
        Dictionary with:
            - r_an_overall: Pearson correlation on anomalies
            - rmse_anomaly: RMSE on anomalies
            - mae_anomaly: MAE on anomalies
            - nrmse_anomaly: NRMSE on anomalies
            - smape_anomaly: SMAPE on anomalies
            - bias_anomaly: Mean error on anomalies (systematic over/under prediction)
            - r2_anomaly: R² on anomalies
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'year': years,
        'region': regions
    })

    # Compute regional means from observed yields
    regional_means = df.groupby('region')['y_true'].mean()

    # De-mean the data
    df['y_true_anomaly'] = df.apply(
        lambda row: row['y_true'] - regional_means[row['region']],
        axis=1
    )
    df['y_pred_anomaly'] = df.apply(
        lambda row: row['y_pred'] - regional_means[row['region']],
        axis=1
    )

    y_t_anomaly = df['y_true_anomaly'].values
    y_p_anomaly = df['y_pred_anomaly'].values

    # Pearson correlation on anomalies
    r_an_overall = pearsonr(y_t_anomaly, y_p_anomaly)[0]

    # RMSE
    rmse_anomaly = np.sqrt(mean_squared_error(y_t_anomaly, y_p_anomaly))

    # MAE
    mae_anomaly = mean_absolute_error(y_t_anomaly, y_p_anomaly)

    # NRMSE (Normalized RMSE): RMSE / mean(|y_true_anomaly|)
    # For anomalies, use mean of absolute values since mean of anomalies is ~0
    mean_abs_y_t_anomaly = np.mean(np.abs(y_t_anomaly))
    nrmse_anomaly = rmse_anomaly / mean_abs_y_t_anomaly if mean_abs_y_t_anomaly != 0 else np.nan

    # SMAPE on anomalies
    smape_anomaly = 100.0 / len(y_t_anomaly) * np.sum(
        2 * np.abs(y_t_anomaly - y_p_anomaly) / (np.abs(y_t_anomaly) + np.abs(y_p_anomaly) + 1e-8)
    )

    # Bias (mean error)
    bias_anomaly = np.mean(y_p_anomaly - y_t_anomaly)  # Positive = over-prediction

    # R² on anomalies
    if len(y_t_anomaly) >= 2:
        r2_anomaly = r2_score(y_t_anomaly, y_p_anomaly)
    else:
        r2_anomaly = np.nan

    results = {
        'r_an_overall': r_an_overall,
        'rmse_anomaly': rmse_anomaly,
        'mae_anomaly': mae_anomaly,
        'nrmse_anomaly': nrmse_anomaly,
        'smape_anomaly': smape_anomaly,
        'bias_anomaly': bias_anomaly,
        'r2_anomaly': r2_anomaly
    }

    return results


def compute_all_spatiotemporal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute all spatial, temporal, and anomaly metrics.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)

    Returns:
        Dictionary with keys 'spatial', 'temporal', 'anomaly', each containing
        the respective metrics dictionary.
    """
    results = {
        'spatial': compute_spatial_correlation(y_true, y_pred, years, regions),
        'temporal': compute_temporal_correlation(y_true, y_pred, years, regions),
        'anomaly': compute_anomaly_correlation(y_true, y_pred, years, regions)
    }
    return results


def flatten_spatiotemporal_metrics(metrics: Dict[str, Dict]) -> Dict[str, float]:
    """
    Flatten nested spatiotemporal metrics dictionary for logging.

    Converts:
        {'spatial': {'r_sp_overall': 0.85, 'r_sp_by_year': {...}}}
    To:
        {'spatial/r_sp_overall': 0.85, 'spatial/r_sp_2018': 0.82, ...}

    Args:
        metrics: Nested dictionary from compute_all_spatiotemporal_metrics

    Returns:
        Flattened dictionary with slash-separated keys
    """
    flat = {}

    for category, cat_metrics in metrics.items():
        for key, value in cat_metrics.items():
            if isinstance(value, dict):
                # Nested dict (e.g., r_sp_by_year)
                for sub_key, sub_value in value.items():
                    flat[f'{category}/{key}_{sub_key}'] = sub_value
            else:
                flat[f'{category}/{key}'] = value

    return flat


def flatten_spatiotemporal_metrics_simple(metrics: Dict[str, Dict]) -> Dict[str, float]:
    """
    Flatten spatiotemporal metrics to only overall correlation values.

    Returns only the key correlation metrics (r_sp, r_tm, r_an) without
    breakdowns by year/region or additional metrics like NRMSE, R², etc.

    Args:
        metrics: Nested dictionary from compute_all_spatiotemporal_metrics

    Returns:
        Simplified flattened dictionary with only overall correlation values
    """
    return {
        'spatial/r_sp_overall': metrics['spatial']['r_sp_overall'],
        'temporal/r_tm_overall': metrics['temporal']['r_tm_overall'],
        'anomaly/r_an_overall': metrics['anomaly']['r_an_overall']
    }


def save_spatiotemporal_metrics(
    metrics: Dict[str, Dict],
    save_dir: str,
    run_id: str,
    timestamp: str
):
    """
    Save spatiotemporal metrics to organized CSV files.

    Creates the following structure:
        save_dir/
            spatial/
                correlation_by_year.csv      # Spatial correlation (r_sp) by year
                normal_by_year.csv          # Spatial normal metrics by year
            temporal/
                correlation_by_region.csv  # Temporal correlation (r_tm) by region
                normal_by_region.csv      # Temporal normal metrics by region
            anomaly/
                anomaly_overall.csv        # Anomaly correlation (r_an)

    Note: Spatial metrics are organized by year because spatial correlation
          is computed across regions FOR EACH YEAR.
          Temporal metrics are organized by region because temporal correlation
          is computed across years FOR EACH REGION.

    Args:
        metrics: Dictionary from compute_all_spatiotemporal_metrics
        save_dir: Base directory to save results
        run_id: Unique run identifier
        timestamp: Timestamp string
    """
    # Create folders
    spatial_dir = os.path.join(save_dir, 'spatial')
    temporal_dir = os.path.join(save_dir, 'temporal')
    anomaly_dir = os.path.join(save_dir, 'anomaly')

    for dir_path in [spatial_dir, temporal_dir, anomaly_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Save spatial metrics
    spatial = metrics['spatial']
    if 'r_sp_by_year' in spatial:
        # Correlation by year
        r_sp_df = pd.DataFrame([
            {'year': year, 'r_sp': corr}
            for year, corr in spatial['r_sp_by_year'].items()
        ])
        r_sp_df['run_id'] = run_id
        r_sp_df['timestamp'] = timestamp
        r_sp_path = os.path.join(spatial_dir, 'correlation_by_year.csv')
        r_sp_df.to_csv(r_sp_path, index=False)

        # Normal metrics by year
        normal_by_year_df = pd.DataFrame([
            {
                'year': year,
                'rmse': spatial['rmse_by_year'].get(year, np.nan),
                'mae': spatial['mae_by_year'].get(year, np.nan),
                'nrmse': spatial['nrmse_by_year'].get(year, np.nan),
                'smape': spatial['smape_by_year'].get(year, np.nan),
                'r2': spatial['r2_by_year'].get(year, np.nan)
            }
            for year in spatial['r_sp_by_year'].keys()
        ])
        normal_by_year_df['run_id'] = run_id
        normal_by_year_df['timestamp'] = timestamp
        normal_path = os.path.join(spatial_dir, 'normal_by_year.csv')
        normal_by_year_df.to_csv(normal_path, index=False)

    # Save temporal metrics
    temporal = metrics['temporal']
    if 'r_tm_by_region' in temporal:
        # Correlation by region
        r_tm_df = pd.DataFrame([
            {'region': region, 'r_tm': corr}
            for region, corr in temporal['r_tm_by_region'].items()
        ])
        r_tm_df['run_id'] = run_id
        r_tm_df['timestamp'] = timestamp
        r_tm_path = os.path.join(temporal_dir, 'correlation_by_region.csv')
        r_tm_df.to_csv(r_tm_path, index=False)

        # Normal metrics by region
        normal_by_region_df = pd.DataFrame([
            {
                'region': region,
                'rmse': temporal['rmse_by_region'].get(region, np.nan),
                'mae': temporal['mae_by_region'].get(region, np.nan),
                'nrmse': temporal['nrmse_by_region'].get(region, np.nan),
                'smape': temporal['smape_by_region'].get(region, np.nan),
                'r2': temporal['r2_by_region'].get(region, np.nan)
            }
            for region in temporal['r_tm_by_region'].keys()
        ])
        normal_by_region_df['run_id'] = run_id
        normal_by_region_df['timestamp'] = timestamp
        normal_path = os.path.join(temporal_dir, 'normal_by_region.csv')
        normal_by_region_df.to_csv(normal_path, index=False)

    # Save anomaly metrics
    anomaly = metrics['anomaly']
    anomaly_df = pd.DataFrame([{
        'r_an_overall': anomaly['r_an_overall'],
        'rmse_anomaly': anomaly['rmse_anomaly'],
        'mae_anomaly': anomaly['mae_anomaly'],
        'nrmse_anomaly': anomaly['nrmse_anomaly'],
        'smape_anomaly': anomaly['smape_anomaly'],
        'bias_anomaly': anomaly['bias_anomaly'],
        'r2_anomaly': anomaly['r2_anomaly'],
        'run_id': run_id,
        'timestamp': timestamp
    }])
    anomaly_path = os.path.join(anomaly_dir, 'anomaly_overall.csv')
    anomaly_df.to_csv(anomaly_path, index=False)

    print(f"[Spatiotemporal Metrics] Saved to:")
    print(f"  Spatial: {spatial_dir}/")
    print(f"  Temporal: {temporal_dir}/")
    print(f"  Anomaly: {anomaly_dir}/")
