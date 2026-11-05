# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: SINGLE SOURCE OF TRUTH for every yield-forecasting metric.

This module consolidates what used to live in two separate files:
  * validateModel.py       — in-training metric machinery (nn.Module wrappers used
                             inside the LightningModules, printing helpers, the
                             walk-forward orchestration, sklearn OOD evaluation).
  * evaluationProtocol.py  — the post-hoc research protocol (reference forecasts,
                             skill scores, the tiered one-row-per-cell suite,
                             cross-country transfer).

The two files computed the SAME point metrics (rmse/mae/r2/mape/smape/nrmse) and the
SAME uncertainty metrics (pinball/crps/picp/pinaw/winkler/width) up to THREE times
each, through different libraries and with silently diverging NRMSE / SMAPE
conventions. Here every metric is defined ONCE, library-sourced where a well-
established implementation exists, and BOTH the training path and the post-hoc path
call the identical function:

  * point_metrics(y_true, y_pred)             — mse/rmse/mae/mape/smape/r2/nrmse
  * uncertainty_metrics(y_true, q_pred, qs)   — pinball/crps/picp/pinaw/winkler/width

Library sourcing (standing user preference — never hand-roll a metric a standard
library provides):
  * point error / r2 / mape / pinball / precision/recall/f1 -> scikit-learn
  * CRPS + Winkler(=interval score)                         -> scoringrules
  * correlations                                            -> scipy (spatiotemporal_metrics)
Kept CUSTOM only where there is no q10/q50/q90 library equivalent: skill-vs-reference
scores, nrmse/smape, tolerance within_*, PICP/coverage/calibration-error/width/pinaw
(uncertainty_toolbox assumes Gaussian mean+/-std), anomaly_spread_ratio,
shortfall_capture_C, and the split-conformal (CQR) margin.

Pinned conventions (were previously inconsistent across the two files):
  * NRMSE = rmse / mean(y_true)   (the convention used by 3 of the 4 old copies and
                                   by torchmetrics NormalizedRootMeanSquaredError)
  * SMAPE = mean( 2*|y-yhat| / (|y|+|yhat|+eps) )

Python version: 3.12.0
--------------------
"""

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchmetrics

import scoringrules as sr
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    r2_score, mean_absolute_percentage_error, mean_pinball_loss,
)

from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET

logger = logging.getLogger(__name__)

# Make the architectures dir importable for the (lazy) TrendModel import used by
# ReferenceForecasts, mirroring the old evaluationProtocol.py path setup.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ARCH_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'architectures')
if _ARCH_DIR not in sys.path:
    sys.path.append(_ARCH_DIR)

# Optional deps kept for interface compatibility with the old UncertaintyMetrics.
try:
    from neuralforecast.losses.pytorch import MQLoss
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False
    MQLoss = None
    logging.warning("neuralforecast not available. MQLoss will use fallback implementation.")

EPSILON = 1e-8


# =============================================================================
# ===============  SPATIAL / TEMPORAL / ANOMALY CORRELATIONS  ==================
#
# Merged verbatim from the former process/spatiotemporal_metrics.py so that
# metrics.py is the single home for all evaluation code. These compute whether a
# model reproduces (1) the geographic yield pattern within a season (r_sp),
# (2) year-to-year variability within a region (r_tm), and (3) deviations from a
# reference level (r_an). Correlations are sourced from scipy; the error metrics
# from scikit-learn. Used by the tier classes below and by the in-training
# logging in architectures/{linearLayer,tstLayer,multitaskModels}.py.
# =============================================================================
def compute_spatial_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute spatial correlation metrics.

    For each test year t, compute Pearson correlation across administrative regions:
        r_sp(t) = r({y_r,t}_{r in R}, {yhat_r,t}_{r in R})
    Then report median across test years:
        r_sp = median_{t in T} r_sp(t)

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
            - r2_by_year: Dict of {year: R2} for each year
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

        # R2 (same as r2 for correlation, but computed differently)
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
    regions: np.ndarray,
    min_years: int = 3,
) -> Dict[str, float]:
    """
    Compute temporal correlation metrics.

    For each region r, compute Pearson correlation across the available test years:
        r_tm(r) = r({y_r,t}_{t in T}, {yhat_r,t}_{t in T})
    Then report median across regions:
        r_tm = median_{r in R} r_tm(r)

    This metric characterizes the ability to reproduce inter-annual yield
    variability within a typical region.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)
        min_years: Minimum number of years a region must have before its
            correlation is counted. With 2 points Pearson r is always +/-1, so
            the default of 3 is the smallest value that carries information.
            Regions below the threshold still get the error metrics but
            r_tm is NaN for them.

    Returns:
        Dictionary with:
            - r_tm_overall: Median temporal correlation across regions
            - r_tm_by_region: Dict of {region: correlation} for each region
            - rmse_by_region: Dict of {region: RMSE} for each region
            - mae_by_region: Dict of {region: MAE} for each region
            - nrmse_by_region: Dict of {region: NRMSE} for each region
            - smape_by_region: Dict of {region: SMAPE} for each region
            - r2_by_region: Dict of {region: R2} for each region
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

        # Need at least 2 years to compute error metrics
        if len(y_t) < 2:
            r_tm_by_region[region] = np.nan
            rmse_by_region[region] = np.nan
            mae_by_region[region] = np.nan
            nrmse_by_region[region] = np.nan
            smape_by_region[region] = np.nan
            r2_by_region[region] = np.nan
            continue

        # Pearson correlation (guarded: +/-1 is guaranteed with 2 points)
        if len(y_t) >= min_years and np.std(y_t) > 0 and np.std(y_p) > 0:
            r_tm_by_region[region] = pearsonr(y_t, y_p)[0]
        else:
            r_tm_by_region[region] = np.nan

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

        # R2
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
    regions: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute anomaly correlation metrics.

    Observed and predicted yields are centred on a per-sample reference level
    and the Pearson correlation is computed over all de-meaned region-year
    observations:
        y'_r,t  = y_r,t  - ref_r,t
        yhat'_r,t  = yhat_r,t  - ref_r,t
        r_an    = r({y'_{r,t}}_{r,t}, {yhat'_{r,t}}_{r,t})

    Two reference conventions are supported:

    * reference=None  ("window" reference, the original behaviour):
        ref_r,t = ybar_r = mean over the evaluated window of the OBSERVED yields
        of region r. This removes persistent between-region productivity
        differences AND any drift over the evaluated window, so r_an measures
        pure within-window year ranking. Note that the reference is built from
        the observed test targets, so it is only usable as a diagnostic, not as
        an out-of-sample skill claim.

    * reference=array (per-sample, e.g. a trend fitted on TRAINING years only):
        ref_r,t = trend_r(t). Anomalies are then "deviation from what a trend
        model would have predicted", which is the quantity the forecast has to
        earn. This is the convention to use for out-of-sample skill statements.

    Error metrics on anomalies are deliberately NOT returned for RMSE/MAE/bias:
    because the same reference is subtracted from y and yhat,
        (y - ref) - (yhat - ref) = y - yhat,
    so they are algebraically identical to the plain RMSE/MAE/bias. Only the
    metrics whose value actually changes under centring are reported.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)
        reference: Optional per-sample reference level, shape (n_samples,)

    Returns:
        Dictionary with:
            - r_an_overall: Pearson correlation on anomalies
            - r2_anomaly: R2 on anomalies (SS_tot is the anomaly variance,
              so this is "variance of deviations-from-reference explained")
            - nrmse_anomaly: RMSE / mean(|y_true_anomaly|)
            - std_true_anomaly, std_pred_anomaly: spread of observed vs
              predicted anomalies (ratio < 1 flags a mean-reverting model)
            - reference_type: 'window' or 'external'
            - n_samples
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regions = np.asarray(regions)

    if reference is None:
        df = pd.DataFrame({'y_true': y_true, 'region': regions})
        regional_means = df.groupby('region')['y_true'].transform('mean').values
        ref = regional_means
        reference_type = 'window'
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != y_true.shape:
            raise ValueError(
                f"reference must have shape {y_true.shape}, got {ref.shape}"
            )
        reference_type = 'external'

    y_t_anomaly = y_true - ref
    y_p_anomaly = y_pred - ref

    valid = ~(np.isnan(y_t_anomaly) | np.isnan(y_p_anomaly))
    y_t_anomaly = y_t_anomaly[valid]
    y_p_anomaly = y_p_anomaly[valid]
    n = int(valid.sum())

    nan_result = {
        'r_an_overall': np.nan,
        'r2_anomaly': np.nan,
        'nrmse_anomaly': np.nan,
        'std_true_anomaly': np.nan,
        'std_pred_anomaly': np.nan,
        'reference_type': reference_type,
        'n_samples': n,
    }
    if n < 3 or np.std(y_t_anomaly) == 0 or np.std(y_p_anomaly) == 0:
        return nan_result

    r_an_overall = pearsonr(y_t_anomaly, y_p_anomaly)[0]
    rmse_anomaly = np.sqrt(mean_squared_error(y_t_anomaly, y_p_anomaly))
    mean_abs_y_t_anomaly = np.mean(np.abs(y_t_anomaly))
    nrmse_anomaly = rmse_anomaly / mean_abs_y_t_anomaly if mean_abs_y_t_anomaly != 0 else np.nan
    r2_anomaly = r2_score(y_t_anomaly, y_p_anomaly)

    return {
        'r_an_overall': float(r_an_overall),
        'r2_anomaly': float(r2_anomaly),
        'nrmse_anomaly': float(nrmse_anomaly),
        'std_true_anomaly': float(np.std(y_t_anomaly)),
        'std_pred_anomaly': float(np.std(y_p_anomaly)),
        'reference_type': reference_type,
        'n_samples': n,
    }


def compute_all_spatiotemporal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    years: np.ndarray,
    regions: np.ndarray,
    trend_reference: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Compute all spatial, temporal, and anomaly metrics.

    Args:
        y_true: Ground truth yields, shape (n_samples,)
        y_pred: Predicted yields, shape (n_samples,)
        years: Year for each sample, shape (n_samples,)
        regions: Region ID for each sample, shape (n_samples,)
        trend_reference: Optional per-sample reference level (e.g. the
            training-fitted trend evaluated at each test region-year). When
            given, an additional 'anomaly_trend' block is returned.

    Returns:
        Dictionary with keys 'spatial', 'temporal', 'anomaly' (window-referenced)
        and, if trend_reference is given, 'anomaly_trend'.
    """
    results = {
        'spatial': compute_spatial_correlation(y_true, y_pred, years, regions),
        'temporal': compute_temporal_correlation(y_true, y_pred, years, regions),
        'anomaly': compute_anomaly_correlation(y_true, y_pred, years, regions),
    }
    if trend_reference is not None:
        results['anomaly_trend'] = compute_anomaly_correlation(
            y_true, y_pred, years, regions, reference=trend_reference
        )
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
    breakdowns by year/region or additional metrics like NRMSE, R2, etc.

    Args:
        metrics: Nested dictionary from compute_all_spatiotemporal_metrics

    Returns:
        Simplified flattened dictionary with only overall correlation values
    """
    out = {
        'spatial/r_sp_overall': metrics['spatial']['r_sp_overall'],
        'temporal/r_tm_overall': metrics['temporal']['r_tm_overall'],
        'anomaly/r_an_overall': metrics['anomaly']['r_an_overall'],   # window-referenced
    }
    if 'anomaly_trend' in metrics:
        out['anomaly/r_an_trend'] = metrics['anomaly_trend']['r_an_overall']
    return out


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

    # Save anomaly metrics (one row per reference convention)
    anomaly_rows = []
    for block_name in ('anomaly', 'anomaly_trend'):
        if block_name not in metrics:
            continue
        a = metrics[block_name]
        anomaly_rows.append({
            'reference': a.get('reference_type', 'window' if block_name == 'anomaly' else 'external'),
            'r_an_overall': a['r_an_overall'],
            'r2_anomaly': a['r2_anomaly'],
            'nrmse_anomaly': a['nrmse_anomaly'],
            'std_true_anomaly': a.get('std_true_anomaly', np.nan),
            'std_pred_anomaly': a.get('std_pred_anomaly', np.nan),
            'n_samples': a.get('n_samples', np.nan),
            'run_id': run_id,
            'timestamp': timestamp
        })
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_path = os.path.join(anomaly_dir, 'anomaly_overall.csv')
    anomaly_df.to_csv(anomaly_path, index=False)

    print(f"[Spatiotemporal Metrics] Saved to:")
    print(f"  Spatial: {spatial_dir}/")
    print(f"  Temporal: {temporal_dir}/")
    print(f"  Anomaly: {anomaly_dir}/")


# =============================================================================
# ======================  SINGLE-SOURCE METRIC PRIMITIVES  ====================
# Every metric below is defined exactly once. Training modules AND the post-hoc
# tiers both call these — there is no second implementation anywhere.
# =============================================================================
def _rmse(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(root_mean_squared_error(a, b)) if len(a) else np.nan


def _mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(mean_absolute_error(a, b)) if len(a) else np.nan


def point_metrics(y_true, y_pred, include_nrmse: bool = True,
                  min_samples_for_r2: int = 2) -> Dict[str, float]:
    """The ONE point-metric implementation (library-sourced).

    Returns mse, rmse, mae, mape, smape, r2, and (optionally) nrmse as plain
    floats. Called identically during training (ModelMetrics, the per-year test
    block) and post-hoc (measureInDomainAccuracy).

    Conventions: NRMSE = rmse/mean(y_true); SMAPE = mean(2|y-yhat|/(|y|+|yhat|+eps)).
    r2 is NaN when there are fewer than `min_samples_for_r2` samples (sklearn needs 2).
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    out: Dict[str, float] = {}
    if len(y_true) == 0:
        return out
    out['mse'] = float(mean_squared_error(y_true, y_pred))
    out['rmse'] = float(root_mean_squared_error(y_true, y_pred))
    out['mae'] = float(mean_absolute_error(y_true, y_pred))
    out['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
    out['smape'] = float(np.mean(2.0 * np.abs(y_true - y_pred) /
                                 (np.abs(y_true) + np.abs(y_pred) + EPSILON)))
    out['r2'] = float(r2_score(y_true, y_pred)) if len(y_true) >= min_samples_for_r2 else np.nan
    if include_nrmse:
        mu = float(np.mean(y_true))
        out['nrmse'] = float(out['rmse'] / mu) if mu else np.nan
    return out


def skill_score(err_model: float, err_reference: float) -> float:
    """1 - err_model/err_reference; >0 means better than the reference."""
    if err_reference is None or not np.isfinite(err_reference) or err_reference == 0:
        return np.nan
    return float(1.0 - err_model / err_reference)


def tolerance_accuracy(y_true, y_pred,
                       rel_tols: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
                       abs_tols: Sequence[float] = (0.1, 0.5,)) -> Dict[str, float]:
    """Fraction of samples whose error is within the tolerance (custom; no library)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = np.abs(y_true - y_pred)
    out = {}
    for t in rel_tols:
        denom = np.maximum(np.abs(y_true), 1e-8)
        out[f'within_{int(round(t * 100))}pct'] = float(np.mean(err / denom <= t))
    for t in abs_tols:
        out[f'within_{t:g}abs'] = float(np.mean(err <= t))
    return out


def crps_from_quantiles(y, q_pred: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Per-sample CRPS from a finite set of quantiles, via scoringrules.

    `scoringrules.crps_quantile` implements the quantile-decomposition CRPS
    (CRPS = 2 * integral of the pinball loss over tau, approximated by the mean
    over the available levels). This is the ONE CRPS path used everywhere.
    """
    y = np.asarray(y, float)
    q_pred = np.asarray(q_pred, float)
    alpha = np.asarray(list(quantiles), float)
    return np.asarray(sr.crps_quantile(y, q_pred, alpha), dtype=float)


def _picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (custom; no q-based library)."""
    y_true = np.asarray(y_true, float)
    inside = (y_true >= np.asarray(y_lower, float)) & (y_true <= np.asarray(y_upper, float))
    return float(np.mean(inside)) if len(y_true) else np.nan


def _pinaw(y_lower: np.ndarray, y_upper: np.ndarray,
           y_true: Optional[np.ndarray] = None) -> float:
    """Prediction Interval Normalized Average Width (custom; no q-based library)."""
    y_lower = np.asarray(y_lower, float)
    y_upper = np.asarray(y_upper, float)
    mean_width = float(np.mean(y_upper - y_lower))
    if y_true is not None and len(y_true) > 0:
        rng = float(np.max(y_true) - np.min(y_true))
    else:
        rng = float(np.max(y_upper) - np.min(y_lower))
    if rng == 0 or not np.isfinite(rng):
        return 0.0
    return float(mean_width / rng)


def quantile_coverage(y, q_pred: np.ndarray, quantiles: Sequence[float]) -> Dict[str, float]:
    """Empirical P(y <= q_tau) per tau and the mean absolute calibration error."""
    y = np.asarray(y, float)
    cov = {t: float(np.mean(y <= q_pred[:, i])) for i, t in enumerate(quantiles)}
    out = {f'coverage_q{int(round(t * 100))}': c for t, c in cov.items()}
    out['quantile_calibration_error'] = float(np.mean([abs(c - t) for t, c in cov.items()]))
    return out


def uncertainty_metrics(y_true, y_pred_quantiles, quantiles: Sequence[float]) -> Dict[str, float]:
    """The ONE uncertainty-metric implementation for the shared quantile set.

    Returns pinball_q*, pinball_avg, crps, picp, pinaw, winkler_score,
    mean_interval_width. Called identically during training (UncertaintyMetrics /
    compute_uncertainty_metrics) and post-hoc (measureCalibratedUncertainity adds
    coverage / ref / conformal blocks ON TOP of this base).

    pinball -> sklearn.mean_pinball_loss; crps + winkler -> scoringrules;
    picp/pinaw -> custom (no q10/q50/q90 library equivalent).
    """
    y_true = np.asarray(y_true, float)
    Q = np.asarray(y_pred_quantiles, float)
    quantiles = list(quantiles)
    results: Dict[str, float] = {}

    # 1. Pinball per quantile (sklearn) + average
    pin = [float(mean_pinball_loss(y_true, Q[:, i], alpha=q)) for i, q in enumerate(quantiles)]
    for q, v in zip(quantiles, pin):
        results[f'pinball_q{int(round(q * 100))}'] = v
    results['pinball_avg'] = float(np.mean(pin))

    # 2. CRPS (scoringrules quantile-decomposition, via the shared primitive)
    results['crps'] = float(np.mean(crps_from_quantiles(y_true, Q, quantiles)))

    # 3-6. Interval diagnostics using the outer (first/last) quantiles
    lo, hi = 0, -1
    results['picp'] = _picp(y_true, Q[:, lo], Q[:, hi])
    results['pinaw'] = _pinaw(Q[:, lo], Q[:, hi], y_true)
    coverage_level = 1.0 - (quantiles[hi] - quantiles[lo])
    results['winkler_score'] = float(np.mean(sr.interval_score(
        y_true, Q[:, lo], Q[:, hi], coverage_level)))
    results['mean_interval_width'] = float(np.mean(Q[:, hi] - Q[:, lo]))
    return results


def bad_year_detection(anom_true: np.ndarray, score: np.ndarray, pred_event: np.ndarray,
                       threshold: np.ndarray) -> Dict[str, float]:
    """Event = observed anomaly below -threshold. AUC/precision/recall/f1 via sklearn."""
    event = anom_true < -threshold
    n_ev = int(event.sum())
    out = {'n_events': n_ev, 'base_rate': float(event.mean()) if len(event) else np.nan}
    if n_ev == 0 or n_ev == len(event):
        out.update({'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
        return out
    out['auc'] = float(roc_auc_score(event, score))
    out['precision'] = float(precision_score(event, pred_event, zero_division=0))
    out['recall'] = float(recall_score(event, pred_event, zero_division=0))
    out['f1'] = float(f1_score(event, pred_event, zero_division=0))
    return out


def conformal_margin(y_val, q_lo_val, q_hi_val, alpha: float) -> float:
    """Split-conformal margin for a two-sided interval (CQR, Romano et al. 2019)."""
    y_val = np.asarray(y_val, float)
    s = np.maximum(np.asarray(q_lo_val, float) - y_val, y_val - np.asarray(q_hi_val, float))
    n = len(s)
    if n == 0:
        return np.nan
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level))


def _shortfall_capture(anom_true: np.ndarray, anom_pred: np.ndarray,
                       decile: float = 0.10) -> Tuple[float, int]:
    """Shortfall capture C on the bottom-decile observed anomalies (custom).

    C = mean(-anom_pred) / mean(-anom_true) over the worst-decile region-years.
    C=1 captures full depth, C=0 none, C<0 wrong sign. Returns (C, n_tail).
    """
    a, ah = np.asarray(anom_true, float), np.asarray(anom_pred, float)
    if len(a) == 0:
        return np.nan, 0
    thr = float(np.quantile(a, decile))
    tail = a <= thr
    n = int(tail.sum())
    if n == 0:
        return np.nan, 0
    denom = float(np.mean(-a[tail]))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, n
    return float(np.mean(-ah[tail]) / denom), n


# =============================================================================
# ======================  IN-TRAINING METRIC MODULES  =========================
# nn.Module wrappers used inside the LightningModules. They accumulate
# predictions over the epoch and then call the SAME primitives above, so the
# numbers are identical to the post-hoc protocol's.
# =============================================================================
class ModelMetrics(nn.Module):
    """
    Point metrics for in-/end-of-season yield forecasting (MSE, MAE, RMSE, R2,
    MAPE, SMAPE, NRMSE), computed by accumulating predictions over the epoch and
    calling the shared `point_metrics()` at compute() time — the identical code
    path used by the post-hoc protocol.

    compute() returns a dict of 0-dim torch tensors so the existing epoch-end
    hooks (torch.sqrt(results['mse']).item(), self.log(...)) keep working
    unchanged. Prefix (train/val/test) is used for wandb logging namespacing.
    Inherits nn.Module so it moves with the parent LightningModule to GPU.
    """

    def __init__(self, prefix: str = "test", include_nrmse: bool = True):
        super().__init__()
        self.prefix = prefix
        self.include_nrmse = include_nrmse
        self._preds: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
        else:
            preds = torch.as_tensor(np.asarray(preds, dtype=float))
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()
        else:
            targets = torch.as_tensor(np.asarray(targets, dtype=float))
        self._preds.append(preds.reshape(-1).float())
        self._targets.append(targets.reshape(-1).float())

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        p = torch.cat(self._preds).numpy() if self._preds else np.array([])
        t = torch.cat(self._targets).numpy() if self._targets else np.array([])
        return t, p

    def compute(self) -> Dict[str, torch.Tensor]:
        if not self._preds:
            return {}
        t, p = self._arrays()
        m = point_metrics(t, p, include_nrmse=self.include_nrmse)
        # Wrap as 0-dim tensors to preserve the historical compute() interface.
        return {k: torch.tensor(float(v)) for k, v in m.items()}

    def reset(self):
        self._preds = []
        self._targets = []

    def log_results(self, step: str = "val"):
        results = self.compute()
        if not results:
            return
        print(f"\n{'-' * 60}")
        print(f"{step.upper()} METRICS ({self.prefix.upper()}):")
        print(f"MSE:   {float(results['mse']):.4f}")
        print(f"MAE:   {float(results['mae']):.4f}")
        print(f"RMSE:  {float(results['rmse']):.4f}")
        print(f"R2:    {float(results['r2']):.4f}")
        print(f"MAPE:  {float(results['mape']):.4f}")
        print(f"SMAPE: {float(results['smape']):.4f}")
        if 'nrmse' in results:
            print(f"NRMSE: {float(results['nrmse']):.4f}")
        print(f"{'-' * 60}")


class UncertaintyMetrics(nn.Module):
    """
    Uncertainty metrics for quantile regression (Pinball, CRPS, PICP, PINAW,
    Winkler, interval width). Accumulates (y_true, y_pred_quantiles) over the
    epoch and calls the shared `uncertainty_metrics()` at compute() time — the
    identical code path used by the post-hoc protocol.
    """

    def __init__(self, quantiles: List[float] = None, prefix: str = ''):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.prefix = prefix
        self.reset()
        # Kept for interface compatibility with the old class (unused in compute()).
        self.mq_loss = MQLoss(quantiles=quantiles) if NEURALFORECAST_AVAILABLE else None

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu()
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self) -> Dict[str, float]:
        if len(self.y_true) == 0:
            return {}
        y_true_all = torch.cat(self.y_true, dim=0).numpy()
        y_pred_all = torch.cat(self.y_pred, dim=0).numpy()
        metrics = uncertainty_metrics(y_true_all, y_pred_all, self.quantiles)
        if self.prefix:
            metrics = {f'{self.prefix}{k}': v for k, v in metrics.items()}
        return metrics


def compute_uncertainty_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred_quantiles: Union[np.ndarray, torch.Tensor],
    quantiles: List[float] = None,
) -> Dict[str, float]:
    """One-shot uncertainty metrics (thin wrapper over the shared primitive)."""
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    if isinstance(y_pred_quantiles, torch.Tensor):
        y_pred_quantiles = y_pred_quantiles.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    return uncertainty_metrics(y_true, y_pred_quantiles, quantiles)


# =============================================================================
# =====================  PRINTING / FORMATTING HELPERS  =======================
# =============================================================================
def format_metrics_dict(results: Dict) -> Dict[str, float]:
    """Convert a ModelMetrics.compute() tensor dict to a dict of plain floats."""
    def _f(key):
        if key not in results:
            return None
        v = results[key]
        return float(v.item()) if isinstance(v, torch.Tensor) else float(v)

    rmse = _f('rmse')
    if rmse is None and 'mse' in results:
        mse = _f('mse')
        rmse = math.sqrt(mse) if mse is not None else None
    return {
        'mse': _f('mse'),
        'mae': _f('mae'),
        'rmse': rmse,
        'r2': _f('r2'),
        'mape': _f('mape'),
        'smape': _f('smape'),
        'nrmse': _f('nrmse'),
    }


def print_metrics_table(title: str, metrics: Dict, step: str = "test"):
    """Print a formatted table of core + uncertainty + spatiotemporal + protocol metrics."""
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
        print(f"R2:    {metrics['r2']:.4f}")
    if metrics.get('mape') is not None:
        print(f"MAPE:  {metrics['mape']:.2f}%")
    if metrics.get('smape') is not None:
        print(f"SMAPE: {metrics['smape']:.2f}%")
    if metrics.get('nrmse') is not None:
        print(f"NRMSE: {metrics['nrmse']:.4f}")

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

    st_keys = [k for k in metrics.keys()
               if k.startswith('spatial/') or k.startswith('temporal/') or k.startswith('anomaly/')]
    if st_keys:
        print(f"\n{'=' * 70}")
        print("SPATIOTEMPORAL METRICS")
        print(f"{'=' * 70}")
        if metrics.get('spatial/r_sp_overall') is not None:
            print(f"Spatial Correlation (r_sp):  {metrics['spatial/r_sp_overall']:.4f}")
        if metrics.get('temporal/r_tm_overall') is not None:
            print(f"Temporal Correlation (r_tm): {metrics['temporal/r_tm_overall']:.4f}")
        if metrics.get('anomaly/r_an_overall') is not None:
            print(f"Anomaly Correlation (r_an, window ref):  {metrics['anomaly/r_an_overall']:.4f}")
        if metrics.get('anomaly/r_an_trend') is not None:
            print(f"Anomaly Correlation (r_an, trend ref):   {metrics['anomaly/r_an_trend']:.4f}")

    proto_keys = [k for k in metrics.keys() if k.startswith('protocol/')]
    if proto_keys:
        print(f"\n{'=' * 70}")
        print("EVALUATION PROTOCOL (overall; see protocol/ CSVs for per-year)")
        print(f"{'=' * 70}")
        for label, key in [
            ('Skill RMSE vs trend (MK)', 'protocol/skill_rmse_vs_trend'),
            ('Skill RMSE vs OLS', 'protocol/skill_rmse_vs_ols'),
            ('Skill RMSE vs climatology', 'protocol/skill_rmse_vs_climatology'),
            ('Skill RMSE vs rolling-3', 'protocol/skill_rmse_vs_rolling3'),
            ('Within +/-10%', 'protocol/within_10pct'),
            ('Within +/-0.5 t/ha', 'protocol/within_0.5abs'),
            ('r_an (OLS-detrended)', 'protocol/r_an_detrended'),
            ('Bad-year AUC (1sig)', 'protocol/badyear_1sig_auc'),
            ('PICP', 'protocol/picp'),
            ('PICP (conformal)', 'protocol/picp_conformal'),
            ('CRPS skill vs OLS', 'protocol/skill_crps_vs_ols'),
        ]:
            v = metrics.get(key)
            if v is not None and v == v:  # not NaN
                print(f"{label:<28}{v:.4f}")

    print(f"{'-' * 70}")


# =============================================================================
# =================  SKLEARN PER-YEAR / OUT-OF-DOMAIN EVAL  ====================
# =============================================================================
def evaluate_predictions_by_year(y_true, y_pred, years, metrics=None,
                                 min_samples_per_year=2, epsilon=1e-6):
    """
    Overall + per-year point metrics, robust to small samples / zero variance.

    Uses the shared library primitives (sklearn root_mean_squared_error, r2_score,
    mean_absolute_percentage_error). `normalized_rmse` here keeps its historical
    range-normalized-percent convention (rmse/(max-min)*100) because this sklearn
    OOD path reports it that way; the model NRMSE elsewhere is rmse/mean (see
    point_metrics). metrics: subset of ['r2','rmse','mape','normalized_rmse'].
    """
    if metrics is None:
        metrics = ['r2', 'rmse', 'mape', 'normalized_rmse']

    results = {}

    def compute_r2_or_nan(y_t, y_p):
        if len(y_t) < min_samples_per_year or np.std(y_t) < epsilon:
            return np.nan
        return r2_score(y_t, y_p)

    def compute_rmse(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return float(root_mean_squared_error(y_t, y_p))

    def compute_mape(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return mean_absolute_percentage_error(y_t, y_p)

    def compute_normalized_rmse(y_t, y_p):
        if len(y_t) < min_samples_per_year:
            return compute_rmse(y_t, y_p)
        denom = np.max(y_t) - np.min(y_t)
        return compute_rmse(y_t, y_p) / (denom + epsilon) * 100

    dispatch = {
        'r2': compute_r2_or_nan,
        'rmse': compute_rmse,
        'mape': compute_mape,
        'normalized_rmse': compute_normalized_rmse,
    }

    results['overall'] = {m: dispatch[m](y_true, y_pred) for m in metrics if m in dispatch}

    years = np.array(years)
    for year in sorted(set(years)):
        mask = np.where(years == year)[0]
        y_t, y_p = y_true[mask], y_pred[mask]
        results[year] = {m: dispatch[m](y_t, y_p) for m in metrics if m in dispatch}

    return results


def store_model_results(results_dict, model_name, country, crop,
                        file_path="../output/myoutputs/sklearn_model_results.csv"):
    rows = []
    for year, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            rows.append({
                "model": str(model_name),
                "country": str(country[0]),
                "crop": str(crop),
                "year": str(year),
                "metric": str(metric_name),
                "value": value,
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
    """Evaluate a trained sklearn pipeline on the EU countries in CY-Bench."""
    from cybench.datasets.configured import load_dfs_crop
    from cybench.datasets.dataset import Dataset as CYDataset
    from loadData import prepare_features_and_targets

    countries_to_evaluate = ["AT", "BE", "BG", "CZ", "DE", "DK", "EE", "EL", "ES", "FI",
                             "FR", "HR", "HU", "IE", "IT", "LT", "LV", "NL", "PL", "PT",
                             "RO", "SE", "SK"]
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
        test_years = [y for y in years_sorted if y >= 2018]
        _, test_ds = ds.split_on_years((train_years, test_years))

        X_test, y_test, years_test = prepare_features_and_targets(test_ds)
        y_pred = pipeline.predict(X_test)

        results_by_year = evaluate_predictions_by_year(y_test, y_pred, years_test)
        _ = store_model_results(results_by_year, model_name, country, crop, file_path)


# =============================================================================
# =========================  WALK-FORWARD VALIDATION  =========================
# =============================================================================
def save_walk_forward_results(results_matrix: List[Dict], all_years: List[int],
                              test_years: int, config, run_id: str, timestamp: str,
                              n_regions: Optional[int] = None) -> str:
    """Save walk-forward results (detailed + per-year aggregate) to CSV."""
    os.makedirs(config.results_dir, exist_ok=True)

    rows = []
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        train_end_year = fold_result['train_end_year']
        for test_year, metrics in fold_result['yearly_metrics'].items():
            rows.append({
                'run_id': run_id,
                'crop': config.crop,
                'country': config.country,
                'model_type': getattr(config, 'model_type', 'unknown'),
                'aggregation': getattr(config, 'aggregation', 'unknown'),
                'fold': fold_idx + 1,
                'train_end_year': train_end_year,
                'test_year': test_year,
                'n_regions': n_regions,
                'mse': metrics.get('mse'),
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'r2': metrics.get('r2'),
                'mape': metrics.get('mape'),
                'smape': metrics.get('smape'),
                'nrmse': metrics.get('nrmse'),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.results_dir,
                            f"{config.crop}_{config.country}_walkforward_{timestamp}_{run_id}.csv")
    df.to_csv(csv_path, index=False)

    metric_cols = ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']
    per_year_agg = df.groupby('test_year').agg(
        {**{col: ['mean', 'std'] for col in metric_cols}}).reset_index()
    per_year_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in per_year_agg.columns]
    per_year_agg['run_id'] = run_id
    per_year_agg['crop'] = config.crop
    per_year_agg['country'] = config.country
    per_year_agg['model_type'] = getattr(config, 'model_type', 'unknown')
    per_year_agg['aggregation'] = getattr(config, 'aggregation', 'unknown')

    agg_path = os.path.join(config.results_dir,
                            f"{config.crop}_{config.country}_walkforward_aggregated_{timestamp}_{run_id}.csv")
    per_year_agg.to_csv(agg_path, index=False)

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


def run_walk_forward_validation(all_years: List[int], test_years: int, config,
                                create_model_fn, datamodule_class, trainer_class,
                                max_epochs: int, loggers: List, run_id: str, timestamp: str,
                                save_checkpoint_dir: Optional[str] = None,
                                save_top_k: int = -1, save_last: bool = True) -> Dict:
    """Walk-forward validation where each fold is tested on ALL future years."""
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from eb_criterion import EBCriterionCallbackV2
    from cybench.process.loadData import generate_walk_forward_splits
    from cybench.process.alignment_patch import verify_forecast_horizon_config, test_alignment_patch

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION (Test on All Future Years)")
    print(f"{'=' * 70}\n")

    print("\n[Pre-flight check] Verifying alignment patch...")
    test_alignment_patch()

    wf_splits = generate_walk_forward_splits(all_years, test_years)

    print(f"[Walk-Forward Config]")
    print(f"Number of folds: {len(wf_splits)}")
    print(f"Max epochs per fold: {max_epochs}")
    print(f"\n[Walk-Forward Splits]")
    for s in wf_splits:
        print(f"Fold {s['fold_idx'] + 1}: train_years={len(s['train_years'])} "
              f"({min(s['train_years'])}-{max(s['train_years'])})")

    results_matrix = []
    n_regions: Optional[int] = None

    for fold_idx, split in enumerate(wf_splits):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{len(wf_splits)}")
        print(f"{'=' * 70}")
        print(f"Train years: {split['train_years']}")
        print(f"Train end year: {max(split['train_years'])}")

        fold_config = _update_config_for_fold(config, split['train_years'], max_epochs)
        verify_forecast_horizon_config(fold_config, fold_idx=split['fold_idx'])

        dm_fold = datamodule_class(fold_config)
        dm_fold.setup(train_years=split['train_years'], val_years=[], test_years=[])
        model_fold = create_model_fn(fold_config)

        # Count unique regions once from the first fold; consistent across folds.
        if fold_idx == 0:
            try:
                n_regions = len(set(dm_fold.train_ds.adm_ids))
            except Exception:
                n_regions = None

        fold_callbacks = [
            EBCriterionCallbackV2(
                batch_size=fold_config.batch_size, patience=2, smoothing=0.9,
                min_epochs=10, verbose=True, stopping_threshold=0.0,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        if save_checkpoint_dir:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            train_end_year = max(split['train_years'])
            checkpoint_filename = (f"{config.crop}_{config.country}_{config.model_type}_"
                                   f"fold{fold_idx + 1}_train_end_{train_end_year}_{{epoch}}")
            fold_callbacks.append(
                ModelCheckpoint(dirpath=save_checkpoint_dir, filename=checkpoint_filename,
                                monitor='train_loss', mode='min',
                                save_top_k=save_top_k, save_last=save_last))

        trainer_fold = trainer_class(
            max_epochs=fold_config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1, callbacks=fold_callbacks, logger=loggers,
            log_every_n_steps=10, enable_progress_bar=True, enable_model_summary=False,
        )

        print(f"\n[Fold {fold_idx + 1}] Training...")
        trainer_fold.fit(model_fold, dm_fold)

        train_end_year = max(split['train_years'])
        future_years = [y for y in all_years if y > train_end_year]
        print(f"\n[Fold {fold_idx + 1}] Testing on {len(future_years)} future years: {future_years}")

        fold_yearly_metrics = {}
        first_test_year_logged = False

        for test_year in future_years:
            dm_test = datamodule_class(fold_config)
            dm_test.copy_normalization_from(dm_fold)
            dm_test.setup(train_years=[], val_years=[], test_years=[test_year])

            test_results = trainer_fold.test(model_fold, dm_test, verbose=False)
            if test_results:
                r = test_results[0]
                metrics = {
                    'mse': r.get('test/mse'), 'mae': r.get('test/mae'),
                    'rmse': r.get('test/rmse'), 'r2': r.get('test/r2'),
                    'mape': r.get('test/mape'), 'smape': r.get('test/smape'),
                    'nrmse': r.get('test/nrmse'),
                }
                fold_yearly_metrics[test_year] = metrics
                print(f"Year {test_year}: NRMSE={metrics['nrmse']:.4f}, R2={metrics['r2']:.4f}")
                if not first_test_year_logged:
                    _log_test_metrics_to_wandb(loggers, metrics, fold_idx)
                    first_test_year_logged = True

        results_matrix.append({
            'fold_idx': fold_idx,
            'train_end_year': train_end_year,
            'yearly_metrics': fold_yearly_metrics,
        })

        if hasattr(model_fold, '_yield_predictions_cache'):
            model_fold._yield_predictions_cache.clear()
            logging.info(f"[Fold {fold_idx + 1}] Cleared prediction cache after testing")
        if hasattr(model_fold, '_prediction_cache'):
            model_fold._prediction_cache.clear()
            logging.info(f"[Fold {fold_idx + 1}] Cleared TST prediction cache after testing")

    aggregated = _aggregate_walk_forward_results(results_matrix, all_years, test_years)
    fold_all_year_metrics = _collect_fold_all_year_metrics(results_matrix)

    print(f"\n[DEBUG] Walk-forward summary - {len(results_matrix)} folds processed")
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        yearly_metrics = fold_result['yearly_metrics']
        first_year = min(yearly_metrics.keys()) if yearly_metrics else None
        print(f"Fold {fold_idx + 1}: {len(yearly_metrics)} test years, first year={first_year}")

    csv_path = save_walk_forward_results(results_matrix, all_years, test_years, config, run_id, timestamp, n_regions=n_regions)
    _log_walk_forward_to_wandb(loggers, aggregated, run_id, fold_all_year_metrics)

    return {
        'results_matrix': results_matrix,
        'overall_metrics': aggregated['overall'],
        'per_year_metrics': aggregated['per_year'],
        'first_year_metrics': aggregated.get('first_year'),
        'csv_path': csv_path,
    }


def _update_config_for_fold(config, train_years, max_epochs):
    """Create a per-fold config, preserving data_fraction and model hyperparameters."""
    fold_config = config.__class__(
        crop=config.crop, country=config.country,
        model_type=getattr(config, 'model_type', None),
        aggregation=getattr(config, 'aggregation', None),
        data_fraction=config.data_fraction,
        use_sota_features=getattr(config, 'use_sota_features', False),
        include_spatial_features=getattr(config, 'include_spatial_features', False),
        lag_years=getattr(config, 'lag_years', 1),
        load_checkpoint=None, seed=config.seed, batch_size=config.batch_size,
        num_workers=config.num_workers, max_epochs=max_epochs, lr=config.lr,
        weight_decay=config.weight_decay, test_years=1,
        use_residual_trend=getattr(config, 'use_residual_trend', False),
        use_recursive_lags=getattr(config, 'use_recursive_lags', False),
        use_gdd=getattr(config, 'use_gdd', False),
        use_heat_stress_days=getattr(config, 'use_heat_stress_days', False),
        use_rue=getattr(config, 'use_rue', False),
        use_rtai=getattr(config, 'use_rtai', False),
        use_cwb_feature=getattr(config, 'use_cwb_feature', False),
        drop_tavg=getattr(config, 'drop_tavg', False),
        use_revin=getattr(config, 'use_revin', False),
        results_dir=config.results_dir,
        lr_scheduler_lambda=getattr(config, 'lr_scheduler_lambda', None),
    )
    for attr in ['xlinear_hidden_size', 'xlinear_temporal_ff', 'xlinear_channel_ff',
                 'xlinear_dropout', 'patchtst_d_model', 'patchtst_num_attention_heads',
                 'patchtst_ffn_dim', 'patchtst_num_layers', 'patchtst_dropout']:
        if hasattr(config, attr):
            setattr(fold_config, attr, getattr(config, attr))
    return fold_config


def _aggregate_walk_forward_results(results_matrix: List[Dict], all_years: List[int],
                                    test_years: int) -> Dict:
    """Aggregate into overall (mean of per-year means), per-year, and first-year metrics."""
    metric_names = ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']
    per_year_values = {}
    first_year_values = {m: [] for m in metric_names}

    for fold_result in results_matrix:
        tested_years = sorted(fold_result['yearly_metrics'].keys())
        first_test_year = min(tested_years) if tested_years else None
        for test_year, metrics in fold_result['yearly_metrics'].items():
            if test_year not in per_year_values:
                per_year_values[test_year] = {m: [] for m in metric_names}
            for metric_name, value in metrics.items():
                if value is not None:
                    per_year_values[test_year][metric_name].append(value)
            if test_year == first_test_year:
                for metric_name, value in metrics.items():
                    if value is not None:
                        first_year_values[metric_name].append(value)

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

    overall = {}
    for metric_name in metric_names:
        year_values = [per_year[y][metric_name] for y in per_year if per_year[y][metric_name] is not None]
        if year_values:
            overall[metric_name] = np.mean(year_values)
            overall[f'{metric_name}_std'] = np.std(year_values)
        else:
            overall[metric_name] = None
            overall[f'{metric_name}_std'] = None

    first_year = {}
    for metric_name in metric_names:
        if first_year_values[metric_name]:
            first_year[metric_name] = np.mean(first_year_values[metric_name])
            first_year[f'{metric_name}_std'] = np.std(first_year_values[metric_name])
        else:
            first_year[metric_name] = None
            first_year[f'{metric_name}_std'] = None

    return {'overall': overall, 'per_year': per_year, 'first_year': first_year}


def _collect_fold_all_year_metrics(results_matrix: List[Dict]) -> Dict:
    """Map fold_idx -> {test_year -> metrics} for fold-level averaging."""
    fold_metrics = {}
    for fold_result in results_matrix:
        fold_metrics[fold_result['fold_idx']] = fold_result['yearly_metrics']
    return fold_metrics


def _log_test_metrics_to_wandb(loggers: List, metrics: Dict, fold_idx: int):
    """Log test metrics to WandB with a fold-specific prefix (first test year only)."""
    for logger_obj in loggers:
        if hasattr(logger_obj, 'experiment'):
            metrics_to_log = {f'test/fold_{fold_idx + 1}/{k}': v
                              for k, v in metrics.items() if v is not None}
            if metrics_to_log:
                logger_obj.experiment.log(metrics_to_log)
                print(f"[WandB] Logged fold_{fold_idx + 1} first-year metrics: "
                      f"R2={metrics.get('r2'):.4f}, NRMSE={metrics.get('nrmse'):.4f}")


def _log_walk_forward_to_wandb(loggers: List, aggregated: Dict, run_id: str,
                               fold_all_year_metrics: Dict = None):
    """Log walk-forward overall + first-year averages to WandB."""
    for logger_obj in loggers:
        if hasattr(logger_obj, 'experiment'):
            metrics_to_log = {}
            if aggregated and 'overall' in aggregated:
                for metric_name, value in aggregated['overall'].items():
                    if value is not None and not metric_name.endswith('_std'):
                        metrics_to_log[f'test/{metric_name}'] = value
            if aggregated and 'first_year' in aggregated:
                for metric_name, value in aggregated['first_year'].items():
                    if value is not None and not metric_name.endswith('_std'):
                        metrics_to_log[f'test/first_year_avg/{metric_name}'] = value
            if metrics_to_log:
                logger_obj.experiment.log(metrics_to_log)
                print(f"[WandB] Logged overall metrics to WandB")


# =============================================================================
# =====================  REFERENCE ("FREE") FORECASTS  ========================
# =============================================================================
class ReferenceForecasts:
    """
    Reference ("free") forecasts and residual spread estimated from TRAINING
    region-years only. Nothing here touches validation or test targets.
    """

    MIN_YEARS_FOR_OLS = 5      # same as TrendModel.MIN_TREND_WINDOW_SIZE
    MIN_YEARS_FOR_SIGMA = 5

    def __init__(self, quantiles: Optional[Sequence[float]] = None):
        from trendLayer import TrendModel  # lazy: keeps `import metrics` cheap
        self.quantiles = list(quantiles) if quantiles is not None else []
        self._train = None
        self._clim: Dict[str, float] = {}
        self._roll3: Dict[str, float] = {}
        self._ols: Dict[str, Tuple[float, float]] = {}
        self._sigma: Dict[str, float] = {}
        self._pooled_sigma: float = np.nan
        self._country_mean: float = np.nan
        self._z_quantiles: Dict[float, float] = {}
        self._trend_model = TrendModel()
        self.last_train_year: Optional[int] = None
        self.origin_year: Optional[int] = None

    def fit(self, adm_ids: Sequence[str], years: Sequence[int], y: Sequence[float],
            origin_year: Optional[int] = None) -> 'ReferenceForecasts':
        df = pd.DataFrame({'region': list(adm_ids),
                           'year': [int(v) for v in years],
                           'y': np.asarray(y, dtype=float)})
        df = df.dropna(subset=['y'])
        self._train = df
        self.last_train_year = int(df['year'].max())
        self.origin_year = int(origin_year) if origin_year is not None else self.last_train_year
        self._country_mean = float(df['y'].mean())

        self._trend_model.fit([{KEY_LOC: r, KEY_YEAR: int(t), KEY_TARGET: float(v)}
                               for r, t, v in zip(df['region'], df['year'], df['y'])])

        all_resid = []
        for region, g in df.groupby('region'):
            g = g.sort_values('year')
            self._clim[region] = float(g['y'].mean())
            self._roll3[region] = float(g['y'].tail(3).mean())
            self._ols[region] = self._fit_ols(g['year'].values, g['y'].values)
            resid = self._in_sample_residuals(g['year'].values, g['y'].values)
            all_resid.append(resid)
            if len(resid) >= self.MIN_YEARS_FOR_SIGMA:
                self._sigma[region] = float(np.std(resid, ddof=1))
        all_resid = np.concatenate(all_resid) if all_resid else np.array([])
        self._pooled_sigma = float(np.std(all_resid, ddof=1)) if len(all_resid) > 1 else np.nan

        if self.quantiles and len(all_resid) > 1:
            z = []
            for region, g in df.groupby('region'):
                g = g.sort_values('year')
                resid = self._in_sample_residuals(g['year'].values, g['y'].values)
                s = self.sigma(region)
                if s and s > 0:
                    z.append(resid / s)
            z = np.concatenate(z)
            for q in self.quantiles:
                self._z_quantiles[q] = float(np.quantile(z, q))
        return self

    def _fit_ols(self, years: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """(slope, intercept) of a per-region OLS line; slope 0 if too few years."""
        if len(y) >= self.MIN_YEARS_FOR_OLS:
            slope, intercept = np.polyfit(years.astype(float), y, deg=1)
            return float(slope), float(intercept)
        return 0.0, float(np.mean(y))

    def _in_sample_residuals(self, years: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Residuals around the per-region OLS line (mean if too few years)."""
        slope, intercept = self._fit_ols(years, y)
        return y - (slope * years.astype(float) + intercept)

    def ols(self, region: str, year: int) -> float:
        if region in self._ols:
            slope, intercept = self._ols[region]
            return slope * float(year) + intercept
        return self._country_mean

    def sigma(self, region: str) -> float:
        return self._sigma.get(region, self._pooled_sigma)

    def predict(self, adm_ids: Sequence[str], years: Sequence[int]) -> pd.DataFrame:
        """Reference forecasts for the given region-years."""
        adm_ids = list(adm_ids)
        years = [int(v) for v in years]
        items = [{KEY_LOC: r, KEY_YEAR: t} for r, t in zip(adm_ids, years)]
        trend = self._trend_model._predict_trend(items).flatten()
        out = pd.DataFrame({
            'region': adm_ids,
            'year': years,
            'ref_climatology': [self._clim.get(r, self._country_mean) for r in adm_ids],
            'ref_rolling3': [self._roll3.get(r, self._country_mean) for r in adm_ids],
            'ref_trend': trend.astype(float),
            'ref_ols': [self.ols(r, t) for r, t in zip(adm_ids, years)],
            'ref_sigma': [self.sigma(r) for r in adm_ids],
            'horizon': [t - self.origin_year for t in years],
        })
        for q in self.quantiles:
            zq = self._z_quantiles.get(q, np.nan)
            # OLS-centred reference band (existing, used by legacy crps_ref_ols)
            out[f'ref_q{int(round(q * 100))}'] = out['ref_ols'] + out['ref_sigma'] * zq
            # Trend / climatology / rolling-3 centred reference bands
            out[f'ref_trend_q{int(round(q * 100))}'] = out['ref_trend'] + out['ref_sigma'] * zq
            out[f'ref_clim_q{int(round(q * 100))}'] = out['ref_climatology'] + out['ref_sigma'] * zq
            out[f'ref_roll3_q{int(round(q * 100))}'] = out['ref_rolling3'] + out['ref_sigma'] * zq
        return out


# =============================================================================
# =====================  LEGACY POST-HOC PROTOCOL  ============================
# Kept intact for the run_evaluation_protocol() path; it now calls the shared
# metric primitives above instead of re-implementing them.
# =============================================================================
@dataclass
class ProtocolConfig:
    rel_tols: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    abs_tols: Tuple[float, ...] = (0.1, 0.5,)
    bad_year_sigmas: Tuple[float, ...] = (0.5, 1.0)
    quantiles: List[float] = field(default_factory=list)   # empty => point model


class YieldEvaluationProtocol:
    """Runs the full (legacy) protocol on a prediction frame."""

    def __init__(self, refs: ReferenceForecasts, cfg: ProtocolConfig):
        self.refs = refs
        self.cfg = cfg
        self.q_cols = [f'q{int(round(q * 100))}' for q in cfg.quantiles]
        self.ref_q_cols = [f'ref_q{int(round(q * 100))}' for q in cfg.quantiles]

    @property
    def is_quantile(self) -> bool:
        return len(self.cfg.quantiles) >= 2

    def _outer(self) -> Tuple[int, int, float]:
        qs = self.cfg.quantiles
        lo, hi = int(np.argmin(qs)), int(np.argmax(qs))
        alpha = 1.0 - (qs[hi] - qs[lo])
        return lo, hi, alpha

    def attach_references(self, df: pd.DataFrame) -> pd.DataFrame:
        ref = self.refs.predict(df['region'].values, df['year'].values)
        ref = ref.drop(columns=['region', 'year'])
        out = pd.concat([df.reset_index(drop=True), ref.reset_index(drop=True)], axis=1)
        out['anom_true'] = out['y_true'] - out['ref_ols']
        out['anom_pred'] = out['y_pred'] - out['ref_ols']
        return out

    def conformalize(self, test_df: pd.DataFrame, val_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, float]:
        if not self.is_quantile or val_df is None or len(val_df) == 0:
            return test_df, np.nan
        lo, hi, alpha = self._outer()
        m = conformal_margin(val_df['y_true'].values,
                             val_df[self.q_cols[lo]].values,
                             val_df[self.q_cols[hi]].values, alpha)
        test_df = test_df.copy()
        test_df[f'{self.q_cols[lo]}_conf'] = np.clip(test_df[self.q_cols[lo]] - m, 0.0, None)
        test_df[f'{self.q_cols[hi]}_conf'] = test_df[self.q_cols[hi]] + m
        return test_df, m

    def metrics(self, df: pd.DataFrame, conformal_margin_value: float = np.nan) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(df) == 0:
            return out
        y, yp = df['y_true'].values, df['y_pred'].values
        out['n_samples'] = int(len(df))

        # --- 1. point error and skill vs references (shared point_metrics)
        pm = point_metrics(y, yp)
        out['rmse'] = pm['rmse']
        out['mae'] = pm['mae']
        for name in ('trend', 'ols', 'climatology', 'rolling3'):
            r = df[f'ref_{name}'].values
            out[f'rmse_ref_{name}'] = _rmse(y, r)
            out[f'mae_ref_{name}'] = _mae(y, r)
            out[f'skill_rmse_vs_{name}'] = skill_score(out['rmse'], out[f'rmse_ref_{name}'])
            out[f'skill_mae_vs_{name}'] = skill_score(out['mae'], out[f'mae_ref_{name}'])

        # --- 2. tolerance
        out.update(tolerance_accuracy(y, yp, self.cfg.rel_tols, self.cfg.abs_tols))

        # --- 3. anomaly skill (OLS-detrended) + window-referenced diagnostic
        st = compute_all_spatiotemporal_metrics(y, yp, df['year'].values, df['region'].values,
                                                trend_reference=df['ref_ols'].values)
        out['r_sp'] = st['spatial']['r_sp_overall']
        out['r_tm'] = st['temporal']['r_tm_overall']
        out['r_an_window'] = st['anomaly']['r_an_overall']
        out['r_an_detrended'] = st['anomaly_trend']['r_an_overall']
        out['r2_anomaly_detrended'] = st['anomaly_trend']['r2_anomaly']
        out['nrmse_anomaly_detrended'] = st['anomaly_trend']['nrmse_anomaly']
        s_t, s_p = st['anomaly_trend']['std_true_anomaly'], st['anomaly_trend']['std_pred_anomaly']
        out['anomaly_spread_ratio'] = float(s_p / s_t) if s_t and np.isfinite(s_t) and s_t > 0 else np.nan

        # --- 4. bad-year detection
        sigma = df['ref_sigma'].values
        anom_true, anom_pred = df['anom_true'].values, df['anom_pred'].values
        for k in self.cfg.bad_year_sigmas:
            thr = k * sigma
            tag = f'badyear_{k:g}sig'
            res = bad_year_detection(anom_true, score=-anom_pred / np.where(sigma > 0, sigma, 1.0),
                                     pred_event=anom_pred < -thr, threshold=thr)
            for kk, v in res.items():
                out[f'{tag}_{kk}'] = v
            if self.is_quantile:
                lo, hi, _ = self._outer()
                q_lo = df[self.q_cols[lo]].values
                res_q = bad_year_detection(anom_true, score=-(q_lo - df['ref_ols'].values) / np.where(sigma > 0, sigma, 1.0),
                                           pred_event=q_lo - df['ref_ols'].values < -thr, threshold=thr)
                out[f'{tag}_auc_qlo'] = res_q['auc']

        # --- 5. probabilistic calibration and CRPS skill
        if self.is_quantile:
            qs = self.cfg.quantiles
            Q = df[self.q_cols].values
            out.update(quantile_coverage(y, Q, qs))
            lo, hi, alpha = self._outer()
            out['picp_nominal'] = float(1 - alpha)
            out['picp'] = float(np.mean((y >= Q[:, lo]) & (y <= Q[:, hi])))
            out['mean_interval_width'] = float(np.mean(Q[:, hi] - Q[:, lo]))
            out['crps_pinball'] = float(np.mean(crps_from_quantiles(y, Q, qs)))
            Qref = df[self.ref_q_cols].values
            out['crps_pinball_ref_ols'] = float(np.mean(crps_from_quantiles(y, Qref, qs)))
            out['skill_crps_vs_ols'] = skill_score(out['crps_pinball'], out['crps_pinball_ref_ols'])
            out['picp_ref_ols'] = float(np.mean((y >= Qref[:, lo]) & (y <= Qref[:, hi])))
            out['mean_interval_width_ref_ols'] = float(np.mean(Qref[:, hi] - Qref[:, lo]))

            lo_c, hi_c = f'{self.q_cols[lo]}_conf', f'{self.q_cols[hi]}_conf'
            if lo_c in df.columns:
                out['conformal_margin'] = float(conformal_margin_value)
                out['picp_conformal'] = float(np.mean((y >= df[lo_c].values) & (y <= df[hi_c].values)))
                out['mean_interval_width_conformal'] = float(np.mean(df[hi_c].values - df[lo_c].values))
                Qc = Q.copy()
                Qc[:, lo] = df[lo_c].values
                Qc[:, hi] = df[hi_c].values
                out['crps_pinball_conformal'] = float(np.mean(crps_from_quantiles(y, Qc, qs)))
        return out

    def evaluate(self, test_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
                 ) -> Tuple[Dict[str, float], pd.DataFrame]:
        df = self.attach_references(test_df)
        df, margin = self.conformalize(df, val_df)
        flat: Dict[str, float] = {}
        for k, v in self.metrics(df, margin).items():
            flat[f'{k}_overall'] = v
        for year, g in df.groupby('year'):
            for k, v in self.metrics(g, margin).items():
                flat[f'{k}_{int(year)}'] = v
        return flat, df


def build_prediction_frame(per_year_preds: Dict, quantiles: Optional[Sequence[float]]) -> pd.DataFrame:
    """Flatten the base layers' `_per_year_preds` into a frame."""
    rows = []
    for year, data in per_year_preds.items():
        preds = data.get('preds', [])
        if not preds:
            continue
        adm = data.get('adm_ids', [None] * len(preds))
        qp = data.get('quantile_preds', None)
        for i in range(len(preds)):
            row = {'region': adm[i], 'year': int(year),
                   'y_true': float(data['targets'][i]), 'y_pred': float(preds[i])}
            if quantiles and qp:
                for j, q in enumerate(quantiles):
                    row[f'q{int(round(q * 100))}'] = float(qp[i][j])
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_predictions(model, dataloader, quantiles: Optional[Sequence[float]]) -> pd.DataFrame:
    """Run `model.predict` over a dataloader and return a prediction frame."""
    was_training = model.training
    model.eval()
    device = model.device
    rows = []
    for batch in dataloader:
        x_ts, x_static, y_z, years, adm_ids, lats, lons, mask = batch
        batch_dev = (x_ts.to(device), x_static.to(device), y_z.to(device), years.to(device),
                     list(adm_ids), lats.to(device), lons.to(device), mask.to(device))
        out = model.predict(batch_dev)
        preds = out['predictions'].detach().cpu().numpy()
        targets = out['targets'].detach().cpu().numpy()
        yrs = out['years'].detach().cpu().numpy()
        for i in range(len(targets)):
            row = {'region': adm_ids[i], 'year': int(yrs[i]), 'y_true': float(targets[i])}
            if preds.ndim > 1 and quantiles:
                med = list(quantiles).index(0.5) if 0.5 in quantiles else preds.shape[1] // 2
                row['y_pred'] = float(preds[i, med])
                for j, q in enumerate(quantiles):
                    row[f'q{int(round(q * 100))}'] = float(preds[i, j])
            else:
                row['y_pred'] = float(np.ravel(preds[i])[0])
            rows.append(row)
    if was_training:
        model.train()
    return pd.DataFrame(rows)


def print_protocol_table(title: str, flat: Dict[str, float], years: Sequence[int]):
    years = sorted(int(y) for y in years)
    cols = ['overall'] + [str(y) for y in years]
    rows = [
        ('RMSE', 'rmse'), ('RMSE trend ref (MK)', 'rmse_ref_trend'), ('RMSE OLS ref', 'rmse_ref_ols'),
        ('Skill RMSE vs trend (MK)', 'skill_rmse_vs_trend'),
        ('Skill RMSE vs OLS', 'skill_rmse_vs_ols'),
        ('Skill RMSE vs climatology', 'skill_rmse_vs_climatology'),
        ('Skill RMSE vs rolling-3', 'skill_rmse_vs_rolling3'),
        ('Within +/-10%', 'within_10pct'), ('Within +/-0.5 t/ha', 'within_0.5abs'),
        ('r_an (OLS-detrended)', 'r_an_detrended'), ('r_an (window ref)', 'r_an_window'),
        ('Anomaly spread ratio', 'anomaly_spread_ratio'),
        ('Bad-year AUC (1sig)', 'badyear_1sig_auc'), ('Bad-year recall (1sig)', 'badyear_1sig_recall'),
        ('Bad-year base rate (1sig)', 'badyear_1sig_base_rate'),
        ('PICP (nominal)', 'picp'), ('PICP conformal', 'picp_conformal'),
        ('Interval width', 'mean_interval_width'), ('Width conformal', 'mean_interval_width_conformal'),
        ('Quantile calib. error', 'quantile_calibration_error'),
        ('CRPS (pinball)', 'crps_pinball'), ('CRPS skill vs OLS', 'skill_crps_vs_ols'),
    ]
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"{'metric':<28}" + ''.join(f"{c:>12}" for c in cols))
    for label, key in rows:
        if f'{key}_overall' not in flat:
            continue
        vals = []
        for c in cols:
            v = flat.get(f'{key}_{c}', np.nan)
            vals.append(f"{v:>12.4f}" if v is not None and np.isfinite(v) else f"{'nan':>12}")
        print(f"{label:<28}" + ''.join(vals))
    if 'picp_nominal_overall' in flat:
        print(f"(nominal interval coverage = {flat['picp_nominal_overall']:.2f}; "
              f"conformal margin = {flat.get('conformal_margin_overall', float('nan')):.4f} t/ha)")
    print('-' * 100)


PROTOCOL_METRICS_TO_SAVE = [
    'rmse', 'mae', 'rmse_ref_trend', 'rmse_ref_ols', 'rmse_ref_climatology', 'rmse_ref_rolling3',
    'skill_rmse_vs_trend', 'skill_rmse_vs_ols', 'skill_rmse_vs_climatology', 'skill_rmse_vs_rolling3',
    'skill_mae_vs_trend', 'skill_mae_vs_ols', 'skill_mae_vs_climatology', 'skill_mae_vs_rolling3',
    'within_5pct', 'within_10pct', 'within_20pct', 'within_0.5abs',
    'r_sp', 'r_tm', 'r_an_window', 'r_an_detrended', 'r2_anomaly_detrended', 'nrmse_anomaly_detrended',
    'anomaly_spread_ratio',
    'badyear_0.5sig_auc', 'badyear_0.5sig_precision', 'badyear_0.5sig_recall', 'badyear_0.5sig_f1',
    'badyear_0.5sig_base_rate', 'badyear_0.5sig_auc_qlo',
    'badyear_1sig_auc', 'badyear_1sig_precision', 'badyear_1sig_recall', 'badyear_1sig_f1',
    'badyear_1sig_base_rate', 'badyear_1sig_auc_qlo',
    'coverage_q10', 'coverage_q50', 'coverage_q90', 'quantile_calibration_error',
    'picp', 'picp_nominal', 'mean_interval_width', 'crps_pinball', 'crps_pinball_ref_ols',
    'skill_crps_vs_ols', 'picp_ref_ols', 'mean_interval_width_ref_ols',
    'conformal_margin', 'picp_conformal', 'mean_interval_width_conformal', 'crps_pinball_conformal',
]


def _fit_reference_forecasts(dm, quantiles):
    """Fit ReferenceForecasts on TRAIN + VAL years, denormalised to original yield
    scale, with the forecast origin set to the last observed (val) year.

    The baselines must see the same recent history the model is allowed to use: the
    validation years feed model selection and supply the first test year's recursive
    lags, and origin_year is the last val year. Fitting references on train-only
    would deny them those 2 recent years, understate rmse_ref_*, and inflate every
    skill-vs-reference score. Test years are never included here, so there is no
    leakage -- this only levels the information set between model and baselines.
    """
    adm = list(dm.train_ds.adm_ids)
    years = [int(v) for v in dm.train_ds.years.numpy().tolist()]
    y = (dm.train_ds.y.numpy() * dm.y_std + dm.y_mean).tolist()
    if getattr(dm, 'val_ds', None) is not None and len(dm.val_ds) > 0:
        adm = adm + list(dm.val_ds.adm_ids)
        years = years + [int(v) for v in dm.val_ds.years.numpy().tolist()]
        y = y + (dm.val_ds.y.numpy() * dm.y_std + dm.y_mean).tolist()
    origin_year = max(years)
    return ReferenceForecasts(quantiles=quantiles).fit(adm, years, y, origin_year=origin_year)


def run_evaluation_protocol(model, dm, config, run_id: str, timestamp: str,
                            save: bool = True, verbose: bool = True) -> Dict[str, float]:
    """End-to-end legacy protocol: references from dm.train_ds, test frame from
    model._per_year_preds, validation predictions for conformal calibration, print + save."""
    quantiles = list(getattr(model, 'quantiles', [])) if getattr(model, 'loss_type', 'mse') == 'mql' else []

    per_year = getattr(model, '_per_year_preds', None)
    if not per_year:
        logger.warning('[Protocol] model has no _per_year_preds; skipping evaluation protocol')
        return {}
    test_df = build_prediction_frame(per_year, quantiles)
    if test_df.empty or test_df['region'].isna().any():
        logger.warning('[Protocol] test predictions lack region ids; skipping evaluation protocol')
        return {}

    # References fit on TRAIN + VAL (all observed years before the test block);
    # see _fit_reference_forecasts for the rationale.
    refs = _fit_reference_forecasts(dm, quantiles)

    val_df = None
    if quantiles:
        try:
            val_df = collect_predictions(model, dm.val_dataloader(), quantiles)
        except Exception as e:
            logger.warning(f'[Protocol] could not collect validation predictions for conformal calibration: {e}')

    proto = YieldEvaluationProtocol(refs, ProtocolConfig(quantiles=quantiles))
    flat, per_sample = proto.evaluate(test_df, val_df)

    years = sorted(test_df['year'].unique())
    if verbose:
        print_protocol_table(f"EVALUATION PROTOCOL: {config.crop}-{config.country} | {config.model_type} "
                             f"| last train year {refs.last_train_year} | forecast origin {refs.origin_year} "
                             f"(horizons {min(years) - refs.origin_year}..{max(years) - refs.origin_year})",
                             flat, years)

    if save:
        from helpers import save_consolidated_results
        save_consolidated_results(config, flat, years, run_id, timestamp,
                                  metric_names=PROTOCOL_METRICS_TO_SAVE,
                                  prefix='protocol_', write_config=False,
                                  skip_missing=True)
        pred_dir = os.path.join(config.results_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        n_regions = int(test_df['region'].nunique())
        per_sample.insert(0, 'run_id', run_id)
        per_sample.insert(0, 'timestamp', timestamp)
        for col in ('model_type', 'crop', 'country', 'data_fraction', 'seed'):
            per_sample.insert(2, col, getattr(config, col, None))
        per_sample.insert(2, 'n_regions', n_regions)
        path = os.path.join(pred_dir, f'{config.model_type}_{config.crop}_{config.country}_{run_id}.csv')
        per_sample.to_csv(path, index=False)
        print(f"[Protocol] Saved per-sample predictions to {path}")
    return flat


# =============================================================================
# ============  Tiered evaluation suite (canonical one-row-per-cell)  ==========
#
# The four reporting tiers the study uses, each writing one canonical row per
# cell (helpers.CELL_KEY_COLS) into its own file:
#   measureInDomainAccuracy        -> accuracy.csv        (Tier 0)
#   measureCalibratedUncertainity  -> uncertainity.csv    (Tier 1, MQL only)
#   measureGeneralizationShift     -> generalization.csv  (Tier 2: anomaly + extremes)
#   measureOperationalValue        -> operational.csv     (Tier 3)
#   measureCrossCountryTransfer    -> transfer_*.csv       (Tier 2b)
# The dispatcher run_evaluation() selects which tiers run from --evaluation_tier.
# All tiers call the SAME point_metrics / uncertainty_metrics primitives above.
# =============================================================================
class _TierBase:
    """Shared quantile bookkeeping for the tier measurement classes."""

    def __init__(self, cfg: ProtocolConfig):
        self.cfg = cfg
        self.quantiles = list(cfg.quantiles)
        self.q_cols = [f'q{int(round(q * 100))}' for q in self.quantiles]
        self.ref_q_cols = [f'ref_q{int(round(q * 100))}' for q in self.quantiles]

    @property
    def is_quantile(self) -> bool:
        return len(self.quantiles) >= 2

    def _outer(self) -> Tuple[int, int, float]:
        qs = self.quantiles
        lo, hi = int(np.argmin(qs)), int(np.argmax(qs))
        return lo, hi, 1.0 - (qs[hi] - qs[lo])


class measureInDomainAccuracy(_TierBase):
    """TIER 0 - point error, skill vs the free forecasts, tolerance accuracy."""

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(df) == 0:
            return out
        y, yp = df['y_true'].values, df['y_pred'].values
        out['n_samples'] = int(len(df))
        # Single-source point metrics (mse/rmse/mae/mape/smape/r2/nrmse)
        out.update(point_metrics(y, yp))
        for name in ('trend', 'ols', 'climatology', 'rolling3'):
            r = df[f'ref_{name}'].values
            out[f'rmse_ref_{name}'] = _rmse(y, r)
            out[f'mae_ref_{name}'] = _mae(y, r)
            out[f'skill_rmse_vs_{name}'] = skill_score(out['rmse'], out[f'rmse_ref_{name}'])
            out[f'skill_mae_vs_{name}'] = skill_score(out['mae'], out[f'mae_ref_{name}'])
        out.update(tolerance_accuracy(y, yp, self.cfg.rel_tols, self.cfg.abs_tols))
        return out


class measureCalibratedUncertainity(_TierBase):
    """TIER 1 - calibration, sharpness, Winkler, CRPS, conformal (MQL only)."""

    def compute(self, df: pd.DataFrame, conformal_margin_value: float = np.nan) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(df) == 0 or not self.is_quantile or self.q_cols[0] not in df.columns:
            return out
        y = df['y_true'].values
        Q = df[self.q_cols].values
        qs = self.quantiles
        out['n_samples'] = int(len(df))

        # Single-source uncertainty base (pinball_q*, pinball_avg, crps, picp,
        # pinaw, winkler_score, mean_interval_width) — identical to the training path.
        base = uncertainty_metrics(y, Q, qs)
        out.update(base)
        out['crps_pinball'] = base['crps']   # protocol alias

        # Coverage + calibration error, and nominal interval target
        out.update(quantile_coverage(y, Q, qs))
        lo, hi, alpha = self._outer()
        out['picp_nominal'] = float(1 - alpha)

        # Reference (OLS) distribution + CRPS skill
        Qref = df[self.ref_q_cols].values
        out['crps_pinball_ref_ols'] = float(np.mean(crps_from_quantiles(y, Qref, qs)))
        out['skill_crps_vs_ols'] = skill_score(out['crps_pinball'], out['crps_pinball_ref_ols'])
        out['picp_ref_ols'] = float(np.mean((y >= Qref[:, lo]) & (y <= Qref[:, hi])))
        out['mean_interval_width_ref_ols'] = float(np.mean(Qref[:, hi] - Qref[:, lo]))

        # CRPS skill vs trend / climatology / rolling-3 reference bands
        for ref_name, col_prefix in [('trend', 'ref_trend_q'), ('climatology', 'ref_clim_q'), ('rolling3', 'ref_roll3_q')]:
            ref_cols = [f'{col_prefix}{int(round(q * 100))}' for q in qs]
            if ref_cols[0] in df.columns:
                Qr = df[ref_cols].values
                crps_ref = float(np.mean(crps_from_quantiles(y, Qr, qs)))
                out[f'crps_pinball_ref_{ref_name}'] = crps_ref
                out[f'skill_crps_vs_{ref_name}'] = skill_score(out['crps_pinball'], crps_ref)

        # Split-conformal (CQR) block, if present
        lo_c, hi_c = f'{self.q_cols[lo]}_conf', f'{self.q_cols[hi]}_conf'
        if lo_c in df.columns:
            out['conformal_margin'] = float(conformal_margin_value)
            out['picp_conformal'] = float(np.mean((y >= df[lo_c].values) & (y <= df[hi_c].values)))
            out['mean_interval_width_conformal'] = float(np.mean(df[hi_c].values - df[lo_c].values))
            Qc = Q.copy()
            Qc[:, lo] = df[lo_c].values
            Qc[:, hi] = df[hi_c].values
            out['crps_pinball_conformal'] = float(np.mean(crps_from_quantiles(y, Qc, qs)))
        return out


class measureGeneralizationShift(_TierBase):
    """TIER 2 - anomaly skill (in_domain) + harder extreme-year strata."""

    def _anomaly_block(self, df: pd.DataFrame) -> Dict[str, float]:
        y, yp = df['y_true'].values, df['y_pred'].values
        out = {'n_samples': int(len(df)), 'rmse': _rmse(y, yp)}
        out['skill_rmse_vs_trend'] = skill_score(out['rmse'], _rmse(y, df['ref_trend'].values))
        try:
            st = compute_all_spatiotemporal_metrics(
                y, yp, df['year'].values, df['region'].values,
                trend_reference=df['ref_ols'].values)
            out['r_sp'] = st['spatial']['r_sp_overall']
            out['r_tm'] = st['temporal']['r_tm_overall']
            out['r_an_window'] = st['anomaly']['r_an_overall']
            out['r_an_detrended'] = st['anomaly_trend']['r_an_overall']
            out['r2_anomaly_detrended'] = st['anomaly_trend']['r2_anomaly']
            out['nrmse_anomaly_detrended'] = st['anomaly_trend']['nrmse_anomaly']
            s_t = st['anomaly_trend']['std_true_anomaly']
            s_p = st['anomaly_trend']['std_pred_anomaly']
            out['anomaly_spread_ratio'] = float(s_p / s_t) if s_t and np.isfinite(s_t) and s_t > 0 else np.nan
        except Exception as e:
            logger.warning(f'[Generalization] spatiotemporal metrics failed: {e}')
        return out

    def compute_in_domain(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) == 0:
            return {}
        return self._anomaly_block(df)

    def compute_extreme(self, df: pd.DataFrame, kind: str = 'label',
                        driver_values: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
        if len(df) == 0:
            return None
        sigma = df['ref_sigma'].values
        anom_true = df['anom_true'].values
        if kind == 'label':
            mask = anom_true < -1.0 * sigma
        elif kind == 'driver' and driver_values is not None:
            dv = np.asarray(driver_values, float)
            mask = dv >= np.nanquantile(dv, 0.90)
        else:
            return None
        out = {'base_rate_extreme': float(mask.mean()) if len(mask) else np.nan}
        sub = df[mask]
        out['n_samples'] = int(len(sub))
        if len(sub) == 0:
            return out
        blk = self._anomaly_block(sub)
        for k in ('rmse', 'skill_rmse_vs_trend', 'r_an_detrended', 'anomaly_spread_ratio'):
            if k in blk:
                out[k] = blk[k]
        return out


class measureOperationalValue(_TierBase):
    """TIER 3 - bad-year detection (ranking) + shortfall capture (magnitude)."""

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(df) == 0:
            return out
        out['n_samples'] = int(len(df))
        sigma = df['ref_sigma'].values
        anom_true, anom_pred = df['anom_true'].values, df['anom_pred'].values
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        for k in self.cfg.bad_year_sigmas:
            thr = k * sigma
            tag = f'badyear_{k:g}sig'
            res = bad_year_detection(anom_true, score=-anom_pred / safe_sigma,
                                     pred_event=anom_pred < -thr, threshold=thr)
            for kk, v in res.items():
                out[f'{tag}_{kk}'] = v
            if self.is_quantile and self.q_cols[0] in df.columns:
                lo, _, _ = self._outer()
                q_lo = df[self.q_cols[lo]].values
                res_q = bad_year_detection(
                    anom_true,
                    score=-(q_lo - df['ref_ols'].values) / safe_sigma,
                    pred_event=q_lo - df['ref_ols'].values < -thr, threshold=thr)
                out[f'{tag}_auc_qlo'] = res_q['auc']
        c, n_tail = _shortfall_capture(anom_true, anom_pred)
        out['shortfall_capture_c'] = c
        out['shortfall_n_tail'] = n_tail
        return out


def _prep_frame(refs: ReferenceForecasts, cfg: ProtocolConfig,
                test_df: pd.DataFrame, val_df: Optional[pd.DataFrame]
                ) -> Tuple[pd.DataFrame, float]:
    """Attach references/anomalies and conformal outer quantiles (reuses the
    legacy protocol's attach/conformalize so the numbers match exactly)."""
    proto = YieldEvaluationProtocol(refs, cfg)
    df = proto.attach_references(test_df)
    vdf = None
    if val_df is not None and len(val_df) > 0:
        try:
            vdf = proto.attach_references(val_df)
        except Exception:
            vdf = None
    df, margin = proto.conformalize(df, vdf)
    return df, margin


def compute_weather_driver_summaries(dm, train_years: Sequence[int]
                                      ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Compute per (region, year) seasonal weather summaries from the datamodule.

    Returns
    -------
    summaries : DataFrame with columns [region, year, prec_sum, tmax_mean]
        One row per (region, year) across ALL years (train + val + test).
    train_thresholds : {region: {'prec_p10': float, 'tmax_p90': float}}
        Per-region decile thresholds from TRAINING years only.
    """
    # The datamodule stores all_adm_ids, all_years and the raw time-series
    # tensor (all_X_ts).  The feature order is in config.time_series_vars;
    # 'prec' and 'tmax' are always present in the CY-Bench weather features.
    config = dm.config
    ts_vars = list(config.time_series_vars)
    try:
        prec_idx = ts_vars.index('prec')
    except ValueError:
        logger.warning('[Driver] "prec" not in time_series_vars; cannot compute driver-extreme')
        return pd.DataFrame(), {}
    try:
        tmax_idx = ts_vars.index('tmax')
    except ValueError:
        logger.warning('[Driver] "tmax" not in time_series_vars; cannot compute driver-extreme')
        return pd.DataFrame(), {}

    X = dm.all_X_ts           # (N, seq_len, n_features)  — raw (pre-normalised) values
    years_all = dm.all_years  # tensor or array, length N
    adm_all = dm.all_adm_ids  # list of str, length N

    if X is None or years_all is None:
        return pd.DataFrame(), {}

    X_np = X.numpy() if hasattr(X, 'numpy') else np.asarray(X)
    years_np = years_all.numpy() if hasattr(years_all, 'numpy') else np.asarray(years_all)

    rows = []
    for i in range(len(years_np)):
        prec_sum = float(np.nansum(X_np[i, :, prec_idx]))
        tmax_mean = float(np.nanmean(X_np[i, :, tmax_idx]))
        rows.append({'region': adm_all[i], 'year': int(years_np[i]),
                     'prec_sum': prec_sum, 'tmax_mean': tmax_mean})
    summaries = pd.DataFrame(rows)

    # Thresholds from training years only
    train_set = set(int(y) for y in train_years)
    train_df = summaries[summaries['year'].isin(train_set)]
    thresholds: Dict[str, Dict[str, float]] = {}
    for region, g in train_df.groupby('region'):
        thresholds[region] = {
            'prec_p10': float(np.nanquantile(g['prec_sum'].values, 0.10)),
            'tmax_p90': float(np.nanquantile(g['tmax_mean'].values, 0.90)),
        }
    return summaries, thresholds


def compute_driver_values(pred_df: pd.DataFrame, summaries: pd.DataFrame,
                          thresholds: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Return a per-row driver-stress score for the prediction frame.

    A row is driver-extreme if cumulative precip is below that region's
    training-decile-10 OR mean tmax is above that region's training-decile-90.
    Returns an array of 0/1 floats aligned to pred_df rows (1 = driver-extreme).
    """
    if summaries.empty or not thresholds:
        return np.full(len(pred_df), np.nan)
    merged = pred_df[['region', 'year']].merge(summaries, on=['region', 'year'], how='left')
    vals = np.zeros(len(merged), dtype=float)
    for i, (_, row) in enumerate(merged.iterrows()):
        t = thresholds.get(row['region'])
        if t is None or np.isnan(row.get('prec_sum', np.nan)):
            vals[i] = np.nan
            continue
        prec_extreme = row['prec_sum'] <= t['prec_p10']
        tmax_extreme = row['tmax_mean'] >= t['tmax_p90']
        vals[i] = 1.0 if (prec_extreme or tmax_extreme) else 0.0
    return vals


def _save_predictions(config, refs, cfg, test_df, val_df, run_id, timestamp,
                      prefix: str = '', from_to: Optional[Dict[str, str]] = None):
    """Write the per-sample prediction frame (bootstrap substrate)."""
    df, _ = _prep_frame(refs, cfg, test_df, val_df)
    pred_dir = os.path.join(config.results_dir, f'{prefix}predictions')
    os.makedirs(pred_dir, exist_ok=True)
    df = df.copy()
    df.insert(0, 'run_id', run_id)
    df.insert(0, 'timestamp', timestamp)
    for col in ('model_type', 'crop', 'country', 'data_fraction', 'seed'):
        df.insert(2, col, getattr(config, col, None))
    if from_to:
        df['from'] = from_to['from']
        df['to'] = from_to['to']
        tag = f"{from_to['from']}_to_{from_to['to']}"
    else:
        tag = str(config.country)
    path = os.path.join(pred_dir, f'{config.model_type}_{config.crop}_{tag}_{run_id}.csv')
    df.to_csv(path, index=False)
    print(f"[Predictions] -> {path}")


def _evaluate_and_save(config, refs, cfg, test_df, val_df, tiers, run_id, timestamp,
                       prefix: str = '', verbose: bool = True,
                       from_to: Optional[Dict[str, str]] = None,
                       driver_values: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Run the requested tier classes on one prepared frame and upsert each
    tier's canonical row(s). Returns a flat dict for console/wandb summary."""
    from helpers import save_tier_results, TRANSFER_KEY_COLS
    df, margin = _prep_frame(refs, cfg, test_df, val_df)
    key_cols = TRANSFER_KEY_COLS if from_to else None
    base = dict(from_to) if from_to else {}
    flat: Dict[str, float] = {}

    if 'accuracy' in tiers:
        m = measureInDomainAccuracy(cfg).compute(df)
        if m:
            save_tier_results(config, 'accuracy', [{**m, **base}], run_id, timestamp,
                              prefix=prefix, key_cols=key_cols)
        flat.update({f'accuracy/{k}': v for k, v in m.items()})

    if 'uncertainity' in tiers:
        m = measureCalibratedUncertainity(cfg).compute(df, conformal_margin_value=margin)
        if m:
            save_tier_results(config, 'uncertainity', [{**m, **base}], run_id, timestamp,
                              prefix=prefix, key_cols=key_cols)
        flat.update({f'uncertainity/{k}': v for k, v in m.items()})

    if 'generalization' in tiers:
        gen = measureGeneralizationShift(cfg)
        rows = []
        m_in = gen.compute_in_domain(df)
        if m_in:
            rows.append({**m_in, 'partition': 'in_domain', **base})
        m_lab = gen.compute_extreme(df, kind='label')
        if m_lab:
            rows.append({**m_lab, 'partition': 'extreme_label', **base})
        # Driver-extreme partition from raw precipitation + temperature
        if driver_values is not None:
            m_drv = gen.compute_extreme(df, kind='driver', driver_values=driver_values)
            if m_drv:
                # Agreement rate: Jaccard of label-extreme vs driver-extreme masks
                sigma = df['ref_sigma'].values
                anom_true = df['anom_true'].values
                label_mask = anom_true < -1.0 * sigma
                driver_mask = np.asarray(driver_values, float) >= 0.5  # 1.0 = extreme
                union = label_mask | driver_mask
                intersection = label_mask & driver_mask
                agreement = float(intersection.sum() / union.sum()) if union.sum() > 0 else np.nan
                m_drv['agreement_rate'] = agreement
                rows.append({**m_drv, 'partition': 'extreme_driver', **base})
        if rows:
            save_tier_results(config, 'generalization', rows, run_id, timestamp,
                              prefix=prefix, key_cols=key_cols)
        flat.update({f'generalization/{k}': v for k, v in m_in.items()})

    if 'operational' in tiers:
        m = measureOperationalValue(cfg).compute(df)
        if m:
            save_tier_results(config, 'operational', [{**m, **base}], run_id, timestamp,
                              prefix=prefix, key_cols=key_cols)
        flat.update({f'operational/{k}': v for k, v in m.items()})

    return flat


@torch.no_grad()
def collect_transfer_predictions(model, dataloader, y_mean_target, y_std_target,
                                 quantiles: Optional[Sequence[float]]) -> pd.DataFrame:
    """Zero-shot X->Y frame: the frozen X-specialist predicts over target-country
    batches (predictions denormalized with X's stats inside model.predict), while
    y_true is recovered in the TARGET country's real scale from the batch."""
    was_training = model.training
    model.eval()
    device = model.device
    rows = []
    for batch in dataloader:
        x_ts, x_static, y_z, years, adm_ids, lats, lons, mask = batch
        batch_dev = (x_ts.to(device), x_static.to(device), y_z.to(device), years.to(device),
                     list(adm_ids), lats.to(device), lons.to(device), mask.to(device))
        out = model.predict(batch_dev)
        preds = out['predictions'].detach().cpu().numpy()          # X-scale predictions
        yrs = out['years'].detach().cpu().numpy()
        y_true = y_z.detach().cpu().numpy() * y_std_target + y_mean_target  # Y real scale
        for i in range(len(y_true)):
            row = {'region': adm_ids[i], 'year': int(yrs[i]), 'y_true': float(y_true[i])}
            if preds.ndim > 1 and quantiles:
                med = list(quantiles).index(0.5) if 0.5 in quantiles else preds.shape[1] // 2
                row['y_pred'] = float(preds[i, med])
                for j, q in enumerate(quantiles):
                    row[f'q{int(round(q * 100))}'] = float(preds[i, j])
            else:
                row['y_pred'] = float(np.ravel(preds[i])[0])
            rows.append(row)
    if was_training:
        model.train()
    return pd.DataFrame(rows)


class measureCrossCountryTransfer:
    """TIER 2b - specialist trained on source X scored zero-shot on each target Y."""

    def __init__(self, model, dm_source, config, quantiles):
        self.model = model
        self.dm_source = dm_source
        self.config = config
        self.quantiles = list(quantiles)

    def _targets(self, transfer_countries) -> List[str]:
        from helpers import select_country
        src = self.config.country
        if transfer_countries:
            return [c for c in transfer_countries if c != src]
        return [c for c in select_country(self.config.crop, 'all') if c != src]

    def run(self, tiers_to_run, transfer_countries, run_id, timestamp,
            generate_predictions: bool = False, verbose: bool = True):
        import dataclasses
        from loadData import DailyCYBenchSeqDataModule, calculate_fixed_split
        from cybench.config import KEY_YEAR as _KEY_YEAR
        from cybench.datasets.configured import load_dfs_crop
        from cybench.datasets.dataset import Dataset as CYDataset
        from helpers import save_hyperparameters

        src = self.config.country
        targets = self._targets(transfer_countries)
        print(f"\n[Transfer] source={src} -> targets={targets}")
        cfg_proto = ProtocolConfig(quantiles=self.quantiles)

        for Y in targets:
            try:
                cfg_Y = dataclasses.replace(self.config, country=Y)
                df_y, dfs_x = load_dfs_crop(cfg_Y.crop, [Y])
                if df_y is None or len(df_y) == 0:
                    logger.warning(f'[Transfer] no data for {Y}; skipping')
                    continue
                ds = CYDataset(cfg_Y.crop, df_y, dfs_x)
                all_years = sorted(set(ds[i][_KEY_YEAR] for i in range(len(ds))))
                splits = calculate_fixed_split(all_years, test_years=cfg_Y.test_years, val_years=2)
                dm_Y = DailyCYBenchSeqDataModule(cfg_Y)
                dm_Y.setup(train_years=splits['train_years'], val_years=splits['val_years'],
                           test_years=splits['test_years'])

                test_df = collect_transfer_predictions(
                    self.model, dm_Y.test_dataloader(), dm_Y.y_mean, dm_Y.y_std, self.quantiles)
                if test_df.empty or test_df['region'].isna().any():
                    logger.warning(f'[Transfer] {src}->{Y}: empty/invalid predictions; skipping')
                    continue

                # References fit on TARGET-country TRAIN + VAL years.
                refs_Y = _fit_reference_forecasts(dm_Y, self.quantiles)

                val_df = None
                if self.quantiles:
                    try:
                        val_df = collect_transfer_predictions(
                            self.model, dm_Y.val_dataloader(), dm_Y.y_mean, dm_Y.y_std, self.quantiles)
                    except Exception as e:
                        logger.warning(f'[Transfer] {src}->{Y}: val predictions failed: {e}')

                from_to = {'from': src, 'to': Y}
                _evaluate_and_save(cfg_Y, refs_Y, cfg_proto, test_df, val_df, tiers_to_run,
                                   run_id, timestamp, prefix='transfer_', verbose=verbose,
                                   from_to=from_to)
                save_hyperparameters(cfg_Y, run_id, timestamp, prefix='transfer_', extra=from_to)
                if generate_predictions:
                    _save_predictions(cfg_Y, refs_Y, cfg_proto, test_df, val_df, run_id, timestamp,
                                      prefix='transfer_', from_to=from_to)
                print(f"[Transfer] {src}->{Y}: done")
            except Exception as e:
                logger.warning(f'[Transfer] {src}->{Y} failed: {e}')


# tier tokens accepted on the CLI (see the baseline scripts' --evaluation_tier)
EVALUATION_TIERS = ['accuracy', 'uncertainity', 'generalization', 'operational']
ALL_EVALUATION_TIERS = EVALUATION_TIERS + ['cross_country_transfer']


def run_evaluation(model, dm, config, tiers, run_id: str, timestamp: str,
                   generate_predictions: bool = False,
                   transfer_countries: Optional[Sequence[str]] = None,
                   verbose: bool = True) -> Dict[str, float]:
    """Dispatcher driven by --evaluation_tier.

    Builds the in-domain prediction frame + training references once, runs the
    selected tier classes, optionally saves per-sample predictions, and optionally
    runs cross-country transfer. Returns a flat dict of in-domain metrics.
    """
    from helpers import save_hyperparameters
    tiers = [t.strip() for t in tiers]
    quantiles = list(getattr(model, 'quantiles', [])) if getattr(model, 'loss_type', 'mse') == 'mql' else []

    per_year = getattr(model, '_per_year_preds', None)
    if not per_year:
        logger.warning('[Eval] model has no _per_year_preds; skipping evaluation')
        return {}
    test_df = build_prediction_frame(per_year, quantiles)
    if test_df.empty or test_df['region'].isna().any():
        logger.warning('[Eval] test predictions lack region ids; skipping evaluation')
        return {}

    # References fit on TRAIN + VAL (all observed years before the test block).
    refs = _fit_reference_forecasts(dm, quantiles)

    val_df = None
    if quantiles:
        try:
            val_df = collect_predictions(model, dm.val_dataloader(), quantiles)
        except Exception as e:
            logger.warning(f'[Eval] could not collect validation predictions for conformal: {e}')

    cfg_proto = ProtocolConfig(quantiles=quantiles)
    in_domain_tiers = [t for t in tiers if t in EVALUATION_TIERS]

    # Compute driver-extreme values from raw weather if generalization tier is requested
    drv_vals = None
    if 'generalization' in in_domain_tiers:
        try:
            train_years_list = dm.train_ds.years.numpy().tolist()
            summaries, thresholds = compute_weather_driver_summaries(dm, train_years_list)
            if not summaries.empty:
                drv_vals = compute_driver_values(test_df, summaries, thresholds)
        except Exception as e:
            logger.warning(f'[Eval] driver-extreme computation failed (non-fatal): {e}')

    flat = _evaluate_and_save(config, refs, cfg_proto, test_df, val_df, in_domain_tiers,
                              run_id, timestamp, prefix='', verbose=verbose,
                              driver_values=drv_vals)

    save_hyperparameters(config, run_id, timestamp, prefix='')

    if generate_predictions:
        _save_predictions(config, refs, cfg_proto, test_df, val_df, run_id, timestamp)

    if 'cross_country_transfer' in tiers:
        transfer_tiers = in_domain_tiers or list(EVALUATION_TIERS)
        measureCrossCountryTransfer(model, dm, config, quantiles=quantiles).run(
            tiers_to_run=transfer_tiers, transfer_countries=transfer_countries,
            run_id=run_id, timestamp=timestamp,
            generate_predictions=generate_predictions, verbose=verbose)

    if verbose:
        print(f"\n[Eval] tiers run: {tiers}  (in-domain cells written to "
              f"{config.results_dir})")
    return flat
