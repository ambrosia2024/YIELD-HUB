"""
--------------------
Author: XYZ
Description: Helper functions for the CY-BENCH yield forecasting pipeline.

This file contains utility functions organized into sections:
1. Parameter verification and selection
2. Checkpoint management
3. HPO (Hyperparameter Optimization) results loading
4. Lead time calculation helpers

Python version: 3.12.0
"""

import os
import re
import sys
import random
import hashlib
import logging
from typing import Union, Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import fields

import numpy as np
import pandas as pd

import torch
from lightning.pytorch import seed_everything

# Custom functions and classes
_helpers_dir = Path(__file__).parent.resolve()
sys.path.append(str(_helpers_dir.parent / 'architectures'))
from modelconfig import TSTModelConfig, LinearModelConfig


def verify_parameters(crop, model, country):
    """
    Verifies if the pipeline for selected crop, model, and country is implemented.
    """
    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    if crop == "maize":
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']
    else:
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']

def select_country(crop, country):
    """
    A function to collect all the countries if country is set to "all". Otherwise,
    it just returns [country]
    """
    if country == "all":
        if crop == "maize":
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 
                                'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE']
        else:
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 
                                'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 
                                'RO', 'SE']
    else:
        country = [country]
    return country

def seed_uniformly(seed):
    # Setting seed value for reproducibility    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    seed_everything(seed)

def generate_checkpoint_name(args) -> str:
    """
    Generate a descriptive checkpoint filename with all hyperparameters.

    Args:
        args: ArgumentParser namespace with all hyperparameters

    Returns:
        Descriptive checkpoint filename without extension
    """
    base_name = f"{args.crop}_{args.country}"

    # Core hyperparameters
    hyperparams = [
        f"model:{args.model_type}",
        f"agg:{args.aggregation}",
        f"epochs:{args.epochs}",
        f"lr:{args.lr}",
        f"wd:{args.weight_decay}",
        f"batch:{args.batch_size}",
        f"seed:{args.seed}",
    ]

    # Feature flags
    if args.use_sota_features:
        hyperparams.append("sota")
    if args.include_spatial_features:
        hyperparams.append("spatial")
    if args.use_residual_trend:
        hyperparams.append("trend")
    hyperparams.append(f"lag:{args.lag_years}")

    # Combine all parts
    parts = [base_name] + hyperparams
    name = "_".join(parts)

    # Protect against OS filename length limits (typically 255 bytes)
    # Leave room for epoch/val_loss suffix (~20 chars) and .ckpt extension (~5 chars)
    MAX_NAME_LENGTH = 230
    if len(name.encode('utf-8')) > MAX_NAME_LENGTH:
        # Use hash for long names while preserving core info
        short_hash = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        original_name = name
        hashed_name = f"{base_name}_run_{short_hash}"
        logging.warning(
            f"Checkpoint name exceeded {MAX_NAME_LENGTH} chars and was hashed:\n"
            f"  Original ({len(name.encode('utf-8'))} chars): {original_name}\n"
            f"  Hashed to: {hashed_name}\n"
            f"  To recover params: check wandb run config or args log"
        )
        name = hashed_name

    return name

def _config_row(config, run_id: str, timestamp: str) -> Dict[str, Any]:
    """Metadata + config fields that prefix every results row."""
    base_data = {'timestamp': timestamp, 'run_id': run_id}
    # Exclude results_dir from CSV columns (it's metadata, not a model hyperparameter)
    for field in fields(config):
        if field.name != 'results_dir':
            base_data[field.name] = getattr(config, field.name)
    # Pick up dynamically-set attributes (e.g. tab_strategy, resolution from tabFMBaselines)
    declared = {f.name for f in fields(config)}
    for k, v in vars(config).items():
        if k not in declared and k != 'results_dir' and not k.startswith('_'):
            base_data[k] = v
    return base_data


def _append_metric_row(csv_path: str, row_data: Dict[str, Any]):
    """Append one run-row to a per-metric CSV, keeping a stable column order."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        new_df = pd.DataFrame([row_data])
        # pandas automatically aligns columns and adds new ones
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    # Enforce consistent column order: metadata -> config -> years (sorted) -> overall
    def get_column_order(cols):
        metadata_cols = ['timestamp', 'run_id']
        year_cols = sorted([c for c in cols if c.isdigit() and len(c) == 4])
        other_cols = [c for c in cols if c not in metadata_cols + year_cols + ['overall']]
        return metadata_cols + other_cols + year_cols + ['overall']

    df = df[get_column_order(df.columns)]
    df.to_csv(csv_path, index=False)


def save_metric_tables(
    config: Union[TSTModelConfig, LinearModelConfig],
    results: Dict[str, float],
    test_years: List[int],
    run_id: str,
    timestamp: str,
    metric_names: List[str],
    subdir: Optional[str] = None,
    skip_missing: bool = True,
):
    """
    Generic per-metric CSV writer: one file per metric, one row per run,
    columns = metadata + config + one column per test year + 'overall'.

    `results` must hold keys '<metric>_<year>' and '<metric>_overall'.
    Metrics with no key at all in `results` are skipped when skip_missing=True
    (e.g. quantile-only metrics for a point model).
    """
    out_dir = os.path.join(config.results_dir, subdir) if subdir else config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    base_data = _config_row(config, run_id, timestamp)

    for metric in metric_names:
        if skip_missing and not any(k.startswith(f'{metric}_') for k in results.keys()):
            continue
        year_columns = {str(year): results.get(f'{metric}_{year}', None) for year in test_years}
        year_columns['overall'] = results.get(f'{metric}_overall', None)
        _append_metric_row(os.path.join(out_dir, f'{metric}.csv'), {**base_data, **year_columns})
    print(f"[CSV Results] Saved {len(metric_names)} metric tables under {out_dir}")


# ---------------------------------------------------------------------------
# Consolidated results writer
#
# Instead of one CSV per metric (which produced ~76 files per results dir and
# repeated the full config block in each), all runs are collapsed into a small,
# fixed set of files, each upserted (not appended blindly) by run_id:
#
#   results_summary.csv  - one row per run_id; compact identity columns + every
#                          metric's OVERALL value side by side (core + uncertainty
#                          + protocol_*). This is the cross-run comparison table.
#   results_by_year.csv  - one row per (run_id, year); same metric columns, for
#                          per-horizon detail without a file explosion.
#   runs_config.csv      - full hyperparameter dump keyed by run_id (joinable,
#                          kept out of the comparison view).
#
# Both save_test_results_to_csv() (standard + uncertainty) and the evaluation
# protocol writer feed the SAME run_id row; the upsert merges their columns when
# they share a timestamp, and replaces the row on a genuine re-run (new timestamp).
# ---------------------------------------------------------------------------

# model_type -> coarse family, so results can be grouped/compared by family.
_MODEL_FAMILY = {
    'nlinear': 'linear', 'dlinear': 'linear', 'xlinear': 'linear',
    'rlinear': 'linear', 'olinear': 'linear',
    'autoformer': 'transformer', 'patchtst': 'transformer', 'tsmixer': 'transformer',
    'informer': 'transformer', 'tst': 'transformer', 'itransformer': 'transformer',
    'timexer': 'transformer', 'timesnet': 'transformer',
    'cnnlstm': 'cnn_lstm', 'cnn_lstm': 'cnn_lstm',
}


def _model_family(model_type) -> str:
    return _MODEL_FAMILY.get(str(model_type).lower(), 'other')


# Compact identity block that prefixes every summary / by-year row.
_SUMMARY_IDENTITY = [
    'timestamp', 'run_id', 'model_family', 'model_type', 'crop', 'country',
    'aggregation', 'forecast_fraction', 'loss_type', 'lag_years', 'test_years',
    'use_residual_trend', 'use_recursive_lags', 'use_sota_features',
    'use_exponential_weighting', 'if_tokenize', 'use_wfan', 'multi_year_summaries',
]

# Preferred left-to-right order for the core metric columns; everything else
# (protocol_*, quantile pinball, etc.) is appended alphabetically after these.
_METRIC_ORDER = [
    'r2', 'rmse', 'nrmse', 'mae', 'mape', 'smape', 'mse',
    'crps', 'picp', 'pinaw', 'pinball_avg', 'winkler_score', 'mean_interval_width',
    'pinball_q10', 'pinball_q50', 'pinball_q90',
]


def _is_present(val) -> bool:
    """Scalar-safe 'not missing' check (pd.notna on a list raises)."""
    try:
        return bool(pd.notna(val))
    except (ValueError, TypeError):
        return val is not None


def _identity_row(config, run_id: str, timestamp: str) -> Dict[str, Any]:
    return {
        'timestamp': timestamp,
        'run_id': run_id,
        'model_family': _model_family(getattr(config, 'model_type', None)),
        'model_type': getattr(config, 'model_type', None),
        'crop': getattr(config, 'crop', None),
        'country': getattr(config, 'country', None),
        'aggregation': getattr(config, 'aggregation', None),
        'forecast_fraction': getattr(config, 'data_fraction', None),
        'loss_type': getattr(config, 'loss_type', None),
        'lag_years': getattr(config, 'lag_years', None),
        'test_years': getattr(config, 'test_years', None),
        'use_residual_trend': getattr(config, 'use_residual_trend', None),
        'use_recursive_lags': getattr(config, 'use_recursive_lags', None),
        'use_sota_features': getattr(config, 'use_sota_features', None),
        'use_exponential_weighting': getattr(config, 'use_exponential_weighting', None),
        'if_tokenize': getattr(config, 'if_tokenize', None),
        'use_wfan': getattr(config, 'use_wfan', None),
        'multi_year_summaries': getattr(config, 'multi_year_summaries', None),
    }


def _order_with(identity_cols: List[str], cols: List[str]) -> List[str]:
    head = [c for c in identity_cols if c in cols]
    metrics = [c for c in cols if c not in head]
    ordered = [m for m in _METRIC_ORDER if m in metrics]
    rest = sorted(m for m in metrics if m not in ordered)
    return head + ordered + rest


def _summary_order(cols: List[str]) -> List[str]:
    return _order_with(_SUMMARY_IDENTITY, cols)


def _by_year_order(cols: List[str]) -> List[str]:
    ident = ['timestamp', 'run_id', 'year'] + [c for c in _SUMMARY_IDENTITY
                                               if c not in ('timestamp', 'run_id')]
    return _order_with(ident, cols)


def _config_order(cols: List[str]) -> List[str]:
    head = [c for c in ('timestamp', 'run_id') if c in cols]
    return head + [c for c in cols if c not in head]


def _upsert_rows(csv_path: str, rows: List[Dict[str, Any]], key_cols: List[str],
                 column_order_fn=None):
    """Upsert rows into a CSV keyed by key_cols, unioning columns across runs.

    Merge semantics let several writers for the SAME execution accumulate columns
    into one row:
      - existing row with the SAME 'timestamp' -> merge incoming non-null columns
        (existing columns preserved);
      - existing row with a DIFFERENT 'timestamp' -> fresh re-run, drop and replace;
      - no existing row for the key -> append.
    Missing cells stay blank. Column order is normalised via column_order_fn.
    """
    if os.path.exists(csv_path):
        try:
            out = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception:
            out = pd.DataFrame()
    else:
        out = pd.DataFrame()

    for inc in rows:
        if out.empty or not all(k in out.columns for k in key_cols):
            out = pd.concat([out, pd.DataFrame([inc])], ignore_index=True)
            continue
        mask = pd.Series(True, index=out.index)
        for k in key_cols:
            mask &= out[k].astype(str) == str(inc[k])
        idx = list(out.index[mask])
        if not idx:
            out = pd.concat([out, pd.DataFrame([inc])], ignore_index=True)
            continue
        i = idx[0]
        same_ts = ('timestamp' in out.columns and 'timestamp' in inc
                   and str(out.at[i, 'timestamp']) == str(inc['timestamp']))
        if same_ts:
            for col, val in inc.items():
                if _is_present(val):
                    if col not in out.columns:
                        out[col] = np.nan
                    out.at[i, col] = val
            for j in idx[1:]:
                out = out.drop(index=j)
        else:
            out = out.drop(index=idx)
            out = pd.concat([out, pd.DataFrame([inc])], ignore_index=True)
        out = out.reset_index(drop=True)

    if column_order_fn is not None and not out.empty:
        out = out.reindex(columns=column_order_fn(list(out.columns)))
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    out.to_csv(csv_path, index=False)


def save_consolidated_results(
    config: Union[TSTModelConfig, LinearModelConfig],
    results: Dict[str, float],
    test_years: List[int],
    run_id: str,
    timestamp: str,
    metric_names: List[str],
    prefix: str = '',
    write_config: bool = True,
    skip_missing: bool = True,
):
    """Upsert one run's metrics into the consolidated summary / by-year / config files.

    `results` holds flat keys '<metric>_<year>' and '<metric>_overall'. `prefix`
    namespaces metric columns (e.g. 'protocol_') to avoid collisions in the shared
    summary. Metrics absent from `results` are skipped when skip_missing=True.
    """
    out_dir = config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    identity = _identity_row(config, run_id, timestamp)

    present = [m for m in metric_names
               if (not skip_missing) or any(k.startswith(f'{m}_') for k in results)]

    # Summary: one row per run_id, overall values only.
    summary_row = dict(identity)
    for m in present:
        summary_row[f'{prefix}{m}'] = results.get(f'{m}_overall', None)
    _upsert_rows(os.path.join(out_dir, 'results_summary.csv'),
                 [summary_row], key_cols=['run_id'], column_order_fn=_summary_order)

    # By-year: one row per (run_id, year) that actually has values.
    year_rows = []
    for year in test_years:
        row = dict(identity)
        row['year'] = year
        has_any = False
        for m in present:
            v = results.get(f'{m}_{year}', None)
            row[f'{prefix}{m}'] = v
            has_any = has_any or (v is not None)
        if has_any:
            year_rows.append(row)
    if year_rows:
        _upsert_rows(os.path.join(out_dir, 'results_by_year.csv'),
                     year_rows, key_cols=['run_id', 'year'],
                     column_order_fn=_by_year_order)

    # Full config dump (written once per run, by the first/standard writer).
    if write_config:
        _upsert_rows(os.path.join(out_dir, 'runs_config.csv'),
                     [_config_row(config, run_id, timestamp)],
                     key_cols=['run_id'], column_order_fn=_config_order)


def save_test_results_to_csv(
    config: Union[TSTModelConfig, LinearModelConfig],
    test_results: Dict[str, float],
    test_years: List[int],
    run_id: str,
    timestamp: str
):
    """Save test results into the consolidated CSV files (summary + by-year + config).

    Standard metrics are always written (blank cells if absent); uncertainty
    metrics only when present (MQLoss). The evaluation protocol later merges its
    own 'protocol_*' columns into the same run_id row.
    """
    standard_metrics = ['nrmse', 'mape', 'r2', 'rmse', 'mae', 'mse', 'smape']
    save_consolidated_results(config, test_results, test_years, run_id, timestamp,
                              metric_names=standard_metrics, prefix='',
                              write_config=True, skip_missing=False)

    # Uncertainty metrics (only if present, i.e., MQLoss was enabled)
    uncertainty_metrics = ['pinball_avg', 'crps', 'picp', 'pinaw', 'winkler_score', 'mean_interval_width']
    uncertainty_metrics.extend([f'pinball_q{int(q*100)}' for q in [0.1, 0.5, 0.9]])  # Individual quantile pinball losses
    save_consolidated_results(config, test_results, test_years, run_id, timestamp,
                              metric_names=uncertainty_metrics, prefix='',
                              write_config=False, skip_missing=True)


# ---------------------------------------------------------------------------
# Canonical tier schema  (one row per evaluation CELL)
#
# The four evaluation tiers (accuracy / uncertainity / generalization /
# operational) and the hyperparameter dump all share ONE identity key so that
# every delta, bootstrap draw and ranking table can be built by joining rows on
# that key. The key is the smallest set of columns that uniquely identifies a
# single trained-and-scored cell:
#
#     crop, arm, model_name, country, origin, seed, partition
#
#   arm        : fusion-ladder label (B0..B6); today a plain baseline is "B0".
#   model_name : the architecture name from the script (defaults to model_type).
#   origin     : forecast origin as a season fraction (data_fraction: 1.0/0.75/..).
#   partition  : evaluation stratum (in_domain / extreme_label / extreme_driver).
#
# run_id is carried as a CONTEXT column (not part of the key) so any row can be
# traced back to its full hyperparameter record in hyperparameters.csv.
#
# Cross-country transfer reuses the exact same metric columns but swaps the
# single `country` key for a (from, to) pair; its files get a `transfer_` prefix.
# ---------------------------------------------------------------------------

# Identity key — one row per cell.
CELL_KEY_COLS = ['crop', 'arm', 'model_name', 'country', 'origin', 'seed', 'partition']
# Transfer identity key — specialist trained on `from`, scored on `to`.
TRANSFER_KEY_COLS = ['crop', 'arm', 'model_name', 'from', 'to', 'origin', 'seed', 'partition']
# Context columns carried on every tier row (traceability + sample count).
CELL_CONTEXT_COLS = ['timestamp', 'run_id', 'model_type', 'model_family',
                     'loss_type', 'n_samples']

# Ordered metric columns per tier (defines the on-disk column order; the writer
# reindexes to this, appending any unexpected metric keys alphabetically).
ACCURACY_METRICS = [
    'r2', 'rmse', 'nrmse', 'mae', 'mse', 'mape', 'smape',
    'rmse_ref_trend', 'rmse_ref_ols', 'rmse_ref_climatology', 'rmse_ref_rolling3',
    'mae_ref_trend', 'mae_ref_ols', 'mae_ref_climatology', 'mae_ref_rolling3',
    'skill_rmse_vs_trend', 'skill_rmse_vs_ols', 'skill_rmse_vs_climatology', 'skill_rmse_vs_rolling3',
    'skill_mae_vs_trend', 'skill_mae_vs_ols', 'skill_mae_vs_climatology', 'skill_mae_vs_rolling3',
    'within_5pct', 'within_10pct', 'within_20pct', 'within_0.5abs',
]
UNCERTAINTY_METRICS = [
    'pinball_q10', 'pinball_q50', 'pinball_q90', 'pinball_avg',
    'coverage_q10', 'coverage_q50', 'coverage_q90', 'quantile_calibration_error',
    'picp', 'picp_nominal', 'mean_interval_width', 'pinaw', 'winkler_score',
    'crps', 'crps_pinball', 'crps_pinball_ref_ols', 'skill_crps_vs_ols',
    'picp_ref_ols', 'mean_interval_width_ref_ols',
    'conformal_margin', 'picp_conformal', 'mean_interval_width_conformal', 'crps_pinball_conformal',
]
GENERALIZATION_METRICS = [
    'rmse', 'skill_rmse_vs_trend',
    'r_sp', 'r_tm', 'r_an_window', 'r_an_detrended', 'r2_anomaly_detrended',
    'nrmse_anomaly_detrended', 'anomaly_spread_ratio',
    'base_rate_extreme', 'agreement_rate',
]
OPERATIONAL_METRICS = [
    'badyear_0.5sig_auc', 'badyear_0.5sig_precision', 'badyear_0.5sig_recall',
    'badyear_0.5sig_f1', 'badyear_0.5sig_base_rate', 'badyear_0.5sig_n_events',
    'badyear_0.5sig_auc_qlo',
    'badyear_1sig_auc', 'badyear_1sig_precision', 'badyear_1sig_recall',
    'badyear_1sig_f1', 'badyear_1sig_base_rate', 'badyear_1sig_n_events',
    'badyear_1sig_auc_qlo',
    'shortfall_capture_c', 'shortfall_n_tail',
]

# tier name -> (metric column list). Used by the writer and the ranking tool.
TIER_METRIC_COLS = {
    'accuracy': ACCURACY_METRICS,
    'uncertainity': UNCERTAINTY_METRICS,
    'generalization': GENERALIZATION_METRICS,
    'operational': OPERATIONAL_METRICS,
}


def _cell_identity(config, run_id: str, timestamp: str, partition: str,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the canonical key + context block for one cell row.

    `extra` overrides/augments fields (e.g. {'from': 'FR', 'to': 'IT'} for the
    transfer writer, which supplies its own key columns in place of `country`).
    """
    ident = {
        'crop': getattr(config, 'crop', None),
        'arm': getattr(config, 'arm', 'B0'),
        'model_name': getattr(config, 'model_name', None) or getattr(config, 'model_type', None),
        'country': getattr(config, 'country', None),
        'origin': getattr(config, 'data_fraction', None),
        'seed': getattr(config, 'seed', None),
        'partition': partition,
        # context
        'timestamp': timestamp,
        'run_id': run_id,
        'model_type': getattr(config, 'model_type', None),
        'model_family': _model_family(getattr(config, 'model_type', None)),
        'loss_type': getattr(config, 'loss_type', None),
    }
    if extra:
        ident.update(extra)
    return ident


def _tier_order(key_cols: List[str], metric_cols: List[str]):
    """Return an ordering function: key cols, then context, then tier metrics."""
    def order(cols: List[str]) -> List[str]:
        head = [c for c in key_cols if c in cols]
        ctx = [c for c in CELL_CONTEXT_COLS if c in cols and c not in head]
        metrics = [m for m in metric_cols if m in cols]
        rest = sorted(c for c in cols if c not in head + ctx + metrics)
        return head + ctx + metrics + rest
    return order


def save_tier_results(
    config: Union[TSTModelConfig, LinearModelConfig],
    tier: str,
    rows: List[Dict[str, Any]],
    run_id: str,
    timestamp: str,
    partition: str = 'in_domain',
    prefix: str = '',
    key_cols: Optional[List[str]] = None,
):
    """Upsert one canonical row per cell into <prefix><tier>.csv.

    `rows` is a list of {metric_key: value, ...} dicts already reduced to one
    entry per cell (overall values only — NO per-year columns). Each dict may
    carry its own 'partition' and key overrides (e.g. 'from'/'to' for transfer);
    otherwise `partition` and the config identity are used. Upsert is keyed on
    `key_cols` (default CELL_KEY_COLS) so re-runs replace and sibling writers
    merge, exactly like the existing consolidated writer.
    """
    if not rows:
        return
    tier = tier.strip()
    metric_cols = TIER_METRIC_COLS.get(tier, [])
    key_cols = key_cols or CELL_KEY_COLS
    out_rows = []
    for r in rows:
        part = r.pop('partition', partition)
        extra = {k: r.pop(k) for k in list(r.keys()) if k in ('from', 'to', 'country')}
        row = _cell_identity(config, run_id, timestamp, part, extra=extra)
        row.update(r)  # metric values (+ n_samples if present)
        out_rows.append(row)
    out_dir = config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{prefix}{tier}.csv')
    _upsert_rows(path, out_rows, key_cols=key_cols,
                 column_order_fn=_tier_order(key_cols, metric_cols))
    print(f"[Tier CSV] {len(out_rows)} row(s) -> {path}")


def save_hyperparameters(
    config: Union[TSTModelConfig, LinearModelConfig],
    run_id: str,
    timestamp: str,
    prefix: str = '',
    extra: Optional[Dict[str, Any]] = None,
):
    """Write the full hyperparameter record keyed by the cell identity + run_id.

    Supersedes runs_config.csv: holds the complete config dump so any tier row
    (identified by its cell key + run_id) is fully traceable here. `extra`
    injects transfer key columns (from/to) when called by the transfer path.
    """
    out_dir = config.results_dir
    os.makedirs(out_dir, exist_ok=True)
    # Key block first (so the file is self-describing), then full config dump.
    ident = _cell_identity(config, run_id, timestamp, 'in_domain', extra=extra)
    cfg = _config_row(config, run_id, timestamp)
    row = {**ident, **cfg}
    key_cols = TRANSFER_KEY_COLS if (extra and 'from' in extra) else CELL_KEY_COLS
    path = os.path.join(out_dir, f'{prefix}hyperparameters.csv')

    def order(cols: List[str]) -> List[str]:
        head = [c for c in key_cols if c in cols]
        ctx = [c for c in CELL_CONTEXT_COLS if c in cols and c not in head]
        rest = sorted(c for c in cols if c not in head + ctx)
        return head + ctx + rest

    _upsert_rows(path, [row], key_cols=key_cols, column_order_fn=order)
    print(f"[Hyperparameters CSV] -> {path}")


def load_best_hps(results_file: str) -> Dict[str, Any]:
    """Load best hyperparameters from a single HPO results file."""
    with open(results_file, 'r') as f:
        content = f.read()

    match = re.search(
        r'BEST TRIAL:.*?Hyperparameters:(.*?)ALL TRIALS',
        content, re.DOTALL
    )

    if not match:
        return {}

    hps = {}
    for line in match.group(1).split('\n'):
        if ':' in line and not line.strip().startswith('='):
            key, val = line.split(':', 1)
            key, val = key.strip(), val.strip()
            try:
                hps[key] = int(val) if '.' not in val else float(val)
            except ValueError:
                hps[key] = val
    return hps


def load_all_hps(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all HPO results from directory.
    Returns dict with keys like 'patchtst-AO-maize' -> hyperparameters
    """
    results_path = Path(results_dir)
    all_hps = {}

    for file_path in results_path.glob("**/*HPO_results*.txt"):
        folder_name = file_path.parent.parent.name
        parts = folder_name.replace('hpo_', '').split('_')

        model = parts[0]
        crop = parts[-1]
        region = '_'.join(parts[1:-1])

        key = f"{model}-{region}-{crop}"
        hps = load_best_hps(str(file_path))

        if hps:
            all_hps[key] = hps

    return all_hps


def get_hps_for_model_crop_region(
    results_dir: str,
    model: str,
    crop: str,
    region: str
) -> Dict[str, Any]:
    """Get hyperparameters for a specific model, crop, and region combination."""
    key = f"{model}-{region}-{crop}"
    all_hps = load_all_hps(results_dir)
    return all_hps.get(key, {})


# Lead time function
def add_cutoff_days(df: pd.DataFrame, lead_time: str) -> pd.DataFrame:
    """Add a column with cutoff days relative to end of season.

    This function converts a lead_time string into the number of days before
    harvest when the forecast should be made.

    Args:
        df (pd.DataFrame): DataFrame with season_length column
        lead_time (str): Lead time option. Choices:
            - "end-of-season": Forecast at harvest (cutoff_days = 0)
            - "three-quarter-of-season": Forecast at 75% through season
            - "middle-of-season": Forecast at 50% through season (default)
            - "quarter-of-season": Forecast at 25% through season
            - "N-days": Forecast N days before harvest (e.g., "60-days")

    Returns:
        pd.DataFrame: The same DataFrame with cutoff_days column added

    Examples:
        >>> df = pd.DataFrame({'season_length': [120]})
        >>> add_cutoff_days(df, "middle-of-season")
        # Adds cutoff_days = 60 (half of 120)
        >>> add_cutoff_days(df, "60-days")
        # Adds cutoff_days = 60
    """
    if "day" in lead_time:
        df["cutoff_days"] = int(lead_time.split("-")[0])
    else:
        assert "season" in lead_time, f'Unrecognized lead time "{lead_time}"'
        if lead_time == "end-of-season":
            df["cutoff_days"] = 0
        elif lead_time == "three-quarter-of-season":
            df["cutoff_days"] = (df["season_length"] // 4).astype(int)
        elif lead_time == "middle-of-season":
            df["cutoff_days"] = (df["season_length"] // 2).astype(int)
        elif lead_time == "quarter-of-season":
            df["cutoff_days"] = (df["season_length"] * 3 // 4).astype(int)
        else:
            raise Exception(f'Unrecognized lead time "{lead_time}"')

    return df


def compute_cutoff_date(eos_date: pd.Timestamp, lead_time: str,
                       season_length: int) -> pd.Timestamp:
    """Compute the cutoff date for a given lead time.

    This function calculates when (what date) to make the forecast based on
    the lead time setting and end of season date.

    Args:
        eos_date (pd.Timestamp): End of season date
        lead_time (str): Lead time option
        season_length (int): Length of the season in days

    Returns:
        pd.Timestamp: The cutoff date when the forecast should be made

    Examples:
        >>> from datetime import datetime
        >>> eos = pd.Timestamp("2024-09-30")
        >>> compute_cutoff_date(eos, "middle-of-season", 120)
        # Returns Timestamp around mid-July (60 days before Sept 30)
        >>> compute_cutoff_date(eos, "end-of-season", 120)
        # Returns Timestamp("2024-09-30") (no change)
    """
    if "day" in lead_time:
        cutoff_days = int(lead_time.split("-")[0])
    elif lead_time == "end-of-season":
        cutoff_days = 0
    elif lead_time == "three-quarter-of-season":
        cutoff_days = season_length // 4
    elif lead_time == "middle-of-season":
        cutoff_days = season_length // 2
    elif lead_time == "quarter-of-season":
        cutoff_days = season_length * 3 // 4
    else:
        raise ValueError(f'Unrecognized lead time "{lead_time}"')

    return eos_date - pd.Timedelta(days=cutoff_days)


def get_effective_lead_time() -> str:
    """Get the effective forecast lead time from config.

    This function imports and calls get_forecast_type() from cybench.config
    to get the current forecast type setting (either the override or default).

    Returns:
        str: The effective forecast type setting

    Note:
        This function is a wrapper that allows the process folder to access
        forecast type settings without directly importing from cybench.datasets.alignment.
        The function name uses 'lead_time' for backward compatibility with existing code.
    """
    from cybench.config import get_forecast_type
    return get_forecast_type()
