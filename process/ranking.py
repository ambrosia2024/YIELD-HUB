# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Cross-run ranking / significance aggregator for the tiered
evaluation substrate (accuracy/uncertainity/generalization/operational.csv +
their per-sample predictions/). It is the separate downstream step that turns a
raw difference into a CLAIM, per the evaluation protocol §6:

  * Point delta  Δ = m(arm) - m(B0), averaged over seeds, per
                 (crop, model_name, country, origin, partition, metric).
  * Year-block bootstrap (resample whole YEARS, not region-years, because all
    regions share one weather year): B draws, recompute Δ on each draw from the
    per-sample predictions by RE-RUNNING the same tier measurement classes, take
    the [2.5, 97.5] percentile interval.
  * Wilcoxon signed-rank across (crop, country) pairs for the panel-wide claim
    at each (crop, origin, partition, metric, arm_vs_B0) group.
  * WIN := bootstrap interval excludes 0 AND Wilcoxon p < 0.05.

Output: one row per (crop, model_name, country, origin, partition, arm_vs_B0,
metric) in <results_dir>/ranking.csv.

This tool READS the substrate only; it never trains. The bootstrap needs runs
executed with --generate_predictions (so predictions/<run>.csv exists); without
them only the point delta is filled and the interval is left blank.

Usage:
  python ranking.py --results_dir checkpoints/results
  python ranking.py --results_dir checkpoints/results --baseline_arm B0 \
                    --n_boot 1000 --alpha 0.05

Python version: 3.12.0
--------------------
"""

import os
import re
import sys
import glob
import argparse
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Make the tier measurement classes and ProtocolConfig importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)
sys.path.append(os.path.join(os.path.dirname(_THIS_DIR), 'architectures'))

# Headline metric per tier used for the Wilcoxon panel test (the bootstrap
# itself covers every metric the tier recomputes).
TIER_HEADLINE = {
    'accuracy': 'skill_rmse_vs_trend',
    'uncertainity': 'winkler_score',
    'generalization': 'r_an_detrended',
    'operational': 'badyear_1sig_auc',
}
# Lower-is-better metrics (so "beats B0" means a NEGATIVE delta).
LOWER_BETTER = {
    'rmse', 'mae', 'mse', 'nrmse', 'mape', 'smape',
    'pinball_q10', 'pinball_q50', 'pinball_q90', 'pinball_avg',
    'quantile_calibration_error', 'mean_interval_width', 'pinaw',
    'winkler_score', 'crps', 'crps_pinball', 'crps_pinball_ref_ols',
    'mean_interval_width_ref_ols', 'conformal_margin',
    'mean_interval_width_conformal', 'crps_pinball_conformal',
    'nrmse_anomaly_detrended', 'shortfall_n_tail',
}
# Columns that are context / counts, never scored as a delta.
NON_METRIC = {
    'timestamp', 'run_id', 'model_type', 'model_family', 'loss_type',
    'n_samples', 'picp_nominal', 'base_rate_extreme', 'agreement_rate',
    'badyear_0.5sig_n_events', 'badyear_1sig_n_events',
    'badyear_0.5sig_base_rate', 'badyear_1sig_base_rate',
}
# The cell key minus `arm` and `seed`: a comparison is arm-vs-B0 within this.
COMPARE_KEY = ['crop', 'model_name', 'country', 'origin', 'partition']

_TIERS = ['accuracy', 'uncertainity', 'generalization', 'operational']


# --------------------------------------------------------------------------- IO
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f'[ranking] could not read {path}: {e}')
        return pd.DataFrame()


def _metric_cols(df: pd.DataFrame, key_cols: Sequence[str]) -> List[str]:
    out = []
    for c in df.columns:
        if c in key_cols or c in NON_METRIC or c in ('arm', 'seed', 'from', 'to'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


# ------------------------------------------------------------------ point delta
def _point_deltas(tier: str, df: pd.DataFrame, baseline_arm: str) -> pd.DataFrame:
    """Δ = mean_seed m(arm) - mean_seed m(B0) for every metric column."""
    if df.empty or 'arm' not in df.columns:
        return pd.DataFrame()
    present_key = [k for k in COMPARE_KEY if k in df.columns]
    metrics = _metric_cols(df, present_key)
    if not metrics:
        return pd.DataFrame()
    # average over seeds (and any stray duplicates)
    agg = df.groupby(present_key + ['arm'], dropna=False)[metrics].mean().reset_index()
    base = agg[agg['arm'] == baseline_arm].drop(columns=['arm'])
    rows = []
    for arm, g in agg.groupby('arm'):
        if arm == baseline_arm:
            continue
        merged = g.drop(columns=['arm']).merge(base, on=present_key, suffixes=('_arm', '_b0'))
        for _, r in merged.iterrows():
            for m in metrics:
                va, vb = r.get(f'{m}_arm'), r.get(f'{m}_b0')
                if pd.isna(va) or pd.isna(vb):
                    continue
                rows.append({
                    **{k: r[k] for k in present_key},
                    'tier': tier, 'arm_vs_b0': f'{arm}_vs_{baseline_arm}',
                    'arm': arm, 'metric': m,
                    'm_arm': float(va), 'm_b0': float(vb),
                    'delta': float(va - vb),
                    'lower_is_better': m in LOWER_BETTER,
                })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- bootstrap
def _load_predictions(results_dir: str) -> pd.DataFrame:
    """Load and concatenate all per-sample prediction frames with run_id."""
    frames = []
    for path in glob.glob(os.path.join(results_dir, 'predictions', '*.csv')):
        d = _read_csv(path)
        if not d.empty:
            d['__file'] = os.path.basename(path)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _tier_compute(tier: str, frame: pd.DataFrame, quantiles) -> Dict[str, float]:
    """Recompute a tier's metrics on a (resampled) prediction frame."""
    from metrics import (
        ProtocolConfig, measureInDomainAccuracy, measureCalibratedUncertainity,
        measureGeneralizationShift, measureOperationalValue,
    )
    cfg = ProtocolConfig(quantiles=list(quantiles) if quantiles else [])
    if tier == 'accuracy':
        return measureInDomainAccuracy(cfg).compute(frame)
    if tier == 'uncertainity':
        return measureCalibratedUncertainity(cfg).compute(frame, conformal_margin_value=np.nan)
    if tier == 'generalization':
        return measureGeneralizationShift(cfg).compute_in_domain(frame)
    if tier == 'operational':
        return measureOperationalValue(cfg).compute(frame)
    return {}


def _resample_years(frame: pd.DataFrame, years: np.ndarray, rng) -> pd.DataFrame:
    """Year-block resample: draw whole years with replacement, concat their rows."""
    drawn = rng.choice(years, size=len(years), replace=True)
    by_year = {y: g for y, g in frame.groupby('year')}
    parts = [by_year[y] for y in drawn if y in by_year]
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]


def _bootstrap_cell(tier: str, arm_frame: pd.DataFrame, b0_frame: pd.DataFrame,
                    quantiles, n_boot: int, alpha: float, seed: int = 0
                    ) -> Dict[str, Dict[str, float]]:
    """Paired year-block bootstrap of Δ for every metric the tier recomputes.

    Returns {metric: {ci_lo, ci_hi, n_boot_eff}}.
    """
    years = np.array(sorted(set(arm_frame['year'].unique()) & set(b0_frame['year'].unique())))
    if len(years) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {}
    for _ in range(n_boot):
        drawn = rng.choice(years, size=len(years), replace=True)
        # reuse the SAME drawn years for arm and baseline (paired)
        a_by = {y: g for y, g in arm_frame.groupby('year')}
        b_by = {y: g for y, g in b0_frame.groupby('year')}
        a_sub = pd.concat([a_by[y] for y in drawn if y in a_by], ignore_index=True)
        b_sub = pd.concat([b_by[y] for y in drawn if y in b_by], ignore_index=True)
        if len(a_sub) == 0 or len(b_sub) == 0:
            continue
        ma = _tier_compute(tier, a_sub, quantiles)
        mb = _tier_compute(tier, b_sub, quantiles)
        for m in set(ma) & set(mb):
            va, vb = ma[m], mb[m]
            if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
                continue
            draws.setdefault(m, []).append(float(va - vb))
    out = {}
    for m, vals in draws.items():
        if len(vals) < max(20, n_boot // 10):   # too few valid draws -> unreliable
            continue
        lo = float(np.percentile(vals, 100 * alpha / 2))
        hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        out[m] = {'ci_lo': lo, 'ci_hi': hi, 'n_boot_eff': len(vals)}
    return out


def _infer_quantiles(frame: pd.DataFrame) -> List[float]:
    return [0.1, 0.5, 0.9] if {'q10', 'q50', 'q90'}.issubset(frame.columns) else []


# ---------------------------------------------------------------------- Wilcoxon
def _wilcoxon_panel(point: pd.DataFrame) -> pd.DataFrame:
    """Signed-rank test of Δ across (crop, country) pairs for each
    (crop, origin, partition, tier, metric, arm) group (the panel claim)."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        logger.warning('[ranking] scipy not available; skipping Wilcoxon')
        return pd.DataFrame()
    rows = []
    group_cols = ['crop', 'model_name', 'origin', 'partition', 'tier', 'metric', 'arm_vs_b0']
    for keys, g in point.groupby(group_cols):
        deltas = g['delta'].dropna().values
        if len(deltas) < 3 or np.allclose(deltas, 0):
            p = np.nan
        else:
            try:
                p = float(wilcoxon(deltas)[1])
            except Exception:
                p = np.nan
        rows.append({**dict(zip(group_cols, keys)), 'wilcoxon_p': p, 'n_countries': len(deltas)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run
def build_ranking(results_dir: str, baseline_arm: str = 'B0',
                  n_boot: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
    # 1. point deltas from the four tier files
    point_frames = []
    for tier in _TIERS:
        df = _read_csv(os.path.join(results_dir, f'{tier}.csv'))
        pf = _point_deltas(tier, df, baseline_arm)
        if not pf.empty:
            point_frames.append(pf)
    if not point_frames:
        print(f'[ranking] no tier files with an arm != {baseline_arm} found in {results_dir}; nothing to rank.')
        return pd.DataFrame()
    point = pd.concat(point_frames, ignore_index=True)

    # 2. year-block bootstrap intervals (in_domain only; needs predictions/)
    preds = _load_predictions(results_dir)
    boot_rows = []
    if not preds.empty and 'run_id' in preds.columns:
        # map run_id -> (arm, cell) via the tier files
        idmap = {}
        for tier in _TIERS:
            df = _read_csv(os.path.join(results_dir, f'{tier}.csv'))
            if df.empty or 'run_id' not in df.columns:
                continue
            for _, r in df.iterrows():
                idmap.setdefault(str(r['run_id']), {
                    'arm': r.get('arm'), 'crop': r.get('crop'),
                    'model_name': r.get('model_name'), 'country': r.get('country'),
                    'origin': r.get('origin'),
                })
        preds = preds.copy()
        preds['arm'] = preds['run_id'].astype(str).map(lambda rid: idmap.get(rid, {}).get('arm'))
        for col in ('crop', 'model_name', 'country', 'origin'):
            preds[col] = preds['run_id'].astype(str).map(lambda rid, c=col: idmap.get(rid, {}).get(c))
        # pool seeds within each (compare key, arm); partition is in_domain here
        cell_cols = ['crop', 'model_name', 'country', 'origin']
        pools = {k: g for k, g in preds.dropna(subset=['arm']).groupby(cell_cols + ['arm'])}
        for key, g_arm in pools.items():
            *cell, arm = key
            if arm == baseline_arm:
                continue
            b0_key = tuple(cell) + (baseline_arm,)
            if b0_key not in pools:
                continue
            g_b0 = pools[b0_key]
            quantiles = _infer_quantiles(g_arm)
            for tier in _TIERS:
                res = _bootstrap_cell(tier, g_arm, g_b0, quantiles, n_boot, alpha)
                for metric, ci in res.items():
                    boot_rows.append({
                        'crop': cell[0], 'model_name': cell[1], 'country': cell[2],
                        'origin': cell[3], 'partition': 'in_domain', 'tier': tier,
                        'arm_vs_b0': f'{arm}_vs_{baseline_arm}', 'metric': metric, **ci,
                    })
    boot = pd.DataFrame(boot_rows)

    # 3. merge point + bootstrap
    merge_cols = ['crop', 'model_name', 'country', 'origin', 'partition', 'tier', 'arm_vs_b0', 'metric']
    out = point.merge(boot, on=merge_cols, how='left') if not boot.empty else point.copy()
    for c in ('ci_lo', 'ci_hi', 'n_boot_eff'):
        if c not in out.columns:
            out[c] = np.nan

    # 4. Wilcoxon panel p-values, attached per group
    wilc = _wilcoxon_panel(point)
    if not wilc.empty:
        out = out.merge(wilc, on=['crop', 'model_name', 'origin', 'partition', 'tier', 'metric', 'arm_vs_b0'], how='left')
    else:
        out['wilcoxon_p'] = np.nan
        out['n_countries'] = np.nan

    # 5. win flag: bootstrap CI excludes 0 AND Wilcoxon p < 0.05
    def _ci_excludes_zero(r):
        if pd.isna(r['ci_lo']) or pd.isna(r['ci_hi']):
            return np.nan
        return bool(r['ci_lo'] > 0 or r['ci_hi'] < 0)
    out['boot_excludes_0'] = out.apply(_ci_excludes_zero, axis=1)
    # "improvement" direction: negative delta is good for lower-is-better metrics
    out['beats_b0_direction'] = out.apply(
        lambda r: bool(r['delta'] < 0) if r['lower_is_better'] else bool(r['delta'] > 0), axis=1)
    out['is_win'] = out.apply(
        lambda r: bool(r['boot_excludes_0'] is True and pd.notna(r['wilcoxon_p'])
                       and r['wilcoxon_p'] < alpha and r['beats_b0_direction']), axis=1)

    front = merge_cols + ['arm', 'm_arm', 'm_b0', 'delta', 'lower_is_better',
                          'beats_b0_direction', 'ci_lo', 'ci_hi', 'n_boot_eff',
                          'wilcoxon_p', 'n_countries', 'boot_excludes_0', 'is_win']
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    return out[cols]


# -------------------------------------------------------- earliest useful origin
def earliest_useful_origin(results_dir: str, auc_threshold: float = 0.70,
                           sigma: str = '1sig') -> pd.DataFrame:
    """Minimum forecast origin where bad-year AUC >= threshold, per (crop, arm, country).

    Reads operational.csv, groups by (crop, arm, country), sorts by origin,
    and returns the first origin meeting the AUC threshold (or 'never').
    This is a post-hoc aggregate — it only makes sense after runs at multiple
    origins have been collected.

    Returns a DataFrame with columns:
        crop, arm, model_name, country, earliest_origin, auc_at_earliest
    """
    path = os.path.join(results_dir, 'operational.csv')
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    auc_col = f'badyear_{sigma}_auc'
    if auc_col not in df.columns or 'origin' not in df.columns:
        logger.warning(f'[ranking] operational.csv lacks "{auc_col}" or "origin"; '
                       f'cannot compute earliest_useful_origin')
        return pd.DataFrame()
    group_cols = [c for c in ['crop', 'arm', 'model_name', 'country'] if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols):
        g = g.sort_values('origin')
        found = g[g[auc_col] >= auc_threshold]
        if len(found):
            best = found.iloc[0]
            rows.append({**dict(zip(group_cols, keys)),
                         'earliest_origin': float(best['origin']),
                         'auc_at_earliest': float(best[auc_col])})
        else:
            rows.append({**dict(zip(group_cols, keys)),
                         'earliest_origin': np.nan,   # "never"
                         'auc_at_earliest': np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------ shift-stratified performance
# Join the per-(crop, country) DATASET shift characterisation produced by
# pre-compute/compute_shift_distance.py onto the model-performance deltas and
# aggregate within each shift bucket (far / near / none).  This is the post-hoc
# step that answers "does the model generalise better where the distribution
# shift is larger?" — the Δ-skill-vs-shift reading of H1/H4 in the protocol §2.C/2.D.
#
# Design: shift distance is a DATASET property and lives in its own pre-compute
# CSV; the join belongs HERE (post-hoc, cross-run), never in metrics.py (which
# would bloat the per-run output) nor in compute_shift_distance.py (which has no
# model results).  It mirrors earliest_useful_origin: read CSVs, aggregate, write.
_SHIFT_CONTEXT_COLS = ['shift_class', 'shift_distance_auc', 'auc_debiased',
                       'perm_p', 'n_features_shifted', 'frac_features_shifted',
                       'n_target']
# Print/sort order for the shift buckets (hardest first).
_SHIFT_ORDER = {'far': 0, 'near': 1, 'none': 2, 'undetermined': 3, 'unknown': 4}


def _default_shift_dir() -> str:
    """cybench/data/shift_distance, resolved relative to this file.

    process/ -> YIELD-HUB/ -> AgML-CY-Bench/ -> cybench/data/shift_distance
    (mirrors the CYBENCH_ROOT logic in pre-compute/compute_shift_distance.py).
    """
    repo_root = os.path.dirname(os.path.dirname(_THIS_DIR))
    return os.path.join(repo_root, 'cybench', 'data', 'shift_distance')


def _load_shift_distance(shift_dir: str, crops: Sequence[str]) -> pd.DataFrame:
    """Concatenate shift_distance_{crop}.csv (context columns only) over crops."""
    keep = ['crop', 'country'] + _SHIFT_CONTEXT_COLS
    frames = []
    for crop in sorted({str(c) for c in crops if pd.notna(c)}):
        d = _read_csv(os.path.join(shift_dir, f'shift_distance_{crop}.csv'))
        if d.empty:
            continue
        frames.append(d[[c for c in keep if c in d.columns]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Rename n_target -> n_regions for clarity in downstream outputs.
    if 'n_target' in out.columns:
        out = out.rename(columns={'n_target': 'n_regions'})
    return out


def shift_stratified_performance(ranking: pd.DataFrame, shift_dir: str):
    """Attach shift context to the ranking rows and bucket performance by shift_class.

    Returns (enriched, stratified):
      enriched   : every ranking row + the shift context columns (on crop, country).
      stratified : one row per (crop, tier, metric, arm_vs_b0, shift_class) with
                   bucketed performance and the member country names — the
                   paper-ready "performance across far/near/none" table.
    Empty stratified (and unchanged ranking) if no shift CSVs are found.
    """
    if ranking.empty or 'country' not in ranking.columns:
        return ranking, pd.DataFrame()
    shift = _load_shift_distance(shift_dir, ranking['crop'].unique())
    if shift.empty:
        logger.warning('[ranking] no shift_distance_{crop}.csv under %s; '
                       'skipping shift-stratified table.', shift_dir)
        return ranking, pd.DataFrame()

    enriched = ranking.merge(shift, on=['crop', 'country'], how='left')
    # Countries absent from the shift CSV (e.g. too few regions to score) are
    # kept as 'unknown' so nothing silently vanishes from the panel.
    enriched['shift_class'] = enriched['shift_class'].fillna('unknown')

    group = ['crop', 'tier', 'metric', 'arm_vs_b0', 'shift_class']
    rows = []
    for keys, g in enriched.groupby(group, dropna=False):
        deltas    = g['delta'].dropna()
        beats     = g['beats_b0_direction'].dropna() if 'beats_b0_direction' in g else pd.Series(dtype=float)
        wins      = g['is_win'].dropna() if 'is_win' in g else pd.Series(dtype=float)
        countries = sorted({str(c) for c in g['country'].dropna().unique()})
        rows.append({
            **dict(zip(group, keys)),
            'n_countries':      len(countries),
            'countries':        ';'.join(countries),
            'mean_delta':       float(deltas.mean())   if len(deltas) else np.nan,
            'median_delta':     float(deltas.median()) if len(deltas) else np.nan,
            'mean_m_arm':       float(g['m_arm'].dropna().mean()) if 'm_arm' in g else np.nan,
            'mean_m_b0':        float(g['m_b0'].dropna().mean())  if 'm_b0'  in g else np.nan,
            'frac_beats_b0':    float(beats.mean()) if len(beats) else np.nan,
            'n_wins':           int(wins.sum())     if len(wins)  else 0,
            # within-bucket shift context (how OOD this bucket's countries are)
            'mean_shift_auc':      float(g['shift_distance_auc'].dropna().mean()) if 'shift_distance_auc' in g else np.nan,
            'mean_auc_debiased':   float(g['auc_debiased'].dropna().mean())       if 'auc_debiased'       in g else np.nan,
            'mean_feats_shifted':  float(g['n_features_shifted'].dropna().mean()) if 'n_features_shifted' in g else np.nan,
        })
    strat = pd.DataFrame(rows)
    if not strat.empty:
        strat['__o'] = strat['shift_class'].map(lambda s: _SHIFT_ORDER.get(s, 9))
        strat = (strat.sort_values(['crop', 'tier', 'metric', 'arm_vs_b0', '__o'])
                      .drop(columns='__o').reset_index(drop=True))
    return enriched, strat


def main():
    ap = argparse.ArgumentParser(description='Cross-run ranking / significance aggregator -> ranking.csv')
    ap.add_argument('--results_dir', required=True,
                    help='Directory holding accuracy.csv/.../operational.csv and predictions/.')
    ap.add_argument('--baseline_arm', default='B0', help='Arm to compare every other arm against (default: B0).')
    ap.add_argument('--n_boot', type=int, default=1000, help='Year-block bootstrap draws (default: 1000).')
    ap.add_argument('--alpha', type=float, default=0.05, help='Two-sided interval / Wilcoxon level (default: 0.05).')
    ap.add_argument('--shift_distance_dir', default=None,
                    help='Directory holding shift_distance_{crop}.csv from '
                         'pre-compute/compute_shift_distance.py (default: '
                         'cybench/data/shift_distance relative to the repo root).')
    ap.add_argument('--out', default=None, help='Output path (default: <results_dir>/ranking.csv).')
    args = ap.parse_args()

    rank = build_ranking(args.results_dir, args.baseline_arm, args.n_boot, args.alpha)
    if rank.empty:
        return
    out_path = args.out or os.path.join(args.results_dir, 'ranking.csv')
    rank.to_csv(out_path, index=False)
    n_win = int(rank['is_win'].sum()) if 'is_win' in rank.columns else 0
    print(f'[ranking] wrote {len(rank)} comparison rows ({n_win} wins) -> {out_path}')

    # Earliest useful origin (only meaningful when multiple origins exist)
    euo = earliest_useful_origin(args.results_dir)
    if not euo.empty:
        euo_path = os.path.join(args.results_dir, 'earliest_useful_origin.csv')
        euo.to_csv(euo_path, index=False)
        n_never = int(euo['earliest_origin'].isna().sum())
        print(f'[ranking] earliest useful origin: {len(euo)} cells ({n_never} "never") -> {euo_path}')

    # Shift-stratified performance: join the dataset shift characterisation and
    # bucket the per-country deltas by far / near / none (protocol §2.C/2.D).
    shift_dir = args.shift_distance_dir or _default_shift_dir()
    enriched, strat = shift_stratified_performance(rank, shift_dir)
    if not strat.empty:
        enr_path = os.path.join(args.results_dir, 'ranking_with_shift.csv')
        enriched.to_csv(enr_path, index=False)
        strat_path = os.path.join(args.results_dir, 'shift_stratified_performance.csv')
        strat.to_csv(strat_path, index=False)
        by_class = strat.drop_duplicates('shift_class').set_index('shift_class')['n_countries'].to_dict()
        print(f'[ranking] shift-stratified: {len(strat)} (crop,tier,metric,arm,class) rows '
              f'[countries/class: {by_class}] -> {strat_path}')
        print(f'[ranking] ranking + shift context -> {enr_path}')


if __name__ == '__main__':
    main()
