#!/usr/bin/env python3
"""
generate_lpjml_results.py
-------------------------
Evaluate LPJmL (rainfed and irrigated, separately) against CY-Bench observed
yields using the same train/val/test split as the ML benchmark, then append
the results to both benchmark CSV files.

Methodology
-----------
* Source priority: region_aggregated (sub-national adm_id level) first; falls
  back to country_aggregated (national level, one row per year) for countries
  not covered by region_aggregated.  Country-aggregated results are flagged with
  spatial_scale='country' vs 'region' so they can be compared fairly.
* Bias correction: per-adm_id OLS (observed ~ lpjml) fit on TRAIN years only.
  If fewer than 3 train overlap points, fall back to country-level OLS.
* Reference forecasts (rolling-3, climatology, OLS/Theil-Sen trend) built from
  TRAIN+VAL observed yields — consistent with metrics.py _fit_reference_forecasts.
* All metrics computed on TEST years (last 5 years per years_dict.json).
* Zero-yield adm_ids (mean LPJmL train yield == 0) are excluded before calibration.

Usage
-----
  /home/vsaxena1/miniconda3/envs/cybench/bin/python \
      YIELD-HUB/pre-compute/generate_lpjml_results.py [--dry-run]

Outputs
-------
  YIELD-HUB/results/lpjml_accuracy.csv          — standalone results
  appends to: YIELD-HUB/results/benchmarkArchitectureResults.csv
  appends to: YIELD-HUB/results/benchmarkResolutionResults.csv
"""

import os, sys, json, glob, csv, argparse, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path('/gpfs/home4/vsaxena1/AgML-CY-Bench')
YIELD_HUB    = REPO_ROOT / 'YIELD-HUB'
CYBENCH_DATA = REPO_ROOT / 'cybench' / 'data'
LPJML_BASE   = CYBENCH_DATA / 'LPJmL Yield Data' / 'region_aggregated'
LPJML_CTRY   = CYBENCH_DATA / 'LPJmL Yield Data' / 'country_aggregated'
YEARS_DICT   = YIELD_HUB / 'setups' / 'configurations' / 'years_dict.json'
RESULTS_DIR  = YIELD_HUB / 'results'
ARCH_CSV     = RESULTS_DIR / 'benchmarkArchitectureResults.csv'
RES_CSV      = RESULTS_DIR / 'benchmarkResolutionResults.csv'
OUT_CSV      = RESULTS_DIR / 'lpjml_accuracy.csv'

# ── Water type config ──────────────────────────────────────────────────────────
WATER_TYPES = {
    'maize': {
        'rainfed':  {'folder': 'rainfed_maize_yield',          'col': 'rainfed_maize_yield'},
        'irrigated':{'folder': 'irrigated_maize_yield',         'col': 'irrigated_maize_yield'},
    },
    'wheat': {
        'rainfed':  {'folder': 'rainfed_temperate_cereals_yield',   'col': 'rainfed_temperate_cereals_yield'},
        'irrigated':{'folder': 'irrigated_temperate_cereals_yield',  'col': 'irrigated_temperate_cereals_yield'},
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def theil_sen_predict(x_train, y_train, x_pred):
    """Theil-Sen slope + median intercept, extrapolated to x_pred."""
    if len(x_train) < 2:
        return np.full(len(x_pred), np.nanmean(y_train))
    slope, intercept, *_ = stats.theilslopes(y_train, x_train)
    return slope * np.array(x_pred) + intercept


def ols_predict(x_train, y_train, x_pred):
    """OLS linear trend, extrapolated to x_pred."""
    if len(x_train) < 2:
        return np.full(len(x_pred), np.nanmean(y_train))
    slope, intercept, *_ = stats.linregress(x_train, y_train)
    return slope * np.array(x_pred) + intercept


def compute_metrics(y_true, y_pred):
    """
    Returns dict of error metrics.
    y_true, y_pred: 1-D arrays of the same length.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return None

    n     = len(yt)
    res   = yt - yp
    rmse  = np.sqrt(np.mean(res**2))
    mae   = np.mean(np.abs(res))
    mse   = np.mean(res**2)
    mean_obs = np.mean(yt)
    nrmse = rmse / mean_obs if mean_obs else np.nan

    # r²
    ss_res = np.sum(res**2)
    ss_tot = np.sum((yt - np.mean(yt))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # percentage errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_pct = np.abs(res) / np.where(yt > 0, yt, np.nan)
    mape  = np.nanmean(abs_pct) * 100
    smape = np.nanmean(2 * np.abs(res) / (np.abs(yt) + np.abs(yp) + 1e-9)) * 100

    # tolerance accuracy
    def within_pct(tol): return np.nanmean(abs_pct <= tol)
    def within_abs(tol): return np.mean(np.abs(res) <= tol)

    return dict(
        n_samples=n, r2=r2, rmse=rmse, mae=mae, mse=mse, nrmse=nrmse,
        mape=mape, smape=smape,
        within_1pct=within_pct(0.01),
        within_5pct=within_pct(0.05),
        within_10pct=within_pct(0.10),
        within_20pct=within_pct(0.20),
        within_0p1abs=within_abs(0.1),
        within_0p5abs=within_abs(0.5),
    )


def compute_ref_metrics(y_true, y_ref):
    y_true = np.asarray(y_true, float)
    y_ref  = np.asarray(y_ref,  float)
    mask   = np.isfinite(y_true) & np.isfinite(y_ref)
    yt, yr = y_true[mask], y_ref[mask]
    if len(yt) == 0:
        return np.nan, np.nan
    rmse_r = np.sqrt(np.mean((yt - yr)**2))
    mae_r  = np.mean(np.abs(yt - yr))
    return rmse_r, mae_r


def build_references(obs_df, train_val_years, test_years):
    """
    Build per-adm_id reference forecasts from TRAIN+VAL observed data.
    Returns a DataFrame indexed by (adm_id, test_year) with columns:
        ref_clim, ref_roll3, ref_ols, ref_trend
    """
    rows = []
    tv_df = obs_df[obs_df['harvest_year'].isin(train_val_years)]
    for adm, grp in tv_df.groupby('adm_id'):
        yrs   = sorted(grp['harvest_year'].values)
        ylds  = grp.set_index('harvest_year')['yield']

        clim   = ylds.mean()
        roll3  = ylds[ylds.index >= (max(yrs) - 2)].mean()   # last 3 train+val years
        x_tr   = np.array(yrs, float)
        y_tr   = np.array([ylds[yr] for yr in yrs], float)

        for ty in test_years:
            ols_pred   = ols_predict(x_tr, y_tr, [ty])[0]
            trend_pred = theil_sen_predict(x_tr, y_tr, [ty])[0]
            rows.append(dict(adm_id=adm, test_year=ty,
                             ref_clim=clim, ref_roll3=roll3,
                             ref_ols=ols_pred, ref_trend=trend_pred))
    return pd.DataFrame(rows).set_index(['adm_id', 'test_year'])


def apply_bias_correction(lpjml_df, obs_df, train_years, lpjml_col):
    """
    Per-adm_id OLS: obs_train = a * lpjml_train + b.
    Returns lpjml_df (test rows) with a 'pred_bc' column added.
    Falls back to country-level OLS when fewer than 3 train points.
    Zero-yield adm_ids (mean lpjml train = 0) are dropped from the output.
    """
    # Train observed joined with LPJmL
    obs_train = obs_df[obs_df['harvest_year'].isin(train_years)][['adm_id','harvest_year','yield']]
    lpj_train = lpjml_df[lpjml_df['year'].isin(train_years)][['adm_id','year', lpjml_col]]
    merged_tr = pd.merge(obs_train, lpj_train,
                         left_on=['adm_id','harvest_year'],
                         right_on=['adm_id','year'], how='inner')

    # Exclude zero-yield adm_ids
    mean_lpj = merged_tr.groupby('adm_id')[lpjml_col].mean()
    valid_adm = mean_lpj[mean_lpj > 0].index
    merged_tr = merged_tr[merged_tr['adm_id'].isin(valid_adm)]

    # Country-level fallback OLS (all valid adm_ids pooled)
    try:
        if len(merged_tr) >= 3 and merged_tr[lpjml_col].nunique() > 1:
            slope_c, intercept_c, *_ = stats.linregress(
                merged_tr[lpjml_col].values, merged_tr['yield'].values)
        else:
            raise ValueError("constant x")
    except (ValueError, TypeError):
        # Fall back to mean-shift: scale=1, shift by mean difference
        mean_lpj = merged_tr[lpjml_col].mean()
        mean_obs = merged_tr['yield'].mean()
        slope_c, intercept_c = 1.0, (mean_obs - mean_lpj) if mean_lpj else 0.0

    # Per-adm_id OLS
    slopes, intercepts = {}, {}
    for adm, grp in merged_tr.groupby('adm_id'):
        if len(grp) >= 3 and grp[lpjml_col].nunique() > 1:
            try:
                s, i, *_ = stats.linregress(grp[lpjml_col].values, grp['yield'].values)
                slopes[adm], intercepts[adm] = s, i
            except (ValueError, TypeError):
                slopes[adm], intercepts[adm] = slope_c, intercept_c
        else:
            slopes[adm], intercepts[adm] = slope_c, intercept_c

    # Apply to test rows
    test_rows = lpjml_df[lpjml_df['adm_id'].isin(valid_adm)].copy()
    test_rows['pred_bc'] = test_rows.apply(
        lambda r: max(0.0, slopes.get(r['adm_id'], slope_c) * r[lpjml_col]
                      + intercepts.get(r['adm_id'], intercept_c)), axis=1)
    return test_rows


def _evaluate_country_national(crop, country, water_type, lpjml_col,
                               lpj, obs, train_years, train_val, val_years, test_years):
    """
    Country-aggregated evaluation path.
    lpj: DataFrame with columns ['adm_id','year', lpjml_col] — one row per year.
    obs: DataFrame with columns ['adm_id','harvest_year','yield'] — sub-national rows.
    Aggregates obs to country mean per year, applies country-level OLS bias correction,
    evaluates on test_years.  Returns 5 data points (one per test year).
    """
    # Aggregate observed to country mean per year
    obs_mean = obs.groupby('harvest_year')['yield'].mean().reset_index()
    obs_mean.columns = ['year', 'obs_mean']

    # Country-level LPJmL (single value per year)
    lpj_yr = lpj[['year', lpjml_col]].drop_duplicates('year').set_index('year')[lpjml_col]

    # Train OLS: obs_mean ~ lpjml on train years
    tv_obs = obs_mean[obs_mean['year'].isin(train_years)].set_index('year')['obs_mean']
    tv_lpj = lpj_yr.reindex(tv_obs.index).dropna()
    tv_obs = tv_obs.reindex(tv_lpj.index)
    tv_obs = tv_obs.dropna()
    tv_lpj = tv_lpj.reindex(tv_obs.index)

    if len(tv_obs) < 3:
        return None  # too few train points

    if tv_lpj.nunique() <= 1:
        # constant LPJmL → mean-shift correction
        slope, intercept = 1.0, float(tv_obs.mean() - tv_lpj.mean())
    else:
        try:
            slope, intercept, *_ = stats.linregress(tv_lpj.values, tv_obs.values)
        except ValueError:
            slope, intercept = 1.0, float(tv_obs.mean() - tv_lpj.mean())

    # Apply to test years
    test_obs_all  = obs_mean[obs_mean['year'].isin(test_years)].set_index('year')['obs_mean']
    test_lpj      = lpj_yr.reindex(test_years).dropna()
    common_test   = test_obs_all.index.intersection(test_lpj.index)
    if len(common_test) == 0:
        return None

    y_true = test_obs_all.reindex(common_test).values
    y_pred = np.clip(slope * test_lpj.reindex(common_test).values + intercept, 0, None)

    # Metrics
    m = compute_metrics(y_true, y_pred)
    if m is None:
        return None

    # Reference forecasts from TRAIN+VAL country means
    tv_obs_all = obs_mean[obs_mean['year'].isin(train_val)].set_index('year')['obs_mean']
    tv_yrs = sorted(tv_obs_all.index)
    tv_vals = tv_obs_all.values
    clim   = float(tv_obs_all.mean())
    roll3  = float(tv_obs_all[tv_obs_all.index >= (max(tv_yrs) - 2)].mean())
    ols_pred   = ols_predict(np.array(tv_yrs, float), tv_vals,
                              np.array(list(common_test), float))
    trend_pred = theil_sen_predict(np.array(tv_yrs, float), tv_vals,
                                   np.array(list(common_test), float))

    ref_clim  = np.full(len(common_test), clim)
    ref_roll3 = np.full(len(common_test), roll3)

    rmse_ref_clim,  mae_ref_clim  = compute_ref_metrics(y_true, ref_clim)
    rmse_ref_roll3, mae_ref_roll3 = compute_ref_metrics(y_true, ref_roll3)
    rmse_ref_ols,   mae_ref_ols   = compute_ref_metrics(y_true, ols_pred)
    rmse_ref_trend, mae_ref_trend = compute_ref_metrics(y_true, trend_pred)

    def skill(rmse_m, rmse_r):
        return 1 - rmse_m / rmse_r if (rmse_r and np.isfinite(rmse_r) and rmse_r > 0) else np.nan

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return dict(
        arm='B0', model_name=f'lpjml_{water_type}', model_type='lpjml',
        model_family='process_model', loss_type='deterministic',
        crop=crop, country=country, origin=1.0, seed=0,
        partition='in_domain', timestamp=ts,
        run_id=f'daily-{water_type}-{crop}-{country}-lpjml-ctry',
        n_samples=m['n_samples'], r2=m['r2'], rmse=m['rmse'], nrmse=m['nrmse'],
        mae=m['mae'], mse=m['mse'], mape=m['mape'], smape=m['smape'],
        rmse_ref_trend=rmse_ref_trend, rmse_ref_ols=rmse_ref_ols,
        rmse_ref_climatology=rmse_ref_clim, rmse_ref_rolling3=rmse_ref_roll3,
        mae_ref_trend=mae_ref_trend, mae_ref_ols=mae_ref_ols,
        mae_ref_climatology=mae_ref_clim, mae_ref_rolling3=mae_ref_roll3,
        skill_rmse_vs_trend=skill(m['rmse'], rmse_ref_trend),
        skill_rmse_vs_ols=skill(m['rmse'], rmse_ref_ols),
        skill_rmse_vs_climatology=skill(m['rmse'], rmse_ref_clim),
        skill_rmse_vs_rolling3=skill(m['rmse'], rmse_ref_roll3),
        skill_mae_vs_trend=skill(m['mae'], mae_ref_trend),
        skill_mae_vs_ols=skill(m['mae'], mae_ref_ols),
        skill_mae_vs_climatology=skill(m['mae'], mae_ref_clim),
        skill_mae_vs_rolling3=skill(m['mae'], mae_ref_roll3),
        within_5pct=m['within_5pct'], within_10pct=m['within_10pct'],
        within_20pct=m['within_20pct'],
        **{'within_0.5abs': m['within_0p5abs'],
           'within_0.1abs': m['within_0p1abs'],
           'within_1pct':   m['within_1pct']},
        n_valid_adm=1,  # country-level = 1 aggregate region
        water_type=water_type,
    )


def evaluate_country(crop, country, water_type, lpjml_col, lpjml_path,
                     years_dict, dry_run=False, spatial_scale='region'):
    """
    Full evaluation for one (crop, country, water_type) combination.
    Returns a results dict or None if data is insufficient.
    """
    # Years split
    all_years  = sorted(years_dict[crop][country])
    if len(all_years) < 8:
        print(f"  SKIP {crop} {country}: only {len(all_years)} years")
        return None
    test_years  = all_years[-5:]
    val_years   = all_years[-7:-5]
    train_years = all_years[:-7]
    train_val   = train_years + val_years

    # Load LPJmL
    lpj = pd.read_csv(lpjml_path, dtype={'date': str})
    lpj['year'] = lpj['date'].str[:4].astype(int)
    # Keep only benchmark years
    lpj = lpj[lpj['year'].isin(all_years)][['adm_id', 'year', lpjml_col]]

    if lpj.empty or lpj[lpjml_col].isna().all():
        print(f"  SKIP {crop} {country} {water_type}: empty LPJmL file")
        return None

    # Load observed yields
    obs_path = CYBENCH_DATA / crop / country / f'yield_{crop}_{country}.csv'
    if not obs_path.exists():
        print(f"  SKIP {crop} {country}: no observed yield CSV")
        return None
    obs = pd.read_csv(obs_path)[['adm_id', 'harvest_year', 'yield']]
    obs = obs[obs['harvest_year'].isin(all_years)]

    if obs.empty:
        print(f"  SKIP {crop} {country}: empty observed yield in benchmark years")
        return None

    # ── Country-aggregated path ────────────────────────────────────
    if spatial_scale == 'country':
        return _evaluate_country_national(
            crop, country, water_type, lpjml_col, lpj, obs,
            train_years, train_val, val_years, test_years)

    # Bias correction on train years
    lpj_test = lpj[lpj['year'].isin(test_years)].copy()
    bc_test  = apply_bias_correction(lpj, obs, train_years, lpjml_col)
    if bc_test.empty:
        print(f"  SKIP {crop} {country} {water_type}: no valid adm_ids after zero-yield filter")
        return None

    # Merge with observed test yields
    obs_test = obs[obs['harvest_year'].isin(test_years)][['adm_id','harvest_year','yield']]
    merged   = pd.merge(bc_test[['adm_id','year','pred_bc']],
                        obs_test,
                        left_on=['adm_id','year'],
                        right_on=['adm_id','harvest_year'], how='inner')
    if merged.empty:
        print(f"  SKIP {crop} {country} {water_type}: no matched test samples")
        return None

    # Metrics: model
    m = compute_metrics(merged['yield'].values, merged['pred_bc'].values)
    if m is None:
        return None

    # Reference forecasts (from TRAIN+VAL)
    refs = build_references(obs, train_val, test_years)
    merged = merged.set_index(['adm_id', 'harvest_year'])
    merged = merged.join(refs, how='left')
    merged = merged.reset_index()

    yt = merged['yield'].values
    yp = merged['pred_bc'].values

    rmse_ref_clim,  mae_ref_clim  = compute_ref_metrics(yt, merged['ref_clim'].values)
    rmse_ref_roll3, mae_ref_roll3 = compute_ref_metrics(yt, merged['ref_roll3'].values)
    rmse_ref_ols,   mae_ref_ols   = compute_ref_metrics(yt, merged['ref_ols'].values)
    rmse_ref_trend, mae_ref_trend = compute_ref_metrics(yt, merged['ref_trend'].values)

    def skill(rmse_m, rmse_r):
        return 1 - rmse_m / rmse_r if (rmse_r and np.isfinite(rmse_r) and rmse_r > 0) else np.nan

    n_valid_adm = bc_test['adm_id'].nunique()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f'daily-{water_type}-{crop}-{country}-lpjml'

    return dict(
        # Schema fields
        arm='B0',
        model_name=f'lpjml_{water_type}',
        model_type='lpjml',
        model_family='process_model',
        loss_type='deterministic',
        crop=crop,
        country=country,
        origin=1.0,
        seed=0,
        partition='in_domain',
        timestamp=ts,
        run_id=run_id,
        # Core metrics
        n_samples=m['n_samples'],
        r2=m['r2'],
        rmse=m['rmse'],
        nrmse=m['nrmse'],
        mae=m['mae'],
        mse=m['mse'],
        mape=m['mape'],
        smape=m['smape'],
        # Reference RMSEs
        rmse_ref_trend=rmse_ref_trend,
        rmse_ref_ols=rmse_ref_ols,
        rmse_ref_climatology=rmse_ref_clim,
        rmse_ref_rolling3=rmse_ref_roll3,
        mae_ref_trend=mae_ref_trend,
        mae_ref_ols=mae_ref_ols,
        mae_ref_climatology=mae_ref_clim,
        mae_ref_rolling3=mae_ref_roll3,
        # Skills
        skill_rmse_vs_trend=skill(m['rmse'], rmse_ref_trend),
        skill_rmse_vs_ols=skill(m['rmse'], rmse_ref_ols),
        skill_rmse_vs_climatology=skill(m['rmse'], rmse_ref_clim),
        skill_rmse_vs_rolling3=skill(m['rmse'], rmse_ref_roll3),
        skill_mae_vs_trend=skill(m['mae'], mae_ref_trend),
        skill_mae_vs_ols=skill(m['mae'], mae_ref_ols),
        skill_mae_vs_climatology=skill(m['mae'], mae_ref_clim),
        skill_mae_vs_rolling3=skill(m['mae'], mae_ref_roll3),
        # Tolerance accuracy
        within_5pct=m['within_5pct'],
        within_10pct=m['within_10pct'],
        within_20pct=m['within_20pct'],
        **{'within_0.5abs': m['within_0p5abs'],
           'within_0.1abs': m['within_0p1abs'],
           'within_1pct':   m['within_1pct']},
        # Extra context
        n_valid_adm=n_valid_adm,
        water_type=water_type,
    )


def find_lpjml_file(crop, country, water_type):
    cfg    = WATER_TYPES[crop][water_type]
    folder = LPJML_BASE / crop / country / cfg['folder']
    if not folder.exists():
        return None, None, 'region'
    files = list(folder.glob('*.csv'))
    return (files[0] if files else None), cfg['col'], 'region'


def find_lpjml_file_country(crop, country, water_type):
    """Fallback: country_aggregated (one value per year per country)."""
    cfg    = WATER_TYPES[crop][water_type]
    folder = LPJML_CTRY / crop / country / cfg['folder']
    if not folder.exists():
        return None, None, 'country'
    files = list(folder.glob('*.csv'))
    return (files[0] if files else None), cfg['col'], 'country'


def append_to_csv(result_rows, csv_path, is_resolution=False):
    """Append rows to an existing CSV, keeping its column order."""
    df_exist = pd.read_csv(csv_path)
    new_rows  = pd.DataFrame(result_rows)

    if is_resolution:
        new_rows['resolution'] = 'daily'

    # Align columns: add missing cols as NaN, drop extras not in existing
    for col in df_exist.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    new_rows = new_rows[df_exist.columns]

    combined = pd.concat([df_exist, new_rows], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"  → Appended {len(new_rows)} rows to {csv_path.name} (total {len(combined)})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(dry_run=False):
    years_dict = json.load(open(YEARS_DICT))
    all_results = []

    for crop in ('maize', 'wheat'):
        bench_countries = set(years_dict.get(crop, {}).keys())
        lpjml_region    = set(os.listdir(LPJML_BASE / crop))
        lpjml_country   = set(os.listdir(LPJML_CTRY / crop)) if (LPJML_CTRY / crop).exists() else set()
        countries = sorted(bench_countries)   # all benchmark countries
        region_only   = bench_countries & lpjml_region
        fallback_only = (bench_countries - lpjml_region) & lpjml_country
        no_coverage   = bench_countries - lpjml_region - lpjml_country
        print(f"\n{'='*60}")
        print(f"  {crop.upper()} — {len(countries)} benchmark countries")
        print(f"  Region-aggregated (sub-national): {len(region_only)} countries")
        print(f"  Country-aggregated fallback:      {len(fallback_only)} — {sorted(fallback_only)}")
        print(f"  No LPJmL coverage at all:         {sorted(no_coverage)}")
        print(f"{'='*60}")

        for country in countries:
            for water_type in ('rainfed', 'irrigated'):
                lpjml_file, lpjml_col, spatial_scale = find_lpjml_file(crop, country, water_type)
                # Fallback to country_aggregated if region data not available
                if lpjml_file is None:
                    lpjml_file, lpjml_col, spatial_scale = find_lpjml_file_country(crop, country, water_type)
                if lpjml_file is None:
                    print(f"  SKIP {crop} {country} {water_type}: file not found in region or country aggregated")
                    continue

                print(f"  Evaluating {crop} {country} {water_type} ...", end=' ', flush=True)
                try:
                    result = evaluate_country(
                        crop, country, water_type, lpjml_col,
                        lpjml_file, years_dict, dry_run=dry_run,
                        spatial_scale=spatial_scale)
                    if result:
                        result['spatial_scale'] = spatial_scale
                        all_results.append(result)
                        print(f"NRMSE={result['nrmse']:.3f}  "
                              f"skill_roll3={result['skill_rmse_vs_rolling3']:.3f}  "
                              f"n_adm={result['n_valid_adm']}  [{spatial_scale}]")
                    else:
                        print("skipped")
                except Exception as ex:
                    print(f"ERROR: {ex}")
                    import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total results: {len(all_results)}")
    if not all_results:
        print("No results generated.")
        return

    if dry_run:
        print("Dry run — not writing files.")
        for r in all_results:
            print(f"  {r['crop']} {r['country']} {r['model_name']}: "
                  f"NRMSE={r['nrmse']:.3f} skill_roll3={r['skill_rmse_vs_rolling3']:.3f}")
        return

    # Write standalone CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote standalone: {OUT_CSV}")

    # Append to architecture CSV (no resolution column)
    if ARCH_CSV.exists():
        append_to_csv(all_results, ARCH_CSV, is_resolution=False)
    else:
        print(f"  WARNING: {ARCH_CSV} not found — skipping architecture append")

    # Append to resolution CSV (adds resolution='daily')
    if RES_CSV.exists():
        append_to_csv(all_results, RES_CSV, is_resolution=True)
    else:
        print(f"  WARNING: {RES_CSV} not found — skipping resolution append")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute metrics but do not write to any CSV')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
