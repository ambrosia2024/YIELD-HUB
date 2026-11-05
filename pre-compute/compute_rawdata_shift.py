#!/usr/bin/env python3
"""
Compute per-variable distribution shift between train and test years
for every (crop, country) in the benchmark.

WHAT THIS MEASURES
    For each time-series variable (meteo, soil moisture, NDVI, FPAR, yield):
      1. For each (adm_id, year), summarise the daily values within the
         growing season by computing QUANTILES [2.25, 25, 50, 75, 97.25].
         This preserves the shape of the within-year daily distribution
         (lower tail, quartiles, upper tail) rather than collapsing it to
         a single seasonal mean, which would miss changes in variance and
         extreme events.
      2. Split into train pool vs test pool using the same fixed-split rule
         as the model training:
             test  = last `test_years` years        (default 5)
             val   = 2 years before test            (not used; excluded from both pools)
             train = all remaining earlier years
      3. For each quantile separately, compare the distribution of that
         quantile's values across all training (adm_id, year) pairs vs all
         test (adm_id, year) pairs using:
             KS statistic   : maximum absolute CDF difference (shape-sensitive;
                              catches any distributional change regardless of direction)
             Wasserstein-1  : Earth Mover's Distance in the original variable units
         This gives 5 rows per (crop, country, variable) — one per quantile.
         Both metrics are O(n log n) via scipy and negligible cost.

WHY QUANTILES INSTEAD OF SEASONAL MEAN
    The seasonal mean discards within-year distributional information.  A year
    with five extreme heat-wave days followed by average temperatures and a year
    with uniformly elevated temperatures can have identical seasonal means but
    very different daily distributions — and the model, which processes daily
    sequences, would experience them differently.

    Using quantiles per region-year preserves the i.i.d. assumption of the KS
    test (each region-year is one draw from the climate distribution) while
    exposing tail shifts:
      q2.25  : lower-tail shift (cold extremes, dry spells at bottom of precip)
      q25    : below-average conditions
      q50    : median — robust analogue of the mean
      q75    : above-average conditions
      q97.25 : upper-tail shift (heat waves, intense rainfall events)

    Yield is already one value per (adm_id, year) so it is treated as a single
    'annual' quantity with no within-year distribution to summarise.

WHY BOTH KS AND WASSERSTEIN
    KS detects *whether* the distribution changed (shape, spread, tail).
    Wasserstein measures *how much* the mass moved (interpretable in the
    original variable units).  They are complementary: a shift in the tail
    with the same median shows high KS but moderate Wasserstein; a uniform
    mean shift shows high Wasserstein with moderate KS.

SKIPPED FEATURES
    - soil     : static per adm_id (no year dimension; same regions in train + test)
    - crop_cal : static sos/eos per adm_id (no year dimension)
    - location : lat/lon — geographic identity, not a distribution to compare
    All would yield 0 shift and add noise.

USAGE
    python compute_rawdata_shift.py --crop maize
    python compute_rawdata_shift.py --crop wheat --countries NL DE FR
    python compute_rawdata_shift.py --crop maize --test_years 5 --n_jobs 8

OUTPUT
    cybench/data/variable_shift/variable_shift_{crop}.csv
    Columns: crop, country, source, variable, quantile,
             ks_stat, ks_pval, wasserstein,
             mean_train, mean_test, std_train, std_test,
             n_train, n_test, computed_at
    One row per (crop, country, variable, quantile).
    n_train / n_test count (adm_id × year) pairs in each pool (after val
    exclusion) — the same value for every quantile of the same variable.
    Yield rows carry quantile='annual' (no daily distribution to summarise).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

# ─────────────────────────────────────────────── paths ───────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent   # AgML-CY-Bench/
DATA_DIR      = REPO_ROOT / "cybench" / "data"
YEARS_JSON    = REPO_ROOT / "YIELD-HUB" / "setups" / "configurations" / "years_dict.json"
OUTPUT_DIR    = REPO_ROOT / "cybench" / "data" / "variable_shift"

MIN_YEARS_THRESHOLD = 8   # countries with fewer total years are skipped

# Quantiles used to summarise the within-year daily distribution per (adm_id, year).
# Lower tail (q2.25) captures cold/dry extremes; upper tail (q97.25) captures heat
# waves and intense rainfall events.  The median (q50) is a robust centre estimate.
QUANTILES = [2.25, 25, 50, 75, 97.25]
QUANTILE_LABELS = [f"q{q:g}" for q in QUANTILES]   # ['q2.25','q25','q50','q75','q97.25']

# ─────────────────────────────────────── daily-file registry ─────────────────
# Each entry: (source_label, filename_template, list_of_value_columns)
# filename_template uses {crop} and {country} placeholders.
DAILY_FILES = [
    ("meteo",         "meteo_{crop}_{country}.csv",         ["tmin", "tmax", "tavg", "prec", "rad", "et0", "vpd", "cwb"]),
    ("soil_moisture", "soil_moisture_{crop}_{country}.csv", ["ssm", "rsm"]),
    ("ndvi",          "ndvi_{crop}_{country}.csv",          ["ndvi"]),
    ("fpar",          "fpar_{crop}_{country}.csv",          ["fpar"]),
]

# ─────────────────────────────────────────────── helpers ─────────────────────
def _get_countries(crop: str, requested: list) -> list:
    with open(YEARS_JSON) as f:
        years_dict = json.load(f)
    crop_years = years_dict.get(crop, {})
    if requested:
        return [c for c in requested if c in crop_years]
    return sorted(c for c, yrs in crop_years.items()
                  if len(yrs) >= MIN_YEARS_THRESHOLD)


def _split_years(all_years: list, test_years: int, val_years: int = 2):
    """
    Sort years and return (train_set, val_set, test_set).
    Last test_years → test, 2 before that → val, rest → train.
    """
    yrs = sorted(all_years)
    n = len(yrs)
    test_set  = set(yrs[n - test_years:])
    val_set   = set(yrs[n - test_years - val_years: n - test_years])
    train_set = set(yrs[:n - test_years - val_years])
    return train_set, val_set, test_set


def _year_from_date(date_series: pd.Series) -> pd.Series:
    """
    Extract year as int from a date column (string 'YYYY-MM-DD' or datetime).
    Uses fast string slicing when possible — avoids full date parsing overhead.
    """
    s = date_series
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year
    # assume ISO string 'YYYY-...'
    return s.str[:4].astype(int)


def _shift_metrics(train_vals: np.ndarray, test_vals: np.ndarray) -> dict:
    """
    Compute KS and Wasserstein between two 1-D arrays.
    Returns NaNs for empty / all-NaN inputs.
    """
    train_vals = train_vals[np.isfinite(train_vals)]
    test_vals  = test_vals[np.isfinite(test_vals)]
    if len(train_vals) < 2 or len(test_vals) < 2:
        return dict(ks_stat=np.nan, ks_pval=np.nan, wasserstein=np.nan,
                    mean_train=np.nan, mean_test=np.nan,
                    std_train=np.nan,  std_test=np.nan,
                    n_train=len(train_vals), n_test=len(test_vals))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ks_stat, ks_pval = ks_2samp(train_vals, test_vals)
        wass = wasserstein_distance(train_vals, test_vals)
    return dict(
        ks_stat=round(float(ks_stat), 6),
        ks_pval=round(float(ks_pval), 6),
        wasserstein=round(float(wass), 6),
        mean_train=round(float(np.mean(train_vals)), 6),
        mean_test=round(float(np.mean(test_vals)),  6),
        std_train=round(float(np.std(train_vals)),  6),
        std_test=round(float(np.std(test_vals)),   6),
        n_train=len(train_vals),
        n_test=len(test_vals),
    )


# ─────────────────────────────────── per-country worker ──────────────────────
def _process_country(crop: str, country: str, test_years: int,
                     val_years: int = 2, verbose: bool = True) -> list:
    """
    Load raw CSVs for one (crop, country), split years, compute per-variable
    shift metrics.  Returns list of row dicts (one per variable).
    """
    country_dir = DATA_DIR / crop / country

    # ── 1. get years from yield CSV ──────────────────────────────────────────
    yield_path = country_dir / f"yield_{crop}_{country}.csv"
    if not yield_path.exists():
        if verbose:
            print(f"  [skip] {crop}-{country}: no yield CSV")
        return []

    ydf = pd.read_csv(yield_path, usecols=["adm_id", "harvest_year", "yield"])
    all_years = sorted(ydf["harvest_year"].unique())

    # Need at least test_years + val_years + 1 train year
    if len(all_years) < test_years + val_years + 1:
        if verbose:
            print(f"  [skip] {crop}-{country}: only {len(all_years)} years "
                  f"(need {test_years + val_years + 1})")
        return []

    train_set, val_set, test_set = _split_years(all_years, test_years, val_years)

    rows = []
    ts = datetime.utcnow().isoformat() + "Z"

    # ── 2. yield shift ───────────────────────────────────────────────────────
    # yield is already one value per (adm_id, year) — no daily distribution
    # to summarise with quantiles.  Label it quantile='annual'.
    ydf = ydf.rename(columns={"harvest_year": "year"})
    tr = ydf[ydf["year"].isin(train_set)]["yield"].values.astype(float)
    te = ydf[ydf["year"].isin(test_set)]["yield"].values.astype(float)
    m = _shift_metrics(tr, te)
    rows.append({"crop": crop, "country": country,
                 "source": "yield", "variable": "yield", "quantile": "annual",
                 **m, "computed_at": ts})

    # ── 3. daily time-series files ───────────────────────────────────────────
    for source, fname_tmpl, val_cols in DAILY_FILES:
        fpath = country_dir / fname_tmpl.format(crop=crop, country=country)
        if not fpath.exists():
            continue

        # Read only adm_id, date, and the value columns we need
        try:
            df = pd.read_csv(fpath, usecols=["adm_id", "date"] + val_cols,
                             dtype={"date": str}, low_memory=False)
        except (ValueError, KeyError):
            # Some columns might be missing (e.g. partial meteo files)
            try:
                df = pd.read_csv(fpath, dtype={"date": str}, low_memory=False)
                val_cols = [c for c in val_cols if c in df.columns]
                if not val_cols:
                    continue
            except Exception:
                continue

        df["year"] = _year_from_date(df["date"])

        # Filter to train + test only (drop val years to keep aggregation clean)
        keep = train_set | test_set
        df = df[df["year"].isin(keep)]
        if df.empty:
            continue

        # ── Aggregate daily → per-quantile values per (adm_id, year) ────────
        # For each region-year we compute QUANTILES of the within-season daily
        # values.  This preserves distributional shape (tails, spread) rather
        # than collapsing everything to a single seasonal mean.
        #
        # np.nanpercentile is applied column-wise inside a groupby using a
        # named-aggregation dict so each quantile gets its own column.
        # Naming: col_q2.25, col_q25, col_q50, col_q75, col_q97.25
        #
        # We then loop over quantiles and run KS + Wasserstein on the
        # distribution of that quantile's value across all (adm_id, year) pairs
        # in the train pool vs the test pool.

        # Build column names for each (variable, quantile) pair
        quant_cols = {}   # {var: {label: agg_col_name}}
        agg_dict   = {}   # passed to .agg()
        for var in val_cols:
            quant_cols[var] = {}
            for q, lbl in zip(QUANTILES, QUANTILE_LABELS):
                col = f"{var}__{lbl}"
                quant_cols[var][lbl] = col
                # capture q in the lambda via default-arg trick
                agg_dict[col] = pd.NamedAgg(
                    column=var,
                    aggfunc=lambda s, _q=q: np.nanpercentile(s.values, _q)
                            if s.notna().any() else np.nan
                )

        agg = (df.groupby(["adm_id", "year"], sort=False)
                 .agg(**agg_dict)
                 .reset_index())

        for var in val_cols:
            # n_train / n_test are the same for every quantile of this variable
            n_tr = int((agg["year"].isin(train_set)).sum())
            n_te = int((agg["year"].isin(test_set)).sum())

            for lbl in QUANTILE_LABELS:
                col = quant_cols[var][lbl]
                tr_vals = agg.loc[agg["year"].isin(train_set), col].values.astype(float)
                te_vals = agg.loc[agg["year"].isin(test_set), col].values.astype(float)
                m = _shift_metrics(tr_vals, te_vals)
                rows.append({"crop": crop, "country": country,
                             "source": source, "variable": var, "quantile": lbl,
                             **m, "computed_at": ts})

    if verbose:
        n_vars = len(rows)
        print(f"  {crop}-{country}: {len(train_set)} train yrs, "
              f"{len(val_set)} val yrs, {len(test_set)} test yrs → "
              f"{n_vars} variables")

    return rows


# ──────────────────────────────────────────────── main ───────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-variable train/test distribution shift for each country",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--crop", required=True, choices=["maize", "wheat"])
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Country codes (default: all with >=%d years)" % MIN_YEARS_THRESHOLD)
    parser.add_argument("--test_years", type=int, default=5,
                        help="Number of test years from the end (default: 5)")
    parser.add_argument("--val_years", type=int, default=2,
                        help="Val years before test, excluded from both pools (default: 2)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel workers (default: 1; use -1 for all cores)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    countries = _get_countries(args.crop, args.countries)
    if not countries:
        print(f"[ERROR] No countries found for crop='{args.crop}'"); sys.exit(1)

    print(f"\n[Variable Shift]  crop={args.crop}  n_countries={len(countries)}"
          f"  test_years={args.test_years}  val_years={args.val_years}")
    print(f"Countries: {countries}\n")

    # ── parallel or serial ───────────────────────────────────────────────────
    def _worker(country):
        return _process_country(args.crop, country, args.test_years,
                                args.val_years, verbose=verbose)

    if args.n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            all_rows = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                delayed(_worker)(c) for c in countries
            )
            all_rows = [r for batch in all_rows for r in batch]
        except ImportError:
            print("[warn] joblib not found, falling back to serial")
            all_rows = [r for c in countries for r in _worker(c)]
    else:
        all_rows = [r for c in countries for r in _worker(c)]

    if not all_rows:
        print("[ERROR] No results produced."); sys.exit(1)

    # ── write output ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"variable_shift_{args.crop}.csv"

    COL_ORDER = [
        "crop", "country", "source", "variable", "quantile",
        "ks_stat", "ks_pval", "wasserstein",
        "mean_train", "mean_test", "std_train", "std_test",
        "n_train", "n_test", "computed_at",
    ]
    df_out = pd.DataFrame(all_rows)[COL_ORDER]
    df_out.to_csv(out_path, index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    n_countries = df_out["country"].nunique()
    n_vars      = df_out["variable"].nunique()
    n_rows      = len(df_out)
    valid       = df_out.dropna(subset=["ks_stat"])

    print(f"\n{'='*70}")
    print(f"VARIABLE SHIFT SUMMARY  ({args.crop})")
    print(f"{'='*70}")
    print(f"Countries : {n_countries}   Variables : {n_vars}   "
          f"Quantiles : {len(QUANTILE_LABELS)}   Rows : {n_rows}")

    if len(valid):
        # Summarise by (variable, quantile) — most informative cut
        top = (valid.groupby(["variable", "quantile"])[["ks_stat", "wasserstein"]]
                    .median()
                    .sort_values("ks_stat", ascending=False))
        print("\nMedian KS stat and Wasserstein by (variable, quantile) — sorted by KS:")
        print(top.round(4).to_string())

        # Also print the upper-tail (q97.25) separately — clearest signal for extremes
        tail = (valid[valid["quantile"] == "q97.25"]
                .groupby("variable")[["ks_stat", "wasserstein"]]
                .median()
                .sort_values("ks_stat", ascending=False))
        print(f"\nUpper-tail (q97.25) only — extremes / heat-wave / intense-rainfall signal:")
        print(tail.round(4).to_string())

    print(f"\nWrote {n_rows} rows → {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
