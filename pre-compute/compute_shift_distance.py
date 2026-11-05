#!/usr/bin/env python3
"""
Compute domain-classifier shift distance per (crop, country) for YIELD-HUB.

DEFINITION  (evaluation_protocol.md §Tier 2 / §4)
    shift_distance(country_Y) = AUC of a binary classifier trained to
    distinguish "this region's feature vector comes from country_Y" (label 1)
    vs "it comes from any other country in the pool" (label 0).

    AUC ~ 0.50  ->  country_Y looks like the rest of the pool.  Easy transfer.
    AUC ~ 1.00  ->  country_Y is highly OOD.  Hard transfer; expect large Delta-skill.

    This is a DATASET property, not a model property.  Pre-compute once per crop
    and attach to the cross-country transfer results as a context column.  This
    script is self-contained: running it to completion produces the full shift
    characterisation with no dependency on the model-evaluation pipeline.

    Full rationale for every design choice below is logged in
    YIELD-HUB/documentation/design-decisions/distribution-shift.md.

WHY A PERMUTATION NULL AND NOT AUC ALONE
    A raw cross-validated AUC is a point estimate with no reference to chance.
    With many features and few regions in the target country (p close to or
    above n), even random country labels yield AUC > 0.5, and the magnitude is
    not comparable across countries with different region counts.  We therefore
    wrap every country's AUC in a label-permutation test:

      - perm_p        = P(AUC_null >= AUC_obs) over n_perm label shuffles.
                        Answers "is this shift beyond chance?" (significance).
      - auc_debiased  = AUC_obs - mean(AUC_null).  Subtracting each country's
                        own null mean removes the finite-sample inflation, so
                        the magnitude is comparable across countries (effect size).
      - shift_class   = none / near / far, derived from (perm_p, auc_debiased):
                          none : perm_p >= alpha                 (indistinguishable)
                          near : perm_p < alpha, auc_debiased <  near_far_threshold
                          far  : perm_p < alpha, auc_debiased >= near_far_threshold

KUIPER BREADTH (complements the classifier)
    The classifier AUC measures OVERALL separability.  To measure the BREADTH of
    a shift (how many features individually differ), we run a two-sample Kuiper
    test per feature (country_Y vs pool), Benjamini-Hochberg-correct across
    features, and report:
      - n_features_shifted    : # features with BH q < breadth_alpha
      - frac_features_shifted : that count / n_features_tested
    Kuiper (V = D+ + D-) is used over KS because it is sensitive to differences
    in BOTH tails, matching the 2.5/97.5 quantile emphasis.  (astropy's
    kuiper_two_sample is used if installed; otherwise a self-contained
    implementation with the standard asymptotic p-value is used.)

WHY LOGISTIC REGRESSION (NOT A GRADIENT-BOOSTED TREE)
    The temporal features below are already a QUANTILE expansion, so the
    distributional information a nonlinear model would otherwise have to
    discover (e.g. a hotter tail) is present as explicit linear dimensions
    (the q97.5 feature).  A regularized LINEAR classifier therefore already
    sees tail/shape shifts, while giving a lower-variance, more comparable
    "distance" and a permutation null cheap enough to run on the CPU-only
    login node.  A tree (XGBoost / RandomForest) overfits the tiny positive
    class, inflates AUC toward 1.0 for almost everyone, and compresses the
    dynamic range.  See the design-decisions file for the reversal rationale.

FEATURES USED
    Static (one value per region, NO quantiles):
        soil     : awc, bulk_density                 (dfs_x["soil"],  1-level index)
    Per-year (2-level [adm_id, year], single mean, no daily axis to quantile):
        crop_season calendar columns.
    Daily temporal (3-level [adm_id, year, date]) -> QUANTILES:
        meteo : tmin, tmax, tavg, prec, cwb, rad
        fpar, ndvi
        For each (region, year) the growing-season daily values are summarized
        by QUANTILES = [2.5, 25, 50, 75, 97.5]; the per-year quantile vectors
        are then averaged across years to one row per region.  The 2.5/97.5
        tails carry the extreme-case information that a plain mean discards.

    latitude/longitude (dfs_x["location"]) are ALWAYS EXCLUDED: a country is a
    contiguous geographic box, so lat/lon make every country trivially separable
    and would make shift distance measure geography rather than the climate /
    feature distribution we care about.

CLASSIFIER
    Pipeline: StandardScaler -> LogisticRegression(class_weight='balanced', C=1.0)
    AUC via stratified k-fold CV (k=5, auto-reduced to n_target if < 5).
    NaNs imputed with per-column median before scaling.

USAGE
    python compute_shift_distance.py --crop maize
    python compute_shift_distance.py --crop wheat --countries NL DE FR
    python compute_shift_distance.py --crop maize --n_perm 1000 \
        --extra_output_dir /path/to/paper_tables

OUTPUT  (cybench/data/shift_distance/shift_distance_{crop}.csv; also copied to
         --extra_output_dir if given)
    crop, country, shift_distance_auc, auc_std, auc_null_mean, auc_debiased,
    perm_p, shift_class, n_features_shifted, frac_features_shifted,
    n_target, n_pool, n_features, n_perm, note, computed_at

NEXT STEPS
    Join shift_distance_{crop}.csv with ranking.py output on (crop, country) to
    plot the Delta-skill-vs-shift-distance curve described in evaluation_protocol.md.
    far-shift countries that still show good Delta-skill support H1
    (generalisation); far-shift with poor Delta-skill are the hard cases that
    motivate B4-B6 fusion arms.  near-shift countries are the borderline cases.

REQUIREMENTS
    scikit-learn, scipy  (pip install scikit-learn scipy)
    astropy is OPTIONAL (used for the Kuiper p-value if present).
    CyBench dataset (cybench/data/{crop}/{country}/ must exist)
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

# ──────────────────────────────────────────────────── CY-BENCH SETUP ─────────
# pre-compute/ → YIELD-HUB/ → AgML-CY-Bench/  (CyBench root)
CYBENCH_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CYBENCH_ROOT))

import cybench.config
cybench.config.FORECAST_LEAD_TIME = "0-days"
from cybench.config import set_forecast_type, KEY_LOC, KEY_YEAR, KEY_TARGET
set_forecast_type("0-days")

# Alignment patch must be applied before importing datasets (mirrors tstBaselines.py)
try:
    from cybench.process.alignment_patch import patch_alignment
    patch_alignment()
except (ImportError, AttributeError):
    pass   # not available in all environments; "0-days" is safe without it

from cybench.datasets.configured import load_dfs_crop

# ──────────────────────────────────────────────────── optional-dep guards ─────
try:
    import sklearn as _sklearn_check  # noqa: F401
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import scipy as _scipy_check  # noqa: F401
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ──────────────────────────────────────────────────── PATHS ──────────────────
DATA_DIR   = CYBENCH_ROOT / "cybench" / "data"
OUTPUT_DIR = CYBENCH_ROOT / "cybench" / "data" / "shift_distance"

MIN_YEARS_THRESHOLD = 8

# Quantiles used to summarize each daily temporal variable per (region, year).
# Tails (2.5/97.5) carry the extreme-case information a plain mean discards;
# the mid-quantiles are the quartiles (collinear neighbours were dropped on
# purpose — see the design-decisions file).
QUANTILES = [2.5, 25.0, 50.0, 75.0, 97.5]

# dfs_x keys dropped unconditionally: lat/lon make countries trivially separable
# and would make shift distance measure geography, not feature distribution.
EXCLUDE_KEYS = {"location"}

# shift_class thresholds (all overridable on the CLI)
DEFAULT_ALPHA              = 0.05   # perm_p below this => shift is "real"
DEFAULT_NEAR_FAR_THRESHOLD = 0.25   # auc_debiased at/above this => "far"
DEFAULT_BREADTH_ALPHA      = 0.05   # BH q below this => feature counts as shifted


# ──────────────────────────────────────────────────── COUNTRY DISCOVERY ──────
def _get_countries(crop: str, requested: list) -> list:
    """
    Return the list of countries to process.
    If `requested` is given, use it directly.
    Otherwise, load years_dict.json and return all countries that have
    >= MIN_YEARS_THRESHOLD years of data for this crop.
    """
    if requested:
        return requested
    years_path = CYBENCH_ROOT / "YIELD-HUB" / "setups" / "configurations" / "years_dict.json"
    with open(years_path) as f:
        years_dict = json.load(f)
    crop_years = years_dict.get(crop, {})
    return sorted(c for c, yrs in crop_years.items()
                  if len(yrs) >= MIN_YEARS_THRESHOLD)


# ──────────────────────────────────────────────────── FEATURE EXTRACTION ─────
def _numeric_cols(feat_df: pd.DataFrame) -> list:
    """Numeric columns, excluding any time-index columns that leaked in."""
    EXCLUDE_NAMES = {KEY_YEAR, "year", "date", "dates"}
    return [
        c for c in feat_df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_NAMES
    ]


def _features_from_df(feat_name: str, feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-region (adm_id) features from one dfs_x DataFrame, branching on
    the index layout:

      - Static        (1-level index: soil)
            -> single value per region, NO quantiles.
      - Per-year      (2-level MultiIndex [adm_id, year]: crop_season)
            -> single mean per region over years, NO quantiles
               (no daily axis to take quantiles over).
      - Daily temporal (3-level MultiIndex [adm_id, year, date]: meteo/fpar/ndvi)
            -> QUANTILES over the dates within each (region, year), then mean
               across years -> one row per region.

    Returns DataFrame indexed by adm_id, columns prefixed by feat_name.
    Empty DataFrame if there are no usable numeric columns.
    """
    num_cols = _numeric_cols(feat_df)
    if not num_cols:
        return pd.DataFrame()

    nlevels = feat_df.index.nlevels if isinstance(feat_df.index, pd.MultiIndex) else 1

    if nlevels >= 3:
        # Daily temporal: quantiles over dates per (adm_id, year), then mean/yr.
        grp = feat_df[num_cols].groupby(level=[0, 1])   # (adm_id, year)
        q_parts = []
        for p in QUANTILES:
            qdf = grp.quantile(p / 100.0)               # index (adm_id, year)
            qdf.columns = [f"{feat_name}__{c}__p{p:g}" for c in num_cols]
            q_parts.append(qdf)
        per_year = pd.concat(q_parts, axis=1)
        by_region = per_year.groupby(level=0).mean()    # average across years
    elif nlevels == 2:
        # Per-year static-like (crop_season): mean across years, no quantiles.
        by_region = feat_df[num_cols].groupby(level=0).mean()
        by_region.columns = [f"{feat_name}__{c}" for c in num_cols]
    else:
        # Static single-index (soil): already one row per region.
        by_region = feat_df[num_cols].copy()
        by_region.columns = [f"{feat_name}__{c}" for c in num_cols]

    by_region.index.name = KEY_LOC
    return by_region


def load_region_features(crop: str, country: str,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Load all features for one (crop, country) and collapse to one row per region.
    Returns DataFrame indexed by adm_id with a 'country' column, or empty DF.
    """
    try:
        df_y, dfs_x = load_dfs_crop(crop, [country])
    except Exception as e:
        if verbose:
            print(f"[skip] {crop}-{country}: load error — {e}")
        return pd.DataFrame()

    if df_y is None or df_y.empty:
        if verbose:
            print(f"[skip] {crop}-{country}: empty yield data")
        return pd.DataFrame()

    # Regions present in df_y (MultiIndex [adm_id, year])
    regions = df_y.index.get_level_values(KEY_LOC).unique()

    parts = []
    for feat_name, feat_df in dfs_x.items():
        if feat_df is None or feat_df.empty:
            continue
        if str(feat_name) in EXCLUDE_KEYS:
            continue   # lat/lon dropped unconditionally (see EXCLUDE_KEYS)
        part = _features_from_df(str(feat_name), feat_df)
        if not part.empty:
            parts.append(part)

    if not parts:
        if verbose:
            print(f"[skip] {crop}-{country}: no numeric features extracted")
        return pd.DataFrame()

    combined = parts[0]
    for part in parts[1:]:
        combined = combined.join(part, how="outer")

    # Keep only regions present in df_y
    combined = combined.loc[combined.index.isin(regions)]
    combined["country"] = country
    return combined


def build_feature_matrix(crop: str, countries: list,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Concatenate per-region feature vectors across all countries.
    Returns a DataFrame with one row per region and a 'country' column.
    """
    frames = []
    for country in countries:
        if verbose:
            print(f"  Loading {crop}-{country} ...", end=" ", flush=True)
        frame = load_region_features(crop, country, verbose=False)
        if frame.empty:
            if verbose:
                print("skipped")
            continue
        frames.append(frame)
        if verbose:
            print(f"{len(frame)} regions, {frame.shape[1] - 1} feature cols")

    if not frames:
        raise ValueError(
            f"No data loaded for crop='{crop}'. "
            f"Check that cybench/data/{crop}/ exists and is populated."
        )

    mat = pd.concat(frames)
    if verbose:
        n_feats = mat.shape[1] - 1   # exclude 'country'
        print(f"\n[info] Feature matrix: {len(mat)} total regions x {n_feats} feature cols")
    return mat


# ──────────────────────────────────────────────────── IMPUTATION ─────────────
def _prepare_X(matrix: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Extract X array, drop all-NaN columns, impute remaining NaNs with col-median.
    Computed ONCE for the whole matrix — X is shared across countries, only the
    label vector y changes.  Returns (X_array, n_valid_features).
    """
    X = matrix[feature_cols].values.astype(float)

    # Drop columns that are entirely NaN
    valid_mask = ~np.all(np.isnan(X), axis=0)
    X = X[:, valid_mask]
    n_feat = int(valid_mask.sum())

    # Impute remaining NaN with per-column median (robust to the heavy tails
    # that the 2.5/97.5 quantile features deliberately capture).
    col_med = np.nanmedian(X, axis=0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_med, nan_idx[1])

    return X, n_feat


# ──────────────────────────────────────────────────── KUIPER BREADTH ─────────
def _kuiper_pvalue(V: float, ne: float) -> float:
    """
    Asymptotic two-sample Kuiper p-value (Press et al., Numerical Recipes).
    ne = effective sample size = n1*n2/(n1+n2).
    """
    if not np.isfinite(V) or ne <= 0:
        return np.nan
    en  = np.sqrt(ne)
    lam = (en + 0.155 + 0.24 / en) * V
    if lam < 0.4:
        return 1.0
    j = np.arange(1, 101)
    s = np.sum((4.0 * j**2 * lam**2 - 1.0) * np.exp(-2.0 * j**2 * lam**2))
    return float(min(max(2.0 * s, 0.0), 1.0))


def _kuiper_two_sample(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Two-sample Kuiper statistic V = D+ + D- from the two empirical CDFs, with
    the asymptotic p-value.  Prefers astropy.stats.kuiper_two_sample when it is
    installed (vetted implementation); otherwise uses the self-contained path.
    Returns (V, p).  NaNs are dropped before testing.
    """
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    try:
        from astropy.stats import kuiper_two_sample
        V, p = kuiper_two_sample(x, y)
        return float(V), float(p)
    except Exception:
        pass   # fall through to the built-in implementation

    xs = np.sort(x); ys = np.sort(y)
    allv = np.concatenate([xs, ys]); allv.sort()
    cdf_x = np.searchsorted(xs, allv, side="right") / n1
    cdf_y = np.searchsorted(ys, allv, side="right") / n2
    diff = cdf_x - cdf_y
    V = float(np.max(diff) + np.max(-diff))   # D+ + D-
    ne = n1 * n2 / (n1 + n2)
    return V, _kuiper_pvalue(V, ne)


def _kuiper_breadth(X: np.ndarray, mask: np.ndarray, alpha: float) -> tuple:
    """
    Per-feature two-sample Kuiper test (country vs pool) + Benjamini-Hochberg.
    Returns (n_features_shifted, frac_features_shifted, n_features_tested).
    """
    from scipy.stats import false_discovery_control
    pos  = X[mask]
    pool = X[~mask]
    pvals = np.array([_kuiper_two_sample(pos[:, j], pool[:, j])[1]
                      for j in range(X.shape[1])], dtype=float)
    valid = np.isfinite(pvals)
    n_tested = int(valid.sum())
    if n_tested == 0:
        return 0, 0.0, 0
    q = false_discovery_control(pvals[valid], method="bh")
    n_sig = int(np.sum(q < alpha))
    return n_sig, float(n_sig / n_tested), n_tested


# ──────────────────────────────────────────────────── CLASSIFIER / CV ────────
def _make_clf(random_state: int):
    """StandardScaler -> balanced L2 logistic regression."""
    from sklearn.linear_model  import LogisticRegression
    from sklearn.pipeline      import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def _cv_auc(X: np.ndarray, y: np.ndarray, k: int, random_state: int) -> tuple:
    """Mean (and std) AUC over stratified k-fold CV."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        scores = cross_val_score(_make_clf(random_state), X, y, cv=cv,
                                 scoring="roc_auc")
    return float(np.mean(scores)), float(np.std(scores))


def _classify_shift(perm_p: float, auc_debiased: float,
                    alpha: float, near_far_threshold: float) -> str:
    if not np.isfinite(perm_p):
        return "undetermined"
    if perm_p >= alpha:
        return "none"
    return "far" if auc_debiased >= near_far_threshold else "near"


# ──────────────────────────────────────────────────── SHIFT DISTANCE ─────────
def compute_shift_distances(
    feature_matrix: pd.DataFrame,
    n_folds:            int   = 5,
    min_regions:        int   = 3,
    n_perm:             int   = 500,
    alpha:              float = DEFAULT_ALPHA,
    near_far_threshold: float = DEFAULT_NEAR_FAR_THRESHOLD,
    breadth_alpha:      float = DEFAULT_BREADTH_ALPHA,
    random_state:       int   = 42,
    verbose:            bool  = True,
) -> pd.DataFrame:
    """
    For each country in feature_matrix:
      - label that country's regions as 1, all others as 0
      - observed AUC via stratified k-fold CV of a balanced logistic regression
      - permutation null: shuffle labels n_perm times, recompute CV AUC
      - perm_p, auc_debiased (= AUC_obs - mean null), and a shift_class label
      - Kuiper breadth: per-feature two-sample test + BH -> n_features_shifted
    Returns a results DataFrame with one row per country.
    """
    if not _SKLEARN_OK:
        raise ImportError(
            "scikit-learn is required.  Install with:  pip install scikit-learn"
        )
    if not _SCIPY_OK:
        raise ImportError(
            "scipy is required (Kuiper breadth / FDR).  Install:  pip install scipy"
        )

    countries    = sorted(feature_matrix["country"].unique())
    feature_cols = [c for c in feature_matrix.columns if c != "country"]

    # X is identical for every country; only the label vector changes.
    X, n_feat = _prepare_X(feature_matrix, feature_cols)
    country_arr = feature_matrix["country"].values
    rng = np.random.default_rng(random_state)

    rows = []
    for country_y in countries:
        target_mask = country_arr == country_y
        n_target    = int(target_mask.sum())
        n_pool      = int((~target_mask).sum())

        if verbose:
            print(f"  {country_y:6s}  target={n_target:4d}  pool={n_pool:4d}", end="")

        base_row = {
            "country": country_y, "shift_distance_auc": np.nan, "auc_std": np.nan,
            "auc_null_mean": np.nan, "auc_debiased": np.nan, "perm_p": np.nan,
            "shift_class": "undetermined",
            "n_features_shifted": np.nan, "frac_features_shifted": np.nan,
            "n_target": n_target, "n_pool": n_pool,
            "n_features": int(n_feat), "n_perm": 0, "note": "",
        }

        # Guard: too few regions for CV.
        # Note: small n_target usually means data is collected at a coarse spatial
        # resolution (e.g. national level), NOT that the country lacks yield data.
        # The note records this so downstream readers don't conflate "cannot assess"
        # with "not shifted".
        if n_target < min_regions:
            base_row["note"] = (
                f"coarse spatial resolution — only {n_target} region(s) in data, "
                f"cannot assess shift"
            )
            if verbose:
                print(f"  → SKIP  ({base_row['note']})")
            rows.append(base_row)
            continue

        k = min(n_folds, n_target)   # can't have more folds than positive samples
        if k < 2:
            base_row["note"] = (
                f"coarse spatial resolution — only {n_target} region(s) in data, "
                f"cannot assess shift"
            )
            if verbose:
                print(f"  → SKIP  ({base_row['note']})")
            rows.append(base_row)
            continue

        y = target_mask.astype(int)

        try:
            auc_obs, auc_std = _cv_auc(X, y, k, random_state)

            # Permutation null: shuffle labels, recompute CV AUC.
            null = np.empty(n_perm, dtype=float)
            for b in range(n_perm):
                y_perm = rng.permutation(y)
                null[b], _ = _cv_auc(X, y_perm, k, random_state)

            auc_null_mean = float(np.mean(null))
            auc_debiased  = float(auc_obs - auc_null_mean)
            # +1 smoothing => perm_p is never exactly 0 (min = 1/(n_perm+1)).
            perm_p = float((1 + np.sum(null >= auc_obs)) / (n_perm + 1))
            shift_class = _classify_shift(perm_p, auc_debiased, alpha,
                                          near_far_threshold)

            # Kuiper breadth (model-free, complements the classifier).
            n_shift, frac_shift, _ = _kuiper_breadth(X, target_mask, breadth_alpha)

            base_row.update({
                "shift_distance_auc": auc_obs, "auc_std": auc_std,
                "auc_null_mean": auc_null_mean, "auc_debiased": auc_debiased,
                "perm_p": perm_p, "shift_class": shift_class,
                "n_features_shifted": n_shift, "frac_features_shifted": frac_shift,
                "n_perm": n_perm,
                "note": "" if k == n_folds else f"auto-reduced to {k}-fold CV",
            })
            if verbose:
                folds_note = f" ({k}-fold)" if k < n_folds else ""
                print(f"  → AUC={auc_obs:.3f}  debiased={auc_debiased:+.3f}  "
                      f"p={perm_p:.3f}  [{shift_class}]  "
                      f"breadth={n_shift}/{n_feat}{folds_note}")
        except Exception as exc:
            base_row["note"] = str(exc)
            if verbose:
                print(f"  → ERROR: {exc}")

        rows.append(base_row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────── MAIN ───────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compute domain-classifier shift distance per country",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--crop", required=True, choices=["maize", "wheat"],
        help="Crop to process",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None,
        help="Country codes (default: all with >= 8 years in years_dict.json)",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help=f"Primary output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--extra_output_dir", default=None,
        help="Optional second directory to ALSO write shift_distance_{crop}.csv "
             "to (e.g. a paper-tables folder). Created if missing.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="CV folds for AUC estimation (default: 5; auto-reduced if n_target < 5)",
    )
    parser.add_argument(
        "--min_regions", type=int, default=3,
        help="Skip countries with fewer than this many regions (default: 3)",
    )
    parser.add_argument(
        "--n_perm", type=int, default=500,
        help="Label permutations for the null distribution (default: 500). "
             "Runtime scales linearly with this; min perm_p = 1/(n_perm+1).",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"perm_p below this => shift is real, not 'none' (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--near_far_threshold", type=float, default=DEFAULT_NEAR_FAR_THRESHOLD,
        help="auc_debiased at/above this (and significant) => 'far', else 'near' "
             f"(default: {DEFAULT_NEAR_FAR_THRESHOLD})",
    )
    parser.add_argument(
        "--breadth_alpha", type=float, default=DEFAULT_BREADTH_ALPHA,
        help="BH q-value cutoff for a feature to count as shifted in the Kuiper "
             f"breadth measure (default: {DEFAULT_BREADTH_ALPHA})",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for CV shuffling, permutations, and logistic regression",
    )
    args = parser.parse_args()

    if not _SKLEARN_OK:
        print(
            "[ERROR] scikit-learn not found.\n"
            "Install it with:  pip install scikit-learn\n"
            "In the cybench conda env:  conda install scikit-learn"
        )
        sys.exit(1)
    if not _SCIPY_OK:
        print("[ERROR] scipy not found (needed for Kuiper breadth / FDR).\n"
              "Install it with:  pip install scipy")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = _get_countries(args.crop, args.countries)
    print(f"\n[Shift Distance]  crop={args.crop}  n_countries={len(countries)}")
    print(f"Countries: {countries}")
    print(f"lat/lon: EXCLUDED  quantiles={QUANTILES}  n_perm={args.n_perm}\n")

    # 1. Build feature matrix
    print("[1/3] Loading per-region features ...")
    feature_matrix = build_feature_matrix(args.crop, countries)

    # 2. Compute shift distances
    print(f"\n[2/3] Computing shift distances  "
          f"(LogisticReg AUC + {args.n_perm}-perm null + Kuiper breadth, "
          f"up to {args.n_folds}-fold CV) ...")
    results = compute_shift_distances(
        feature_matrix,
        n_folds=args.n_folds,
        min_regions=args.min_regions,
        n_perm=args.n_perm,
        alpha=args.alpha,
        near_far_threshold=args.near_far_threshold,
        breadth_alpha=args.breadth_alpha,
        random_state=args.random_state,
    )
    results.insert(0, "crop", args.crop)
    results["computed_at"] = datetime.utcnow().isoformat() + "Z"

    # 3. Write (primary + optional extra directory)
    out_path = output_dir / f"shift_distance_{args.crop}.csv"
    results.to_csv(out_path, index=False)
    print(f"\n[3/3] Wrote {len(results)} rows → {out_path}")
    if args.extra_output_dir:
        extra_dir = Path(args.extra_output_dir)
        extra_dir.mkdir(parents=True, exist_ok=True)
        extra_path = extra_dir / f"shift_distance_{args.crop}.csv"
        results.to_csv(extra_path, index=False)
        print(f"      also copied → {extra_path}")

    # Summary
    valid = results.dropna(subset=["shift_distance_auc"])
    print(f"\n{'='*74}")
    print(f"SHIFT DISTANCE SUMMARY  ({args.crop})")
    print(f"{'='*74}")
    print(f"Countries with AUC : {len(valid)}/{len(results)}")

    cols = ["country", "shift_distance_auc", "auc_debiased", "perm_p",
            "n_features_shifted", "n_target"]
    if len(valid):
        counts = valid["shift_class"].value_counts().to_dict()
        print(f"shift_class        : "
              f"far={counts.get('far', 0)}  near={counts.get('near', 0)}  "
              f"none={counts.get('none', 0)}")
        print(f"\nfar-shift (significant + large debiased AUC → hardest transfer):")
        far = valid[valid["shift_class"] == "far"].nlargest(10, "auc_debiased")[cols]
        print(far.to_string(index=False) if len(far) else "  (none)")
        print(f"\nnear-shift (significant but small debiased AUC → borderline):")
        near = valid[valid["shift_class"] == "near"].sort_values(
            "auc_debiased", ascending=False)[cols]
        print(near.to_string(index=False) if len(near) else "  (none)")
        print(f"\nno-shift (not distinguishable from the pool):")
        none = valid[valid["shift_class"] == "none"][cols]
        print(none.to_string(index=False) if len(none) else "  (none)")

    skipped = results[results["shift_distance_auc"].isna()]
    if len(skipped):
        print(f"\nSkipped ({len(skipped)}):")
        for _, row in skipped.iterrows():
            print(f"  {row['country']:6s}  {row['note']}")

    print(f"{'='*74}")
    print(f"\nNext: join {out_path.name} with ranking.py output on (crop, country)")
    print(f"      to plot the Delta-skill-vs-shift-distance curve.")


if __name__ == "__main__":
    main()
