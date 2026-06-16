"""
--------------------
Author: XYZ
Description: Contains helper functions and classes to load the cybench data.
Python version: 3.12.0

--------------------
"""

import sys
import logging
from typing import Union

import numpy as np
import pandas as pd

from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from cybench.config import (LOCATION_PROPERTIES, SOIL_PROPERTIES, CROP_CALENDAR_DATES)
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Custom functions
from featureEngineering import build_daily_input_sequence, _get_static_feature_names

sys.path.append('../architectures/')
from modelconfig import TSTModelConfig, LinearModelConfig

# %% Global constants
# Weather feature lists - used as defaults by TSTModelConfig or LinearModelConfig.weather_features property
# These are module-level constants; actual features used come from config
WEATHER_FEATURES_BASE = ['tmin', 'tmax', 'tavg', 'prec', 'rad']
WEATHER_FEATURES_WITH_CWB = ['tmin', 'tmax', 'tavg', 'prec', 'cwb', 'rad']

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']

STANDARD_STATIC_VARS = SOIL_PROPERTIES + LOCATION_PROPERTIES
# CROP_CALENDAR_DATES imported from cybench.config

SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

# based on config.weather_features and config.time_series_vars properties
print(f"[Feature Config] Static vars ({len(STANDARD_STATIC_VARS)}): {STANDARD_STATIC_VARS}")
print(f"[Feature Config] SOTA Temporal vars ({len(SOTA_TEMPORAL_VARS_LIST)}): {SOTA_TEMPORAL_VARS_LIST}")


def prepare_features_and_targets(dataset):
    """
    Prepared features and target from the raw data.
    """
    X_list, y_list, years_list = [], [], []

    targets_array = dataset.targets()
    indices_list = list(dataset.indices())  # [(adm_id, year), ...]

    for i, idx in enumerate(indices_list):
        adm_id, year = idx
        target = targets_array[i]

        features = {}

        # Soil
        soil_row = dataset._dfs_x['soil'].loc[adm_id]
        for col in soil_row.index:
            features[f'soil_{col}'] = soil_row[col]

        # Meteorological
        meteo_rows = dataset._dfs_x['meteo'].loc[adm_id].loc[year]
        features['meteo_tmin_mean'] = meteo_rows['tmin'].mean()
        features['meteo_tmax_mean'] = meteo_rows['tmax'].mean()
        features['meteo_tavg_mean'] = meteo_rows['tavg'].mean()
        features['meteo_prec_sum'] = meteo_rows['prec'].sum()
        features['meteo_cwb_sum'] = meteo_rows['cwb'].sum()
        features['meteo_rad_sum'] = meteo_rows['rad'].sum()

        # Remote sensing
        for key in ['fpar', 'ndvi', 'ssm']:
            try:
                rs_rows = dataset._dfs_x[key].loc[adm_id].loc[year]
                features[f'{key}_mean'] = rs_rows.iloc[:, 0].mean() if not rs_rows.empty else np.nan
            except KeyError:
                features[f'{key}_mean'] = np.nan

        # Crop season
        try:
            cs_row = dataset._dfs_x['crop_season'].loc[(adm_id, year)]
            for col in cs_row.index:
                value = cs_row[col]
                if isinstance(value, pd.Timestamp):
                    value = (value - pd.Timestamp("1970-01-01")).days
                elif pd.isnull(value):
                    value = np.nan
                features[f'crop_{col}'] = value
        except KeyError:
            for col in dataset._dfs_x['crop_season'].columns:
                features[f'crop_{col}'] = np.nan

        X_list.append(list(features.values()))
        y_list.append(target)
        years_list.append(year)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, years_list

# %% Dataset Wrapper
class DailyYieldDataset(Dataset):
    """PyTorch Dataset wrapping pre-computed arrays for one data split."""

    def __init__(self, X_ts, X_static, y, years=None, adm_ids=None,
                 lats=None, lons=None, validity_masks=None):
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.years = torch.tensor(years, dtype=torch.long) if years is not None else None
        self.adm_ids = list(adm_ids) if adm_ids is not None else None

        # replace None with nan before passing to torch.tensor().
        # _extract_static_features() can return lat=None / lon=None when location data is missing. 
        # In those cases, torch.tensor([1.2, None, 3.4]) raises a TypeError.
        def _safe_tensor(lst):
            if lst is None:
                return None
            cleaned = [v if v is not None else float('nan') for v in lst]
            return torch.tensor(cleaned, dtype=torch.float32)

        self.lats = _safe_tensor(lats)
        self.lons = _safe_tensor(lons)
        self.validity_masks = (torch.tensor(validity_masks, dtype=torch.bool)
                               if validity_masks is not None else None)

    def __len__(self):
        return len(self.X_ts)

    def __getitem__(self, idx):
        vm = (self.validity_masks[idx] if self.validity_masks is not None
              else torch.ones(self.X_ts.shape[1], dtype=torch.bool))
        return (self.X_ts[idx], self.X_static[idx], self.y[idx],
                self.years[idx], self.adm_ids[idx],
                self.lats[idx], self.lons[idx], vm)
    
# %% Data Module
class DailyCYBenchSeqDataModule(pl.LightningDataModule):
    """Lightning DataModule: loads CY-Bench data, builds features, normalises."""

    def __init__(self, config: Union[TSTModelConfig, LinearModelConfig]):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.y_mean = self.y_std = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.feature_norm_params = None
        # Store all data for dynamic splitting
        self.all_X_ts = None
        self.all_X_static = None
        self.all_y = None
        self.all_years = None
        self.all_adm_ids = None
        self.all_lats = None
        self.all_lons = None
        self.all_masks = None
        # Recursive lag prediction cache
        self._prediction_cache = {}  # {(adm_id, year): predicted_yield}
        self._test_years = None  # Track which years are in test set
        self._train_years = None  # Track which years are in train set

    def setup(self, stage: Optional[str] = None,
              train_years: Optional[List[int]] = None,
              val_years: Optional[List[int]] = None,
              test_years: Optional[List[int]] = None,
              features_only: bool = False):
        """
        Setup datasets and features.

        Args:
            stage: 'fit', 'validate', 'test', or None
            train_years: Years for training split
            val_years: Years for validation split
            test_years: Years for test split
            features_only: If True, only build feature arrays and skip split/normalization
                          (useful for caching features across CV folds)
        """
        cfg = self.config
        # If called by Lightning internals with no explicit splits, and we already have datasets, skip re-setup to preserve the current split.
        # This prevents Lightning from overriding our carefully configured splits.
        if (train_years is None and self.train_ds is not None):
            return

        # Always recompute normalization for the current split
        # Prevents stale params from a previous setup() call leaking into a new split
        # EXCEPT if we're setting up for testing only (no train years), in which case
        # we reuse existing normalization stats
        if train_years is not None and len(train_years) > 0:
            self.feature_norm_params = None
            self.y_mean = None
            self.y_std = None

        print(f"\n[DataModule] {cfg.crop}-{cfg.country} | {cfg.aggregation.upper()} | "
              f"Spatial={cfg.include_spatial_features} | Lag={cfg.lag_years}")

        df_y, dfs_x = load_dfs_crop(cfg.crop, [cfg.country])
        if df_y is None or len(df_y) == 0:
            raise ValueError(f"No data for {cfg.crop}-{cfg.country}")

        ds = CYDataset(cfg.crop, df_y, dfs_x)

        # Build all features (only done once)
        if self.all_X_ts is None:
            all_X_ts, all_X_static, all_y = [], [], []
            all_years_list, all_adm_ids, all_lats, all_lons, all_masks = [], [], [], [], []

            for i in range(len(ds)):
                sample = ds[i]
                X_ts, X_static, y, meta, mask = build_daily_input_sequence(
                    ds, sample[KEY_LOC], sample[KEY_YEAR],
                    aggregation=cfg.aggregation,
                    data_fraction=cfg.data_fraction,
                    use_sota_features=cfg.use_sota_features,
                    include_spatial_features=cfg.include_spatial_features,
                    lag_years=cfg.lag_years,
                    weather_features_list=cfg.weather_features,  # Pass from config
                    use_gdd=cfg.use_gdd,
                    use_heat_stress_days=cfg.use_heat_stress_days,
                    use_rue=cfg.use_rue,
                    use_farquhar=cfg.use_farquhar,
                    crop=cfg.crop,
                    multi_year_config=cfg.multi_year_config if hasattr(cfg, 'multi_year_config') and cfg.multi_year_config is not None else None,
                )
                all_X_ts.append(X_ts)
                all_X_static.append(X_static)
                all_y.append(y)
                all_years_list.append(sample[KEY_YEAR])
                all_adm_ids.append(sample[KEY_LOC])
                all_lats.append(meta["lat"])
                all_lons.append(meta["lon"])
                all_masks.append(mask)

            # Convert to numpy arrays
            # With compute_summaries now returning consistent feature structure,
            # all_X_static should have consistent length across samples
            self.all_X_ts = np.array(all_X_ts)
            self.all_X_static = np.array(all_X_static)
            self.all_y = np.array(all_y)
            self.all_years = np.array(all_years_list)
            self.all_adm_ids = np.array(all_adm_ids)
            self.all_lats = np.array(all_lats, dtype=object)
            self.all_lons = np.array(all_lons, dtype=object)
            self.all_masks = np.array(all_masks)

            # Validate static feature count
            expected = cfg._compute_expected_static_features()
            actual = self.all_X_static.shape[1]
            if actual != expected:
                raise ValueError(
                    f"\n[ERROR] Static feature mismatch: expected {expected}, got {actual}.\n"
                    f"A feature creation step likely failed silently. Check debug logs.\n"
                    f"Config: spatial={cfg.include_spatial_features}, lag={cfg.lag_years}"
                )
            print(f"  Static features validated: {actual}/{expected}")

        # Early return for features-only mode (avoids wasteful split/normalization)
        # Moved OUTSIDE the `if self.all_X_ts is None` block so it works even when arrays are cached
        if features_only:
            logging.info(f"[DataModule] features_only=True - skipping split/normalization, "
                       f"cached {len(self.all_X_ts)} samples")
            return

        # Use provided splits or compute default
        if train_years is not None and val_years is not None and test_years is not None:
            # Dynamic splits provided
            train_yrs = set(train_years)
            val_yrs = set(val_years)
            test_yrs = set(test_years)
        else:
            # Default split for backward compatibility
            years_sorted = np.unique(self.all_years)
            if len(years_sorted) < 6:
                raise ValueError(
                    f"Need ≥ 6 years for split (3 test + 3 val + train); got {len(years_sorted)}: {sorted(years_sorted)}"
                )
            test_yrs = set(years_sorted[-3:])
            val_yrs = set(years_sorted[-6:-3])
            train_yrs = set(years_sorted[:-6])

        print(f"  Split: Train {sorted(train_yrs)}, Val {sorted(val_yrs)}, Test {sorted(test_yrs)}")
        print(f"  Years Summary:")
        print(f"    Train years ({len(train_yrs)}): {sorted(train_yrs)}")
        print(f"    Val years ({len(val_yrs)}):   {sorted(val_yrs)}")
        print(f"    Test years ({len(test_yrs)}):  {sorted(test_yrs)}")

        def idx(yr_set):
            return np.where(np.isin(self.all_years, list(yr_set)))[0]

        train_idx, val_idx, test_idx = idx(train_yrs), idx(val_yrs), idx(test_yrs)

        # Normalise targets using training statistics only (no leakage)
        # If train_idx is empty (e.g., testing only), reuse existing stats if available
        if len(train_idx) > 0:
            self.y_mean = float(np.mean(self.all_y[train_idx]))
            self.y_std = float(np.std(self.all_y[train_idx])) or 1.0
        elif self.y_mean is None or self.y_std is None:
            # No training data and no existing stats - this is an error case
            raise ValueError("Cannot compute normalization: no training data available")
        # If train_idx is empty but we have existing stats, reuse them (for testing only)
        print(f"  Y norm: mean={self.y_mean:.4f}, std={self.y_std:.4f}")

        # Compute or reuse feature normalization
        if len(train_idx) > 0:
            self.feature_norm_params = self._compute_feature_normalization(
                self.all_X_ts[train_idx], self.all_X_static[train_idx],
                self.all_masks[train_idx] if self.all_masks is not None else None
            )
        elif self.feature_norm_params is None:
            raise ValueError("Cannot compute feature normalization: no training data available")
        print(f"  Feature norm params: {len(self.feature_norm_params)} features")

        all_y_norm = (self.all_y - self.y_mean) / self.y_std

        def make_ds(idxs):
            return DailyYieldDataset(
                self.all_X_ts[idxs], self.all_X_static[idxs], all_y_norm[idxs],
                self.all_years[idxs], self.all_adm_ids[idxs],
                self.all_lats[idxs], self.all_lons[idxs], self.all_masks[idxs],
            )

        self.train_ds = make_ds(train_idx)
        self.val_ds = make_ds(val_idx)
        self.test_ds = make_ds(test_idx)
        print(f"  Samples: train={len(self.train_ds)}, val={len(self.val_ds)}, "
              f"test={len(self.test_ds)}")

        # Store train/test years for recursive lag prediction
        self._train_years = train_yrs
        self._test_years = test_yrs
        self._val_years = val_yrs

        # Warn about potential lag yield leakage in test set
        if self.config.lag_years > 0:
            test_year_set = test_yrs
            lag_overlap_years = set()
            for test_year in test_yrs:
                for lag in range(1, self.config.lag_years + 1):
                    if (test_year - lag) in test_year_set:
                        lag_overlap_years.add(test_year - lag)
            if lag_overlap_years:
                msg = (
                    f"[WARNING] Lag Leakage: Test years {sorted(lag_overlap_years)} are used as lag "
                    f"inputs for later test years. Reported test metrics may be optimistic because "
                    f"the model has access to observed test-set yields. "
                    f"Use --lag_years 0 for no lag features, or --use_recursive_lags for true "
                    f"out-of-sample evaluation (uses predicted yields as lags)."
                )
                print(msg)
                logging.warning(msg)

    def _compute_feature_normalization(self, X_ts, X_static, observed_masks=None):
        """Compute z-score params from training data only.

        Args:
            X_ts: Time series features (n_samples, seq_len, n_features)
            X_static: Static features (n_samples, n_static_features)
            observed_masks: Boolean masks (n_samples, seq_len) indicating valid timesteps
        """
        params = {}
        for i, name in enumerate(self._get_ts_feature_names()):
            col = X_ts[:, :, i].flatten()
            # Exclude padded zeros if masks provided
            if observed_masks is not None:
                col = X_ts[:, :, i][observed_masks]
            params[f"ts_{name}"] = {
                "mean": float(np.nanmean(col)) if col.size > 0 and not np.all(np.isnan(col)) else 0.0,
                "std": (float(np.nanstd(col)) or 1.0) if col.size > 0 and not np.all(np.isnan(col)) else 1.0,
            }
        for i, name in enumerate(self._get_static_feature_names()):
            col = X_static[:, i]
            params[f"static_{name}"] = {
                "mean": float(np.nanmean(col)) if col.size > 0 and not np.all(np.isnan(col)) else 0.0,
                "std": (float(np.nanstd(col)) or 1.0) if col.size > 0 and not np.all(np.isnan(col)) else 1.0,
            }
        return params

    def _get_ts_feature_names(self) -> List[str]:
        """
        Return ordered list of time series feature names.

        Order matches the column order in the assembled time series array:
          1. Base weather features
          2. Domain features (GDD, RUE, Farquhar) - when enabled
          3. Remote sensing features
          4. SOTA temporal features - when enabled
        """
        names = [f'weather_{n}' for n in self.config.weather_features]
        # Domain time series channels, in the order they are appended
        if self.config.use_gdd:
            names.append('domain_cum_gdd')
        if self.config.use_rue:
            names.append('domain_rue_index')
        if self.config.use_farquhar:
            names.append('domain_farquhar_proxy')
        names += [f'rs_{n}' for n in REMOTE_SENSING_FEATURES]
        if self.config.use_sota_features:
            names += [f'sota_{n}' for n in SOTA_TEMPORAL_VARS_LIST]
        return names

    def _get_static_feature_names(self) -> List[str]:
        """Thin wrapper around the module-level helper (single source of truth)."""
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
            self.config.use_heat_stress_days,
            multi_year_config=self.config.multi_year_config if hasattr(self.config, 'multi_year_config') else None,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self):
        """
        Returns the test dataloader.

        shuffle=False processes samples in dataset construction order.
        For --use_recursive_lags to work correctly, samples within each location must
        appear in chronological order. This holds as long as ds[i] iterates chronologically
        within each location, which is assumed but not enforced by CYDataset.
        """
        return DataLoader(self.test_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.num_workers)

    def copy_normalization_from(self, other_dm):
        """Copy normalization statistics from another datamodule.

        Useful when creating a test datamodule that should use the same
        normalization as the training datamodule.

        Args:
            other_dm: Another DailyCYBenchSeqDataModule with computed stats
        """
        self.y_mean = other_dm.y_mean
        self.y_std = other_dm.y_std
        self.feature_norm_params = other_dm.feature_norm_params


def calculate_fixed_split(
    all_years: List[int],
    test_years: int = 5,
    val_years: int = 2
) -> Dict:
    """
    Calculate fixed train/val/test splits for non-CV mode.
    
    Uses last N years for test set, last M years of remaining for validation,
    and everything else for training.
    """
    sorted_years = sorted(all_years)
    total_years = len(sorted_years)
    
    min_required = test_years + val_years + 1
    if total_years < min_required:
        raise ValueError(
            f"Insufficient data for fixed split mode: {total_years} years available, "
            f"but {min_required} years required (test={test_years} + val={val_years} + min_train=1). "
            f"Please use a country-crop combination with at least {min_required} years of data."
        )
    
    test_years_list = sorted_years[-test_years:]
    remaining = sorted_years[:-test_years]
    val_years_list = remaining[-val_years:]
    train_years_list = remaining[:-val_years]
    
    return {
        'train_years': train_years_list,
        'val_years': val_years_list,
        'test_years': test_years_list,
        'total_years': total_years,
        'can_split': True,
        'skip_reason': None
    }

def generate_walk_forward_splits(all_years, test_years):
    """
    Generate walk-forward (expanding window) splits.

    Train on all years except last N, then walk forward one year at a time.

    Example: years=[2000-2020], test_years=5
    - Fold 1 (fold_idx=0): train=2000-2015, test=2016
    - Fold 2 (fold_idx=1): train=2000-2016, test=2017
    - Fold 3 (fold_idx=2): train=2000-2017, test=2018
    - Fold 4 (fold_idx=3): train=2000-2018, test=2019
    - Fold 5 (fold_idx=4): train=2000-2019, test=2020

    Note: Internal fold_idx is 0-indexed, but displayed as 1-indexed to users.

    Args:
        all_years: List of all available years
        test_years: Number of years to walk forward (N)

    Returns:
        List of dicts with train_years, test_years, fold_idx (0-indexed)
    """
    splits = []
    initial_train_cutoff = len(all_years) - test_years

    for i in range(test_years):
        train_years = all_years[:initial_train_cutoff + i]
        test_year = all_years[initial_train_cutoff + i]

        splits.append({
            'train_years': train_years,
            'test_years': [test_year],
            'fold_idx': i
        })

    return splits