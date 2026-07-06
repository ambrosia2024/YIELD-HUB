"""
--------------------
Author: XYZ
Description: Feature caching system for HPO to avoid rebuilding features on every trial
Python version: 3.12.0

--------------------
Caches pre-built features (X_ts, X_static, y, masks) keyed by feature configuration.
Allows reuse across Optuna trials when only model hyperparameters change.

Key insight: Most HPO trials only vary model params (lr, batch_size), not features.
By caching features, we avoid expensive recomputation of GDD, RUE, Farquhar, etc.
"""

import hashlib
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

from cybench.datasets.dataset import Dataset as CYDataset
from cybench.config import KEY_LOC, KEY_YEAR, KEY_DATES

logger = logging.getLogger(__name__)

# Import the main feature building function
from featureEngineering import build_daily_input_sequence


class FeatureCache:
    """
    Caches pre-built features keyed by feature configuration hash.

    The cache stores:
    - X_ts: Time series features (n_samples, seq_len, n_ts_features)
    - X_static: Static features (n_samples, n_static_features)
    - y: Target values (n_samples,)
    - masks: Validity masks (n_samples, seq_len)
    - years: Sample years (n_samples,)
    - adm_ids: Administrative IDs (n_samples,)
    - lats/lons: Coordinates (n_samples,)
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._prebuild_stats = {}

    @staticmethod
    def make_feature_hash(
        crop: str,
        country: str,
        aggregation: str,
        data_fraction: float,
        use_sota_features: bool,
        include_spatial_features: bool,
        lag_years: int,
        use_gdd: bool,
        use_heat_stress_days: bool,
        use_rue: bool,
        use_farquhar: bool,
        use_cwb_feature: bool,
        drop_tavg: bool,
        use_exponential_weighting: bool,
        exponential_tau: float,
        multi_year_summaries: bool,
        multi_year_window: int,
        multi_year_features: List[str],
        weather_features: List[str],
    ) -> str:
        """
        Generate a deterministic hash from all feature-related configuration parameters.

        Model hyperparameters (lr, batch_size, etc.) are NOT included because they
        don't affect feature computation.
        """
        # Sort multi_year_features for deterministic hashing
        multi_year_features_sorted = sorted(multi_year_features) if multi_year_features else []

        hash_input = (
            f"{crop}|{country}|{aggregation}|{data_fraction}|"
            f"{use_sota_features}|{include_spatial_features}|{lag_years}|"
            f"{use_gdd}|{use_heat_stress_days}|{use_rue}|{use_farquhar}|"
            f"{use_cwb_feature}|{drop_tavg}|"
            f"{use_exponential_weighting}|{exponential_tau}|"
            f"{multi_year_summaries}|{multi_year_window}|{multi_year_features_sorted}|"
            f"{sorted(weather_features)}"
        )

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def get(self, feature_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features for a given hash."""
        if feature_hash in self._cache:
            self._hits += 1
            logger.debug(f"[FeatureCache] HIT for hash {feature_hash[:8]}... (hits: {self._hits})")
            return self._cache[feature_hash]
        self._misses += 1
        return None

    def set(self, feature_hash: str, features: Dict[str, Any]) -> None:
        """Store features in cache."""
        self._cache[feature_hash] = features
        logger.debug(f"[FeatureCache] CACHED features for hash {feature_hash[:8]}...")

    def has(self, feature_hash: str) -> bool:
        """Check if features are cached."""
        return feature_hash in self._cache

    def prebuild_features(
        self,
        ds: CYDataset,
        feature_configs: List[Dict[str, Any]],
    ) -> None:
        """
        Pre-build features for multiple configurations and cache them.

        Args:
            ds: CYDataset instance
            feature_configs: List of dicts containing feature parameters for each config
        """
        total_configs = len(feature_configs)
        logger.info(f"[FeatureCache] Pre-building features for {total_configs} configurations...")

        for i, config in enumerate(feature_configs):
            feature_hash = self.make_feature_hash(
                crop=config['crop'],
                country=config['country'],
                aggregation=config['aggregation'],
                data_fraction=config['data_fraction'],
                use_sota_features=config['use_sota_features'],
                include_spatial_features=config['include_spatial_features'],
                lag_years=config['lag_years'],
                use_gdd=config['use_gdd'],
                use_heat_stress_days=config['use_heat_stress_days'],
                use_rue=config['use_rue'],
                use_farquhar=config['use_farquhar'],
                use_cwb_feature=config.get('use_cwb_feature', False),
                drop_tavg=config.get('drop_tavg', False),
                use_exponential_weighting=config.get('use_exponential_weighting', False),
                exponential_tau=config.get('exponential_tau', 10.0),
                multi_year_summaries=config.get('multi_year_summaries', False),
                multi_year_window=config.get('multi_year_window', 1),
                multi_year_features=config.get('multi_year_features', []),
                weather_features=config.get('weather_features', []),
            )

            # Check if already cached
            if self.has(feature_hash):
                logger.info(f"  [{i+1}/{total_configs}] Already cached: {feature_hash[:8]}...")
                continue

            logger.info(f"  [{i+1}/{total_configs}] Building: {feature_hash[:8]}... "
                        f"(SOTA={config['use_sota_features']}, GDD={config['use_gdd']}, "
                        f"RUE={config['use_rue']}, Farquhar={config['use_farquhar']}, "
                        f"HeatStress={config['use_heat_stress_days']}, Lag={config['lag_years']})")

            # Build features for all samples
            all_X_ts, all_X_static, all_y = [], [], []
            all_years, all_adm_ids, all_lats, all_lons, all_masks = [], [], [], [], []

            # Build multi-year config if needed
            multi_year_config = None
            if config.get('multi_year_summaries', False):
                multi_year_config = {
                    'window': config.get('multi_year_window', 1),
                    'features': config.get('multi_year_features', ['weather']),
                }

            for sample_idx in range(len(ds)):
                sample = ds[sample_idx]

                X_ts, X_static, y, meta, mask = build_daily_input_sequence(
                    ds,
                    sample[KEY_LOC],
                    sample[KEY_YEAR],
                    aggregation=config['aggregation'],
                    data_fraction=config['data_fraction'],
                    use_sota_features=config['use_sota_features'],
                    include_spatial_features=config['include_spatial_features'],
                    lag_years=config['lag_years'],
                    weather_features_list=config.get('weather_features', []),
                    use_gdd=config['use_gdd'],
                    use_heat_stress_days=config['use_heat_stress_days'],
                    use_rue=config['use_rue'],
                    use_farquhar=config['use_farquhar'],
                    crop=config['crop'],
                    multi_year_config=multi_year_config,
                )

                all_X_ts.append(X_ts)
                all_X_static.append(X_static)
                all_y.append(y)
                all_years.append(sample[KEY_YEAR])
                all_adm_ids.append(sample[KEY_LOC])
                all_lats.append(meta["lat"])
                all_lons.append(meta["lon"])
                all_masks.append(mask)

            # Convert to numpy arrays
            cached_data = {
                'X_ts': np.array(all_X_ts),
                'X_static': np.array(all_X_static),
                'y': np.array(all_y),
                'years': np.array(all_years),
                'adm_ids': np.array(all_adm_ids),
                'lats': np.array(all_lats, dtype=object),
                'lons': np.array(all_lons, dtype=object),
                'masks': np.array(all_masks),
                'n_samples': len(all_X_ts),
            }

            self.set(feature_hash, cached_data)
            self._prebuild_stats[feature_hash] = {
                'n_samples': len(all_X_ts),
                'X_ts_shape': np.array(all_X_ts).shape,
                'X_static_shape': np.array(all_X_static).shape,
            }

        logger.info(f"[FeatureCache] Pre-build complete. Cache size: {len(self._cache)}")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        hit_rate = self._hits / (self._hits + self._misses) * 100 if (self._hits + self._misses) > 0 else 0
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'prebuild_entries': len(self._prebuild_stats),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._prebuild_stats.clear()

    def save(self, path: str) -> None:
        """Save cache to disk."""
        cache_data = {
            'cache': self._cache,
            'prebuild_stats': self._prebuild_stats,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"[FeatureCache] Saved cache to {path}")

    def load(self, path: str) -> None:
        """Load cache from disk."""
        if not Path(path).exists():
            logger.warning(f"[FeatureCache] Cache file not found: {path}")
            return
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
        self._cache = cache_data['cache']
        self._prebuild_stats = cache_data.get('prebuild_stats', {})
        logger.info(f"[FeatureCache] Loaded cache from {path} ({len(self._cache)} entries)")


# Global cache instance
_global_cache: Optional[FeatureCache] = None


def get_global_cache() -> FeatureCache:
    """Get or create the global feature cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache()
    return _global_cache


def reset_global_cache() -> None:
    """Reset the global feature cache."""
    global _global_cache
    _global_cache = None


def generate_common_feature_configs(
    crop: str,
    country: str,
    aggregation: str,
    data_fraction: float,
    weather_features: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate common feature configuration combinations for pre-building.

    This creates a representative set of feature configs that cover the HPO search space.
    The idea is to pre-build features for the most common combinations so that
    most trials can use cached features.

    Args:
        crop: Crop name (maize, wheat)
        country: Country code
        aggregation: Temporal aggregation (daily, weekly, dekad)
        data_fraction: Portion of season data (1.0, 0.75, 0.5, 0.25)
        weather_features: Base weather feature list

    Returns:
        List of feature configuration dictionaries
    """
    configs = []

    # Get all combinations of boolean feature flags
    # We use a smart subset rather than full cartesian product to avoid explosion

    # Base config (all features OFF, minimal lag)
    configs.append({
        'crop': crop, 'country': country, 'aggregation': aggregation,
        'data_fraction': data_fraction, 'use_sota_features': False,
        'include_spatial_features': False, 'lag_years': 0,
        'use_gdd': False, 'use_heat_stress_days': False,
        'use_rue': False, 'use_farquhar': False,
        'use_cwb_feature': False, 'drop_tavg': False,
        'use_exponential_weighting': False, 'exponential_tau': 10.0,
        'multi_year_summaries': False, 'multi_year_window': 1,
        'multi_year_features': [], 'weather_features': weather_features,
    })

    # Common combinations - vary a few features at a time
    feature_combinations = [
        # SOTA features ON
        {'use_sota_features': True},

        # GDD variations
        {'use_gdd': True},

        # Heat stress days
        {'use_heat_stress_days': True},

        # RUE
        {'use_rue': True},

        # Farquhar
        {'use_farquhar': True},

        # Spatial features
        {'include_spatial_features': True},

        # Lag years
        {'lag_years': 1},
        {'lag_years': 2},

        # Common combos
        {'use_sota_features': True, 'lag_years': 1},
        {'use_gdd': True, 'use_heat_stress_days': True},
        {'use_rue': True, 'use_farquhar': True},
        {'use_sota_features': True, 'include_spatial_features': True, 'lag_years': 1},

        # Exponential weighting variations
        {'use_exponential_weighting': True, 'exponential_tau': 10.0},
        {'use_exponential_weighting': True, 'exponential_tau': 25.0},

        # Multi-year summaries
        {'multi_year_summaries': True, 'multi_year_window': 1, 'multi_year_features': ['weather']},

        # "kitchen sink" - many features ON
        {'use_sota_features': True, 'use_gdd': True, 'use_heat_stress_days': True,
         'include_spatial_features': True, 'lag_years': 1},
    ]

    for combo in feature_combinations:
        config = {
            'crop': crop, 'country': country, 'aggregation': aggregation,
            'data_fraction': data_fraction,
            'use_sota_features': False, 'include_spatial_features': False,
            'lag_years': 0, 'use_gdd': False, 'use_heat_stress_days': False,
            'use_rue': False, 'use_farquhar': False,
            'use_cwb_feature': False, 'drop_tavg': False,
            'use_exponential_weighting': False, 'exponential_tau': 10.0,
            'multi_year_summaries': False, 'multi_year_window': 1,
            'multi_year_features': [], 'weather_features': weather_features,
        }
        config.update(combo)
        configs.append(config)

    return configs


def prebuild_features_for_hpo(
    ds: CYDataset,
    crop: str,
    country: str,
    aggregation: str,
    data_fraction: float,
    weather_features: List[str],
    cache: Optional[FeatureCache] = None,
) -> FeatureCache:
    """
    Pre-build common feature configurations for HPO.

    This should be called before starting Optuna HPO to populate the cache
    with frequently-used feature combinations.

    Args:
        ds: CYDataset instance
        crop: Crop name
        country: Country code
        aggregation: Temporal aggregation
        data_fraction: Portion of season data
        weather_features: Base weather feature list
        cache: Optional FeatureCache (uses global if None)

    Returns:
        The FeatureCache instance (newly created if cache was None)
    """
    if cache is None:
        cache = get_global_cache()

    configs = generate_common_feature_configs(
        crop=crop,
        country=country,
        aggregation=aggregation,
        data_fraction=data_fraction,
        weather_features=weather_features,
    )

    cache.prebuild_features(ds, configs)

    return cache
