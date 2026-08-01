"""
--------------------
Author: XYZ
Description: PyTorch Lightning DataModule for contrastive pre-training.
             Handles multi-country data loading with per-country temporal splits,
             loads pre-computed AEZ codes and contrastive pairs, and provides
             batches with (anchor, positives, negatives) structure.

Key features:
- Multi-country support with per-country temporal splits
- Loads pre-computed AEZ codes and contrastive pairs
- Handles core vs extended features
- Supports both contrastive pre-training and supervised fine-tuning modes

Python version: 3.12.0
--------------------
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning.pytorch as pl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# CY-BENCH dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES, KEY_LOC, KEY_YEAR, KEY_TARGET,
    FORECAST_LEAD_TIME
)
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

# Custom functions
sys.path.append('../../process/')
from loadData import calculate_fixed_split
from featureEngineering import build_daily_input_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContrastiveModelConfig:
    """Configuration for contrastive pre-training model."""
    # Basic crop/country settings
    crop: str = "maize"
    countries: List[str] = None  # List of country codes or ["all"]

    # Data settings
    aggregation: str = "dekad"  # daily, weekly, dekad
    data_fraction: float = 1.0
    lag_years: int = 1

    # Feature settings (core features only for pre-training)
    use_gdd: bool = False
    use_heat_stress_days: bool = False
    use_rue: bool = False
    use_farquhar: bool = False
    use_cwb_feature: bool = False
    drop_tavg: bool = False
    include_spatial_features: bool = False

    # Multi-year summaries (extended features, for fine-tuning only)
    multi_year_summaries: bool = False
    multi_year_window: int = 1
    multi_year_features: List[str] = None

    # Exponential weighting (extended feature, for fine-tuning only)
    use_exponential_weighting: bool = False
    exponential_tau: float = 10.0

    # SOTA features
    use_sota_features: bool = False

    # Split settings
    test_years: int = 3
    val_years: int = 2

    # Pre-computed file paths
    aez_lookup_path: Optional[str] = None
    pairs_cache_path: Optional[str] = None

    # Training settings
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    pin_memory: bool = True

    # Mode: 'pretrain' for contrastive learning, 'finetune' for supervised
    mode: str = 'pretrain'

    # Contrastive sampling settings
    num_hard_positives: int = 2
    num_soft_positives: int = 2
    num_weak_positives: int = 2
    num_negatives: int = 4

    def __post_init__(self):
        if self.countries is None:
            self.countries = ["all"]
        if self.multi_year_features is None:
            self.multi_year_features = ['weather']

    @property
    def seq_len(self):
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        from cybench.architectures.modelconfig import WEATHER_FEATURES_BASE, WEATHER_FEATURES_WITH_CWB
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        from cybench.architectures.modelconfig import REMOTE_SENSING_FEATURES
        return self.weather_features + REMOTE_SENSING_FEATURES


class ContrastiveDataset(Dataset):
    """
    PyTorch Dataset for contrastive pre-training.

    Returns batches with:
    - anchor: (X_ts, X_static, metadata)
    - positives: List of positive samples
    - negatives: List of negative samples
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],  # (adm_id, year, country)
        pairs: Dict[int, Dict],
        features_dict: Dict[Tuple[str, int, str], Dict],  # (adm_id, year, country) -> features
        aez_codes: Dict[str, str],  # adm_id -> aez_code
        mode: str = 'pretrain'
    ):
        """
        Args:
            samples: List of (adm_id, year, country) tuples
            pairs: Dict mapping sample index to {hard_positives, soft_positives, weak_positives, negatives}
            features_dict: Pre-computed features for each sample
            aez_codes: AEZ code mapping
            mode: 'pretrain' or 'finetune'
        """
        self.samples = samples
        self.pairs = pairs
        self.features_dict = features_dict
        self.aez_codes = aez_codes
        self.mode = mode

        # Create sample index mapping
        self.sample_to_idx = {sample: i for i, sample in enumerate(samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a batch for contrastive learning.

        For pretrain mode:
        - anchor features
        - anchor metadata (year, country, aez_code)
        - positive indices (for batch collation)
        - negative indices (for batch collation)

        For finetune mode:
        - features
        - target
        - metadata
        """
        adm_id, year, country = self.samples[idx]

        if self.mode == 'pretrain':
            # Get features
            feat = self.features_dict[(adm_id, year, country)]

            # Get pairs
            pair_info = self.pairs.get(idx, {})

            return {
                'anchor': {
                    'X_ts': feat['X_ts'],
                    'X_static': feat['X_static'],
                    'year': year,
                    'country': country,
                    'aez_code': self.aez_codes.get(adm_id, 'UNKNOWN'),
                },
                'pairs': pair_info,
                'idx': idx,
            }
        else:  # finetune mode
            feat = self.features_dict[(adm_id, year, country)]
            return {
                'X_ts': feat['X_ts'],
                'X_static': feat['X_static'],
                'y': feat['y'],
                'year': year,
                'country': country,
                'adm_id': adm_id,
            }


class MultiCountryContrastiveDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multi-country contrastive pre-training.

    Workflow:
    1. Load data for each country separately
    2. Apply per-country temporal split
    3. Combine train/val/test sets across countries
    4. Load pre-computed AEZ codes and contrastive pairs
    5. Build features for all samples
    6. Create PyTorch Datasets and DataLoaders
    """

    def __init__(self, config: ContrastiveModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])

        # Data storage
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None

        # Feature storage
        self.features_dict = {}  # (adm_id, year, country) -> features
        self.aez_lookup = None  # DataFrame with AEZ codes
        self.aez_codes = {}  # adm_id -> aez_code

        # Tensor storage for fast indexing (used by collate function)
        self.train_X_ts = None  # (n_train, seq_len, n_features)
        self.train_X_static = None  # (n_train, n_static)
        self.val_X_ts = None
        self.val_X_static = None
        self.test_X_ts = None
        self.test_X_static = None

        # Pairs storage
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None

        # Normalization parameters (computed from train set)
        self.feature_norm_params = None
        self.y_mean = None
        self.y_std = None

        # Raw datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and features."""

        logger.info(f"\n[DataModule] Multi-country contrastive setup")
        logger.info(f"  Crop: {self.config.crop}")
        logger.info(f"  Countries: {self.config.countries}")
        logger.info(f"  Mode: {self.config.mode}")

        # Step 1: Load AEZ lookup
        self._load_aez_lookup()

        # Step 2: Load pre-computed pairs if available
        if self.config.pairs_cache_path:
            self._load_precomputed_pairs()
        else:
            logger.warning("No pairs cache path provided. Will need to compute pairs.")

        # Step 3: Load data per country with temporal splits
        self._load_multi_country_data()

        # Step 4: Build features for all samples
        self._build_features()

        # Step 5: Compute normalization from train set
        self._compute_normalization()

        # Step 6: Apply normalization
        self._apply_normalization()

        # Step 7: Create datasets
        self._create_datasets()

    def _load_aez_lookup(self):
        """Load pre-computed AEZ codes."""
        if self.config.aez_lookup_path:
            path = self.config.aez_lookup_path
        else:
            # Find latest AEZ lookup file
            aez_cache_dir = Path(__file__).parent.parent / 'pre-compute' / 'aez_cache'
            aez_files = list(aez_cache_dir.glob(f'{self.config.crop}_aez_*_lookup.csv'))
            if not aez_files:
                raise FileNotFoundError(
                    f"No AEZ lookup file found for {self.config.crop}. "
                    f"Please run precompute_aez.py first."
                )
            path = str(max(aez_files, key=os.path.getctime))

        logger.info(f"Loading AEZ lookup from {path}")
        self.aez_lookup = pd.read_csv(path)

        # Build adm_id -> aez_code mapping
        for _, row in self.aez_lookup.iterrows():
            self.aez_codes[row['adm_id']] = row['aez_code']

        logger.info(f"  Loaded {len(self.aez_codes)} AEZ codes")

    def _load_precomputed_pairs(self):
        """Load pre-computed contrastive pairs."""
        base_path = self.config.pairs_cache_path

        # Load train pairs
        train_path = f"{base_path}_train_pairs.pkl"
        if os.path.exists(train_path):
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)
            self.train_samples = train_data['samples']
            self.train_pairs = train_data['pairs']
            logger.info(f"  Loaded {len(self.train_samples)} train samples with pairs")

        # Load val pairs
        val_path = f"{base_path}_val_pairs.pkl"
        if os.path.exists(val_path):
            with open(val_path, 'rb') as f:
                val_data = pickle.load(f)
            self.val_samples = val_data['samples']
            self.val_pairs = val_data['pairs']
            logger.info(f"  Loaded {len(self.val_samples)} val samples with pairs")

        # Load test pairs
        test_path = f"{base_path}_test_pairs.pkl"
        if os.path.exists(test_path):
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            self.test_samples = test_data['samples']
            self.test_pairs = test_data['pairs']
            logger.info(f"  Loaded {len(self.test_samples)} test samples with pairs")

    def _load_multi_country_data(self):
        """Load data for each country with temporal splits."""
        # Handle 'all' countries
        if "all" in self.config.countries:
            if self.config.crop == "maize":
                countries = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR',
                            'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE']
            else:  # wheat
                countries = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
                            'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL',
                            'PL', 'PT', 'RO', 'SE']
        else:
            countries = self.config.countries

        # If we already have samples from pre-computed pairs, use those
        if self.train_samples is not None:
            logger.info(f"Using pre-computed samples ({len(self.train_samples)} train, "
                       f"{len(self.val_samples)} val, {len(self.test_samples)} test)")
            return

        # Otherwise, load fresh data
        logger.info("Loading fresh data with temporal splits...")

        train_samples = {}
        val_samples = {}
        test_samples = {}

        for country in countries:
            try:
                df_y, dfs_x = load_dfs_crop(self.config.crop, [country])
                if df_y is None or len(df_y) == 0:
                    logger.warning(f"No data for {self.config.crop}-{country}")
                    continue

                ds = CYDataset(self.config.crop, df_y, dfs_x)
                all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))

                # Compute temporal split
                splits = calculate_fixed_split(
                    all_years,
                    test_years=self.config.test_years,
                    val_years=self.config.val_years
                )

                # Extract samples
                all_indices = list(ds.indices())
                train_samples[country] = [(idx[0], idx[1], country)
                                        for idx in all_indices if idx[1] in splits['train_years']]
                val_samples[country] = [(idx[0], idx[1], country)
                                      for idx in all_indices if idx[1] in splits['val_years']]
                test_samples[country] = [(idx[0], idx[1], country]
                                       for idx in all_indices if idx[1] in splits['test_years']]

                logger.info(f"  {country}: {len(train_samples[country])} train, "
                           f"{len(val_samples[country])} val, {len(test_samples[country])} test")

            except Exception as e:
                logger.error(f"Error loading {self.config.crop}-{country}: {e}")

        # Combine across countries
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        for country in countries:
            self.train_samples.extend(train_samples.get(country, []))
            self.val_samples.extend(val_samples.get(country, []))
            self.test_samples.extend(test_samples.get(country, []))

        logger.info(f"Combined: {len(self.train_samples)} train, "
                   f"{len(self.val_samples)} val, {len(self.test_samples)} test")

    def _build_features(self):
        """Build features for all samples."""
        logger.info("Building features for all samples...")

        # Get unique countries
        all_samples = self.train_samples + self.val_samples + self.test_samples
        unique_countries = set(s[2] for s in all_samples)

        # Load data for each country
        country_data = {}
        for country in unique_countries:
            try:
                df_y, dfs_x = load_dfs_crop(self.config.crop, [country])
                country_data[country] = (CYDataset(self.config.crop, df_y, dfs_x), dfs_x)
            except Exception as e:
                logger.error(f"Error loading data for {country}: {e}")

        # Build features
        all_samples_list = []

        # Process train samples
        for adm_id, year, country in tqdm(self.train_samples, desc="Train features"):
            ds, dfs_x = country_data[country]
            feat = self._build_sample_features(ds, dfs_x, adm_id, year)
            self.features_dict[(adm_id, year, country)] = feat
            all_samples_list.append((adm_id, year, country))

        # Process val samples
        for adm_id, year, country in tqdm(self.val_samples, desc="Val features"):
            ds, dfs_x = country_data[country]
            feat = self._build_sample_features(ds, dfs_x, adm_id, year)
            self.features_dict[(adm_id, year, country)] = feat

        # Process test samples
        for adm_id, year, country in tqdm(self.test_samples, desc="Test features"):
            ds, dfs_x = country_data[country]
            feat = self._build_sample_features(ds, dfs_x, adm_id, year)
            self.features_dict[(adm_id, year, country)] = feat

        logger.info(f"  Built features for {len(self.features_dict)} samples")

    def _build_sample_features(self, ds, dfs_x, adm_id, year):
        """Build features for a single sample."""
        # Get the raw sample
        sample = None
        for i in range(len(ds)):
            idx = ds.indices()[i]
            if idx[0] == adm_id and idx[1] == year:
                sample = ds[i]
                break

        if sample is None:
            raise ValueError(f"Sample {adm_id}, {year} not found in dataset")

        # Build features
        X_ts, X_static, y, meta, mask = build_daily_input_sequence(
            ds, sample[KEY_LOC], sample[KEY_YEAR],
            aggregation=self.config.aggregation,
            data_fraction=self.config.data_fraction,
            use_sota_features=self.config.use_sota_features,
            include_spatial_features=self.config.include_spatial_features,
            lag_years=self.config.lag_years,
            weather_features_list=self.config.weather_features,
            use_gdd=self.config.use_gdd,
            use_heat_stress_days=self.config.use_heat_stress_days,
            use_rue=self.config.use_rue,
            use_farquhar=self.config.use_farquhar,
            crop=self.config.crop,
            multi_year_config=None,  # No multi-year during pre-training
        )

        return {
            'X_ts': X_ts,
            'X_static': X_static,
            'y': y,
            'lat': meta['lat'],
            'lon': meta['lon'],
            'mask': mask,
        }

    def _compute_normalization(self):
        """Compute normalization parameters from train set."""
        logger.info("Computing normalization parameters...")

        # Collect all training features
        train_X_ts = []
        train_X_static = []
        train_y = []

        for adm_id, year, country in self.train_samples:
            feat = self.features_dict[(adm_id, year, country)]
            train_X_ts.append(feat['X_ts'])
            train_X_static.append(feat['X_static'])
            train_y.append(feat['y'])

        train_X_ts = np.array(train_X_ts)
        train_X_static = np.array(train_X_static)
        train_y = np.array(train_y)

        # Y normalization
        self.y_mean = float(np.mean(train_y))
        self.y_std = float(np.std(train_y)) or 1.0

        # Feature normalization (simple z-score per feature)
        self.feature_norm_params = {}

        # Time series features
        n_ts_features = train_X_ts.shape[2]
        for i in range(n_ts_features):
            feat_data = train_X_ts[:, :, i].flatten()
            self.feature_norm_params[f'ts_{i}'] = {
                'mean': float(np.mean(feat_data)),
                'std': float(np.std(feat_data)) or 1.0
            }

        # Static features
        n_static_features = train_X_static.shape[1]
        for i in range(n_static_features):
            feat_data = train_X_static[:, i]
            self.feature_norm_params[f'static_{i}'] = {
                'mean': float(np.mean(feat_data)),
                'std': float(np.std(feat_data)) or 1.0
            }

        logger.info(f"  Y normalization: mean={self.y_mean:.4f}, std={self.y_std:.4f}")
        logger.info(f"  Feature normalization: {len(self.feature_norm_params)} features")

    def _apply_normalization(self):
        """Apply normalization to all features."""
        logger.info("Applying normalization...")

        for key in self.features_dict:
            feat = self.features_dict[key]

            # Normalize time series
            X_ts_norm = np.zeros_like(feat['X_ts'])
            for i in range(feat['X_ts'].shape[2]):
                param = self.feature_norm_params.get(f'ts_{i}', {'mean': 0.0, 'std': 1.0})
                X_ts_norm[:, :, i] = (feat['X_ts'][:, :, i] - param['mean']) / param['std']

            # Normalize static
            X_static_norm = np.zeros_like(feat['X_static'])
            for i in range(feat['X_static'].shape[0]):
                param = self.feature_norm_params.get(f'static_{i}', {'mean': 0.0, 'std': 1.0})
                X_static_norm[i] = (feat['X_static'][i] - param['mean']) / param['std']

            # Normalize target
            y_norm = (feat['y'] - self.y_mean) / self.y_std

            # Update features dict
            self.features_dict[key] = {
                **feat,
                'X_ts': X_ts_norm,
                'X_static': X_static_norm,
                'y': y_norm,
            }

    def _create_datasets(self):
        """Create PyTorch datasets."""
        logger.info("Creating datasets...")

        self.train_dataset = ContrastiveDataset(
            samples=self.train_samples,
            pairs=self.train_pairs or {},
            features_dict=self.features_dict,
            aez_codes=self.aez_codes,
            mode=self.config.mode
        )

        self.val_dataset = ContrastiveDataset(
            samples=self.val_samples,
            pairs=self.val_pairs or {},
            features_dict=self.features_dict,
            aez_codes=self.aez_codes,
            mode=self.config.mode
        )

        self.test_dataset = ContrastiveDataset(
            samples=self.test_samples,
            pairs=self.test_pairs or {},
            features_dict=self.features_dict,
            aez_codes=self.aez_codes,
            mode=self.config.mode
        )

        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")

        # Build tensor arrays for fast indexing during collation
        self._build_tensor_arrays()

    def _build_tensor_arrays(self):
        """Build tensor arrays from features_dict for fast indexing during collation."""
        logger.info("Building tensor arrays for fast indexing...")

        # Build train arrays
        if self.train_samples:
            train_X_ts_list = []
            train_X_static_list = []
            for adm_id, year, country in self.train_samples:
                feat = self.features_dict[(adm_id, year, country)]
                train_X_ts_list.append(feat['X_ts'])
                train_X_static_list.append(feat['X_static'])
            self.train_X_ts = np.array(train_X_ts_list, dtype=np.float32)
            self.train_X_static = np.array(train_X_static_list, dtype=np.float32)
            logger.info(f"  Train arrays: {self.train_X_ts.shape}, {self.train_X_static.shape}")

        # Build val arrays
        if self.val_samples:
            val_X_ts_list = []
            val_X_static_list = []
            for adm_id, year, country in self.val_samples:
                feat = self.features_dict[(adm_id, year, country)]
                val_X_ts_list.append(feat['X_ts'])
                val_X_static_list.append(feat['X_static'])
            self.val_X_ts = np.array(val_X_ts_list, dtype=np.float32)
            self.val_X_static = np.array(val_X_static_list, dtype=np.float32)
            logger.info(f"  Val arrays: {self.val_X_ts.shape}, {self.val_X_static.shape}")

        # Build test arrays
        if self.test_samples:
            test_X_ts_list = []
            test_X_static_list = []
            for adm_id, year, country in self.test_samples:
                feat = self.features_dict[(adm_id, year, country)]
                test_X_ts_list.append(feat['X_ts'])
                test_X_static_list.append(feat['X_static'])
            self.test_X_ts = np.array(test_X_ts_list, dtype=np.float32)
            self.test_X_static = np.array(test_X_static_list, dtype=np.float32)
            logger.info(f"  Test arrays: {self.test_X_ts.shape}, {self.test_X_static.shape}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn if self.config.mode == 'pretrain' else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn if self.config.mode == 'pretrain' else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn if self.config.mode == 'pretrain' else None,
        )

    def _collate_fn(self, batch):
        """
        Custom collate function for contrastive learning.

        Takes a batch of anchor samples and fetches their pre-computed
        positives and negatives using the pair indices.

        Returns:
            Dict with actual feature data for anchors, positives, and negatives.
        """
        batch_size = len(batch)

        # Determine which split we're working with (train/val/test)
        # by checking the first sample's index
        first_idx = batch[0]['idx']

        # Select the appropriate tensor arrays
        if first_idx < len(self.train_X_ts):
            all_X_ts = self.train_X_ts
            all_X_static = self.train_X_static
        elif first_idx < len(self.train_X_ts) + len(self.val_X_ts):
            offset = len(self.train_X_ts)
            all_X_ts = self.val_X_ts
            all_X_static = self.val_X_static
            # Adjust indices for val set
            for item in batch:
                item['idx'] -= offset
                for key in ['hard_positives', 'soft_positives', 'weak_positives', 'negatives']:
                    if key in item['pairs']:
                        item['pairs'][key] = [i - offset for i in item['pairs'][key]]
        else:
            offset = len(self.train_X_ts) + len(self.val_X_ts)
            all_X_ts = self.test_X_ts
            all_X_static = self.test_X_static
            # Adjust indices for test set
            for item in batch:
                item['idx'] -= offset
                for key in ['hard_positives', 'soft_positives', 'weak_positives', 'negatives']:
                    if key in item['pairs']:
                        item['pairs'][key] = [i - offset for i in item['pairs'][key]]

        # Collect anchor data
        anchor_X_ts = []
        anchor_X_static = []
        anchor_years = []
        anchor_countries = []
        anchor_aez_codes = []

        # Collect positive and negative features
        all_positives_X_ts = []
        all_positives_X_static = []
        all_negatives_X_ts = []
        all_negatives_X_static = []

        for item in batch:
            anchor = item['anchor']
            pairs = item['pairs']
            idx = item['idx']

            # Add anchor features
            anchor_X_ts.append(anchor['X_ts'])
            anchor_X_static.append(anchor['X_static'])
            anchor_years.append(anchor['year'])
            anchor_countries.append(anchor['country'])
            anchor_aez_codes.append(anchor['aez_code'])

            # Fetch positive features using indices
            pos_indices = []
            pos_indices.extend(pairs.get('hard_positives', []))
            pos_indices.extend(pairs.get('soft_positives', []))
            pos_indices.extend(pairs.get('weak_positives', []))

            # Clip indices to valid range
            valid_pos_indices = [i for i in pos_indices if 0 <= i < len(all_X_ts)]

            if valid_pos_indices:
                all_positives_X_ts.append(all_X_ts[valid_pos_indices])
                all_positives_X_static.append(all_X_static[valid_pos_indices])
            else:
                # If no valid positives, create empty arrays with correct shape
                all_positives_X_ts.append(np.zeros((0, all_X_ts.shape[1], all_X_ts.shape[2]), dtype=np.float32))
                all_positives_X_static.append(np.zeros((0, all_X_static.shape[1]), dtype=np.float32))

            # Fetch negative features using indices
            neg_indices = pairs.get('negatives', [])
            valid_neg_indices = [i for i in neg_indices if 0 <= i < len(all_X_ts)]

            if valid_neg_indices:
                all_negatives_X_ts.append(all_X_ts[valid_neg_indices])
                all_negatives_X_static.append(all_X_static[valid_neg_indices])
            else:
                # If no valid negatives, create empty arrays with correct shape
                all_negatives_X_ts.append(np.zeros((0, all_X_ts.shape[1], all_X_ts.shape[2]), dtype=np.float32))
                all_negatives_X_static.append(np.zeros((0, all_X_static.shape[1]), dtype=np.float32))

        # Convert anchor data to tensors
        anchor_X_ts = torch.tensor(np.array(anchor_X_ts), dtype=torch.float32)
        anchor_X_static = torch.tensor(np.array(anchor_X_static), dtype=torch.float32)

        # Stack positives and negatives (handle variable lengths)
        # For now, we'll just stack them as-is and handle padding in the model

        return {
            'anchor_X_ts': anchor_X_ts,
            'anchor_X_static': anchor_X_static,
            'anchor_years': anchor_years,
            'anchor_countries': anchor_countries,
            'anchor_aez_codes': anchor_aez_codes,
            'positives_X_ts': all_positives_X_ts,  # List of arrays, variable length
            'positives_X_static': all_positives_X_static,
            'negatives_X_ts': all_negatives_X_ts,
            'negatives_X_static': all_negatives_X_static,
        }


def calculate_fixed_split(all_years, test_years=3, val_years=2):
    """Calculate fixed temporal split."""
    all_years = sorted(all_years)

    if len(all_years) < test_years + val_years + 1:
        raise ValueError(
            f"Need at least {test_years + val_years + 1} years for split, got {len(all_years)}"
        )

    train_years = all_years[:-(test_years + val_years)]
    val_years = all_years[-(test_years + val_years):-test_years]
    test_years = all_years[-test_years:]

    return {
        'train_years': set(train_years),
        'val_years': set(val_years),
        'test_years': set(test_years),
        'total_years': len(all_years),
    }
