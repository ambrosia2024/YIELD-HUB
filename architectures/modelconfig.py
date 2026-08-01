from typing import Optional, Dict, List, Tuple, Callable, Union

from dataclasses import dataclass, field

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# %% Global constants
# Weather feature lists - used as defaults by ModelConfig.weather_features
WEATHER_FEATURES_BASE = ['tmin', 'tmax', 'tavg', 'prec', 'rad']
WEATHER_FEATURES_WITH_CWB = ['tmin', 'tmax', 'tavg', 'prec', 'cwb', 'rad']

# Remote sensing features
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']

STANDARD_STATIC_VARS = SOIL_PROPERTIES + LOCATION_PROPERTIES

# based on config.weather_features and config.time_series_vars properties
print(f"[Feature Config] Static vars ({len(STANDARD_STATIC_VARS)}): {STANDARD_STATIC_VARS}")

@dataclass
class TSTModelConfig:
    """Central configuration for time series forecasting model."""
    crop: str = "maize"
    country: str = "NL"
    model_type: str = "autoformer"
    aggregation: str = "dekad"
    data_fraction: float = 1.0  # Fraction of available data to use (default: 1.0 = all data)
    use_sota_features: bool = False
    include_spatial_features: bool = False
    use_residual_trend: bool = True
    lag_years: int = 1
    load_checkpoint: Optional[str] = None
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    test_years: int = 3
    use_cwb_feature: bool = False
    drop_tavg: bool = False
    use_recursive_lags: bool = False
    use_gdd: bool = False
    use_heat_stress_days: bool = False
    use_rue: bool = False
    use_farquhar: bool = False
    use_revin: bool = False
    use_positional_encoding: bool = True  # Add sinusoidal positional encoding to preserve temporal ordering
    use_exponential_weighting: bool = False  # Enable exponential sample weighting for non-stationarity
    exponential_tau: float = 10.0  # Decay constant for exponential weighting (higher = slower decay)
    results_dir: str = "checkpoints/results"
    lr_scheduler_lambda: Optional[Callable] = None
    patchtst_d_model: int = 64
    patchtst_num_attention_heads: int = 4
    patchtst_ffn_dim: int = 256
    patchtst_num_layers: int = 3
    patchtst_dropout: float = 0.1
    # Multi-year context features
    multi_year_summaries: bool = False  # Enable multi-year summary features from previous growing seasons
    multi_year_window: int = 1  # Years of historical context (1=T-1, 2=T-1,T-2, 3=T-1,T-2,T-3)
    multi_year_features: List[str] = field(default_factory=lambda: ['weather'])  # Which features to summarize
    # Tokenization ablation: fixed average pooling preprocessing
    if_tokenize: bool = False  # Enable fixed average pooling tokenization (ablation study)
    tokenize_kernel: int = 7  # Kernel size for average pooling (default: 7 for weekly aggregation)
    tokenize_stride: int = 7  # Stride for average pooling (default: 7 for non-overlapping weekly tokens)
    # Parallel feature building for HPO speedup
    use_parallel: bool = False  # Enable parallel feature building using joblib (default: False for backward compatibility)
    # WFAN (Frequency-Adaptive Normalization) parameters
    use_wfan: bool = False  # Enable WFAN frequency-adaptive normalization
    wfan_k: int = 2  # Number of dominant frequency components to remove (K)
    wfan_lambda: float = 1.0  # Loss balancing coefficient for pattern-adaptive prediction (λ)
    # Quantile regression / Uncertainty quantification parameters
    loss_type: str = 'mse'  # Loss function: 'mse' for point prediction, 'pinball' for quantile regression
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])  # Quantiles to predict

    @property
    def seq_len(self):
        # Sequence length derived from aggregation frequency.
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        # Compute the list of weather features based on config flags.
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        # Full list of time series variables including remote sensing.
        return self.weather_features + REMOTE_SENSING_FEATURES

    @property
    def multi_year_config(self) -> Optional[Dict]:
        """Build multi-year config dict for feature engineering."""
        if self.multi_year_summaries:
            return {
                'enabled': True,
                'window': self.multi_year_window,
                'features': self.multi_year_features,
            }
        return None

    def _compute_expected_static_features(self) -> int:
        # Compute the total expected static feature count from the current config.
        # Validates if build_daily_input_sequence() is producing the right number of features.
        # Checks for any mismatch at the feature creation step.

        n_soil = len(SOIL_PROPERTIES)
        n_location = len(LOCATION_PROPERTIES)
        # Crop calendar: sos_date and eos_date use cyclic encoding (2 each), other dates use 1 feature each
        n_crop = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop += 2  # sin and cos
            else:
                n_crop += 1
        n_spatial = 2 if self.include_spatial_features else 0
        n_lagged = self.lag_years
        # Heat stress: 7 scalar features when enabled
        n_heat_stress = 7 if self.use_heat_stress_days else 0

        # Multi-year summaries
        n_multi_year = 0
        if self.multi_year_summaries:
            from cybench.process.featureEngineering import MultiYearFeatureEngineer
            feature_types = (['all'] if 'all' in self.multi_year_features
                           else self.multi_year_features)
            n_multi_year = len(MultiYearFeatureEngineer.get_feature_names(
                self.multi_year_window, feature_types
            ))

        return n_soil + n_location + n_crop + n_spatial + n_lagged + n_heat_stress + n_multi_year


@dataclass
class LinearModelConfig:
    """Central configuration for time series forecasting model."""
    crop: str = "maize"
    country: str = "NL"
    model_type: str = "nlinear"
    aggregation: str = "dekad"
    data_fraction: float = 1.0  # Fraction of available data to use (default: 1.0 = all data)
    use_sota_features: bool = False
    include_spatial_features: bool = False
    use_residual_trend: bool = True
    lag_years: int = 1
    load_checkpoint: Optional[str] = None
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    test_years: int = 3
    use_cwb_feature: bool = False
    drop_tavg: bool = False
    use_revin: bool = False
    use_recursive_lags: bool = False
    use_gdd: bool = False
    use_heat_stress_days: bool = False
    use_rue: bool = False
    use_farquhar: bool = False
    use_positional_encoding: bool = True  # Add sinusoidal positional encoding to preserve temporal ordering
    use_exponential_weighting: bool = False  # Enable exponential sample weighting for non-stationarity
    exponential_tau: float = 10.0  # Decay constant for exponential weighting (higher = slower decay)
    results_dir: str = "checkpoints/results"
    lr_scheduler_lambda: Optional[Callable] = None
    xlinear_hidden_size: int = 64
    xlinear_temporal_ff: int = 128
    xlinear_channel_ff: int = 16
    xlinear_dropout: float = 0.1
    # Multi-year context features
    multi_year_summaries: bool = False  # Enable multi-year summary features from previous growing seasons
    multi_year_window: int = 1  # Years of historical context (1=T-1, 2=T-1,T-2, 3=T-1,T-2,T-3)
    multi_year_features: List[str] = field(default_factory=lambda: ['weather'])  # Which features to summarize
    # Tokenization ablation: fixed average pooling preprocessing
    if_tokenize: bool = False  # Enable fixed average pooling tokenization (ablation study)
    tokenize_kernel: int = 7  # Kernel size for average pooling (default: 7 for weekly aggregation)
    tokenize_stride: int = 7  # Stride for average pooling (default: 7 for non-overlapping weekly tokens)
    # Parallel feature building for HPO speedup
    use_parallel: bool = False  # Enable parallel feature building using joblib (default: False for backward compatibility)
    # WFAN (Frequency-Adaptive Normalization) parameters
    use_wfan: bool = False  # Enable WFAN frequency-adaptive normalization
    wfan_k: int = 2  # Number of dominant frequency components to remove (K)
    wfan_lambda: float = 1.0  # Loss balancing coefficient for pattern-adaptive prediction (λ)
    # Quantile regression / Uncertainty quantification parameters
    loss_type: str = 'mse'  # Loss function: 'mse' for point prediction, 'pinball' for quantile regression
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])  # Quantiles to predict

    @property
    def seq_len(self):
        """Sequence length derived from aggregation frequency."""
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        """
        Compute the list of weather features based on config flags.
        """
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        """Full list of time series variables including remote sensing."""
        return self.weather_features + REMOTE_SENSING_FEATURES

    @property
    def multi_year_config(self) -> Optional[Dict]:
        """Build multi-year config dict for feature engineering."""
        if self.multi_year_summaries:
            return {
                'enabled': True,
                'window': self.multi_year_window,
                'features': self.multi_year_features,
            }
        return None

    def _compute_expected_static_features(self) -> int:
        """
        Compute the total expected static feature count from the current config.
        """
        n_soil = len(SOIL_PROPERTIES)
        n_location = len(LOCATION_PROPERTIES)
        n_crop = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop += 2  # sin and cos
            else:
                n_crop += 1
        n_spatial = 2 if self.include_spatial_features else 0
        n_lagged = self.lag_years

        # Heat stress: 7 scalar features when enabled
        n_heat_stress = 7 if self.use_heat_stress_days else 0

        # Multi-year summaries
        n_multi_year = 0
        if self.multi_year_summaries:
            from cybench.process.featureEngineering import MultiYearFeatureEngineer
            feature_types = (['all'] if 'all' in self.multi_year_features
                           else self.multi_year_features)
            n_multi_year = len(MultiYearFeatureEngineer.get_feature_names(
                self.multi_year_window, feature_types
            ))

        return n_soil + n_location + n_crop + n_spatial + n_lagged + n_heat_stress + n_multi_year


@dataclass
class TabModelConfig:
    """Central configuration for tabular foundation models (TabPFN, TabICL, TabDPT).

    Designed to be fully coherent with TSTModelConfig for direct comparison.
    The main difference is that time series features are flattened into tabular format.
    """
    crop: str = "maize"
    country: str = "NL"
    model_type: str = "tabpfn"  # tabpfn, tabicl, tabdpt
    aggregation: str = "dekad"
    data_fraction: float = 1.0  # Fraction of available data to use (default: 1.0 = all data)
    use_sota_features: bool = False
    include_spatial_features: bool = False
    use_residual_trend: bool = True
    lag_years: int = 1
    load_checkpoint: Optional[str] = None
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4  # Not used by tabular models but kept for coherence
    weight_decay: float = 1e-5  # Not used by tabular models but kept for coherence
    max_epochs: int = 50
    test_years: int = 3
    use_cwb_feature: bool = False
    drop_tavg: bool = False
    use_recursive_lags: bool = False
    use_gdd: bool = False
    use_heat_stress_days: bool = False
    use_rue: bool = False
    use_farquhar: bool = False
    use_positional_encoding: bool = True
    use_exponential_weighting: bool = False
    exponential_tau: float = 10.0
    results_dir: str = "checkpoints/results"
    # Tabular model specific parameters
    preprocess: str = "none"  # Preprocessing mode: "none" or "sklearn"
    max_train_samples: Optional[int] = None  # Subsample training data for TabPFN context limits
    subsample: str = "random"  # Subsampling method: "random" or "quantile"
    subsample_bins: int = 10  # Number of bins for quantile subsampling
    allow_cpu_fallback: bool = False  # Allow CPU fallback if GPU fails
    predict_batch_size: int = 256  # Batch size for prediction
    min_train_samples: int = 100  # Minimum samples for TabDPT (pads if needed)
    multi_year_summaries: bool = False
    multi_year_window: int = 1
    multi_year_features: List[str] = field(default_factory=lambda: ['weather'])

    @property
    def seq_len(self):
        """Sequence length derived from aggregation frequency."""
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        """Compute the list of weather features based on config flags."""
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        """Full list of time series variables including remote sensing."""
        return self.weather_features + REMOTE_SENSING_FEATURES

    @property
    def multi_year_config(self) -> Optional[Dict]:
        """Build multi-year config dict for feature engineering."""
        if self.multi_year_summaries:
            return {
                'enabled': True,
                'window': self.multi_year_window,
                'features': self.multi_year_features,
            }
        return None

    def _compute_expected_static_features(self) -> int:
        """Compute the total expected static feature count from the current config."""
        n_soil = len(SOIL_PROPERTIES)
        n_location = len(LOCATION_PROPERTIES)
        n_crop = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop += 2  # sin and cos
            else:
                n_crop += 1
        n_spatial = 2 if self.include_spatial_features else 0
        n_lagged = self.lag_years
        n_heat_stress = 7 if self.use_heat_stress_days else 0

        # Multi-year summaries
        n_multi_year = 0
        if self.multi_year_summaries:
            from cybench.process.featureEngineering import MultiYearFeatureEngineer
            feature_types = (['all'] if 'all' in self.multi_year_features
                           else self.multi_year_features)
            n_multi_year = len(MultiYearFeatureEngineer.get_feature_names(
                self.multi_year_window, feature_types
            ))

        return n_soil + n_location + n_crop + n_spatial + n_lagged + n_heat_stress + n_multi_year

    @property
    def flattened_feature_size(self) -> int:
        """Calculate total flattened feature size for tabular models.

        For tabular models, we flatten time series features:
        - Time series: seq_len × len(time_series_vars)
        - Static: _compute_expected_static_features()
        - Additional engineered features (GDD, RUE, Farquhar) if enabled
        """
        n_ts = self.seq_len * len(self.time_series_vars)
        n_static = self._compute_expected_static_features()

        # Additional engineered features that are time series
        n_engineered = 0
        if self.use_gdd:
            n_engineered += self.seq_len  # GDD per time step
        if self.use_rue:
            n_engineered += self.seq_len  # RUE per time step
        if self.use_farquhar:
            n_engineered += self.seq_len  # Farquhar per time step

        return n_ts + n_static + n_engineered
