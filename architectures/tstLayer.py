import sys
import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

from transformers import (
    AutoformerModel as HFAutoformerModel,
    AutoformerConfig,
    PatchTSTModel as HFPatchTSTModel, PatchTSTConfig,
    InformerModel as HFInformerModel, InformerConfig,
    TimeSeriesTransformerModel as HFTimeSeriesTransformerModel, TimeSeriesTransformerConfig,
)

# Only for TSMixer
TimeSeriesMixerModel = None
TimeSeriesMixerForPrediction = None
TimeSeriesMixerConfig = None
try:
    from transformers.models.patchtsmixer.modeling_patchtsmixer import (
        PatchTSMixerModel as TimeSeriesMixerModel,
        PatchTSMixerForPrediction as TimeSeriesMixerForPrediction,
        PatchTSMixerConfig as TimeSeriesMixerConfig,
    )
except ImportError:
    try:
        from transformers.models.time_series_transformer import (
            TimeSeriesMixerModel,
            TimeSeriesMixerForPrediction,
            TimeSeriesMixerConfig,
        )
    except ImportError:
        try:
            from transformers import (
                PatchTSMixerModel as TimeSeriesMixerModel,
                PatchTSMixerForPrediction as TimeSeriesMixerForPrediction,
                PatchTSMixerConfig as TimeSeriesMixerConfig,
            )
        except ImportError:
            try:
                from transformers import (
                    TimeSeriesMixerModel,
                    TimeSeriesMixerForPrediction,
                    TimeSeriesMixerConfig,
                )
            except ImportError:
                pass 

if TimeSeriesMixerForPrediction is None:
    logging.warning("TSMixer/PatchTSMixer not available in this transformers version. "
                   "'tsmixer' model_type will raise an error at runtime.")

# Custom Classes and functions
from trendLayer import TrendModel
from biasCorrectionLayer import BiasCorrection
from modelconfig import TSTModelConfig

sys.path.append('../process/')
from validateModel import ModelMetrics
from featureEngineering import _get_static_feature_names

# Global variables
SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']
print(f"[Feature Config] SOTA Temporal vars ({len(SOTA_TEMPORAL_VARS_LIST)}): {SOTA_TEMPORAL_VARS_LIST}")

class BaseTimeSeriesModel(ABC, pl.LightningModule):
    """
    Abstract base for all time series forecasting architectures.

    Provides:
    - Feature normalisation (z-score, from training stats)
    - Residual trend learning (per-location OLS, fitted in on_train_start)
    - Weighted loss computation (per-sample validity weighting)
    - Shared train / val / test step logic

    Subclasses must implement:
    - _build_model() -> nn.Module
    - forward(x_ts, x_static) -> Tensor  (shape: batch,)
    """

    def __init__(self, config: TSTModelConfig, lr: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.trend_model = TrendModel()
        # Bias correction: automatic when use_recursive_lags=True
        # Fits on validation predictions, applies to test predictions for recursive lags
        self.bias_correction: Optional[BiasCorrection] = (
            BiasCorrection(method='linear') if config.use_recursive_lags else None
        )
        if self.bias_correction:
            logging.info("[BiasCorrection] ENABLED (automatic with use_recursive_lags=True)")
        self.feature_norm_params: Optional[Dict] = None
        # Track most recent training year for exponential weighting
        self._most_recent_year: Optional[int] = None
        # Flag for first batch logging
        self._weight_log_done = False

        # Log exponential weighting configuration
        if config.use_exponential_weighting:
            print(f"[Exponential Weighting] ENABLED - tau={config.exponential_tau}")
        else:
            print(f"[Exponential Weighting] DISABLED")

        # Tokenization configuration
        if config.if_tokenize:
            print(f"[Tokenization] ENABLED - fixed AvgPool1d (kernel={config.tokenize_kernel}, stride={config.tokenize_stride})")
            self.tokenizer = nn.AvgPool1d(
                kernel_size=config.tokenize_kernel,
                stride=config.tokenize_stride,
            )
            self._tokenized_seq_len = self._compute_tokenized_seq_len(
                config.seq_len, config.tokenize_kernel, config.tokenize_stride
            )
            print(f"[Tokenization] Sequence length: {config.seq_len} -> {self._tokenized_seq_len}")
        else:
            print(f"[Tokenization] DISABLED")
            self.tokenizer = None
            self._tokenized_seq_len = None

        # Uses config.time_series_vars property instead of global WEATHER_FEATURES
        # This ensures feature count matches the actual features being extracted
        use_sota = config.use_sota_features
        n_domain_ts = sum([config.use_gdd, config.use_rue, config.use_farquhar])
        self.n_ts_features = (
            len(config.time_series_vars)
            + n_domain_ts
            + (len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0)
        )
        # When using SOTA features, they're passed through past_values, not as time features
        # This prevents double-counting: num_time_features should be 0 in that case
        self.num_time_features = 0  # No separate time feature embedding needed

        # Static feature count — must match _compute_expected_static_features()
        include_spatial = config.include_spatial_features
        lag_years = config.lag_years
        n_heat_stress = 7 if config.use_heat_stress_days else 0

        # Compute n_crop_calendar dynamically from CROP_CALENDAR_DATES
        # using the same cyclic-encoding logic as _compute_expected_static_features().
        n_crop_calendar = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop_calendar += 2  # sin and cos for cyclic encoding
            else:
                n_crop_calendar += 1

        # Multi-year summaries - must match the config's calculation
        n_multi_year = 0
        if config.multi_year_summaries:
            from cybench.process.featureEngineering import MultiYearFeatureEngineer
            feature_types = (['all'] if 'all' in config.multi_year_features
                           else config.multi_year_features)
            n_multi_year = len(MultiYearFeatureEngineer.get_feature_names(
                config.multi_year_window, feature_types
            ))

        self.n_static_features = (
            len(SOIL_PROPERTIES) + len(LOCATION_PROPERTIES) + n_crop_calendar
            + (2 if include_spatial else 0)
            + lag_years
            + n_heat_stress
            + n_multi_year
        )

        print(f"[Model] TS features={self.n_ts_features}, Static features={self.n_static_features}")

        # Initialize positional encoding for temporal ordering (optional, enabled by default)
        # The max_len is based on aggregation to ensure proper coverage of the season
        self.d_model = 64  # Standard hidden size across all transformer models
        max_len_map = {"daily": 365, "weekly": 52, "dekad": 36}
        pe_max_len = max_len_map.get(config.aggregation, 365)
        self.use_positional_encoding = config.use_positional_encoding

        if self.use_positional_encoding:
            self.positional_encoding = TemporalPositionalEncoding(
                d_model=self.d_model,
                max_len=pe_max_len,
                dropout=0.1
            )
            logging.info(f"[Positional Encoding] Enabled with d_model={self.d_model}, max_len={pe_max_len}")
        else:
            self.positional_encoding = None
            logging.info("[Positional Encoding] Disabled")

        # Flag for tracking model build completion (used by _verify_mask_is_used)
        self._model_ready = False

        self.base_model = self._build_model()

        # self.criterion (nn.MSELoss) is intentionally absent. Training steps use _compute_weighted_loss() which calls F.mse_loss(reduction='none') to get per-sample losses before weighting.
        # Exclude NRMSE from training metrics since targets are in z-score space (mean≈0 causes division issues)

        # Initialize metrics with quantiles if using pinball loss
        metrics_quantiles = config.quantiles if config.loss_type == 'pinball' else None
        if config.loss_type == 'pinball':
            print(f"[Quantile Regression] ENABLED - quantiles: {config.quantiles}")

        self.train_metrics = ModelMetrics(prefix="train", include_nrmse=False, quantiles=metrics_quantiles)
        self.val_metrics = ModelMetrics(prefix="val", quantiles=metrics_quantiles)
        self.test_metrics = ModelMetrics(prefix="test", quantiles=metrics_quantiles)

    @property
    def output_dim(self) -> int:
        """Output dimension: 1 for MSE loss, n_quantiles for pinball loss."""
        if self.config.loss_type == 'pinball':
            return len(self.config.quantiles)
        return 1

    @staticmethod
    def _get_standardized_context_length(seq_len: int, lags_sequence: List[int]) -> int:
        """
        Calculate standardized context length for all models.

        This ensures fair comparison across architectures by using the same
        context length calculation: seq_len - max(lags_sequence)

        Args:
            seq_len: Total sequence length
            lags_sequence: List of lag values

        Returns:
            Context length that satisfies: context_length + max(lags) <= seq_len
        """
        return seq_len - max(lags_sequence)

    @staticmethod
    def _compute_tokenized_seq_len(seq_len: int, kernel: int, stride: int) -> int:
        """
        Compute output sequence length after 1D average pooling.

        Formula: floor((seq_len - kernel) / stride) + 1

        Args:
            seq_len: Input sequence length
            kernel: Kernel size for pooling
            stride: Stride for pooling

        Returns:
            Output sequence length after pooling
        """
        return (seq_len - kernel) // stride + 1

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Instantiate and return the underlying HuggingFace model.

        Returns:
            Configured HuggingFace model instance
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_ts: Time series features of shape (batch, seq_len, n_ts_features)
            x_static: Static features of shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) indicating valid timesteps

        Returns:
            Predictions of shape (batch,) in normalised z-score space
        """
        raise NotImplementedError


    def _get_static_feature_names(self) -> List[str]:
        """
        Thin wrapper so _normalize_and_impute_static() can look up norm params.
        """
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
            self.config.use_heat_stress_days,
            multi_year_config=self.config.multi_year_config if hasattr(self.config, 'multi_year_config') else None,
        )

    def _apply_tokenization(self, x_ts: torch.Tensor,
                            observed_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply fixed average pooling tokenization to time series data.

        This is used for ablation studies to understand if reducing sequence length
        via fixed (non-learned) pooling helps or hurts transformer performance.

        Args:
            x_ts: Time series tensor of shape (batch, seq_len, n_features)
            observed_mask: Optional boolean mask of shape (batch, seq_len)

        Returns:
            Tuple of (tokenized_x_ts, tokenized_observed_mask)
            - tokenized_x_ts: Tokenized time series of shape (batch, new_seq_len, n_features)
            - tokenized_observed_mask: Mask adjusted to tokenized sequence, or None if input was None
        """
        if self.tokenizer is None:
            return x_ts, observed_mask

        B, T, C = x_ts.shape

        # Transpose to (B, C, T) for AvgPool1d which operates on the last dimension
        x_ts_t = x_ts.transpose(1, 2)  # (B, C, T)

        # Apply average pooling
        x_tokenized_t = self.tokenizer(x_ts_t)  # (B, C, new_T)

        # Transpose back to (B, new_T, C)
        x_tokenized = x_tokenized_t.transpose(1, 2)  # (B, new_T, C)

        # Adjust observed mask if provided
        if observed_mask is not None:
            # Reshape mask to (B, T, 1) for pooling
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (B, T, 1)
            mask_t = mask_expanded.transpose(1, 2)  # (B, 1, T)
            # Apply same pooling - any valid observation in the pool makes the token valid
            mask_pooled_t = self.tokenizer(mask_t)  # (B, 1, new_T)
            # Convert back to boolean: token is valid if any observation in pool was valid
            mask_pooled = (mask_pooled_t.squeeze(1) > 0).bool()  # (B, new_T)
            return x_tokenized, mask_pooled

        return x_tokenized, None

    def _init_temporal_attention(self, d_model: int) -> None:
        """
        Initialize temporal attention module with known d_model.

        Called during _build_model() by models that support attention-based pooling (e.g., TimeXer, iTransformer). The current
        implementation uses a simpler pooling strategy-

        Args:
            d_model: The hidden state dimension (e.g., 64 for most models)
        """
        # TimeXer and iTransformer models are using custom pooling strategie (channel projection, patch-based pooling) and don't need temporal attention pooling.
        pass

    def _extract_hidden_state(self, outputs) -> torch.Tensor:
        """
        Extract last hidden state from HuggingFace model outputs.

        Handles different output formats across model architectures by trying
        common attribute names in order of preference.

        Args:
            outputs: HuggingFace model output object

        Returns:
            Hidden state tensor

        Raises:
            ValueError: If no hidden state can be found
        """
        # Try single tensor attributes first
        for attr in ("encoder_last_hidden_state", "last_hidden_state"):
            val = getattr(outputs, attr, None)
            if val is not None:
                return val

        # encoder_hidden_states is a tuple — take the last layer
        val = getattr(outputs, "encoder_hidden_states", None)
        if val is not None:
            return val[-1]

        # No hidden state found — provide helpful error message
        tensor_attrs = [a for a in dir(outputs) if hasattr(getattr(outputs, a, None), 'shape')]
        raise ValueError(
            f"Could not extract hidden state from model outputs. "
            f"Available tensor attributes: {tensor_attrs}"
        )

    def _pool_hidden_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden state to (batch, features) regardless of input shape.

        Handles different hidden state formats across model architectures:
          - (B, seq_len, d_model)          → mean over seq_len → (B, d_model)
          - (B, n_channels, n_patches, d)  → mean over patches, flatten channels → (B, n_channels * d)
          - (B, d_model)                   → already pooled → (B, d_model)

        Args:
            h: Hidden state tensor with shape (B, ...) where last dim is features

        Returns:
            Pooled tensor with shape (B, pooled_dim)

        Raises:
            ValueError: If tensor has unexpected number of dimensions
        """
        if h.dim() == 2:
            # Already pooled: (B, d_model)
            return h
        elif h.dim() == 3:
            # Standard transformer output: (B, seq_len, d_model)
            # Pool over sequence length
            return h.mean(dim=1)  # (B, d_model)
        elif h.dim() == 4:
            # PatchTST/TSMixer multivariate output: (B, n_channels, n_patches, d_model)
            # Pool over patches first, then flatten channels and d_model
            h = h.mean(dim=2)           # (B, n_channels, d_model)
            B = h.shape[0]
            return h.reshape(B, -1)     # (B, n_channels * d_model)
        else:
            raise ValueError(
                f"Unexpected hidden state shape: {h.shape} "
                f"(expected 2D, 3D, or 4D tensor, got {h.dim()}D)"
            )

    def _apply_positional_encoding(self, h: torch.Tensor,
                                    observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to hidden state if enabled.

        This method handles different tensor shapes:
          - (B, seq_len, d_model) → apply PE to seq_len dimension
          - (B, d_model) → no PE needed (already pooled)
          - Other shapes → return as-is (PatchTST special handling)

        Args:
            h: Hidden state tensor
            observed_mask: Optional boolean mask for valid timesteps

        Returns:
            Hidden state with positional encoding added (if applicable)
        """
        if not self.use_positional_encoding or self.positional_encoding is None:
            return h

        if h.dim() == 3:
            # Standard transformer output: (B, seq_len, d_model)
            return self.positional_encoding(h, observed_mask)
        elif h.dim() == 2:
            # Already pooled: (B, d_model) - no PE needed
            return h
        else:
            # 4D tensors (PatchTST multivariate) - skip PE for now
            # Could be extended to add PE per-patch if needed
            return h

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z-score normalise each time series feature using training statistics. Raises RuntimeError / KeyError early if params are missing — intentional,
        to catch silent data pipeline failures before they corrupt training.

        Re-zero padded positions after normalization to prevent spurious signal from padded zeros being normalized to (0 - mean)/std which is non-zero.
        """
        # Try to get feature_norm_params from datamodule
        if self.feature_norm_params is None:
            # Try stored datamodule first (for inference without trainer)
            dm = None
            if hasattr(self, '_datamodule') and self._datamodule is not None:
                dm = self._datamodule
            else:
                # Try trainer.datamodule as fallback (for training/with trainer)
                try:
                    dm = self.trainer.datamodule
                except (RuntimeError, AttributeError):
                    raise RuntimeError("feature_norm_params not set and no datamodule available. "
                                     "Model should store datamodule during training or have datamodule passed to predict().")

            dm_params = dm.feature_norm_params
            if dm_params is not None:
                self.feature_norm_params = dm_params
            else:
                raise RuntimeError("feature_norm_params not set in model or datamodule. "
                                   "Ensure datamodule.setup() has been called.")

        # Use config.weather_features instead of global WEATHER_FEATURES
        names = [f'weather_{f}' for f in self.config.weather_features]
        # Domain time series channels — must match order in _get_ts_feature_names()
        if self.config.use_gdd:
            names.append('domain_cum_gdd')
        if self.config.use_rue:
            names.append('domain_rue_index')
        if self.config.use_farquhar:
            names.append('domain_farquhar_proxy')
        names += [f'rs_{f}' for f in REMOTE_SENSING_FEATURES]
        if self.config.use_sota_features:
            names += [f'sota_{n}' for n in SOTA_TEMPORAL_VARS_LIST]

        if len(names) != x_ts.shape[2]:
            raise ValueError(f"TS feature name count ({len(names)}) != "
                             f"tensor dim ({x_ts.shape[2]})")

        x = x_ts.clone()
        for i, name in enumerate(names):
            key = f"ts_{name}"
            if key not in self.feature_norm_params:
                raise KeyError(f"Missing norm params for TS feature '{name}'")
            p = self.feature_norm_params[key]
            # Protect against zero/near-zero std to prevent inf/NaN
            if p['std'] < 1e-8:
                # Feature has no variance - set to 0 (mean in z-score space)
                x[:, :, i] = torch.zeros_like(x_ts[:, :, i])
            else:
                x[:, :, i] = (x_ts[:, :, i] - p['mean']) / p['std']
            # Handle both NaN AND inf (can be produced by division)
            x[:, :, i] = torch.nan_to_num(x[:, :, i], nan=0.0, posinf=0.0, neginf=0.0)

        # Re-zero padded positions after normalization
        # Without this, padded zeros become (0 - mean)/std which is non-zero
        # and participates in attention computation as spurious signal
        if observed_mask is not None:
            # observed_mask: (batch, seq_len), x: (batch, seq_len, features)
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = x * mask_expanded  # Zero out padded positions

        return x

    def _apply_revin_normalization(self, x_ts: torch.Tensor,
                                   observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Reversible Instance Normalization (RevIN) to time series data.

        RevIN computes per-instance, per-channel statistics:
        - mean = sum(x * mask) / sum(mask)
        - std = sqrt(sum((x - mean)^2 * mask) / sum(mask) + eps)

        Args:
            x_ts: Time series tensor of shape (batch, seq_len, n_features)
            observed_mask: Boolean mask of shape (batch, seq_len) indicating valid timesteps

        Returns:
            Normalized tensor of same shape as x_ts
        """
        B, T, C = x_ts.shape
        eps = 1e-8

        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)  # (B, T, 1)
            valid_counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)

            # Compute instance mean per channel, considering only valid positions
            instance_mean = (x_ts * mask_f).sum(dim=1, keepdim=True) / valid_counts  # (B, 1, C)

            # Compute instance std per channel
            sq_dev = ((x_ts - instance_mean) ** 2 * mask_f).sum(dim=1, keepdim=True)
            instance_std = torch.sqrt(sq_dev / valid_counts + eps)  # (B, 1, C)
        else:
            # No mask: use all positions
            instance_mean = x_ts.mean(dim=1, keepdim=True)  # (B, 1, C)
            instance_std = x_ts.std(dim=1, keepdim=True) + eps  # (B, 1, C)

        # Normalize
        x_norm = (x_ts - instance_mean) / instance_std

        # Re-zero masked positions after normalization
        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)  # (B, T, 1)
            x_norm = x_norm * mask_f

        return x_norm

    def _normalize_and_impute_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalise static features then impute NaN → 0.0.

        ORDER IS CRITICAL — normalise first, impute second:
          1. z = (x - μ) / σ   →  puts data in z-score space
          2. NaN → 0.0          →  0.0 IS the mean in z-score space

        Reversing the order would impute NaN to 0.0 in original space, which
        normalises to (0 - μ)/σ — typically far below the mean, not at it.
        """
        if self.feature_norm_params is None:
            return x_static

        names = self._get_static_feature_names()
        x = x_static.clone()

        for i, name in enumerate(names):
            if i >= x.shape[1]:
                break
            key = f"static_{name}"
            if key not in self.feature_norm_params:
                continue
            p = self.feature_norm_params[key]
            # Protect against zero/near-zero std to prevent inf/NaN
            if p['std'] < 1e-8:
                # Feature has no variance - skip normalization, set to 0 (mean in z-score space)
                x[:, i] = torch.zeros_like(x_static[:, i])
            else:
                x[:, i] = (x_static[:, i] - p['mean']) / p['std']
            # Handle both NaN and inf (from division by near-zero std)
            x[:, i] = torch.nan_to_num(x[:, i], nan=0.0, posinf=0.0, neginf=0.0)

        return x

    def on_train_start(self):
        """
        Fit per-location OLS trend lines and cache (slope, intercept).

        Trend decomposition: yield = trend(location, year) + residual. The model learns residuals and trend is added back at inference.
        Also copies feature_norm_params from the DataModule and builds spatial index for nearest-neighbor trend estimation.

        Added mask verification to ensure HuggingFace models correctly use past_observed_mask to zero out padded positions in attention.
        """
        dm = self.trainer.datamodule
        # Store datamodule reference for inference without trainer
        self._datamodule = dm
        train_y_orig = dm.train_ds.y.numpy() * dm.y_std + dm.y_mean

        train_items = [
            {KEY_LOC: dm.train_ds.adm_ids[i],
             KEY_YEAR: int(dm.train_ds.years[i]),
             KEY_TARGET: float(train_y_orig[i])}
            for i in range(len(train_y_orig))
        ]
        self.trend_model.fit(train_items)
        self.feature_norm_params = dm.feature_norm_params

        # Initialize validation prediction storage for bias correction fitting
        if self.bias_correction is not None:
            self._val_preds = []
            self._val_targets = []
            logging.info("[BiasCorrection] Initialized validation prediction storage")

        # Track most recent training year for exponential weighting
        if self.config.use_exponential_weighting:
            self._most_recent_year = int(dm.train_ds.years.max().item())
            train_years = dm.train_ds.years.numpy()
            year_range = (train_years.min(), train_years.max())

            logging.info(f"[Exponential Weighting] ENABLED")
            logging.info(f"  Most recent training year: {self._most_recent_year}")
            logging.info(f"  Training year range: {year_range[0]} to {year_range[1]}")
            logging.info(f"  Decay constant (tau): {self.config.exponential_tau}")
            logging.info(f"  Sample weights by year distance:")
            for dist_label, year_val in [("Current (0y)", self._most_recent_year),
                                         ("-3 years", self._most_recent_year-3),
                                         ("-5 years", self._most_recent_year-5),
                                         ("-10 years", self._most_recent_year-10)]:
                if year_val >= year_range[0]:
                    w = np.exp(-(self._most_recent_year - year_val) / self.config.exponential_tau)
                    logging.info(f"    {dist_label} (year {year_val}): {w:.4f}")

        train_df = self.trend_model._train_df
        logging.info(f"Fitting trends for {len(train_df[KEY_LOC].unique())} locations")

        # TrendModel is now used for trend prediction in _compute_batch_trends()
        # The Mann-Kendall logic is in TrendModel._predict_trend(), which is delegated to at inference time.
        # This means we don't need loc_trend_params, loc_coords, or _find_nearest_neighbor_trend anymore.

        # Verify that the model's forward pass accepts and uses past_observed_mask
        # by checking that outputs differ when mask zeros out all timesteps
        self._verify_mask_is_used()

    def _verify_mask_is_used(self):
        """Smoke test: masked-out inputs should produce different outputs than unmasked."""
        # Skip mask verification if model hasn't been fully built yet
        # (e.g., during checkpoint load before _build_model completes)
        if not hasattr(self, '_model_ready') or not self._model_ready:
            logging.info(f"[{self.config.model_type}] Skipping mask verification (model not ready).")
            return

        # Skip if feature_norm_params not available yet
        if self.feature_norm_params is None:
            logging.info(f"[{self.config.model_type}] Skipping mask verification (no norm params yet).")
            return

        # Use actual_context_length if available (for transformer models)
        seq_len = self._actual_context_length if hasattr(self, '_actual_context_length') else self.config.seq_len

        # Create normalized dummy data to match what forward() expects during training
        # After normalization, data is in z-score space (mean=0, std=1)
        dummy_ts = torch.randn(2, seq_len, self.n_ts_features, device=self.device)
        dummy_ts = self._normalize_time_series(dummy_ts)  # Normalize like real training data
        dummy_static = torch.zeros(2, self.n_static_features, device=self.device)
        dummy_static = self._normalize_and_impute_static(dummy_static)

        full_mask = torch.ones(2, seq_len, dtype=torch.bool, device=self.device)
        half_mask = full_mask.clone()
        half_mask[:, seq_len // 2:] = False

        with torch.no_grad():
            out_full = self.forward(dummy_ts, dummy_static, observed_mask=full_mask)
            out_half = self.forward(dummy_ts, dummy_static, observed_mask=half_mask)

        if torch.allclose(out_full, out_half, atol=1e-4):
            logging.warning(
                f"[{self.config.model_type}] past_observed_mask appears to have NO effect "
                f"on model output. Padded positions may be contributing spurious signal. "
                f"Verify HuggingFace implementation handles this mask correctly."
            )
        else:
            logging.info(f"[{self.config.model_type}] Mask verification passed.")

    def _compute_batch_trends(self, adm_ids, years: torch.Tensor, dm, lats, lons) -> torch.Tensor:
        """
        Compute normalised trend estimate for each sample in a batch.

        Uses TrendModel's sophisticated prediction logic which includes:
        - Mann-Kendall significance testing for trend validation
        - Optimal window selection for robust trend estimation
        - Forward/backward interpolation for test years between training blocks
        - Nearest-neighbor spatial interpolation for unseen locations

        Args:
            adm_ids: List of administrative region IDs
            years: Tensor of years (batch,)
            dm: DataModule with y_mean and y_std
            lats: Tensor of latitudes (batch,)
            lons: Tensor of longitudes (batch,)

        Returns:
            Tensor of shape (batch, 1) with trends in z-score space
        """
        # Guard against calling when trend is disabled
        if not self.config.use_residual_trend:
            raise RuntimeError(
                "_compute_batch_trends called with use_residual_trend=False. "
                "This is a bug in _shared_step - trend should not be computed or added."
            )

        # Construct test items for TrendModel
        test_items = []
        for i, (loc, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            test_items.append({
                KEY_LOC: loc,
                KEY_YEAR: year_int
            })

        # Use TrendModel's sophisticated prediction logic
        # DEBUG: Log batch info
        print(f"[DEBUG _compute_batch_trends] len(adm_ids)={len(adm_ids)}, len(years)={len(years) if hasattr(years, '__len__') else 'N/A'}, years.shape={years.shape if hasattr(years, 'shape') else 'N/A'}")
        print(f"[DEBUG _compute_batch_trends] test_items sample: {test_items[:3]}")

        trend_predictions_orig = self.trend_model._predict_trend(test_items)

        # DEBUG: Check what _predict_trend actually returned
        print(f"[DEBUG _compute_batch_trends] _predict_trend returned shape={trend_predictions_orig.shape}, dtype={trend_predictions_orig.dtype}")
        if hasattr(trend_predictions_orig, 'flatten'):
            trend_predictions_orig = trend_predictions_orig.flatten()
            print(f"[DEBUG _compute_batch_trends] After flatten, shape={trend_predictions_orig.shape}")

        # Normalize to z-score space
        trends_z = (trend_predictions_orig - dm.y_mean) / dm.y_std
        print(f"[DEBUG _compute_batch_trends] trends_z shape={trends_z.shape if hasattr(trends_z, 'shape') else 'N/A'}")

        # Ensure trends_z is 1D to avoid shape mismatches with quantile predictions
        if hasattr(trends_z, 'ndim') and trends_z.ndim > 1:
            print(f"[DEBUG _compute_batch_trends] WARNING: trends_z is {trends_z.ndim}D, flattening...")
            trends_z = trends_z.flatten()
        trends_tensor = torch.tensor(trends_z, dtype=torch.float32, device=self.device)
        # Ensure 2D output (batch, 1) for consistent squeeze(-1) behavior
        if trends_tensor.dim() == 1:
            trends_tensor = trends_tensor.unsqueeze(1)
        print(f"[DEBUG _compute_batch_trends] Final trends_tensor shape={trends_tensor.shape}")
        return trends_tensor

    def _compute_weighted_loss(self, pred: torch.Tensor, y: torch.Tensor,
                                years: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss between predictions and targets (MSE or Pinball).

        Args:
            pred: Predictions of shape (batch,) for MSE or (batch, n_quantiles) for pinball
            y: Targets of shape (batch,) or (batch, 1)
            years: Sample years of shape (batch,) for exponential weighting

        Returns:
            Loss tensor

        Note: validity_mask is not used for loss weighting since the model
        outputs a SCALAR prediction per sample (yield). Weighting by the
        fraction of valid timesteps would arbitrarily down-weight short-season
        samples. A 191-day season is not inherently less trustworthy than
        a 365-day season. The mask is retained for attention masking in
        forward() but not used for loss computation.
        """
        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Pinball loss for quantile regression
        if self.config.loss_type == 'pinball':
            from cybench.process.uncertainty_metrics import weighted_pinball_loss_torch

            if self.config.use_exponential_weighting and years is not None:
                # Compute per-sample weights
                years_int = years.cpu().numpy() if years.is_cuda else years.numpy()
                weights = np.exp(-(self._most_recent_year - years_int) / self.config.exponential_tau)
                weights_tensor = torch.tensor(weights, dtype=torch.float32, device=pred.device).unsqueeze(-1)

                # Log first batch sample weights (once)
                if not self._weight_log_done:
                    unique_years = np.unique(years_int[:min(16, len(years_int))])  # Sample first 16
                    logging.info(f"[Exponential Weighting] First batch sample weights:")
                    for yr in sorted(unique_years, reverse=True)[:6]:  # Show up to 6 years
                        idx = np.where(years_int == yr)[0][0]
                        dist = self._most_recent_year - yr
                        logging.info(f"    Year {yr} (distance {dist}y): weight={weights[idx]:.4f}")
                    self._weight_log_done = True

                return weighted_pinball_loss_torch(pred, y, self.config.quantiles, weights_tensor)
            else:
                from cybench.process.uncertainty_metrics import pinball_loss_torch
                return pinball_loss_torch(pred, y, self.config.quantiles)

        # MSE loss for point prediction
        if self.config.use_exponential_weighting and years is not None:
            # Compute per-sample losses
            per_sample_loss = F.mse_loss(pred, y, reduction='none')

            # Compute exponential weights based on year distance
            # weight = exp(-(current_year - sample_year) / tau)
            years_int = years.cpu().numpy() if years.is_cuda else years.numpy()
            weights = np.exp(-(self._most_recent_year - years_int) / self.config.exponential_tau)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=pred.device)

            # Normalize weights to sum to batch size (preserves loss scale)
            weights_tensor = weights_tensor / (weights_tensor.mean() + 1e-8)

            # Log first batch sample weights (once)
            if not self._weight_log_done:
                unique_years = np.unique(years_int[:min(16, len(years_int))])  # Sample first 16
                logging.info(f"[Exponential Weighting] First batch sample weights:")
                for yr in sorted(unique_years, reverse=True)[:6]:  # Show up to 6 years
                    idx = np.where(years_int == yr)[0][0]
                    dist = self._most_recent_year - yr
                    logging.info(f"    Year {yr} (distance {dist}y): weight={weights[idx]:.4f}")
                self._weight_log_done = True

            # Apply weighted loss
            weighted_loss = (per_sample_loss * weights_tensor).mean()
            return weighted_loss
        else:
            return F.mse_loss(pred, y)

    def _shared_step(self, batch, metrics: ModelMetrics, loss_key: str):
        x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # DEBUG: Log batch shapes and sizes
        print(f"[DEBUG _shared_step] x_ts.shape={x_ts.shape}, len(adm_ids)={len(adm_ids)}, years.shape={years.shape if hasattr(years, 'shape') else 'N/A'}")

        # Debug: Check for NaN in targets
        if torch.isnan(y).any():
            print(f"[DEBUG TRAIN] NaN in targets (y): {torch.isnan(y).sum().item()} out of {y.numel()}")
            print(f"[DEBUG TRAIN] Sample targets: {y[:5].tolist()}")

        # Apply tokenization if enabled (ablation study)
        x_ts, validity_mask = self._apply_tokenization(x_ts, observed_mask=validity_mask)

        # Compute trends only if use_residual_trend is enabled
        if self.config.use_residual_trend:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        # Pass observed_mask to forward so models can use it for attention masking
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # DEBUG: Log pred shape
        print(f"[DEBUG _shared_step] pred.shape={pred.shape}")

        # Only add trend back when use_residual_trend is True
        # Detach batch_trends to prevent gradient flow through OLS-computed values
        if batch_trends is not None:
            # For pinball loss (quantile regression), pred has shape (batch, n_quantiles)
            # We need to broadcast trends to all quantiles
            # batch_trends is (batch, 1), squeeze to (batch,), then expand to (batch, n_quantiles)
            trends_squeezed = batch_trends.squeeze(-1).detach()  # (batch,)
            if trends_squeezed.dim() == 1 and pred.dim() == 2:
                # Broadcast trends to match quantile dimensions
                trends_broadcast = trends_squeezed.unsqueeze(-1).expand_as(pred)  # (batch, n_quantiles)
                final_pred = pred + trends_broadcast
            else:
                # Fallback to original behavior
                final_pred = pred + trends_squeezed
        else:
            final_pred = pred

        # Pass years for exponential weighting
        loss = self._compute_weighted_loss(final_pred, y, years=years)

        metrics.update(final_pred.detach(), y.detach())
        self.log(loss_key, loss, prog_bar=True)
        return loss

    def _eval_step_with_clipping(self, batch, metrics: ModelMetrics, loss_key: str, stage: str,
                                  return_predictions: bool = False, return_orig: bool = False):
        """
        Evaluation step with clipping for physically meaningful yield predictions.

        Unlike training, this step:
        1. Denormalizes predictions and targets to original scale (tons/ha)
        2. Clips predictions to minimum of 0 (yields cannot be negative)
        3. Logs the rate at which predictions are clipped (diagnostic)
        4. Computes metrics on clipped predictions

        Args:
            batch: Input batch
            metrics: ModelMetrics instance for this stage
            loss_key: Key for logging loss (e.g., 'val_loss', 'test_loss')
            stage: 'val' or 'test' for logging clip rate
            return_predictions: If True, include predictions_z in return
            return_orig: If True, include clipped predictions and targets in return

        Returns:
            Loss tensor (computed in z-score space for consistency with training)
            or (loss, predictions_z) if return_predictions=True
            or (loss, predictions_clipped, targets_orig, years) if return_orig=True
            or (loss, predictions_z, predictions_clipped, targets_orig, years) if both True
        """
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Apply tokenization if enabled (ablation study)
        x_ts, validity_mask = self._apply_tokenization(x_ts, observed_mask=validity_mask)

        # Compute trends if enabled and trend model has been fitted
        # (skip during sanity check validation before on_train_start)
        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        # Forward pass (same as training)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trend back with proper broadcasting for quantile regression
        if batch_trends is not None:
            trends_squeezed = batch_trends.squeeze(-1).detach()  # (batch,)
            if trends_squeezed.dim() == 1 and pred.dim() == 2:
                # Broadcast trends to match quantile dimensions
                trends_broadcast = trends_squeezed.unsqueeze(-1).expand_as(pred)  # (batch, n_quantiles)
                final_pred_z = pred + trends_broadcast
            else:
                # Fallback for MSE case
                final_pred_z = pred + trends_squeezed
        else:
            final_pred_z = pred

        # Compute loss in z-score space (for consistency with training)
        loss = self._compute_weighted_loss(final_pred_z, y_z)

        # Denormalize to original scale for metrics computation
        # Ensure y_std and y_mean are on the same device as the predictions
        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        final_pred_orig = final_pred_z.detach() * y_std + y_mean
        y_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to physically meaningful range (yields ≥ 0)
        final_pred_clipped = torch.clamp(final_pred_orig, min=0.0)

        # Log clip rate as diagnostic (helps identify model issues)
        # Only count predictions that were actually clipped from negative values
        # (not all zeros, since legitimate zero yields are possible)
        clipped_mask = final_pred_orig < 0.0
        clip_rate = clipped_mask.float().mean()
        self.log(f'{stage}/clip_rate', clip_rate, prog_bar=False)

        # Also log stats about negative predictions before clipping
        negative_rate = (final_pred_orig < 0).float().mean()
        self.log(f'{stage}/negative_rate', negative_rate, prog_bar=False)

        # Update metrics with clipped predictions (in original scale)
        metrics.update(final_pred_clipped, y_orig)

        self.log(loss_key, loss, prog_bar=True)

        if return_orig and return_predictions:
            return loss, final_pred_z, final_pred_clipped, y_orig, years
        if return_orig:
            return loss, final_pred_clipped, y_orig, years
        if return_predictions:
            return loss, final_pred_z
        return loss

    def training_step(self, batch, batch_idx):
        # Training uses _shared_step without clipping (honest gradients)
        return self._shared_step(batch, self.train_metrics, 'train_loss')

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log('train/mse', results['mse'], prog_bar=False)
        self.log('train/mae', results['mae'], prog_bar=False)
        self.log('train/r2', results['r2'], prog_bar=False)
        self.log('train/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('train/mape', results['mape'], prog_bar=False)
        self.log('train/smape', results['smape'], prog_bar=False)
        self.train_metrics.log_results(step="train")
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Evaluation uses clipping for physically meaningful predictions
        # Accumulate predictions for bias correction fitting if enabled
        if self.bias_correction is not None and hasattr(self, '_val_preds'):
            loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
                batch, self.val_metrics, 'val_loss', stage='val',
                return_orig=True, return_predictions=True
            )
            # Accumulate predictions in original scale for bias correction fitting
            # When quantile predictions are enabled (shape: [n, 3]), use median (index 1) for bias correction
            preds_np = preds_clipped.detach().cpu()
            if preds_np.dim() == 2 and preds_np.shape[1] == 3:
                # Quantile mode: use median (index 1) for bias correction
                self._val_preds.extend(preds_np[:, 1].tolist())
            else:
                self._val_preds.extend(preds_np.tolist())
            self._val_targets.extend(targets.detach().cpu().tolist())
            return loss
        else:
            return self._eval_step_with_clipping(batch, self.val_metrics, 'val_loss', stage='val')

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.log('val/mse', results['mse'], prog_bar=False)
        self.log('val/mae', results['mae'], prog_bar=False)
        self.log('val/r2', results['r2'], prog_bar=False)
        self.log('val/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('val/mape', results['mape'], prog_bar=False)
        self.log('val/smape', results['smape'], prog_bar=False)
        self.log('val/nrmse', results['nrmse'], prog_bar=False)
        self.val_metrics.log_results(step="val")
        self.val_metrics.reset()

    def on_train_end(self):
        """Fit bias correction on validation predictions at the end of training."""
        if self.bias_correction is not None and hasattr(self, '_val_preds') and len(self._val_preds) > 0:
            # Fit bias correction on validation predictions
            self.bias_correction.fit(np.array(self._val_preds), np.array(self._val_targets))
            params = self.bias_correction.get_correction_params()
            logging.info(f"[BiasCorrection] Fitted on {len(self._val_preds)} validation samples")
            if 'a' in params and 'b' in params:
                logging.info(f"[BiasCorrection] Correction: corrected = {params['a']:.4f} * pred + {params['b']:.4f}")
            # Clear validation predictions to free memory
            self._val_preds = []
            self._val_targets = []

    def _compute_per_year_metrics_from_preds(self) -> dict:
        """Compute per-year metrics from accumulated predictions using torchmetrics."""
        results = {}
        all_preds = []
        all_targets = []

        for year, data in self._per_year_preds.items():
            if len(data['preds']) == 0:
                continue

            # Convert to torch tensors
            preds = torch.tensor(data['preds'])
            targets = torch.tensor(data['targets'])

            # Compute metrics using torchmetrics for consistency
            mse = MeanSquaredError()
            r2 = R2Score()
            mae = MeanAbsoluteError()
            mape = MeanAbsolutePercentageError()

            mse_val = mse(preds, targets)
            mae_val = mae(preds, targets)
            mape_val = mape(preds, targets)
            rmse_val = torch.sqrt(mse_val)

            # SMAPE - manual computation for consistency
            smape_val = torch.mean(2.0 * torch.abs(preds - targets) /
                                  (torch.abs(preds) + torch.abs(targets) + 1e-8))
            nrmse_val = rmse_val / (torch.mean(targets).clamp(min=1e-8))

            # R² score requires at least 2 samples
            if len(preds) >= 2:
                r2_val = r2(preds, targets)
            else:
                r2_val = torch.tensor(float('nan'))  # Cannot compute R² with single sample

            # Store per-year metrics
            results[f'nrmse_{year}'] = nrmse_val.item()
            results[f'mape_{year}'] = mape_val.item()
            results[f'r2_{year}'] = r2_val.item()
            results[f'rmse_{year}'] = rmse_val.item()
            results[f'mae_{year}'] = mae_val.item()
            results[f'mse_{year}'] = mse_val.item()
            results[f'smape_{year}'] = smape_val.item()

            # Accumulate for overall metrics
            all_preds.extend(data['preds'])
            all_targets.extend(data['targets'])

        # Compute overall metrics
        if all_preds and all_targets:
            all_preds_t = torch.tensor(all_preds)
            all_targets_t = torch.tensor(all_targets)

            mse = MeanSquaredError()
            r2 = R2Score()
            mae = MeanAbsoluteError()
            mape = MeanAbsolutePercentageError()

            mse_val = mse(all_preds_t, all_targets_t)
            mae_val = mae(all_preds_t, all_targets_t)
            mape_val = mape(all_preds_t, all_targets_t)
            rmse_val = torch.sqrt(mse_val)

            smape_val = torch.mean(2.0 * torch.abs(all_preds_t - all_targets_t) /
                                  (torch.abs(all_preds_t) + torch.abs(all_targets_t) + 1e-8))
            nrmse_val = rmse_val / (torch.mean(all_targets_t).clamp(min=1e-8))

            # R² score requires at least 2 samples
            if len(all_preds_t) >= 2:
                r2_val = r2(all_preds_t, all_targets_t)
            else:
                r2_val = torch.tensor(float('nan'))  # Cannot compute R² with single sample

            results['nrmse_overall'] = nrmse_val.item()
            results['mape_overall'] = mape_val.item()
            results['r2_overall'] = r2_val.item()
            results['rmse_overall'] = rmse_val.item()
            results['mae_overall'] = mae_val.item()
            results['mse_overall'] = mse_val.item()
            results['smape_overall'] = smape_val.item()

        return results

    def _accumulate_per_year_predictions(self, preds: torch.Tensor, targets: torch.Tensor, years: torch.Tensor):
        """Accumulate predictions and targets per year for later metrics computation."""
        if not hasattr(self, '_per_year_preds') or not self._per_year_preds:
            return

        # Convert to numpy
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy().flatten()
        years_np = years.cpu().numpy() if isinstance(years, torch.Tensor) else years

        # For quantile predictions (shape: [n, 3]), extract only the median (index 1)
        if preds_np.ndim == 2 and preds_np.shape[1] == 3:
            preds_np = preds_np[:, 1]  # Use median (0.5 quantile)
        else:
            preds_np = preds_np.flatten()

        for pred, target, year in zip(preds_np, targets_np, years_np):
            year_int = int(year)
            if year_int in self._per_year_preds:
                self._per_year_preds[year_int]['preds'].append(float(pred))
                self._per_year_preds[year_int]['targets'].append(float(target))

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log('test/mse', results['mse'], prog_bar=False)
        self.log('test/mae', results['mae'], prog_bar=False)
        self.log('test/r2', results['r2'], prog_bar=False)
        self.log('test/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('test/mape', results['mape'], prog_bar=False)
        self.log('test/smape', results['smape'], prog_bar=False)
        self.log('test/nrmse', results['nrmse'], prog_bar=False)
        self.test_metrics.log_results(step="test")

        # Log uncertainty metrics if using quantile regression
        print(f"[on_test_epoch_end] loss_type={self.config.loss_type}, checking uncertainty metrics")
        if self.config.loss_type == 'pinball':
            uq_metrics = self.test_metrics.compute_uncertainty_metrics()
            print(f"[on_test_epoch_end] uq_metrics={uq_metrics}")
            if uq_metrics:
                # Store on model for later retrieval
                self._test_uncertainty_metrics = uq_metrics
                print(f"\n[TEST UNCERTAINTY METRICS]")
                for key, value in sorted(uq_metrics.items()):
                    # Log all uncertainty metrics, regardless of prefix format
                    self.log(key, value, prog_bar=False)
                    # Clean up the key name for printing: remove 'test' or 'test/' prefix
                    if key.startswith('test/'):
                        print(f"{key.replace('test/', '').upper()}: {value:.4f}")
                    elif key.startswith('test'):
                        print(f"{key[4:].upper()}: {value:.4f}")
                    else:
                        print(f"{key.upper()}: {value:.4f}")
            else:
                print(f"[on_test_epoch_end] WARNING: uq_metrics is None or empty!")

        self.test_metrics.reset()

        # Compute per-year metrics and store on model for CSV saving
        if hasattr(self, '_per_year_preds') and self._per_year_preds:
            self._test_results_per_year = self._compute_per_year_metrics_from_preds()

    def predict(self, batch, datamodule=None):
        """
        Generate predictions for a batch of data without updating metrics.

        Args:
            batch: Input batch tuple (x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask)
            datamodule: Optional DataModule for trend computation and denormalization.
                        If not provided, uses stored datamodule from training (if available).

        Returns:
            dict: Dictionary containing:
                - predictions: Predictions in original scale (tons/ha), clipped to >= 0
                - predictions_z: Predictions in z-score space (before denormalization)
                - targets: Ground truth targets in original scale (tons/ha)
                - years: Years for each sample
                - adm_ids: Administrative IDs for each sample
                - lats: Latitudes for each sample
                - lons: Longitudes for each sample
        """
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch

        # Use provided datamodule, fall back to stored reference, then trainer
        if datamodule is not None:
            dm = datamodule
            # Store for future calls (for inference without trainer)
            self._datamodule = datamodule
            # Also store feature_norm_params for _normalize_time_series
            if hasattr(datamodule, 'feature_norm_params'):
                self.feature_norm_params = datamodule.feature_norm_params
        elif hasattr(self, '_datamodule') and self._datamodule is not None:
            dm = self._datamodule
        else:
            try:
                dm = self.trainer.datamodule
            except (RuntimeError, AttributeError):
                raise RuntimeError("No datamodule available. Pass datamodule to predict() or ensure model was trained with a datamodule.")

        # Move inputs to model device (important for predict() after training)
        device = self.device
        x_ts = x_ts.to(device)
        x_static = x_static.to(device)
        y_z = y_z.to(device)
        validity_mask = validity_mask.to(device) if validity_mask is not None else None
        # lats, lons are not used in forward pass, keep as is

        # Apply tokenization if enabled (ablation study)
        x_ts, validity_mask = self._apply_tokenization(x_ts, observed_mask=validity_mask)

        # Compute trends if enabled
        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        # Forward pass (same as training)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trend back with proper broadcasting for quantile regression
        if batch_trends is not None:
            trends_squeezed = batch_trends.squeeze(-1).detach()  # (batch,)
            if trends_squeezed.dim() == 1 and pred.dim() == 2:
                # Broadcast trends to match quantile dimensions
                trends_broadcast = trends_squeezed.unsqueeze(-1).expand_as(pred)  # (batch, n_quantiles)
                final_pred_z = pred + trends_broadcast
            else:
                # Fallback for MSE case
                final_pred_z = pred + trends_squeezed
        else:
            final_pred_z = pred

        # Denormalize to original scale
        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        predictions_orig = final_pred_z.detach() * y_std + y_mean
        targets_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to physically meaningful range (yields >= 0)
        predictions_clipped = torch.clamp(predictions_orig, min=0.0)

        return {
            'predictions': predictions_clipped,
            'predictions_z': final_pred_z,
            'targets': targets_orig,
            'years': years,
            'adm_ids': adm_ids,
            'lats': lats,
            'lons': lons,
        }

    def on_test_start(self):
        """Initialize prediction cache for recursive lag prediction and per-year metrics."""
        # Initialize per-year prediction storage for CSV results
        dm = self.trainer.datamodule
        if hasattr(dm, '_test_years') and dm._test_years is not None:
            self._test_years = dm._test_years
            self._per_year_preds = {year: {'preds': [], 'targets': []} for year in self._test_years}
            logging.info(f"[Per-Year Metrics] Initialized storage for test years: {sorted(self._test_years)}")
        else:
            logging.warning("[Per-Year Metrics] Datamodule has no _test_years set, per-year metrics will not be computed")
            self._test_years = set()  # Initialize as empty set to prevent AttributeError in downstream methods
            self._per_year_preds = {}

        if self.config.use_recursive_lags:
            if not hasattr(self, '_prediction_cache') or not self._prediction_cache:
                self._prediction_cache = {}
                logging.info(f"[Recursive Lags] Initialized prediction cache for true out-of-sample testing")
            else:
                logging.info(f"[Recursive Lags] Preserving cache with {len(self._prediction_cache)} predictions for multi-year testing")
            if self._test_years:  # Empty set is falsy, so this safely checks if test years exist
                logging.info(f"[Recursive Lags] Test years: {sorted(self._test_years)}")
            else:
                logging.warning("[Recursive Lags] No test years available for recursive prediction")

    def _replace_lags_with_predictions(self, x_static: torch.Tensor, years: torch.Tensor,
                                       adm_ids: List[str]) -> torch.Tensor:
        """
        Replace lag features with cached predictions for recursive lag evaluation.
        For test samples where lag years fall within the test set, this replaces the observed lag values (which cause leakage) with previously predicted values.

        Args:
            x_static: Static features tensor (batch, n_static_features)
            years: Tensor of years (batch,)
            adm_ids: List of administrative region IDs

        Returns:
            Modified x_static with lag features replaced by predictions where appropriate
        """
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return x_static

        x_static_modified = x_static.clone()
        static_feature_names = self._get_static_feature_names()

        # Find indices of lag features in the static feature array
        lag_feature_indices = []
        for lag in range(1, self.config.lag_years + 1):
            lag_name = f'lag_yield_{lag}'
            if lag_name in static_feature_names:
                lag_feature_indices.append((lag, static_feature_names.index(lag_name)))

        if not lag_feature_indices:
            return x_static

        # Replace each lag feature with cached prediction if available
        for i, (year, adm_id) in enumerate(zip(years, adm_ids)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            for lag, lag_idx in lag_feature_indices:
                lag_year = year_int - lag

                # Check if this lag year is in the test set
                if self._test_years and lag_year in self._test_years:
                    cache_key = (adm_id, lag_year)

                    if cache_key in self._prediction_cache:
                        # Use cached prediction (already in z-score space)
                        x_static_modified[i, lag_idx] = self._prediction_cache[cache_key]
                    else:
                        # No prediction available yet - use default (mean in z-score space = 0.0)
                        # This can happen if batch isn't perfectly sorted chronologically within each location
                        x_static_modified[i, lag_idx] = 0.0

                        logging.warning(
                            f"[Recursive Lags] No cached prediction for {adm_id} year {lag_year}, "
                            f"using default (mean in z-score space=0.0). "
                            f"This may occur if test batches are not sorted chronologically within each location."
                        )

        return x_static_modified

    def _cache_predictions(self, predictions_z: torch.Tensor, years: torch.Tensor,
                          adm_ids: List[str], dm):
        """
        Cache predictions in ORIGINAL scale for recursive lag replacement.

        Applies bias correction if enabled (automatic with use_recursive_lags=True).
        Bias-corrected predictions are cached to reduce error compounding in recursive forecasting.

        x_static holds raw (un-normalized) lag yields. _normalize_and_impute_static will z-score them later. Storing in original scale ensures only one
        normalization pass occurs.

        Args:
            predictions_z: Predictions in z-score space [B]
            years: Years [B]
            adm_ids: Location IDs [B]
            dm: DataModule for denormalization
        """
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return

        device = predictions_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)

        # Convert to original scale
        predictions_orig = predictions_z.detach() * y_std + y_mean

        # Apply bias correction if fitted (reduces error compounding in recursive lags)
        if self.bias_correction is not None and self.bias_correction.is_fitted:
            predictions_orig_np = predictions_orig.cpu().numpy()
            predictions_orig_np = self.bias_correction.correct(predictions_orig_np)
            predictions_orig = torch.tensor(predictions_orig_np, device=device, dtype=predictions_orig.dtype)

        for pred, year, adm_id in zip(predictions_orig, years, adm_ids):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            # Only cache predictions for test years
            if self._test_years and year_int in self._test_years:
                cache_key = (adm_id, year_int)
                # Extract scalar value: use median (index 1) for quantile predictions, otherwise use the value
                pred_cpu = pred.detach().cpu()
                if pred_cpu.numel() > 1:
                    # Quantile predictions: use median (index 1)
                    pred_value = pred_cpu[1].item()
                else:
                    pred_value = pred_cpu.item()
                self._prediction_cache[cache_key] = pred_value

                # Log first few cached predictions for debugging
                if len(self._prediction_cache) <= 5:
                    bc_status = "bias-corrected" if (self.bias_correction is not None and self.bias_correction.is_fitted) else "raw"
                    logging.debug(
                        f"[Recursive Lags] Cached {bc_status} prediction for {adm_id} year {year_int}: "
                        f"orig_scale={pred_value:.4f}"
                    )

    def test_step(self, batch, batch_idx):
        """Test step with optional recursive lag prediction and per-year accumulation."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            # Use standard evaluation and accumulate per-year predictions
            loss, preds, targets, years = self._eval_step_with_clipping(
                batch, self.test_metrics, 'test_loss', stage='test', return_orig=True
            )
            # Accumulate per-year predictions for CSV results
            self._accumulate_per_year_predictions(preds, targets, years)
            return loss

        # Recursive lag mode: modify batch to use cached predictions
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Replace lag features with cached predictions
        x_static_modified = self._replace_lags_with_predictions(x_static, years, adm_ids)

        # Create modified batch
        modified_batch = (x_ts, x_static_modified, y_z, years, adm_ids, lats, lons, validity_mask)

        # Run evaluation step and get both z-score and original predictions
        loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
            modified_batch, self.test_metrics, 'test_loss', stage='test',
            return_orig=True, return_predictions=True
        )

        # Cache predictions in z-score space for recursive lag prediction
        # The cached predictions are used for subsequent test samples in the same location
        self._cache_predictions(preds_z, years, adm_ids, dm)

        # Accumulate per-year predictions for CSV results (using clipped predictions in original scale)
        self._accumulate_per_year_predictions(preds_clipped, targets, years)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        if self.config.lr_scheduler_lambda is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self.config.lr_scheduler_lambda
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def on_fit_start(self):
        """Log model size to wandb at the start of training."""
        print(f"\n{'=' * 60}")
        print("DEBUG: on_fit_start() called - counting model parameters...")
        print(f"{'=' * 60}")

        # Debug: Check what modules exist
        print(f"DEBUG: Model type: {type(self).__name__}")
        print(f"DEBUG: Has base_model: {hasattr(self, 'base_model')}")
        print(f"DEBUG: Has regression_head: {hasattr(self, 'regression_head')}")

        if hasattr(self, 'base_model'):
            print(f"DEBUG: base_model type: {type(self.base_model).__name__}")
            base_model_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"DEBUG: base_model parameters: {base_model_params:,}")

        if hasattr(self, 'regression_head'):
            regression_head_params = sum(p.numel() for p in self.regression_head.parameters())
            print(f"DEBUG: regression_head parameters: {regression_head_params:,}")

        # Get list of parameters to debug
        params_list = list(self.parameters())
        print(f"DEBUG: Number of parameter groups: {len(params_list)}")

        # Count total parameters
        total_params = sum(p.numel() for p in params_list)
        print(f"DEBUG: Total parameters counted: {total_params:,}")

        # Calculate size
        param_size = sum(p.numel() * p.element_size() for p in params_list)
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        print(f"DEBUG: Parameter size (bytes): {param_size:,}")
        print(f"DEBUG: Buffer size (bytes): {buffer_size:,}")
        print(f"{'=' * 60}")
        print(f"MODEL SIZE: {model_size_mb:.2f} MB")
        print(f"Total parameters: {total_params:,}")
        print(f"{'=' * 60}\n")

        # Log directly to wandb (self.log() is not allowed in on_fit_start)
        if self.logger and hasattr(self.logger, 'experiment'):
            # Only log if the logger supports direct logging (e.g., WandBLogger)
            # CSVLogger and other loggers don't support this
            if hasattr(self.logger.experiment, 'log') and callable(getattr(self.logger.experiment, 'log')):
                self.logger.experiment.log({
                    'model_size_mb': model_size_mb,
                    'total_params': total_params
                })
                print("DEBUG: Logged to logger successfully")


# =========================================================
# MODEL ARCHITECTURES
# =========================================================

class AutoformerYieldModel(BaseTimeSeriesModel):
    """Autoformer: auto-correlation based transformer for yield forecasting."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFAutoformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # Use HFAutoformerModel instead of AutoformerForPrediction as HFAutoformerModel returns raw encoder/decoder outputs with gradients preserved

            # Standardize context length calculation for fair comparison
            # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
            lags_sequence = [1] if self.config.lag_years > 0 else [0]
            # context_length = self._get_standardized_context_length(self.config.seq_len, lags_sequence)
            effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
            context_length = self._get_standardized_context_length(effective_seq_len, lags_sequence)

            # Following baseline approach: process temporal features first,
            # then concatenate with static features after getting pooled representation
            cfg = AutoformerConfig(
                prediction_length=1,
                context_length=context_length,
                lags_sequence=lags_sequence,
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,  # Standardized: all models use 0
                num_static_categorical_features=0,
                # not using num_static_real_features - will be concatenated later
                d_model=64, num_attention_heads=4, ffn_dim=256, num_layers=3,
                dropout=0.1,
            )

            model = HFAutoformerModel(cfg)

            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[Autoformer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"n_ts_features={self.n_ts_features}, n_static_features={self.n_static_features}")

            logging.info(f"[Autoformer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"n_ts_features={self.n_ts_features}, n_static_features={self.n_static_features}")

        # Always create regression head, even when loading from checkpoint
        # This ensures parameters are registered before optimizer is created
        # d_model is hardcoded to 64 in the config above
        d_model = 64
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        logging.info(f"[Autoformer BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}, output_dim={self.output_dim}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        Following baseline pattern: temporal → pool → concat static → regression

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) for valid timesteps

        Returns:
            Predictions of shape (batch,)
        """
        batch_size, seq_len = x_ts.shape[:2]

        # Truncate input to match context_length for fair comparison
        if seq_len > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            seq_len = self._actual_context_length

        #  Process temporal features through Autoformer (NO static features yet)
        # Request encoder outputs explicitly
        outputs = self.base_model(
            past_values=x_ts,
            past_time_features=torch.zeros(batch_size, seq_len, 0, device=x_ts.device),
            past_observed_mask=observed_mask.unsqueeze(-1).expand(-1, -1, self.n_ts_features).float() if observed_mask is not None else None,
            future_values=torch.zeros(batch_size, 1, self.n_ts_features, device=x_ts.device),
            future_time_features=torch.zeros(batch_size, 1, 0, device=x_ts.device),
            return_dict=True,
            output_hidden_states=True,  # Request all hidden states
        )

        #  Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        #  Apply positional encoding if enabled (preserves temporal ordering)
        h = self._apply_positional_encoding(h, observed_mask)

        #  Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        #  Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class PatchTSTModel(BaseTimeSeriesModel):
    """PatchTST: patch-based transformer (linear complexity in sequence length)."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFPatchTSTModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence) if hasattr(model.config, 'lags_sequence') else [1]
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            patch_len = {"daily": 16, "weekly": 4, "dekad": 6}[self.config.aggregation]
            stride = {"daily": 8, "weekly": 2, "dekad": 3}[self.config.aggregation]

            # Standardize context length calculation for fair comparison
            # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
            requested_lags = [1] if self.config.lag_years > 0 else [0]
            # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
            effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
            context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)

            logging.info(f"[PatchTST BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={context_length}, n_ts_features={self.n_ts_features}, "
                        f"n_static_features={self.n_static_features}, "
                        f"patch_length={patch_len}, stride={stride}, aggregation={self.config.aggregation}")
            logging.info(f"[PatchTST BUILD] Hyperparameters: d_model={self.config.patchtst_d_model}, "
                        f"num_attention_heads={self.config.patchtst_num_attention_heads}, "
                        f"ffn_dim={self.config.patchtst_ffn_dim}, num_layers={self.config.patchtst_num_layers}, "
                        f"dropout={self.config.patchtst_dropout}")

            cfg = PatchTSTConfig(
                prediction_length=1,
                context_length=context_length,
                # num_input_channels is the total number of input channels
                # This should match n_ts_features (excluding time features since we pass them separately)
                num_input_channels=self.n_ts_features,
                num_time_features=0,  # Standardized: all models use 0
                # NOT using num_static_real_features - we'll concatenate later
                d_model=self.config.patchtst_d_model,
                num_attention_heads=self.config.patchtst_num_attention_heads,
                ffn_dim=self.config.patchtst_ffn_dim,
                num_layers=self.config.patchtst_num_layers,
                dropout=self.config.patchtst_dropout,
                patch_length=patch_len,
                stride=stride,
            )

            model = HFPatchTSTModel(cfg)

            # Read back what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence) if hasattr(model.config, 'lags_sequence') else [1]
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[PatchTST] Config verification: context_length={model.config.context_length}, "
                        f"num_input_channels={model.config.num_input_channels}, "
                        f"num_time_features={model.config.num_time_features}")

            logging.info(f"[PatchTST] Config verification: context_length={model.config.context_length}, "
                        f"num_input_channels={model.config.num_input_channels}, num_time_features={model.config.num_time_features}")

        # Probe the actual output shape to build regression head correctly
        # PatchTST can output different shapes depending on configuration:
        #   - (B, seq_len, d_model) for some configs
        #   - (B, n_channels, n_patches, d_model) for multivariate patch-based
        with torch.no_grad():
            # Use actual_context_length to match model's expected input size
            dummy = torch.zeros(1, self._actual_context_length, self.n_ts_features)
            out = model(past_values=dummy)
            h = self._extract_hidden_state(out)
            pooled = self._pool_hidden_state(h)
            pooled_dim = pooled.shape[-1]

        logging.info(f"[PatchTST BUILD] Probed actual output: h.shape={h.shape} → pooled.shape={pooled.shape} → pooled_dim={pooled_dim}")

        # Build regression head with correct dimension (always created, even when loading from checkpoint)
        combined_dim = pooled_dim + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        # Store for validation in forward()
        self._pooled_dim = pooled_dim

        logging.info(f"[PatchTST BUILD] Created regression head: pooled_dim={pooled_dim}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through PatchTST model.
        Following baseline pattern: temporal → pool → concat static → regression
        """
        # Truncate input to match context_length for fair comparison
        B, T, C = x_ts.shape
        if T > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            T = self._actual_context_length

        #  Process temporal features through PatchTST
        # Pass past_observed_mask to prevent padded zeros from being used in attention
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None
        outputs = self.base_model(past_values=x_ts, past_observed_mask=past_observed_mask, future_values=None)

        #  Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        #  Apply positional encoding if enabled (preserves temporal ordering)
        h = self._apply_positional_encoding(h, observed_mask)

        #  Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, pooled_dim)

        # Validate against probed dimension (catches shape regressions)
        if pooled.shape[-1] != self._pooled_dim:
            raise RuntimeError(
                f"[PatchTST] Pooled dimension mismatch! "
                f"Expected {self._pooled_dim} (from build-time probe), "
                f"got {pooled.shape[-1]}. This indicates the model output shape changed "
                f"between _build_model() and forward(). h.shape={h.shape}"
            )

        #  Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions



class TSMixerModel(BaseTimeSeriesModel):
    """TSMixer: all-MLP architecture — simple, fast, surprisingly competitive."""

    def _build_model(self) -> nn.Module:
        # Standardize context length calculation for fair comparison
        # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
        requested_lags = [1] if self.config.lag_years > 0 else [0]
        # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
        effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)


        # First, load or create the base model
        # For fair comparison, only use base model (consistent with other architectures)
        if self.config.load_checkpoint:
            if TimeSeriesMixerModel is not None:
                model = TimeSeriesMixerModel.from_pretrained(self.config.load_checkpoint)
            else:
                raise ImportError(
                    "Cannot load TSMixer checkpoint: TimeSeriesMixerModel (base model) not available. "
                    "For fair comparison with other architectures, TSMixer requires the base model, "
                    "not the ForPrediction variant. Please upgrade transformers or use a different model."
                )
        else:
            logging.info(f"[TSMixer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={context_length}, n_ts_features={self.n_ts_features}, "
                        f"n_static_features={self.n_static_features}")

            # For fair comparison, always use base model (not ForPrediction variant)
            if TimeSeriesMixerModel is not None:
                model = TimeSeriesMixerModel(TimeSeriesMixerConfig(
                    prediction_length=1,
                    context_length=context_length,
                    input_size=self.n_ts_features,
                    num_time_features=0,  # Standardized: all models use 0
                    # NOT using num_static_real_features - we'll concatenate later
                    hidden_size=64, num_layers=3, dropout=0.1, expansion_factor=2,
                ))
            else:
                raise ImportError(
                    "TSMixer base model (TimeSeriesMixerModel) not available in this transformers version. "
                    "For fair comparison with other architectures, TSMixer requires the base model. "
                    "Please upgrade transformers or use a different model."
                )

        # Read back what HF actually stored (it may override our values)
        # This needs to happen for both checkpoint loading and new model creation
        self._actual_lags = list(model.config.lags_sequence) if hasattr(model.config, 'lags_sequence') else [1]
        self._actual_num_time_features = int(model.config.num_time_features)
        self._actual_context_length = int(model.config.context_length)
        self._uses_for_prediction = False

        # Probe the actual output shape to build regression head correctly
        # This needs to happen for both checkpoint loading and new model creation
        with torch.no_grad():
            # Use actual_context_length to match model's expected input size
            dummy = torch.zeros(1, self._actual_context_length, self.n_ts_features)
            out = model(past_values=dummy)
            h = self._extract_hidden_state(out)
            pooled = self._pool_hidden_state(h)
            pooled_dim = pooled.shape[-1]

        logging.info(f"[TSMixer BUILD] Probed actual output: h.shape={h.shape} → pooled.shape={pooled.shape} → pooled_dim={pooled_dim}")

        # Build regression head with correct dimension
        combined_dim = pooled_dim + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        # Store for validation in forward()
        self._pooled_dim = pooled_dim

        logging.info(f"[TSMixer BUILD] Created regression head: pooled_dim={pooled_dim}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) for valid timesteps

        Returns:
            Predictions of shape (batch,)
        """
        # Truncate input to match context_length for fair comparison
        B, T, C = x_ts.shape
        if T > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            T = self._actual_context_length

        #  Process temporal features through TSMixer
        # Note: PatchTSMixerModel doesn't support past_observed_mask (MLP-based, no attention)
        # Mask is still applied later via _apply_positional_encoding
        outputs = self.base_model(past_values=x_ts)

        #  Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        #  Apply positional encoding if enabled (preserves temporal ordering)
        h = self._apply_positional_encoding(h, observed_mask)

        #  Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, pooled_dim)

        # Validate against probed dimension (catches shape regressions)
        if pooled.shape[-1] != self._pooled_dim:
            raise RuntimeError(
                f"[TSMixer] Pooled dimension mismatch! "
                f"Expected {self._pooled_dim} (from build-time probe), "
                f"got {pooled.shape[-1]}. This indicates the model output shape changed "
                f"between _build_model() and forward(). h.shape={h.shape}"
            )

        #  Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions

class InformerModel(BaseTimeSeriesModel):
    """Informer: sparse attention transformer, efficient on long sequences."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFInformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # Standardize context length calculation for fair comparison
            # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
            requested_lags = [1] if self.config.lag_years > 0 else [0]
            effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
            context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)

            # Build model with adjusted context_length (BASELINE PATTERN)
            cfg = InformerConfig(
                prediction_length=1,
                context_length=context_length,
                lags_sequence=requested_lags,
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,  # Standardized: all models use 0
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, ffn_dim=256, num_layers=3,
                dropout=0.1,
            )
            # Ensure context_length accommodates lags
            # Already computed: context_length = seq_len - max(lags_sequence)
            # This should be correct, but verify after model creation
            model = HFInformerModel(cfg)  # Use base model for gradient flow

            # Read nacl what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[Informer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"num_time_features={self._actual_num_time_features}, n_ts_features={self.n_ts_features}")

        # Always create regression head, even when loading from checkpoint
        d_model = 64  # from config above
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        logging.info(f"[Informer BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}, output_dim={self.output_dim}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Informer model. Following baseline pattern."""
        batch_size, seq_len = x_ts.shape[:2]

        # Truncate input to match context_length for fair comparison
        if seq_len > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            seq_len = self._actual_context_length

        #  Process temporal features through Informer
        # Standardized: use 0 for num_time_features consistently across all models
        past_time_features = torch.zeros(batch_size, seq_len, 0, device=x_ts.device, dtype=x_ts.dtype)
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None

        outputs = self.base_model(past_values=x_ts, past_time_features=past_time_features, past_observed_mask=past_observed_mask, return_dict=True)

        #  Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        #  Apply positional encoding if enabled (preserves temporal ordering)
        h = self._apply_positional_encoding(h, observed_mask)

        #  Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        #  Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class TSTModel(BaseTimeSeriesModel):
    """
    TimeSeriesTransformer: vanilla encoder-decoder with student-t output.

    Operates on raw normalised float values — no tokenisation.
    distribution_output='student_t' is robust to yield outliers.
    We extract the mean (index 0) of the distribution parameters for a
    deterministic prediction.
    """

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFTimeSeriesTransformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # Standardize context length calculation for fair comparison
            # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
            requested_lags = [1] if self.config.lag_years > 0 else [0]
            # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
            effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
            context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)

            # Build model with adjusted context_length (BASELINE PATTERN)
            cfg = TimeSeriesTransformerConfig(
                prediction_length=1,
                context_length=context_length,
                lags_sequence=requested_lags,
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,  # Standardized: all models use 0
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, num_hidden_layers=3,
                dim_feedforward=256, dropout=0.1, attention_probs_dropout_prob=0.1,
                activation_function="gelu", layer_norm_eps=1e-5,
                scaling="std", loss="nll", distribution_output="student_t",
            )
            # Ensure context_length accommodates lags
            # Already computed: context_length = seq_len - max(lags_sequence)
            # This should be correct, but verify after model creation
            model = HFTimeSeriesTransformerModel(cfg)  # Use base model for gradient flow

            # READ BACK what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[TST BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"num_time_features={self._actual_num_time_features}, n_ts_features={self.n_ts_features}")

        # Always create regression head, even when loading from checkpoint
        d_model = 64  # from config above
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        logging.info(f"[TST BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}, output_dim={self.output_dim}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through TST model. Following baseline pattern."""
        batch_size, seq_len = x_ts.shape[:2]

        # Truncate input to match context_length for fair comparison
        if seq_len > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            seq_len = self._actual_context_length

        #  Process temporal features through TST
        past_time_features = torch.zeros(batch_size, seq_len, self._actual_num_time_features, device=x_ts.device, dtype=x_ts.dtype)
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None

        outputs = self.base_model(past_values=x_ts, past_time_features=past_time_features, past_observed_mask=past_observed_mask, return_dict=True)

        #  Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        #  Apply positional encoding if enabled (preserves temporal ordering)
        h = self._apply_positional_encoding(h, observed_mask)

        #  Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        #  Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


# Helper classes for iTransformer and timexer – defined at module level to ensure proper pickling for checkpoint saving/loading. 
# Reference implementations from nixtla.

class FullAttention(nn.Module):
    """Scaled dot-product attention (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        q = queries.permute(0, 2, 1, 3)  # [B, H, L, E]
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        scale = self.scale or 1.0 / math.sqrt(E)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=scale,
        )
        V = attn_output.permute(0, 2, 1, 3).contiguous()
        return V, None


class AttentionLayer(nn.Module):
    """Multi-head attention layer wrapper (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, attention, hidden_size, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (hidden_size // n_heads)
        d_values = d_values or (hidden_size // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.key_projection = nn.Linear(hidden_size, d_keys * n_heads)
        self.value_projection = nn.Linear(hidden_size, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, hidden_size)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B_q, L_q, _ = queries.shape
        B_k, L_k, _ = keys.shape
        B_v, L_v, _ = values.shape
        H = self.n_heads

        # Project and reshape each tensor with its OWN dimensions
        queries = self.query_projection(queries).view(B_q, L_q, H, -1)
        keys = self.key_projection(keys).view(B_k, L_k, H, -1)
        values = self.value_projection(values).view(B_v, L_v, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B_q, L_q, -1)
        return self.out_projection(out), attn


class TransEncoderLayer(nn.Module):
    """Transformer encoder layer (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, attention, hidden_size, conv_hidden_size=None,
                dropout=0.1, activation="gelu"):
        super().__init__()
        conv_hidden_size = conv_hidden_size or 4 * hidden_size
        self.attention = attention
        self.conv1 = nn.Conv1d(hidden_size, conv_hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(conv_hidden_size, hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class TransEncoder(nn.Module):
    """Transformer encoder stack (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, hidden_size, 2).float() *
                   -(math.log(10000.0) / hidden_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time series transformers.

    Designed for agricultural time series where temporal ordering matters
    for understanding crop growth patterns and weather dependencies.

    Uses standard sinusoidal encoding with configurable max length based
    on aggregation frequency (daily: 365, weekly: 52, dekad: 36).

    Args:
        d_model: Dimension of the model (hidden size)
        max_len: Maximum sequence length (default: 365 for daily data)
        dropout: Dropout probability (default: 0.1)

    The encoding uses the standard transformer formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This creates unique positional embeddings for each timestep while
    allowing the model to learn relative positions through attention.
    """

    def __init__(self, d_model: int, max_len: int = 365, dropout: float = 0.1):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term for different frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: [1, max_len, d_model] for broadcasting
        pe = pe.unsqueeze(0)

        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor, observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            observed_mask: Optional boolean mask [batch, mask_len] indicating valid timesteps.
                           Will be truncated to match seq_len if mask_len > seq_len.

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        B, T, D = x.shape

        # Get positional encoding for this sequence length
        # pe is [1, max_len, d_model], take first T positions
        pos_encoding = self.pe[:, :T, :].expand(B, -1, -1)  # [B, T, D]

        # If mask is provided, zero out positional encoding for padded positions
        # This prevents positional information from leaking into attention for invalid timesteps
        if observed_mask is not None:
            # Truncate mask to match actual sequence length T
            # This handles cases where HuggingFace models internally downsample the sequence
            # but the mask still has the original (longer) length
            if observed_mask.shape[1] > T:
                observed_mask = observed_mask[:, :T]
            mask_expanded = observed_mask.unsqueeze(-1).float()  # [B, T, 1]
            pos_encoding = pos_encoding * mask_expanded

        # Add positional encoding to input and apply dropout
        return self.dropout(x + pos_encoding)


class EncoderLayer(nn.Module):
    """Encoder layer with self- and cross-attention (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, self_attention, cross_attention, d_model,
                d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention on endogenous patches
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask)[0]
        )
        x = self.norm1(x)

        # Cross-attention: global tokens attend to exogenous channels
        x_glb_ori = x[:, -1:, :]

        # Cross-attention: query=[batch*C, 1, D], key/value=[batch*C, C, D]
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb_ori, cross, cross, attn_mask=cross_mask)[0]
        )
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # Concatenate patches and updated global token
        y = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # Feed-forward network
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Encoder(nn.Module):
    """Encoder stack (nixtla implementation).

    References:
        - nixtla neuralforecast: https://github.com/Nixtla/neuralforecast
    """
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


# iTransformers
class iTransformerYieldModel(BaseTimeSeriesModel):
    """
    iTransformer: Inverted transformer for yield forecasting.

    Treats each channel (weather variable) as a token, not each timestep. This allows different variables to attend to each other,
    capturing cross-variable dependencies critical for crop yield prediction.

    Architectural adaptations for crop yield forecasting:
    - Inverted embedding: [B, T, C] -> [B, C, hidden] via projection over time
    - Instance normalization (RevIN) for per-series normalization
    - Masked pooling to handle variable season lengths
    - Static features concatenated after temporal processing
    - Deeper regression head for scalar output

    References:
        - [Yong Liu, Tengge Hu, Haoran Zhang, et al. "iTransformer: Inverted
           Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625)
        - nixtla implementation: https://github.com/Nixtla/neuralforecast
    """

    def _build_model(self) -> nn.Module:
        hidden_size = 64
        n_heads = 4
        e_layers = 2
        d_ff = 256
        dropout = 0.1

        # Calculate effective sequence length for alignment with linear models
        # iTransformer doesn't use lags in the traditional sense, but for consistency
        # we should truncate to the same context length as other models
        requested_lags = [1] if self.config.lag_years > 0 else [0]
        # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
        effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)


        seq_len = context_length  # Use context_length, not full seq_len
        n_channels = self.n_ts_features

        logging.info(
            f"[iTransformer BUILD] seq_len={self.config.seq_len}, context_length={context_length}, "
            f"n_channels={n_channels}, n_static={self.n_static_features}, hidden_size={hidden_size}, "
            f"n_heads={n_heads}, e_layers={e_layers}"
        )

        # Store context_length for forward pass truncation
        self._actual_context_length = context_length
        self._actual_lags = requested_lags

        # Inverted embedding: projects time dimension to hidden for each channel
        # Input: [B, T, C] -> permute to [B, C, T] -> project T to hidden -> [B, C, hidden]
        self.inverted_embedding = nn.Linear(seq_len, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Build encoder layers
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(scale=None, attention_dropout=dropout),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )

        # Initialize temporal attention
        self._init_temporal_attention(hidden_size)

        # Projection head: from hidden to prediction per channel
        # For quantile regression, outputs n_quantiles per channel; for MSE, outputs 1
        # We'll pool across channels in forward()
        self.channel_projection = nn.Linear(hidden_size, self.output_dim)

        # Final regression head: combines pooled channel representations with static features
        # Standardized to 2-layer MLP for consistency with other models
        head_input_dim = n_channels + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.LayerNorm(head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, self.output_dim)
        )

        logging.info(
            f"[iTransformer BUILD] head_input_dim={head_input_dim} "
            f"(n_channels={n_channels} + static={self.n_static_features}), "
            f"hidden_dim={head_input_dim // 2}, output_dim={self.output_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for iTransformer.

        Data flow:
          1. Truncate to context_length for alignment with other models
          2. Apply RevIN normalization (per-instance)
          3. Inverted embedding: [B, T, C] -> [B, C, hidden]
          4. Transformer encoder: channels attend to each other
          5. Project each channel to scalar -> [B, C]
          6. Masked pooling across channels (if needed)
          7. Concatenate with static features -> scalar prediction
        """
        B, T, C = x_ts.shape

        #  Truncate to context_length for alignment with linear models
        if T > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            T = self._actual_context_length

        # Apply instance-level (RevIN) normalization if enabled
        # Input is already z-score normalized by _shared_step, so we only
        # apply RevIN for additional instance-level normalization when enabled.
        # We do NOT call _normalize_time_series here as it would double-normalize.
        if self.config.use_revin:
            x_ts = self._apply_revin_normalization(x_ts, observed_mask)
        # else: x_ts is already normalized by _shared_step, use as-is

        #  Inverted embedding (nixtla's key innovation)
        # Permute from [B, T, C] to [B, C, T], then project T to hidden
        x_ts_permuted = x_ts.permute(0, 2, 1)  # [B, C, T]
        embedded = self.inverted_embedding(x_ts_permuted)  # [B, C, hidden]
        embedded = self.embedding_dropout(embedded)

        #  Transformer encoder (channels as tokens)
        enc_out, _ = self.encoder(embedded, attn_mask=None)  # [B, C, hidden]

        #  Project each channel to scalar representation
        channel_scalars = self.channel_projection(enc_out).squeeze(-1)  # [B, C]

        #  Handle missing channels based on observed_mask
        # After normalization, padded positions are zeros, so we check if a channel
        # has any non-zero (valid) values across the sequence.
        if observed_mask is not None:
            # Channel is valid if it has any observed (non-padded) timesteps
            # observed_mask: (B, T), we check per channel if any position is observed
            # Since all channels share the same mask (temporal masking), we expand it
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (B, T, 1)
            # A channel is valid if any timestep in that channel is observed
            channel_validity = (mask_expanded.sum(dim=1) > 0).float()  # (B, 1)
            channel_validity = channel_validity.expand(B, C).float()  # (B, C)
        else:
            # No mask: all channels are valid
            channel_validity = torch.ones(B, C, device=x_ts.device)

        #  Fill invalid channels with mean of valid ones
        # This handles channels that were entirely padded (all zeros after normalization)
        valid_channel_repr = channel_scalars * channel_validity  # Zero out invalid channels
        n_valid = channel_validity.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        mean_per_sample = (valid_channel_repr.sum(dim=-1, keepdim=True) / n_valid)  # (B, 1)

        filled_channels = torch.where(
            channel_validity > 0.5,
            channel_scalars,
            mean_per_sample.expand(-1, C)  # (B, C)
        )  # (B, C)

        #  Concatenate with static features and predict
        combined = torch.cat([filled_channels, x_static], dim=-1)  # [B, C + n_static]
        return self.regression_head(combined).squeeze(-1)  # [B]



# timexer MODEL
class TimeXerYieldModel(BaseTimeSeriesModel):
    """
    TimeXer: Cross-attention transformer for yield forecasting with exogenous variables.

    Combines endogenous patchinh with cross-attention to exogenous channels. This allows the model to capture both temporal patterns (via patching) and cross-variable interactions.

    Architectural adaptations for crop yield forecasting:
    - Endogenous: lag yields (broadcast over season) -> patched
    - Exogenous: weather variables (inverted embedding, channels as tokens)
    - Cross-attention: endogenous patches attend to exogenous channels
    - Instance normalization (RevIN) for per-series normalization
    - Masked pooling to handle variable season lengths
    - Static features concatenated after temporal processing
    - Deeper regression head for scalar output

    References:
        - [Yuxuan Wang, Haixu Wu, Jiaxiang Dong, et al. "TimeXer: Empowering
           Transformers for Time Series Forecasting with Exogenous Variables"](https://arxiv.org/abs/2402.19072)
        - nixtla implementation: https://github.com/Nixtla/neuralforecast
    """

    def _build_model(self) -> nn.Module:
        hidden_size = 64
        n_heads = 4
        e_layers = 2
        d_ff = 256
        dropout = 0.1

        # Calculate effective sequence length for alignment with linear models
        requested_lags = [1] if self.config.lag_years > 0 else [0]
        # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
        effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)


        seq_len = context_length  # Use context_length, not full seq_len
        n_channels = self.n_ts_features

        # Store hidden_size as instance attribute for use in forward()
        self.hidden_size = hidden_size

        # Store context_length for forward pass truncation
        self._actual_context_length = context_length
        self._actual_lags = requested_lags

        # Patching parameters
        if self.config.aggregation == "daily":
            self.patch_len = 16  # ~2 weeks
        elif self.config.aggregation == "weekly":
            self.patch_len = 4  # ~1 month
        else:  # dekad
            self.patch_len = 3  # ~1 month

        self.patch_num = max(1, seq_len // self.patch_len)

        logging.info(
            f"[TimeXer BUILD] seq_len={self.config.seq_len}, context_length={context_length}, "
            f"patch_len={self.patch_len}, patch_num={self.patch_num}, n_channels={n_channels}, "
            f"n_static={self.n_static_features}, hidden_size={hidden_size}, "
            f"n_heads={n_heads}, e_layers={e_layers}"
        )

        # Endogenous embedding: patching with global token
        # Lag yields will be treated as endogenous series
        self.endo_patch_embedding = nn.Linear(self.patch_len, hidden_size, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, n_channels, 1, hidden_size))
        self.positional_embedding = PositionalEmbedding(hidden_size)
        self.endo_dropout = nn.Dropout(dropout)

        # Exogenous embedding: inverted (channels as tokens)
        self.exo_inverted_embedding = nn.Linear(seq_len, hidden_size)
        self.exo_dropout = nn.Dropout(dropout)

        # Encoder with self- and cross-attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(scale=None, attention_dropout=dropout),
                                  hidden_size, n_heads),
                    AttentionLayer(FullAttention(scale=None, attention_dropout=dropout),
                                  hidden_size, n_heads),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )

        # Initialize temporal attention
        self._init_temporal_attention(hidden_size)

        # Flatten and project head
        # Input: [B, n_channels, (patch_num + 1), hidden_size]
        # After permute: [B, n_channels, hidden_size, (patch_num + 1)]
        # Flatten: [B, n_channels, hidden_size * (patch_num + 1)]
        head_nf = hidden_size * (self.patch_num + 1)
        self.flatten = nn.Flatten(start_dim=-2)
        self.channel_projection = nn.Linear(head_nf, self.output_dim)
        self.head_dropout = nn.Dropout(dropout)

        # Final regression head: combines channel representations with static features
        # Standardized to 2-layer MLP for consistency with other models
        head_input_dim = n_channels + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.LayerNorm(head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim // 2, self.output_dim)
        )

        logging.info(
            f"[TimeXer BUILD] head_input_dim={head_input_dim} "
            f"(n_channels={n_channels} + static={self.n_static_features}), "
            f"head_nf={head_nf}, hidden_dim={head_input_dim // 2}, output_dim={self.output_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for TimeXer.

        Data flow:
          1. Apply RevIN normalization (if enabled)
          2. Endogenous: lag yields (broadcast) -> patch + global token
          3. Exogenous: weather variables -> inverted embedding
          4. Encoder: self-attention on patches, cross-attention to exogenous
          5. Project each channel's patches to scalar
          6. Concatenate with static features -> scalar prediction
        """
        B, T, C = x_ts.shape

        #  Truncate to context_length for alignment with linear models
        if T > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            T = self._actual_context_length

        # Apply instance-level (RevIN) normalization if enabled
        # Input is already z-score normalized by _shared_step, so we only apply RevIN for additional instance-level normalization when enabled.
        # We do not call _normalize_time_series here as it would double-normalize.
        if self.config.use_revin:
            x_ts = self._apply_revin_normalization(x_ts, observed_mask)
        # else: x_ts is already normalized by _shared_step, use as-is

        #  Build endogenous series from lag yields
        # This creates a constant series (broadcast scalar) for patching
        endo = self._build_endogenous_series_lag(x_static, x_ts, observed_mask)  # [B, T, 1]

        #  Endogenous patching (nixtla's approach)
        # unfold: [B, T, 1] -> [B, 1, patch_num, patch_len]
        endo_patched = endo.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        endo_patched = endo_patched.squeeze(2)  # [B, patch_num, patch_len]

        # If T not perfectly divisible, pad
        if endo_patched.shape[1] < self.patch_num:
            pad_len = self.patch_num - endo_patched.shape[1]
            endo_patched = F.pad(endo_patched, (0, 0, 0, pad_len))

        # Expand to n_channels: [B, patch_num, patch_len] -> [B, n_channels, patch_num, patch_len]
        endo_patched = endo_patched.unsqueeze(1).expand(B, C, -1, -1)

        # Reshape for embedding: [B * C, patch_num, patch_len]
        endo_reshaped = endo_patched.reshape(B * C, -1, self.patch_len)

        # Project patches: [B*C, patch_num, patch_len] -> [B*C, patch_num, hidden_size]
        endo_patches = self.endo_patch_embedding(endo_reshaped)

        # Add positional encoding
        endo_patches = endo_patches + self.positional_embedding(endo_patches)

        # Reshape back: [B, C, patch_num, hidden_size]
        endo_patches = endo_patches.reshape(B, C, self.patch_num, self.hidden_size)
        hidden_size = self.hidden_size

        # Add global token: [B, C, 1, hidden_size]
        glb_token = self.global_token.repeat(B, 1, 1, 1)

        # Concatenate patches and global token: [B, C, (patch_num + 1), hidden_size]
        endo_with_glb = torch.cat([endo_patches, glb_token], dim=2)

        # Reshape for encoder: [B*C, (patch_num + 1), hidden_size]
        endo_embed = endo_with_glb.reshape(B * C, -1, hidden_size)
        endo_embed = self.endo_dropout(endo_embed)

        #  Exogenous inverted embedding (nixtla's approach)
        # Permute [B, T, C] to [B, C, T], project T to hidden
        x_ts_permuted = x_ts.permute(0, 2, 1)  # [B, C, T]
        exo_embed = self.exo_inverted_embedding(x_ts_permuted)  # [B, C, hidden_size]
        exo_embed = self.exo_dropout(exo_embed)

        # Expand exo to match endo batch dimension for cross-attention
        # exo_embed: [B, C, hidden] -> expand to [B*C, 1, hidden]
        # Each endogenous channel attends to all exogenous channels
        exo_expanded = exo_embed.unsqueeze(1).expand(B, C, -1, -1)  # [B, C, C, hidden]
        exo_expanded = exo_expanded.reshape(B * C, C, hidden_size)  # [B*C, C, hidden]

        #  Encoder with self- and cross-attention
        enc_out = self.encoder(endo_embed, exo_expanded)  # [B*C, (patch_num+1), hidden]

        #  Reshape and project per channel
        enc_out = enc_out.reshape(B, C, -1, self.hidden_size)  # [B, C, (patch_num+1), hidden]

        # Flatten and project each channel
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, hidden, (patch_num+1)]
        flat = self.flatten(enc_out)  # [B, C, hidden*(patch_num+1)]
        channel_repr = self.channel_projection(flat).squeeze(-1)  # [B, C]
        channel_repr = self.head_dropout(channel_repr)

        #  Handle missing channels based on observed_mask
        # After normalization, padded positions are zeros, so we check if a channel
        # has any non-zero (valid) values across the sequence.
        if observed_mask is not None:
            # Channel is valid if it has any observed (non-padded) timesteps
            # observed_mask: (B, T), we check per channel if any position is observed
            # Since all channels share the same mask (temporal masking), we expand it
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (B, T, 1)
            # A channel is valid if any timestep in that channel is observed
            channel_validity = (mask_expanded.sum(dim=1) > 0).float()  # (B, 1)
            channel_validity = channel_validity.expand(B, C).float()  # (B, C)
        else:
            # No mask: all channels are valid
            channel_validity = torch.ones(B, C, device=x_ts.device)

        valid_channel_repr = channel_repr * channel_validity  # [B, C]
        n_valid = channel_validity.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled_repr = (valid_channel_repr.sum(dim=-1, keepdim=True) / n_valid)  # (B, 1)

        # Fill invalid channels with mean of valid ones
        filled_channels = channel_repr.clone()
        mean_per_sample = pooled_repr.expand(B, C)  # [B, C]
        filled_channels = torch.where(
            channel_validity > 0.5,
            filled_channels,
            mean_per_sample
        )  # [B, C]

        #  Concatenate with static features and predict
        combined = torch.cat([filled_channels, x_static], dim=-1)  # [B, C + n_static]
        return self.regression_head(combined).squeeze(-1)  # [B]

    def _build_endogenous_series_lag(
        self,
        x_static: torch.Tensor,
        x_ts: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build endogenous series for TimeXer.

        Uses lag yield as the endogenous series (the target variable from previous years).
        While this creates a constant series over time, it is semantically closer to the
        original papers' intent where endogenous = the target variable.

        Falls back to mean of exogenous channels if lag yields are not available.
        """
        B, T, C = x_ts.shape

        # Use lag yield as endogenous (target variable from previous years)
        if self.config.lag_years > 0:
            static_names = self._get_static_feature_names()
            lag_indices = [
                i for i, name in enumerate(static_names)
                if name.startswith('lag_yield_')
            ]

            if lag_indices:
                # Use the most recent lag as endogenous
                lag_val = x_static[:, lag_indices[0]:lag_indices[0] + 1]  # [B, 1]
                endo = lag_val.unsqueeze(1).expand(B, T, 1)  # [B, T, 1] - broadcast over time

                if observed_mask is not None:
                    endo = endo * observed_mask.unsqueeze(-1).float()

                return endo

        # Fallback: mean of exogenous channels
        endo = x_ts.mean(dim=-1, keepdim=True)
        return endo


# TimesNet Architecture
class Inception_Block_V1(nn.Module):
    """
    Inception-style convolutional block for TimesNet.

    Uses multiple kernel sizes to capture different temporal patterns,
    then concatenates the outputs.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.layers = nn.ModuleList()

        # Create multiple conv layers with different ODD kernel sizes
        # Using odd kernels (1, 3, 5, 7, ...) ensures proper spatial dimension preservation
        # padding = kernel_size // 2 maintains output size = input size
        for i in range(self.num_kernels):
            kernel_size = 2 * i + 1  # Generates [1, 3, 5, 7, 9, 11] for num_kernels=6
            padding = i  # padding = (kernel_size - 1) // 2 ensures output size = input size
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through inception block.

        Args:
            x: Input tensor of shape (batch, in_channels, time, features)

        Returns:
            Output tensor of shape (batch, out_channels, time, features)
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.layers[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """
    TimesNet block: 1D to 2D transformation using FFT-based period detection.

    The key idea is to transform 1D time series into 2D tensors based on
    detected periods, then apply 2D convolutions to capture temporal variations.
    """
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ffn, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        # Parameter-efficient design with two Inception blocks
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ffn, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ffn, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        """
        Forward pass through TimesBlock.

        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)

        Returns:
            Output tensor of shape (batch, seq_len, n_channels)
        """
        B, T, N = x.size()

        #  FFT-based period detection
        period_list, period_weight = self._fft_for_period(x, self.k)

        #  Apply 2D conv for each detected period with per-period padding
        res = []
        for i in range(self.k):
            period = period_list[i]

            # Padding for this specific period to make it divisible
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, length - T, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x

            # Reshape: treat period as the "height" dimension for 2D conv
            # (B, length, N) -> (B, length//period, period, N) -> (B, N, length//period, period)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # Apply 2D convolution: captures 2D variations in time series
            out = self.conv(out)

            # Reshape back: (B, N, length//period, period) -> (B, length//period, period, N) -> (B, length, N)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # Truncate back to original T (all branches produce same output size)
            res.append(out[:, :T, :])

        #  Adaptive aggregation of multiple period-based representations
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        #  Residual connection
        res = res + x
        return res

    @staticmethod
    def _fft_for_period(x, k=2):
        """
        Use FFT to find top-k dominant periods in the time series.

        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)
            k: Number of top periods to extract

        Returns:
            period_list: List of top-k periods
            period_weight: Importance weights for each period
        """
        # Apply FFT along the time dimension
        xf = torch.fft.rfft(x, dim=1)

        # Find period by analyzing amplitudes
        # Average over batch and channels to get global frequency importance
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # Remove DC component (zero frequency)

        # Get top-k frequencies
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()

        # Convert frequency indices to periods
        period = x.shape[1] // top_list

        # Get amplitude-based weights for adaptive aggregation
        period_weight = abs(xf).mean(-1)[:, top_list]

        return period, period_weight


class TimesNetModel(BaseTimeSeriesModel):
    """
    TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.

    Transforms 1D time series into 2D tensors using FFT-based period detection, then applies 2D convolutions to capture complex temporal patterns.

    Adapted from: https://github.com/thuml/Time-Series-Library

    Modified for crop yield prediction (regression task instead of forecasting):
    - Removed predict_linear layer (used for sequence expansion in forecasting)
    - Added pooling after TimesNet blocks
    - Concatenate pooled representation with static features
    - Single regression head for yield prediction
    """

    def _build_model(self) -> nn.Module:
        """Build TimesNet model for yield prediction."""
        # Standardize context length calculation for fair comparison
        # Use [1] if lag_years > 0, else [0] to ensure alignment with linear models
        requested_lags = [1] if self.config.lag_years > 0 else [0]
        # context_length = self._get_standardized_context_length(self.config.seq_len, requested_lags)
        effective_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        context_length = self._get_standardized_context_length(effective_seq_len,requested_lags)

        # TimesNet hyperparameters
        d_model = 64
        top_k = 3  # Number of periods to detect
        num_kernels = 6  # Number of kernel sizes in Inception block
        num_layers = 3  # Number of TimesNet blocks
        e_layers = num_layers
        d_ffn = 256  # Hidden dimension in Inception blocks
        dropout = 0.1

        logging.info(f"[TimesNet BUILD] CONFIG: seq_len={self.config.seq_len}, "
                    f"context_length={context_length}, n_ts_features={self.n_ts_features}, "
                    f"n_static_features={self.n_static_features}, "
                    f"top_k={top_k}, num_kernels={num_kernels}, num_layers={num_layers}")

        # Store for forward pass
        self._actual_context_length = context_length
        self._actual_lags = requested_lags

        # Create TimesNet blocks
        self.times_net_blocks = nn.ModuleList([
            TimesBlock(
                seq_len=context_length,
                pred_len=1,  # Not used in regression, but needed for block structure
                top_k=top_k,
                d_model=d_model,
                d_ffn=d_ffn,
                num_kernels=num_kernels
            )
            for _ in range(e_layers)
        ])

        # Input embedding: project time series features to d_model
        self.enc_embedding = nn.Linear(self.n_ts_features, d_model)

        # Layer normalization after each block
        self.layer_norm = nn.LayerNorm(d_model)

        # Create regression head: pooled temporal features + static features
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, self.output_dim)
        )

        logging.info(f"[TimesNet BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}, output_dim={self.output_dim}")

        self._model_ready = True

        # Return nn.Identity() for base_model to avoid double-registration
        # The actual model components (enc_embedding, times_net_blocks) are registered
        # as direct attributes and used in forward(). Wrapping them in a Sequential
        # would register parameters twice. nn.Identity() provides a valid module
        # without duplicating parameters.
        return nn.Identity()

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through TimesNet.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) for valid timesteps

        Returns:
            Predictions of shape (batch,)
        """
        batch_size, seq_len = x_ts.shape[:2]

        # Truncate input to match context_length for fair comparison
        if seq_len > self._actual_context_length:
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            seq_len = self._actual_context_length

        #  Input embedding
        # (B, T, n_ts_features) -> (B, T, d_model)
        enc_out = self.enc_embedding(x_ts)

        #  Apply TimesNet blocks
        # TimesNet doesn't use normalization in the original implementation, but we add it for stability
        for block in self.times_net_blocks:
            enc_out = block(enc_out)
            enc_out = self.layer_norm(enc_out)

        #  Pool hidden state over time dimension
        # (B, T, d_model) -> (B, d_model)
        pooled = enc_out.mean(dim=1)

        #  Concatenate with static features
        combined = torch.cat([pooled, x_static], dim=-1)

        #  Regression head
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


def create_model(config: TSTModelConfig) -> BaseTimeSeriesModel:
    model_map = {
        "autoformer": AutoformerYieldModel,
        "patchtst": PatchTSTModel,
        "informer": InformerModel,
        "tst": TSTModel,
        "itransformer": iTransformerYieldModel,
        "timexer": TimeXerYieldModel,
        "timesnet": TimesNetModel,
    }
    # Only register TSMixer if import succeeded
    if TimeSeriesMixerForPrediction is not None:
        model_map["tsmixer"] = TSMixerModel

    if config.model_type.lower() not in model_map:
        raise ValueError(f"Unknown model_type '{config.model_type}'. "
                         f"Choose from: {list(model_map)}")

    # Create base model
    model = model_map[config.model_type.lower()](
        config, lr=config.lr, weight_decay=config.weight_decay)

    # Wrap with WFAN if enabled
    if config.use_wfan:
        from wfan_layer import WFANWrapper
        # Calculate number of time series features for WFAN
        use_sota = config.use_sota_features
        n_domain_ts = sum([config.use_gdd, config.use_rue, config.use_farquhar])
        n_ts_features = (
            len(config.time_series_vars)
            + n_domain_ts
            + (len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0)
        )

        model = WFANWrapper(
            model,
            seq_len=config.seq_len,
            n_vars=n_ts_features,
            K=config.wfan_k,
            lambda_coef=config.wfan_lambda,
            config=config
        )
        logging.info(f"[WFAN] ENABLED - K={config.wfan_k}, λ={config.wfan_lambda}")

    return model
