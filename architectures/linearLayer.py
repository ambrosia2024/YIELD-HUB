import sys
import logging
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

# Custom Classes and functions
from trendLayer import TrendModel
from biasCorrectionLayer import BiasCorrection
from modelconfig import LinearModelConfig

sys.path.append('../process/')
from validateModel import ModelMetrics
from loadData import _get_static_feature_names

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
    Base class for all time series forecasting architectures.
    """
    def __init__(self, config: LinearModelConfig, lr: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.trend_model = TrendModel()
        # Bias correction: automatic when use_recursive_lags=True
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
        # Prediction cache for recursive lag prediction
        self._yield_predictions_cache: Dict[Tuple[str, int], float] = {}

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

        use_sota = config.use_sota_features
        # Domain features (GDD, RUE, Farquhar) – additional TS channels beyond weather
        n_domain_ts = sum([config.use_gdd, config.use_rue, config.use_farquhar])
        self.n_ts_features = (
            len(config.time_series_vars)
            + n_domain_ts
            + (len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0)
        )
        self.num_time_features = 0

        include_spatial = config.include_spatial_features
        lag_years = config.lag_years

        n_crop_calendar = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop_calendar += 2
            else:
                n_crop_calendar += 1

        # Heat stress: 7 scalar features when enabled
        n_heat_stress = 7 if config.use_heat_stress_days else 0

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

        self._model_ready = False

        self.base_model = self._build_model()

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

    @abstractmethod
    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _get_standardized_context_length(seq_len: int, lags_sequence: List[int]) -> int:
        """
        Calculate standardized context length for all models (both linear and transformer).
        context length calculation: seq_len - max(lags_sequence)
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

    def _get_static_feature_names(self) -> List[str]:
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
        via fixed (non-learned) pooling helps or hurts model performance.

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

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z-score normalise each time series feature using training statistics.
        Also includes domain feature names (GDD, RUE, Farquhar) when enabled.
        """
        if self.feature_norm_params is None:
            if hasattr(self, 'trainer') and self.trainer is not None:
                dm_params = self.trainer.datamodule.feature_norm_params
                if dm_params is not None:
                    self.feature_norm_params = dm_params
                else:
                    raise RuntimeError("feature_norm_params not set in model or datamodule.")
            else:
                raise RuntimeError("feature_norm_params not set and no trainer available.")

        names = [f'weather_{f}' for f in self.config.weather_features]

        # Domain features (appended after base weather)
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
            if p['std'] < 1e-8:
                x[:, :, i] = torch.zeros_like(x_ts[:, :, i])
            else:
                x[:, :, i] = (x_ts[:, :, i] - p['mean']) / p['std']
            x[:, :, i] = torch.nan_to_num(x[:, :, i], nan=0.0, posinf=0.0, neginf=0.0)

        if observed_mask is not None:
            mask_expanded = observed_mask.unsqueeze(-1).float()
            x = x * mask_expanded

        return x

    def _normalize_and_impute_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalise static features then impute NaN → 0.0.
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
            if p['std'] < 1e-8:
                x[:, i] = torch.zeros_like(x_static[:, i])
            else:
                x[:, i] = (x_static[:, i] - p['mean']) / p['std']
            x[:, i] = torch.nan_to_num(x[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def on_train_start(self):
        """
        Fit per-location OLS trend lines and cache.
        """
        dm = self.trainer.datamodule
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

            # Compute and log weight distribution
            sample_years = [year_range[1], year_range[1]-3, year_range[1]-5, year_range[1]-10]
            weights = {y: np.exp(-(self._most_recent_year - y) / self.config.exponential_tau)
                      for y in sample_years if y >= year_range[0]}

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

        self._verify_mask_is_used()

    def _verify_mask_is_used(self):
        # Testing if masked-out inputs produces different outputs than unmasked.
        if not hasattr(self, '_model_ready') or not self._model_ready:
            logging.info(f"[{self.config.model_type}] Skipping mask verification (model not ready).")
            return

        if self.feature_norm_params is None:
            logging.info(f"[{self.config.model_type}] Skipping mask verification (norm params not available).")
            return

        # Using effective_seq_len if available (for linear models)
        seq_len = self._effective_seq_len if hasattr(self, '_effective_seq_len') else self.config.seq_len
        dummy_ts = torch.randn(2, seq_len, self.n_ts_features, device=self.device)
        dummy_static = torch.zeros(2, self.n_static_features, device=self.device)
        full_mask = torch.ones(2, seq_len, dtype=torch.bool, device=self.device)
        half_mask = full_mask.clone()
        half_mask[:, seq_len // 2:] = False

        # Normalize dummy data to match training distribution
        # Models expect normalized input as in _shared_step
        dummy_ts_n = self._normalize_time_series(dummy_ts, observed_mask=full_mask)
        dummy_static_n = self._normalize_and_impute_static(dummy_static)

        with torch.no_grad():
            out_full = self.forward(dummy_ts_n, dummy_static_n, observed_mask=full_mask)
            # Also normalize the half-masked data for consistency
            dummy_ts_half_n = self._normalize_time_series(dummy_ts, observed_mask=half_mask)
            out_half = self.forward(dummy_ts_half_n, dummy_static_n, observed_mask=half_mask)

        if torch.allclose(out_full, out_half, atol=1e-4):
            logging.warning(
                f"[{self.config.model_type}] past_observed_mask appears to have NO effect "
                f"on model output."
            )
        else:
            logging.info(f"[{self.config.model_type}] Mask verification passed.")

    def _compute_batch_trends(self, adm_ids, years: torch.Tensor, dm, lats, lons) -> torch.Tensor:
        """
        Compute normalised trend estimate for each sample in a batch.
        """
        if not self.config.use_residual_trend:
            raise RuntimeError(
                "_compute_batch_trends called with use_residual_trend=False."
            )

        test_items = []
        for i, (loc, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            test_items.append({
                KEY_LOC: loc,
                KEY_YEAR: year_int
            })

        trend_predictions_orig = self.trend_model._predict_trend(test_items)

        # DEBUG: Check what _predict_trend actually returned
        print(f"[DEBUG _compute_batch_trends] len(test_items)={len(test_items)}, _predict_trend returned shape={trend_predictions_orig.shape}, dtype={trend_predictions_orig.dtype}")
        if hasattr(trend_predictions_orig, 'flatten'):
            trend_predictions_orig_flat = trend_predictions_orig.flatten()
            print(f"[DEBUG _compute_batch_trends] After flatten, shape={trend_predictions_orig_flat.shape}")
        else:
            trend_predictions_orig_flat = trend_predictions_orig

        # Normalize to z-score space
        trends_z = (trend_predictions_orig_flat - dm.y_mean) / dm.y_std
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

        # Apply tokenization if enabled (ablation study)
        x_ts, validity_mask = self._apply_tokenization(x_ts, observed_mask=validity_mask)

        if self.config.use_residual_trend:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # DEBUG: Log pred shape
        print(f"[DEBUG _shared_step] pred.shape={pred.shape}")

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

    def on_test_start(self):
        # Only clear cache if not using recursive lags, or if cache is empty (first call)
        # For recursive lags in multi-year testing, preserve cache across years
        if self.config.use_recursive_lags and self.config.lag_years > 0:
            if not self._yield_predictions_cache:
                logging.info("[Recursive Lags] Prediction cache initialized for testing")
            else:
                logging.info(f"[Recursive Lags] Preserving cache with {len(self._yield_predictions_cache)} predictions for multi-year testing")
        else:
            self._yield_predictions_cache.clear()

        # Initialize per-year prediction storage for CSV results
        dm = self.trainer.datamodule
        if hasattr(dm, '_test_years') and dm._test_years is not None:
            self._test_years = dm._test_years
            self._per_year_preds = {year: {'preds': [], 'targets': []} for year in self._test_years}
            logging.info(f"[Per-Year Metrics] Initialized storage for test years: {sorted(self._test_years)}")
        else:
            logging.warning("[Per-Year Metrics] Datamodule has no _test_years set, per-year metrics will not be computed")
            self._test_years = set()
            self._per_year_preds = {}

    def _replace_lags_with_predictions(
        self, x_static: torch.Tensor, years: torch.Tensor, adm_ids: List[str]
    ) -> torch.Tensor:
        # Replace observed lag yield features with cached predictions.
        x_static_modified = x_static.clone()

        # Get indices of lag features in static features
        static_names = self._get_static_feature_names()
        lag_indices = [
            i for i, name in enumerate(static_names)
            if name.startswith('lag_yield_')
        ]

        if not lag_indices:
            return x_static_modified

        # Replace each sample's lag features with cached predictions
        for sample_idx, (adm_id, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            # Replace each lag feature (lag_yield_1, lag_yield_2, etc.)
            for lag_offset, lag_idx in enumerate(lag_indices, start=1):
                lag_year = year_int - lag_offset
                cache_key = (adm_id, lag_year)

                if cache_key in self._yield_predictions_cache:
                    # Use cached prediction (in original scale, will be normalized later)
                    cached_pred = self._yield_predictions_cache[cache_key]
                    x_static_modified[sample_idx, lag_idx] = cached_pred
                    # else: No cached prediction available, keep original (will be NaN/imputed)

        return x_static_modified

    def _cache_predictions(
        self, predictions_z: torch.Tensor, years: torch.Tensor,
        adm_ids: List[str], dm
    ):
        """
        Cache predictions in original scale for recursive lag prediction.

        Applies bias correction if enabled (automatic with use_recursive_lags=True).
        Bias-corrected predictions are cached to reduce error compounding in recursive forecasting.

        Args:
            predictions_z: Predictions in z-score space [B]
            years: Years [B]
            adm_ids: Location IDs [B]
            dm: DataModule for denormalization
        """
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
            cache_key = (adm_id, int(year))
            # Extract scalar value: use median (index 1) for quantile predictions, otherwise use the value
            pred_cpu = pred.detach().cpu() if hasattr(pred, 'detach') else pred
            if pred_cpu.numel() > 1:
                # Quantile predictions: use median (index 1)
                pred_value = pred_cpu[1].item()
            else:
                pred_value = pred_cpu.item()
            self._yield_predictions_cache[cache_key] = pred_value

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
        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trends with proper broadcasting for quantile regression
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

        loss = self._compute_weighted_loss(final_pred_z, y_z)

        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        final_pred_orig = final_pred_z.detach() * y_std + y_mean
        y_orig = y_z.detach() * y_std + y_mean

        final_pred_clipped = torch.clamp(final_pred_orig, min=0.0)

        # Log clip rate as diagnostic (helps identify model issues)
        clip_rate = (final_pred_orig < 0.0).float().mean()
        self.log(f'{stage}/clip_rate', clip_rate, prog_bar=False)

        # Only update metrics here — never call compute() mid-epoch
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
        return self._shared_step(batch, self.train_metrics, "train_loss")

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log('train/mse', results['mse'], prog_bar=False)
        self.log('train/mae', results['mae'], prog_bar=False)
        self.log('train/r2', results['r2'], prog_bar=False)
        self.log('train/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('train/mape', results['mape'], prog_bar=False)
        self.log('train/smape', results['smape'], prog_bar=False)
        self.train_metrics.log_results("train")
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Accumulate predictions for bias correction fitting if enabled
        if self.bias_correction is not None and hasattr(self, '_val_preds'):
            loss, preds_clipped, targets, years = self._eval_step_with_clipping(
                batch, self.val_metrics, "val_loss", "val", return_orig=True
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
            self._eval_step_with_clipping(batch, self.val_metrics, "val_loss", "val")

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.log('val/mse', results['mse'], prog_bar=False)
        self.log('val/mae', results['mae'], prog_bar=False)
        self.log('val/r2', results['r2'], prog_bar=False)
        self.log('val/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('val/mape', results['mape'], prog_bar=False)
        self.log('val/smape', results['smape'], prog_bar=False)
        self.log('val/nrmse', results['nrmse'], prog_bar=False)
        self.val_metrics.log_results("val")
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

    def test_step(self, batch, batch_idx):
        """Test step with optional recursive lag prediction and per-year accumulation."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            # Use standard evaluation and get predictions for per-year metrics
            loss, preds_clipped, targets, years = self._eval_step_with_clipping(
                batch, self.test_metrics, 'test_loss', stage='test', return_orig=True
            )
            self._accumulate_per_year_predictions(preds_clipped, targets, years)
            return loss

        # Recursive lag mode: modify batch to use cached predictions
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Replace lag features with cached predictions
        x_static_modified = self._replace_lags_with_predictions(x_static, years, adm_ids)

        # Create modified batch
        modified_batch = (x_ts, x_static_modified, y_z, years, adm_ids, lats, lons, validity_mask)

        # Run evaluation step and get predictions
        loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
            modified_batch, self.test_metrics, 'test_loss', stage='test',
            return_orig=True, return_predictions=True
        )

        # Cache predictions in z-score space for recursive lag prediction
        self._cache_predictions(preds_z, years, adm_ids, dm)

        # Accumulate per-year predictions for CSV results
        self._accumulate_per_year_predictions(preds_clipped, targets, years)

        return loss

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

    def _compute_per_year_metrics_from_preds(self) -> dict:
        # Compute per-year metrics from accumulated predictions using torchmetrics.
        results = {}

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

            # SMAPE - manual computation
            smape_val = torch.mean(2.0 * torch.abs(preds - targets) /
                                  (torch.abs(preds) + torch.abs(targets) + 1e-8))

            # R² score requires at least 2 samples
            if len(preds) >= 2:
                r2_val = r2(preds, targets)
            else:
                r2_val = torch.tensor(float('nan'))  # Cannot compute R² with single sample

            # Store results with year suffix
            results[f'mse_{year}'] = mse_val.item()
            results[f'mae_{year}'] = mae_val.item()
            results[f'rmse_{year}'] = rmse_val.item()
            results[f'r2_{year}'] = r2_val.item()
            results[f'mape_{year}'] = mape_val.item()
            results[f'smape_{year}'] = smape_val.item()
            # Add nrmse for per-year metrics (consistent with FM script)
            nrmse_val = rmse_val / (targets.mean().clamp(min=1e-8))
            results[f'nrmse_{year}'] = nrmse_val.item()

        # Compute overall metrics (across all years)
        all_preds = []
        all_targets = []
        for data in self._per_year_preds.values():
            if len(data['preds']) > 0:
                all_preds.extend(data['preds'])
                all_targets.extend(data['targets'])

        if all_preds:
            all_preds_tensor = torch.tensor(all_preds)
            all_targets_tensor = torch.tensor(all_targets)

            mse_overall = MeanSquaredError()
            r2_overall = R2Score()
            mae_overall = MeanAbsoluteError()
            mape_overall = MeanAbsolutePercentageError()

            mse_val = mse_overall(all_preds_tensor, all_targets_tensor)
            mae_val = mae_overall(all_preds_tensor, all_targets_tensor)
            mape_val = mape_overall(all_preds_tensor, all_targets_tensor)
            rmse_val = torch.sqrt(mse_val)
            smape_val = torch.mean(2.0 * torch.abs(all_preds_tensor - all_targets_tensor) /
                                  (torch.abs(all_preds_tensor) + torch.abs(all_targets_tensor) + 1e-8))
            nrmse_val = rmse_val / (all_targets_tensor.mean().clamp(min=1e-8))

            # R² score requires at least 2 samples
            if len(all_preds_tensor) >= 2:
                r2_val = r2_overall(all_preds_tensor, all_targets_tensor)
            else:
                r2_val = torch.tensor(float('nan'))  # Cannot compute R² with single sample

            results['mse_overall'] = mse_val.item()
            results['mae_overall'] = mae_val.item()
            results['rmse_overall'] = rmse_val.item()
            results['r2_overall'] = r2_val.item()
            results['mape_overall'] = mape_val.item()
            results['smape_overall'] = smape_val.item()
            results['nrmse_overall'] = nrmse_val.item()

        return results

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log('test/mse', results['mse'], prog_bar=False)
        self.log('test/mae', results['mae'], prog_bar=False)
        self.log('test/r2', results['r2'], prog_bar=False)
        self.log('test/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('test/mape', results['mape'], prog_bar=False)
        self.log('test/smape', results['smape'], prog_bar=False)
        self.log('test/nrmse', results['nrmse'], prog_bar=False)
        self.test_metrics.log_results("test")

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
        else:
            batch_trends = None

        # Forward pass
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trends with proper broadcasting for quantile regression
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

        # Clip predictions to physically meaningful range
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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


# Linear Baseline architectures
class NLinearYieldModel(BaseTimeSeriesModel):
    """
    NLinear baseline: single linear layer per channel with last-value subtraction.

    From "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023)
    """

    def _build_model(self) -> nn.Module:
        # Standardize sequence length calculation for fair comparison with transformers
        lags_sequence = [1] if self.config.lag_years > 0 else [0]
        # Use tokenized sequence length if tokenization is enabled
        base_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        effective_seq_len = self._get_standardized_context_length(
            base_seq_len, lags_sequence
        )

        logging.info(
            f"[NLinear BUILD] seq_len={self.config.seq_len}, "
            f"base_seq_len={base_seq_len}, "
            f"effective_seq_len={effective_seq_len}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}, "
            f"output_dim={self.output_dim}"
        )

        # One linear layer maps (effective_seq_len,) = (1,) per channel
        # The regression head handles the final output_dim (e.g., multiple quantiles)
        self.temporal_linear = nn.Linear(effective_seq_len, 1)

        # After pooling across channels: n_ts_features scalars + static features = yield
        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim),
        )

        logging.info(
            f"[NLinear BUILD] temporal_linear: ({effective_seq_len} → {self.output_dim}), "
            f"regression_head input: {combined_dim}"
        )

        # Store effective sequence length for forward pass
        self._effective_seq_len = effective_seq_len

        self._model_ready = True
        return nn.Identity()

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through NLinear model.
        """
        B, T, C = x_ts.shape

        # Truncate to effective sequence length to compare with transformers
        if T > self._effective_seq_len:
            x_ts = x_ts[:, :self._effective_seq_len, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._effective_seq_len]
            T = self._effective_seq_len

        # Find last valid (non-padded) value per sample per channel
        if observed_mask is not None:
            last_valid_idx = observed_mask.long().sum(dim=1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)

            idx_expanded = last_valid_idx.view(B, 1, 1).expand(B, 1, C)
            last_val = x_ts.gather(dim=1, index=idx_expanded)
        else:
            last_val = x_ts[:, -1:, :]

        # Subtract last value (distribution shift normalization)
        x_shifted = x_ts - last_val

        # Apply shared linear layer across sequence dimension per channel
        x_t = x_shifted.transpose(1, 2)
        out = self.temporal_linear(x_t)

        # Add last value back (undo distribution shift)
        last_val_t = last_val.transpose(1, 2)
        out = out + last_val_t

        # Squeeze to (B, C)
        pooled = out.squeeze(-1)

        # Concatenate with static features and predict yield
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class DLinearYieldModel(BaseTimeSeriesModel):
    """
    DLinear baseline: decompose into trend + remainder, apply separate linear layers.

    From "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023)
    """

    KERNEL_SIZES = {
        "daily": 25,
        "weekly": 7,
        "dekad": 5,
    }

    def _build_model(self) -> nn.Module:
        # Standardize sequence length calculation for fair comparison with transformers
        lags_sequence = [1] if self.config.lag_years > 0 else [0]
        # Use tokenized sequence length if tokenization is enabled
        base_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        effective_seq_len = self._get_standardized_context_length(
            base_seq_len, lags_sequence
        )

        kernel_size = self.KERNEL_SIZES.get(self.config.aggregation, 25)

        if kernel_size % 2 == 0:
            kernel_size += 1

        logging.info(
            f"[DLinear BUILD] seq_len={self.config.seq_len}, "
            f"base_seq_len={base_seq_len}, "
            f"effective_seq_len={effective_seq_len}, "
            f"kernel_size={kernel_size}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}, "
            f"output_dim={self.output_dim}"
        )

        self._kernel_size = kernel_size
        self._effective_seq_len = effective_seq_len

        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        # Each linear layer maps (effective_seq_len,) = (1,) per channel
        # The regression head handles the final output_dim (e.g., multiple quantiles)
        self.trend_linear = nn.Linear(effective_seq_len, 1)
        self.remainder_linear = nn.Linear(effective_seq_len, 1)

        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim),
        )

        logging.info(
            f"[DLinear BUILD] trend_linear+remainder_linear: ({effective_seq_len} → {self.output_dim}), "
            f"regression_head input: {combined_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def _extract_trend(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract trend component via symmetric moving average.

        When observed_mask is provided, padded positions (mask=0) are excluded
        from the moving average to avoid diluting the trend estimate with zeros.
        """
        B, C, T = x.shape
        pad = (self._kernel_size - 1) // 2

        if observed_mask is not None:
            # Convert mask to (B, C, T) format to match x
            mask_c = observed_mask.unsqueeze(1).float()  # (B, 1, T) -> (B, 1, T)

            # Pad both x and mask
            x_padded = F.pad(x, (pad, pad), mode='replicate')
            mask_padded = F.pad(mask_c, (pad, pad), mode='constant', value=0.0)

            # Compute masked moving average:
            # - AvgPool1d divides by kernel_size, so we divide both x*mask and mask
            # - The kernel_size divisions cancel: (sum(x*mask)/k) / (sum(mask)/k) = sum(x*mask)/sum(mask)
            # - This gives the correct mean over only valid (non-padded) positions
            trend_padded = self.moving_avg(x_padded * mask_padded)
            mask_sum = self.moving_avg(mask_padded).clamp(min=1e-8)

            # Normalize by actual mask sum to get true masked mean
            trend_padded = trend_padded / mask_sum
            trend = trend_padded[:, :, :T]
        else:
            # Original behavior when no mask is provided
            x_padded = F.pad(x, (pad, pad), mode='replicate')
            trend = self.moving_avg(x_padded)
            trend = trend[:, :, :T]

        return trend

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DLinear model.
        """
        B, T, C = x_ts.shape

        # Truncate to effective sequence length for fair comparison with transformers
        if T > self._effective_seq_len:
            x_ts = x_ts[:, :self._effective_seq_len, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._effective_seq_len]
            T = self._effective_seq_len

        x_t = x_ts.transpose(1, 2)

        trend = self._extract_trend(x_t, observed_mask=observed_mask)
        remainder = x_t - trend

        trend_out = self.trend_linear(trend)
        remainder_out = self.remainder_linear(remainder)

        pooled = (trend_out + remainder_out).squeeze(-1)

        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.

    Computes per-instance, per-channel mean and std:
        mean = sum(x * mask) / sum(mask)
        std = sqrt(sum((x - mean)^2 * mask) / sum(mask) + eps)

    Uses population std (dividing by N, not N-1) for consistency with the Reversible Instance Normalization paper. 
    The affine transform (gamma, beta) is learned during training.
    """

    def __init__(self, num_features: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize input using per-instance, per-channel statistics.
        """
        B, T, C = x.shape

        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)

            instance_mean = (x * mask_f).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)

            sq_dev = ((x - instance_mean) ** 2 * mask_f).sum(dim=1, keepdim=True)
            instance_std = torch.sqrt(sq_dev / valid_counts.unsqueeze(1) + self.eps)
        else:
            instance_mean = x.mean(dim=1, keepdim=True)
            instance_std = x.std(dim=1, keepdim=True) + self.eps

        x_norm = (x - instance_mean) / instance_std

        if observed_mask is not None:
            x_norm = x_norm * observed_mask.float().unsqueeze(-1)

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, instance_mean, instance_std


class RLinearYieldModel(BaseTimeSeriesModel):
    """
    RLinear: NLinear with RevIN instance normalization.
    From "An Analysis of Linear Time Series Forecasting Models" (Li et al. 2024)
    """

    def _build_model(self) -> nn.Module:
        # Standardize sequence length calculation for fair comparison with transformers
        lags_sequence = [1] if self.config.lag_years > 0 else [0]
        # Use tokenized sequence length if tokenization is enabled
        base_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        effective_seq_len = self._get_standardized_context_length(
            base_seq_len, lags_sequence
        )

        logging.info(
            f"[RLinear BUILD] seq_len={self.config.seq_len}, "
            f"base_seq_len={base_seq_len}, "
            f"effective_seq_len={effective_seq_len}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}, "
            f"output_dim={self.output_dim}"
        )

        self.revin = RevIN(
            num_features=self.n_ts_features,
            eps=1e-8,
            affine=True,
        )

        # One linear layer maps (effective_seq_len,) = (1,) per channel
        # The regression head handles the final output_dim (e.g., multiple quantiles)
        self.temporal_linear = nn.Linear(effective_seq_len, 1)

        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, self.output_dim),
        )

        logging.info(
            f"[RLinear BUILD] revin channels={self.n_ts_features} (affine=True), "
            f"temporal_linear: ({effective_seq_len} → {self.output_dim}), "
            f"regression_head input: {combined_dim}"
        )

        self._effective_seq_len = effective_seq_len

        self._model_ready = True
        return nn.Identity()

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through RLinear model.
        """
        B, T, C = x_ts.shape

        # Truncate to effective sequence length for fair comparison with transformers
        if T > self._effective_seq_len:
            x_ts = x_ts[:, :self._effective_seq_len, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._effective_seq_len]
            T = self._effective_seq_len

        # RevIN normalize
        # x_ts is already in z-score space (normalized by _normalize_time_series). Applying RevIN here is double normalization: global z-score + per-instance RevIN.
        # This is intentional for handling per-instance distribution shifts on top of global normalization, but differs from the original RevIN paper which assumes
        # RevIN is applied to raw data.
        x_revin, instance_mean, instance_std = self.revin(x_ts, observed_mask)

        # Apply shared linear across sequence dimension per channel
        x_t = x_revin.transpose(1, 2)
        out = self.temporal_linear(x_t)

        # Squeeze to (B, C)
        pooled = out.squeeze(-1)

        # Concatenate static features and predict yield
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class XLinearGatingBlock(nn.Module):
    # Shared gating block used by XLinear.
    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_mlp(x)
        out = x * gate
        out = self.dropout(out)
        return self.norm(out + x)


class XLinearYieldModel(BaseTimeSeriesModel):
    """
    XLinear adapted for crop yield regression with static features.

    A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs (Chen et al., AAAI 2026)

    Optional RevIN normalization:
    When config.use_revin=True, applies Reversible Instance Normalization to the endogenous series before embedding. This provides per-instance
    normalization (vs. global z-scoring), which can help with distribution shift across locations and years.
    """

    def _build_model(self) -> nn.Module:
        # Standardize sequence length calculation for fair comparison with transformers
        lags_sequence = [1] if self.config.lag_years > 0 else [0]
        # Use tokenized sequence length if tokenization is enabled
        base_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        effective_seq_len = self._get_standardized_context_length(
            base_seq_len, lags_sequence
        )

        n_exo = self.n_ts_features
        hidden = self.config.xlinear_hidden_size
        t_ff = self.config.xlinear_temporal_ff
        c_ff = self.config.xlinear_channel_ff
        drop = self.config.xlinear_dropout

        logging.info(
            f"[XLinear BUILD] seq_len={self.config.seq_len}, "
            f"base_seq_len={base_seq_len}, "
            f"effective_seq_len={effective_seq_len}, "
            f"n_exo_channels={n_exo}, "
            f"n_static={self.n_static_features}, hidden={hidden}, "
            f"temporal_ff={t_ff}, channel_ff={c_ff}, dropout={drop}"
        )

        # Store effective sequence length for forward pass
        self._effective_seq_len = effective_seq_len

        # RevIN for endogenous series normalization (optional)
        if self.config.use_revin:
            self.revin = RevIN(
                num_features=1,  # endogenous series is 1-dimensional
                eps=1e-8,
                affine=True,
            )
            logging.info("[XLinear BUILD] RevIN normalization enabled for endogenous series")

        # Endogenous embedding
        self.endo_embed = nn.Linear(1, hidden)

        # Learnable global token
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.global_token, std=0.02)

        # Exogenous embedding
        self.exo_embed = nn.Linear(1, hidden)

        # Time-wise Gating Module (TGM)
        self.tgm = XLinearGatingBlock(
            input_dim=hidden,
            ff_dim=t_ff,
            dropout=drop,
        )

        # Variate-wise Gating Module (VGM)
        self.vgm = XLinearGatingBlock(
            input_dim=2 * hidden,
            ff_dim=c_ff,
            dropout=drop,
        )

        self.vgm_proj = nn.Linear(2 * hidden, hidden)

        # Prediction head
        head_input_dim = hidden + (n_exo * hidden) + self.n_static_features

        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.LayerNorm(head_input_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.LayerNorm(head_input_dim // 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_input_dim // 4, self.output_dim),
        )

        logging.info(
            f"[XLinear BUILD] head_input_dim={head_input_dim} "
            f"(endo_pooled={hidden} + exo_pooled={n_exo * hidden} "
            f"+ static={self.n_static_features}), output_dim={self.output_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def _build_endogenous_series(
        self,
        x_static: torch.Tensor,
        x_ts: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Construct the endogenous series for XLinear.

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
                # Use only the most recent lag (lag_indices[0]) as the endogenous series
                lag_val = x_static[:, lag_indices[0]:lag_indices[0]+1]  # [B, 1]
                endo = lag_val.unsqueeze(1).expand(B, T, 1)  # [B, T, 1] - broadcast over time

                if observed_mask is not None:
                    endo = endo * observed_mask.unsqueeze(-1).float()

                # Don't apply RevIN on constant lag yield series (std ≈ 0)
                return endo

        # Fallback: mean of all exogenous channels
        endo = x_ts.mean(dim=-1, keepdim=True)

        # Apply RevIN normalization if enabled (only for fallback case)
        # This applies RevIN on already-normalized data, which is double normalization. This is intentional for handling per-instance distribution shifts on top of global normalization.
        if self.config.use_revin:
            endo, _, _ = self.revin(endo, observed_mask)

        return endo

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        XLinear forward pass.
        """
        B, T, C = x_ts.shape

        # Truncate to effective sequence length for fair comparison with transformers
        if T > self._effective_seq_len:
            x_ts = x_ts[:, :self._effective_seq_len, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._effective_seq_len]
            T = self._effective_seq_len

        # Construct endogenous series
        x_endo = self._build_endogenous_series(x_static, x_ts, observed_mask)

        # Embed endogenous
        h_endo = self.endo_embed(x_endo)

        # Initialize global token
        G = self.global_token.expand(B, 1, self.config.xlinear_hidden_size)

        # Embed exogenous channels
        x_exo_reshaped = x_ts.permute(0, 2, 1).reshape(B * C, T, 1)
        h_exo_flat = self.exo_embed(x_exo_reshaped)
        h_exo = h_exo_flat.reshape(B, C, T, self.config.xlinear_hidden_size)

        # Apply TGM to endogenous embeddings
        h_tgm = self.tgm(h_endo)

        # Update global token
        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)
            tgm_mean = (h_tgm * mask_f).sum(dim=1, keepdim=True)
            tgm_mean = tgm_mean / valid_counts.unsqueeze(1)
        else:
            tgm_mean = h_tgm.mean(dim=1, keepdim=True)

        G = G + tgm_mean

        # VGM: G × each exo channel
        G_expanded = G.expand(B, T, self.config.xlinear_hidden_size)
        h_exo_t = h_exo.permute(0, 2, 1, 3)
        G_for_vgm = G_expanded.unsqueeze(2).expand(B, T, C, self.config.xlinear_hidden_size)

        vgm_input = torch.cat([G_for_vgm, h_exo_t], dim=-1)
        vgm_flat = vgm_input.reshape(B * T * C, 2 * self.config.xlinear_hidden_size)
        h_vgm_flat = self.vgm(vgm_flat)
        h_vgm_flat = self.vgm_proj(h_vgm_flat)
        h_vgm = h_vgm_flat.reshape(B, T, C, self.config.xlinear_hidden_size)

        # Pool and predict
        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)
            endo_pooled = (h_tgm * mask_f).sum(dim=1) / valid_counts

            mask_tc = observed_mask.float().unsqueeze(-1).unsqueeze(-1)
            valid_tc = mask_tc.sum(dim=1).clamp(min=1)
            exo_pooled = (h_vgm * mask_tc).sum(dim=1) / valid_tc
        else:
            endo_pooled = h_tgm.mean(dim=1)
            exo_pooled = h_vgm.mean(dim=1)

        exo_pooled_flat = exo_pooled.reshape(B, C * self.config.xlinear_hidden_size)

        combined = torch.cat([
            endo_pooled,
            exo_pooled_flat,
            x_static,
        ], dim=-1)

        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class OLinearYieldModel(BaseTimeSeriesModel):
    """
    OLinear-C adapted for crop yield regression.

    From "OLinear: A Linear Orthogonal Transformation for Time Series Forecasting"
    (Yue et al., 2025)

    OLinear-C variant uses fixed channel correlation matrix computed from
    temporal correlations in the training data, making it well-suited for
    agricultural datasets with limited samples.

    Key adaptations for regression:
    - Removed predict_linear layer (for sequence forecasting)
    - Added pooling over temporal dimension after OLinear blocks
    - Concatenated pooled features with static features
    - Added regression head for single-value yield prediction
    """

    HIDDEN_SIZE = 64
    FF_SIZE = 256
    DROPOUT = 0.1
    E_LAYERS = 2
    EMBED_SIZE = 4

    def _build_model(self) -> nn.Module:
        # Standardize sequence length calculation for fair comparison with transformers
        lags_sequence = [1] if self.config.lag_years > 0 else [0]
        # Use tokenized sequence length if tokenization is enabled
        base_seq_len = self._tokenized_seq_len if self._tokenized_seq_len else self.config.seq_len
        effective_seq_len = self._get_standardized_context_length(
            base_seq_len, lags_sequence
        )

        n_channels = self.n_ts_features
        hidden = self.HIDDEN_SIZE
        d_ff = self.FF_SIZE
        dropout = self.DROPOUT

        logging.info(
            f"[OLinear BUILD] seq_len={self.config.seq_len}, "
            f"base_seq_len={base_seq_len}, "
            f"effective_seq_len={effective_seq_len}, "
            f"n_channels={n_channels}, "
            f"n_static={self.n_static_features}, hidden={hidden}, "
            f"ff_size={d_ff}"
        )

        # Store effective sequence length for forward pass
        self._effective_seq_len = effective_seq_len

        # RevIN for input/output normalization
        self.revin = RevIN(
            num_features=n_channels,
            eps=1e-8,
            affine=True,
        )

        # Embedding dimension for token expansion
        self.embed_size = self.EMBED_SIZE
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        # Channel correlation matrix (will be initialized during first forward pass)
        # Register as buffer to ensure it moves to the same device as the model
        self.register_buffer('channel_corr_mat', None)
        self._corr_mat_initialized = False

        # Build OLinear encoder layers
        encoder_layers = []
        for _ in range(self.E_LAYERS):
            encoder_layers.append(
                OLinearEncoderLayer(
                    d_model=hidden,
                    d_ff=d_ff,
                    n_channels=n_channels,
                    dropout=dropout,
                    activation='gelu',
                )
            )

        # OLinear-style orthogonal transformation adapted for regression
        # Keep the core innovation: process channels as tokens with proper normalization
        self.ortho_trans = nn.Sequential(
            nn.Linear(effective_seq_len * self.embed_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.embed_size),
        )

        # Learnable delta parameters for orthogonal transformation (from OLinear-C)
        self.delta1 = nn.Parameter(torch.zeros(1, n_channels, 1, effective_seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, n_channels, 1, 1))

        # Final projection and regression head
        combined_dim = n_channels + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, self.output_dim),
        )

        logging.info(
            f"[OLinear BUILD] regression_head input: {combined_dim}, "
            f"output: {self.output_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def _compute_channel_correlation(self, x_ts: torch.Tensor):
        """
        Compute temporal correlation matrix across channels.

        Args:
            x_ts: Time series data [B, T, C]

        Returns:
            Correlation matrix [C, C]
        """
        B, T, C = x_ts.shape

        # Compute correlation matrix across time dimension for each channel pair
        # Normalize each channel
        x_norm = x_ts - x_ts.mean(dim=1, keepdim=True)
        x_norm = x_norm / (x_norm.std(dim=1, keepdim=True) + 1e-8)

        # Compute correlation matrix
        # x_norm: [B, T, C] -> [B, C, T]
        x_t = x_norm.transpose(1, 2)
        # Correlation: [B, C, C]
        corr = torch.bmm(x_t, x_t.transpose(1, 2)) / T

        # Average across batch
        corr_mean = corr.mean(dim=0)

        return corr_mean

    def tokenEmb(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Expand dimension with learnable embeddings.

        Args:
            x: Input [B, T, C]
            embeddings: Learnable embeddings [1, D]

        Returns:
            Expanded tensor [B, N, T, D]
        """
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)

        # x: [B, T, N] -> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through OLinear model.
        """
        B, T, C = x_ts.shape

        # Truncate to effective sequence length for fair comparison with transformers
        if T > self._effective_seq_len:
            x_ts = x_ts[:, :self._effective_seq_len, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._effective_seq_len]
            T = self._effective_seq_len

        # Initialize channel correlation matrix if not already done
        if not self._corr_mat_initialized and self.training:
            with torch.no_grad():
                self.channel_corr_mat = self._compute_channel_correlation(x_ts)
                self._corr_mat_initialized = True
                logging.info("[OLinear] Channel correlation matrix initialized")

        # RevIN normalize
        # x_ts is already in z-score space (normalized by _normalize_time_series).
        # Applying RevIN here is double normalization: global z-score + per-instance RevIN.
        # This is intentional for handling per-instance distribution shifts on top
        # of global normalization, but differs from the original RevIN paper which assumes
        # RevIN is applied to raw data.
        x_norm, instance_mean, instance_std = self.revin(x_ts, observed_mask)

        # Token embedding with dimension expansion
        # [B, T, C] -> [B, C, T, D]
        x_emb = self.tokenEmb(x_norm, self.embeddings)
        B, N, D, T = x_emb.shape

        # Follow original OLinear-C architecture more closely
        # Original: [B, N, T, D] -> [B, N, D, T] -> flatten(-2) -> [B, N, D*T]
        #         -> ortho_trans -> [B, N, D*pred_len] -> reshape -> [B, N, D, pred_len]

        # Transpose to match original structure
        x_trans = x_emb.transpose(-1, -2)  # [B, N, T, D] -> [B, N, D, T]

        # Flatten last two dimensions: [B, N, D, T] -> [B, N, D*T]
        x_flat = x_trans.flatten(-2)

        # Reshape to apply ortho_trans to all channels: [B, N, D*T] -> [B*N, D*T]
        x_flat_reshaped = x_flat.reshape(B * N, -1)

        # Apply ortho_trans (MLP with proper normalization
        encoded = self.ortho_trans(x_flat_reshaped)  # [B*N, D*T] -> [B*N, embed_size]

        # Reshape back to [B, N, embed_size]
        encoded = encoded.reshape(B, N, -1)

        # Apply channel correlation matrix
        # Use the computed correlation to guide channel-wise attention
        if self.channel_corr_mat is not None:
            # Normalize correlation matrix with softmax (key OLinear-C component)
            corr_weight = F.softmax(self.channel_corr_mat, dim=-1)  # [N, N]
            # Apply channel attention: weighted combination of channels
            # Expand corr_weight to match batch dimension
            corr_weight = corr_weight.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
            # Apply attention: [B, N, N] @ [B, N, embed_size] -> [B, N, embed_size]
            encoded = torch.bmm(corr_weight, encoded)  # [B, N, embed_size]

        # Pool over embed_size dimension to get channel representation
        pooled = encoded.mean(dim=-1)  # [B, N] - pool over embed_size, not channels

        # Concatenate with static features and predict yield
        combined = torch.cat([pooled, x_static], dim=-1)
        predictions = self.regression_head(combined).squeeze(-1)

        return predictions


class OLinearEncoderLayer(nn.Module):
    """
    Simplified OLinear encoder layer with fixed correlation matrix.

    Based on LinearEncoder_abla_design from OLinear with:
    - CovMatTrans='softmax': Use softmax on correlation matrix
    - WeightTrans='none': No learnable weight matrix
    - NormSet='none': No additional normalization
    - onlyconv=True: Use convolutional layers only
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_channels: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_channels = n_channels

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Value projection
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor [B, N, d_model] where N is number of channels

        Returns:
            Tuple of (output, None)
        """
        # Value projection
        values = self.v_proj(x)

        # Apply identity transformation (correlation matrix handled at model level)
        new_x = values

        # Residual connection with dropout
        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        # Feedforward convolutional layers
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Second residual connection
        output = self.norm2(x + y)

        return output, None


# Creating model function
def create_model(config: LinearModelConfig) -> BaseTimeSeriesModel:
    model_map = {
        "nlinear": NLinearYieldModel,
        "dlinear": DLinearYieldModel,
        "xlinear": XLinearYieldModel,
        "rlinear": RLinearYieldModel,
        "olinear": OLinearYieldModel,
    }

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