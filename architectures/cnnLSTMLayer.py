# -*- coding: utf-8 -*-
"""
CNN-LSTM Architecture for Crop Yield Prediction

This module implements a CNN-LSTM architecture based on:
Khaki, S., Wang, L., Kapos, Z. et al. A CNN-RNN Framework for Crop Yield Prediction.
Computational Intelligence in Neuroscience (2020).
https://doi.org/10.1155/2020/3429157
GitHub: https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction

The architecture processes time series data through parallel CNN branches for
different feature types, followed by an LSTM for temporal modeling.

Architecture (adapted for CY-BENCH):
1. Weather CNN: Each weather variable processed through 4 conv layers
2. Soil/RS CNN: Each soil/remote sensing variable processed through 3 conv layers
3. Practice features: Flattened directly
4. Fusion: Concatenate CNN outputs per feature type, apply FC projection
5. Reshape to sequence and concatenate with static features
6. LSTM: Process sequence and take final timestep output
7. Regression head: Final yield prediction

Key differences from original:
- Original uses 5-year historical context as LSTM input
- This version uses temporal aggregation (daily/weekly/dekad) sequences
- Adapted for CY-BENCH data structure while maintaining CNN-LSTM concept
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Custom Classes and functions
from trendLayer import TrendModel
from biasCorrectionLayer import BiasCorrection
from modelconfig import TSTModelConfig

import sys
sys.path.append('../process/')
from validateModel import ModelMetrics
from featureEngineering import _get_static_feature_names

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']

# SOTA temporal vars
SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

logger = logging.getLogger(__name__)


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for preserving temporal order information.
    """

    def __init__(self, d_model: int, max_len: int = 365, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate div_term for sin/cos computation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        # Compute sin and cos based positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            observed_mask: Optional boolean mask for valid timesteps

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        # Get positional encoding up to current sequence length
        pe = self.pe[:, :seq_len, :]

        # If mask provided, zero out positional encoding for padded positions
        if observed_mask is not None:
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (B, T, 1)
            pe = pe * mask_expanded

        return self.dropout(x + pe)


class WeatherCNN(nn.Module):
    """
    CNN branch for processing weather time series (one variable at a time).

    Architecture based on Khaki et al. 2020:
    - Conv1d(8, kernel=9) + ReLU + AvgPool(2)
    - Conv1d(12, kernel=3) + ReLU + AvgPool(2)
    - Conv1d(16, kernel=3) + ReLU + AvgPool(2)
    - Conv1d(20, kernel=3) + ReLU + AvgPool(2)
    - Flatten output

    Each weather variable is processed independently through the same CNN.
    """

    def __init__(self):
        super().__init__()

        # Layer 1: Large kernel for broad temporal patterns
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=9,
            stride=1,
            padding='valid'
        )

        # Layer 2: Medium kernel
        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding='valid'
        )

        # Layer 3: Small kernel
        self.conv3 = nn.Conv1d(
            in_channels=12,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding='valid'
        )

        # Layer 4: Final convolution
        self.conv4 = nn.Conv1d(
            in_channels=16,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding='valid'
        )

        # Pooling layers (AvgPool with kernel_size=2, stride=2)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through weather CNN.

        Args:
            x: Input tensor of shape (batch, seq_len, 1)

        Returns:
            Flattened features of shape (batch, flattened_dim)
        """
        # Transpose to (B, C=1, T) for Conv1d
        x = x.transpose(1, 2)  # (B, 1, T)

        # Conv + ReLU + Pool sequence
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten: (B, C, T) -> (B, C*T)
        return x.flatten(start_dim=1)


class SoilCNN(nn.Module):
    """
    CNN branch for processing soil and remote sensing time series.

    Architecture based on Khaki et al. 2020:
    - Conv1d(4, kernel=3) + ReLU + AvgPool(2)
    - Conv1d(8, kernel=3) + ReLU + AvgPool(2)
    - Conv1d(12, kernel=2) + ReLU
    - Flatten output

    Each soil/RS variable is processed independently through the same CNN.
    """

    def __init__(self):
        super().__init__()

        # Layer 1
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding='valid'
        )

        # Layer 2
        self.conv2 = nn.Conv1d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding='valid'
        )

        # Layer 3
        self.conv3 = nn.Conv1d(
            in_channels=8,
            out_channels=12,
            kernel_size=2,
            stride=1,
            padding='valid'
        )

        # Pooling layer
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through soil CNN.

        Args:
            x: Input tensor of shape (batch, seq_len, 1)

        Returns:
            Flattened features of shape (batch, flattened_dim)
        """
        # Transpose to (B, C=1, T) for Conv1d
        x = x.transpose(1, 2)  # (B, 1, T)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        # No pooling after layer 3

        return x.flatten(start_dim=1)


class CNNLSTMYieldModel(pl.LightningModule):
    """
    CNN-LSTM model for crop yield prediction.

    Based on Khaki et al. 2020, adapted for PyTorch and CY-BENCH framework.

    Architecture:
    1. Weather features: Each processed through WeatherCNN
    2. Soil/RS features: Each processed through SoilCNN
    3. Domain features (GDD, RUE, Farquhar): Processed through SoilCNN
    4. Concatenate outputs per feature type, apply FC projection
    5. Concatenate all projections with static features
    6. Reshape to sequence and process with LSTM
    7. Take final LSTM output for regression

    Features:
    - Flexible temporal aggregation (daily, weekly, dekad)
    - Optional residual trend modeling
    - Exponential sample weighting for non-stationarity
    - Bias correction for recursive lag evaluation
    """

    def __init__(self, config: TSTModelConfig,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.trend_model = TrendModel()

        # Bias correction for recursive lags
        self.bias_correction: Optional[BiasCorrection] = (
            BiasCorrection(method='linear') if config.use_recursive_lags else None
        )
        if self.bias_correction:
            logging.info("[BiasCorrection] ENABLED (automatic with use_recursive_lags=True)")

        self.feature_norm_params: Optional[dict] = None
        self._most_recent_year: Optional[int] = None
        self._weight_log_done = False

        # Log exponential weighting configuration
        if config.use_exponential_weighting:
            print(f"[Exponential Weighting] ENABLED - tau={config.exponential_tau}")
        else:
            print(f"[Exponential Weighting] DISABLED")

        # Compute feature dimensions
        use_sota = config.use_sota_features
        n_domain_ts = sum([config.use_gdd, config.use_rue, config.use_farquhar])

        # Weather features (base: tmin, tmax, tavg, prec, rad + optional cwb)
        self.weather_features = config.weather_features
        n_weather = len(self.weather_features)

        # Remote sensing features
        self.n_rs = len(REMOTE_SENSING_FEATURES)

        # Domain features (GDD, RUE, Farquhar) - processed through soil CNN
        self.n_domain_ts = n_domain_ts

        # Total time series features
        self.n_ts_features = n_weather + self.n_rs + n_domain_ts
        if use_sota:
            self.n_ts_features += len(SOTA_TEMPORAL_VARS_LIST)

        # Static features count
        n_crop_calendar = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop_calendar += 2
            else:
                n_crop_calendar += 1

        n_heat_stress = 7 if config.use_heat_stress_days else 0

        # Multi-year summaries
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
            + (2 if config.include_spatial_features else 0)
            + config.lag_years
            + n_heat_stress
            + n_multi_year
        )

        logging.info(f"[CNNLSTM] Weather features={n_weather}, RS features={self.n_rs}, "
                    f"Domain features={self.n_domain_ts}, TS total={self.n_ts_features}, "
                    f"Static features={self.n_static_features}")

        # Positional encoding (optional, for sequence order)
        self.d_model = 64
        max_len_map = {"daily": 365, "weekly": 52, "dekad": 36}
        pe_max_len = max_len_map.get(config.aggregation, 365)
        self.use_positional_encoding = config.use_positional_encoding

        if self.use_positional_encoding:
            self.positional_encoding = TemporalPositionalEncoding(
                d_model=self.d_model,
                max_len=pe_max_len,
                dropout=0.1
            )

        # CNN branches (shared across features of same type)
        self.weather_cnn = WeatherCNN()  # For each weather variable
        self.soil_cnn = SoilCNN()  # For each RS/domain variable

        # Helper function to compute CNN output dimension given input timesteps
        def weather_cnn_output_dim(timesteps):
            """Calculate WeatherCNN output dimension after all conv+pool layers."""
            x = timesteps
            x = (x - 8) // 2      # conv1(k=9) + pool
            x = (x - 2) // 2      # conv2(k=3) + pool
            x = (x - 2) // 2      # conv3(k=3) + pool
            x = (x - 2) // 2      # conv4(k=3) + pool
            return x * 20  # 20 channels * remaining timesteps

        def soil_cnn_output_dim(timesteps):
            """Calculate SoilCNN output dimension after all conv+pool layers."""
            x = timesteps
            x = (x - 2) // 2      # conv1(k=3) + pool
            x = (x - 2) // 2      # conv2(k=3) + pool
            x = x - 1             # conv3(k=2), NO pool
            return x * 12  # 12 channels * remaining timesteps

        # Get sequence length for this aggregation
        seq_len = {"daily": 365, "weekly": 52, "dekad": 36}.get(config.aggregation, 365)

        # Calculate actual CNN output dimensions
        weather_per_var = weather_cnn_output_dim(seq_len)
        soil_per_var = soil_cnn_output_dim(seq_len)

        self.weather_cnn_out_dim = n_weather * weather_per_var
        self.soil_cnn_out_dim = (self.n_rs + self.n_domain_ts) * soil_per_var

        logging.info(f"[CNNLSTM] Aggregation={config.aggregation}, seq_len={seq_len}")
        logging.info(f"[CNNLSTM] Weather CNN: {weather_per_var} per var, total {self.weather_cnn_out_dim}")
        logging.info(f"[CNNLSTM] Soil CNN: {soil_per_var} per var, total {self.soil_cnn_out_dim}")

        # Fusion layers (as per Khaki et al.)
        # Weather: concatenate all CNN outputs, project to 60
        self.weather_fc = nn.Linear(self.weather_cnn_out_dim, 60)

        # Soil/RS: concatenate all CNN outputs, project to 40
        self.soil_fc = nn.Linear(self.soil_cnn_out_dim, 40)

        # Total CNN output dimension (before adding static)
        self.cnn_output_dim = 60 + 40  # weather_fc + soil_fc

        # LSTM for temporal modeling
        # Default: 1 layer, 64 hidden units, 0 dropout (matches Khaki et al.)
        self.lstm_hidden_size = config.patchtst_d_model if hasattr(config, 'patchtst_d_model') else 64
        self.lstm_num_layers = config.patchtst_num_layers if hasattr(config, 'patchtst_num_layers') else 1
        self.lstm_dropout = config.patchtst_dropout if hasattr(config, 'patchtst_dropout') else 0.0

        # LSTM input = CNN features + static features
        lstm_input_size = self.cnn_output_dim + self.n_static_features

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout if self.lstm_num_layers > 1 else 0.0,
            batch_first=True
        )

        # Final regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size // 2),
            nn.LayerNorm(self.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.lstm_hidden_size // 2, 1)
        )

        # Verify regression head dimensions
        logging.info(f"[CNNLSTM] Regression head: Linear({self.lstm_hidden_size}, {self.lstm_hidden_size // 2}) -> Linear({self.lstm_hidden_size // 2}, 1)")

        # CRITICAL: Verify the final layer has correct output size
        final_layer = self.regression_head[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(f"Final layer of regression_head should be Linear, got {type(final_layer)}")
        if final_layer.out_features != 1:
            raise ValueError(f"Final layer of regression_head should have 1 output, got {final_layer.out_features}")

        # Metrics
        self.train_metrics = ModelMetrics(prefix="train", include_nrmse=False)
        self.val_metrics = ModelMetrics(prefix="val")
        self.test_metrics = ModelMetrics(prefix="test")

        logging.info(f"[CNNLSTM] Built model with LSTM hidden_size={self.lstm_hidden_size}, "
                    f"num_layers={self.lstm_num_layers}, dropout={self.lstm_dropout}")

    def _get_static_feature_names(self) -> list:
        """Get static feature names for normalization."""
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
            self.config.use_heat_stress_days,
            multi_year_config=self.config.multi_year_config if hasattr(self.config, 'multi_year_config') else None,
        )

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Z-score normalize time series features."""
        if self.feature_norm_params is None:
            if hasattr(self, 'trainer') and self.trainer is not None:
                dm_params = self.trainer.datamodule.feature_norm_params
                if dm_params is not None:
                    self.feature_norm_params = dm_params
                else:
                    raise RuntimeError("feature_norm_params not set in datamodule.")
            else:
                raise RuntimeError("feature_norm_params not set and no trainer available.")

        names = [f'weather_{f}' for f in self.config.weather_features]
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
            raise ValueError(f"TS feature name count ({len(names)}) != tensor dim ({x_ts.shape[2]})")

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
        """Z-score normalize static features and impute NaN to 0.0."""
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
        """Fit trend model and copy normalization parameters."""
        dm = self.trainer.datamodule
        train_y_orig = dm.train_ds.y.numpy() * dm.y_std + dm.y_mean

        from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET
        train_items = [
            {KEY_LOC: dm.train_ds.adm_ids[i],
             KEY_YEAR: int(dm.train_ds.years[i]),
             KEY_TARGET: float(train_y_orig[i])}
            for i in range(len(train_y_orig))
        ]
        self.trend_model.fit(train_items)
        self.feature_norm_params = dm.feature_norm_params

        if self.bias_correction is not None:
            self._val_preds = []
            self._val_targets = []

        if self.config.use_exponential_weighting:
            self._most_recent_year = int(dm.train_ds.years.max().item())
            train_years = dm.train_ds.years.numpy()
            year_range = (train_years.min(), train_years.max())
            logging.info(f"[Exponential Weighting] ENABLED")
            logging.info(f"  Most recent training year: {self._most_recent_year}")
            logging.info(f"  Training year range: {year_range[0]} to {year_range[1]}")

    def _compute_batch_trends(self, adm_ids, years: torch.Tensor, dm, lats, lons) -> torch.Tensor:
        """Compute trend estimates for batch samples."""
        from cybench.config import KEY_LOC, KEY_YEAR

        test_items = []
        for i, (loc, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            test_items.append({KEY_LOC: loc, KEY_YEAR: year_int})

        trend_predictions_orig = self.trend_model._predict_trend(test_items).flatten()
        trends_z = (trend_predictions_orig - dm.y_mean) / dm.y_std
        return torch.tensor(trends_z, dtype=torch.float32, device=self.device).unsqueeze(1)

    def _compute_weighted_loss(self, pred: torch.Tensor, y: torch.Tensor,
                                years: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute weighted MSE loss."""
        if self.config.use_exponential_weighting and years is not None:
            per_sample_loss = F.mse_loss(pred, y, reduction='none')
            years_int = years.cpu().numpy() if years.is_cuda else years.numpy()
            weights = np.exp(-(self._most_recent_year - years_int) / self.config.exponential_tau)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=pred.device)
            weights_tensor = weights_tensor / (weights_tensor.mean() + 1e-8)
            weighted_loss = (per_sample_loss * weights_tensor).mean()
            return weighted_loss
        else:
            return F.mse_loss(pred, y)

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM model.

        Args:
            x_ts: Time series features (batch, seq_len, n_ts_features)
            x_static: Static features (batch, n_static_features)
            observed_mask: Boolean mask for valid timesteps

        Returns:
            Predictions (batch,)
        """
        batch_size = x_ts.size(0)

        # Debug: Log input shapes
        logging.debug(f"[CNNLSTM forward] x_ts shape: {x_ts.shape}, x_static shape: {x_static.shape}")
        logging.debug(f"[CNNLSTM forward] n_weather={len(self.weather_features)}, n_rs={self.n_rs}, n_domain={self.n_domain_ts}")
        logging.debug(f"[CNNLSTM forward] use_sota={self.config.use_sota_features}")

        # Split time series features by type
        # Index ranges for different feature groups
        n_weather = len(self.weather_features)
        n_rs = self.n_rs
        n_domain = self.n_domain_ts
        use_sota = self.config.use_sota_features
        n_sota = len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0

        idx = 0
        weather_features = []
        for i in range(n_weather):
            feat = x_ts[:, :, idx:idx+1]  # (B, T, 1)
            weather_features.append(feat)
            idx += 1

        rs_features = []
        for i in range(n_rs):
            feat = x_ts[:, :, idx:idx+1]  # (B, T, 1)
            rs_features.append(feat)
            idx += 1

        domain_features = []
        for i in range(n_domain):
            feat = x_ts[:, :, idx:idx+1]  # (B, T, 1)
            domain_features.append(feat)
            idx += 1

        # SOTA features are handled differently - skip CNN processing
        if use_sota:
            sota_features = x_ts[:, :, idx:idx+n_sota]  # (B, T, n_sota)
            idx += n_sota
        else:
            sota_features = None

        # Process weather features through WeatherCNN
        weather_outputs = []
        for feat in weather_features:
            out = self.weather_cnn(feat)  # (B, flattened)
            weather_outputs.append(out)

        # Concatenate weather outputs and apply FC projection
        weather_concat = torch.cat(weather_outputs, dim=1)  # (B, total_flattened)
        print(f"[DEBUG] weather_concat shape before FC: {weather_concat.shape}")

        # Adaptive pooling to fixed size before FC (handles varying sequence lengths)
        # Reshape to (B, n_weather, flattened_per_feature) for adaptive pool
        # First, we need to know the per-feature output dimension
        # For simplicity, apply FC directly if dimensions match, otherwise use adaptive approach
        weather_out = self.weather_fc(weather_concat)  # (B, 60)
        weather_out = F.relu(weather_out)
        print(f"[DEBUG] weather_out shape after FC: {weather_out.shape}")

        # Process RS and domain features through SoilCNN
        soil_outputs = []
        for feat in rs_features:
            out = self.soil_cnn(feat)  # (B, flattened)
            soil_outputs.append(out)
        for feat in domain_features:
            out = self.soil_cnn(feat)  # (B, flattened)
            soil_outputs.append(out)

        # Concatenate soil/RS outputs and apply FC projection
        soil_concat = torch.cat(soil_outputs, dim=1)  # (B, total_flattened)
        print(f"[DEBUG] soil_concat shape before FC: {soil_concat.shape}")
        soil_out = self.soil_fc(soil_concat)  # (B, 40)
        soil_out = F.relu(soil_out)
        print(f"[DEBUG] soil_out shape after FC: {soil_out.shape}")

        # Concatenate all CNN outputs
        cnn_concat = torch.cat([weather_out, soil_out], dim=1)  # (B, 100)
        print(f"[DEBUG] cnn_concat shape: {cnn_concat.shape}")

        # Add SOTA features if enabled (they skip CNN processing)
        if sota_features is not None:
            # SOTA features are (B, T, n_sota) - we need to aggregate them
            # Use mean pooling across time dimension
            sota_pooled = sota_features.mean(dim=1)  # (B, n_sota)
            cnn_concat = torch.cat([cnn_concat, sota_pooled], dim=1)  # (B, 100 + n_sota)
            logging.debug(f"[CNNLSTM forward] After adding SOTA: cnn_concat shape: {cnn_concat.shape}")

        # Aggregate static features across time dimension if they're passed as a sequence
        # CY-BENCH passes static features as (B, T, n_static) - we need to pool to (B, n_static)
        logging.debug(f"[CNNLSTM forward] x_static shape before pooling: {x_static.shape}, dim={x_static.dim()}")
        if x_static.dim() == 3:
            # Static features are passed as a sequence - take mean across time
            x_static_pooled = x_static.mean(dim=1)  # (B, n_static)
            logging.debug(f"[CNNLSTM forward] x_static_pooled shape: {x_static_pooled.shape}")
        else:
            x_static_pooled = x_static  # (B, n_static)
            logging.debug(f"[CNNLSTM forward] x_static already 2D: {x_static_pooled.shape}")

        # Concatenate with static features
        combined = torch.cat([cnn_concat, x_static_pooled], dim=1)  # (B, 100 + n_sota + n_static)
        print(f"[DEBUG] cnn_concat shape: {cnn_concat.shape}, x_static_pooled shape: {x_static_pooled.shape}, combined shape: {combined.shape}")

        # Reshape for LSTM as sequence with 1 timestep
        # In original Khaki, they use 5 years of history
        # Here we use a single timestep representing the current season's aggregated features
        lstm_input = combined.unsqueeze(1)  # (B, 1, lstm_input_size)
        print(f"[DEBUG] lstm_input shape: {lstm_input.shape}")

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        # lstm_out shape: (B, 1, hidden_size)
        # h_n shape: (num_layers, B, hidden_size)
        print(f"[DEBUG] lstm_out shape: {lstm_out.shape}")
        logging.debug(f"[CNNLSTM forward] lstm_out shape: {lstm_out.shape}")

        # Take the last timestep's output
        lstm_final = lstm_out[:, -1, :]  # (B, hidden_size)

        # Apply positional encoding if enabled (no-op for single timestep)
        # NOTE: We skip positional encoding here because we're using a single timestep
        # and observed_mask is for the full sequence which would cause broadcasting issues
        # if self.use_positional_encoding:
        #     lstm_final = lstm_final.unsqueeze(1)  # (B, 1, hidden_size)
        #     lstm_final = self.positional_encoding(lstm_final, observed_mask=observed_mask)
        #     lstm_final = lstm_final.squeeze(1)  # (B, hidden_size)

        # Final prediction
        # First, verify lstm_final shape is correct
        if lstm_final.dim() != 2 or lstm_final.size(1) != self.lstm_hidden_size:
            logging.error(f"[CNNLSTM] lstm_final has unexpected shape: {lstm_final.shape}, expected ({batch_size}, {self.lstm_hidden_size})")
            raise ValueError(f"lstm_final shape is {lstm_final.shape}, expected ({batch_size}, {self.lstm_hidden_size})")

        prediction = self.regression_head(lstm_final)  # (B, 1)
        logging.debug(f"[CNNLSTM forward] prediction after regression_head: {prediction.shape}")

        # Verify prediction shape
        if prediction.dim() == 2 and prediction.size(1) != 1:
            logging.error(f"[CNNLSTM] prediction has wrong shape: {prediction.shape}, expected ({batch_size}, 1)")
            logging.error(f"[CNNLSTM] Taking mean across dimension 1 to fix...")
            prediction = prediction.mean(dim=1, keepdim=True)

        # Debug: ensure correct shape
        if prediction.dim() > 2:
            logging.warning(f"prediction has unexpected shape {prediction.shape}, flattening...")
            prediction = prediction.view(prediction.size(0), -1).mean(dim=1, keepdim=True)
        elif prediction.dim() == 2 and prediction.size(1) > 1:
            logging.warning(f"prediction has too many outputs: {prediction.shape}, taking mean across dim 1")
            prediction = prediction.mean(dim=1, keepdim=True)

        prediction = prediction.squeeze(-1)  # (B,)
        logging.debug(f"[CNNLSTM forward] final prediction shape: {prediction.shape}")

        return prediction

    def _shared_step(self, batch, metrics: ModelMetrics, loss_key: str):
        """Shared training/validation step logic."""
        x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        if self.config.use_residual_trend:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        if batch_trends is not None:
            final_pred = pred + batch_trends.squeeze(-1).detach()
        else:
            final_pred = pred

        loss = self._compute_weighted_loss(final_pred, y, years=years)
        metrics.update(final_pred.detach(), y.detach())
        self.log(loss_key, loss, prog_bar=True)
        return loss

    def _eval_step_with_clipping(self, batch, metrics: ModelMetrics, loss_key: str, stage: str,
                                  return_predictions: bool = False, return_orig: bool = False):
        """Evaluation step with output clipping."""
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        final_pred_z = pred + batch_trends.squeeze(-1).detach() if batch_trends is not None else pred
        loss = self._compute_weighted_loss(final_pred_z, y_z)

        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        final_pred_orig = final_pred_z.detach() * y_std + y_mean
        y_orig = y_z.detach() * y_std + y_mean

        final_pred_clipped = torch.clamp(final_pred_orig, min=0.0)
        clip_rate = (final_pred_orig < 0.0).float().mean()
        self.log(f'{stage}/clip_rate', clip_rate, prog_bar=False)

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
        return self._shared_step(batch, self.train_metrics, 'train_loss')

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log('train/mse', results['mse'], prog_bar=False)
        self.log('train/mae', results['mae'], prog_bar=False)
        self.log('train/r2', results['r2'], prog_bar=False)
        self.log('train/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if self.bias_correction is not None and hasattr(self, '_val_preds'):
            loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
                batch, self.val_metrics, 'val_loss', stage='val',
                return_orig=True, return_predictions=True
            )
            self._val_preds.extend(preds_clipped.detach().cpu().tolist())
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
        self.log('val/nrmse', results['nrmse'], prog_bar=False)
        self.val_metrics.reset()

    def on_train_end(self):
        if self.bias_correction is not None and hasattr(self, '_val_preds') and len(self._val_preds) > 0:
            self.bias_correction.fit(np.array(self._val_preds), np.array(self._val_targets))
            params = self.bias_correction.get_correction_params()
            logging.info(f"[BiasCorrection] Fitted on {len(self._val_preds)} validation samples")
            self._val_preds = []
            self._val_targets = []

    def on_test_start(self):
        dm = self.trainer.datamodule
        if hasattr(dm, '_test_years') and dm._test_years is not None:
            self._test_years = dm._test_years
            self._per_year_preds = {year: {'preds': [], 'targets': []} for year in self._test_years}
        else:
            self._test_years = set()
            self._per_year_preds = {}

        if self.config.use_recursive_lags:
            if not hasattr(self, '_prediction_cache') or not self._prediction_cache:
                self._prediction_cache = {}

    def test_step(self, batch, batch_idx):
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            loss, preds, targets, years = self._eval_step_with_clipping(
                batch, self.test_metrics, 'test_loss', stage='test', return_orig=True
            )
            self._accumulate_per_year_predictions(preds, targets, years)
            return loss

        # Recursive lag evaluation
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        x_static_modified = self._replace_lags_with_predictions(x_static, years, adm_ids)
        modified_batch = (x_ts, x_static_modified, y_z, years, adm_ids, lats, lons, validity_mask)

        loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
            modified_batch, self.test_metrics, 'test_loss', stage='test',
            return_orig=True, return_predictions=True
        )

        self._cache_predictions(preds_z, years, adm_ids, dm)
        self._accumulate_per_year_predictions(preds_clipped, targets, years)
        return loss

    def _replace_lags_with_predictions(self, x_static: torch.Tensor, years: torch.Tensor,
                                       adm_ids: list) -> torch.Tensor:
        """Replace lag features with cached predictions."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return x_static

        x_static_modified = x_static.clone()
        static_feature_names = self._get_static_feature_names()

        lag_feature_indices = []
        for lag in range(1, self.config.lag_years + 1):
            lag_name = f'lag_yield_{lag}'
            if lag_name in static_feature_names:
                lag_feature_indices.append((lag, static_feature_names.index(lag_name)))

        if not lag_feature_indices:
            return x_static

        for i, (year, adm_id) in enumerate(zip(years, adm_ids)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            for lag, lag_idx in lag_feature_indices:
                lag_year = year_int - lag
                if self._test_years and lag_year in self._test_years:
                    cache_key = (adm_id, lag_year)
                    if cache_key in self._prediction_cache:
                        x_static_modified[i, lag_idx] = self._prediction_cache[cache_key]
                    else:
                        x_static_modified[i, lag_idx] = 0.0
        return x_static_modified

    def _cache_predictions(self, predictions_z: torch.Tensor, years: torch.Tensor,
                          adm_ids: list, dm):
        """Cache predictions for recursive lag evaluation."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return

        device = predictions_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)

        predictions_orig = predictions_z.detach() * y_std + y_mean

        if self.bias_correction is not None and self.bias_correction.is_fitted:
            predictions_orig_np = predictions_orig.cpu().numpy()
            predictions_orig_np = self.bias_correction.correct(predictions_orig_np)
            predictions_orig = torch.tensor(predictions_orig_np, device=device, dtype=predictions_orig.dtype)

        for pred, year, adm_id in zip(predictions_orig, years, adm_ids):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            if self._test_years and year_int in self._test_years:
                cache_key = (adm_id, year_int)
                self._prediction_cache[cache_key] = pred.detach().cpu().item()

    def _accumulate_per_year_predictions(self, preds: torch.Tensor, targets: torch.Tensor, years: torch.Tensor):
        """Accumulate predictions per year for metrics."""
        if not hasattr(self, '_per_year_preds') or not self._per_year_preds:
            return

        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        years_np = years.cpu().numpy() if isinstance(years, torch.Tensor) else years

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
        self.log('test/nrmse', results['nrmse'], prog_bar=False)
        self.test_metrics.reset()

        if hasattr(self, '_per_year_preds') and self._per_year_preds:
            self._test_results_per_year = self._compute_per_year_metrics()

    def _compute_per_year_metrics(self) -> dict:
        """Compute per-year metrics from accumulated predictions."""
        results = {}
        all_preds = []
        all_targets = []

        for year, data in self._per_year_preds.items():
            if len(data['preds']) == 0:
                continue

            preds = torch.tensor(data['preds'])
            targets = torch.tensor(data['targets'])

            mse = MeanSquaredError()
            r2 = R2Score()
            mae = MeanAbsoluteError()
            mape = MeanAbsolutePercentageError()

            mse_val = mse(preds, targets)
            mae_val = mae(preds, targets)
            mape_val = mape(preds, targets)
            rmse_val = torch.sqrt(mse_val)
            smape_val = torch.mean(2.0 * torch.abs(preds - targets) /
                                  (torch.abs(preds) + torch.abs(targets) + 1e-8))
            nrmse_val = rmse_val / (torch.mean(targets).clamp(min=1e-8))

            if len(preds) >= 2:
                r2_val = r2(preds, targets)
            else:
                r2_val = torch.tensor(float('nan'))

            results[f'nrmse_{year}'] = nrmse_val.item()
            results[f'mape_{year}'] = mape_val.item()
            results[f'r2_{year}'] = r2_val.item()
            results[f'rmse_{year}'] = rmse_val.item()
            results[f'mae_{year}'] = mae_val.item()
            results[f'mse_{year}'] = mse_val.item()
            results[f'smape_{year}'] = smape_val.item()

            all_preds.extend(data['preds'])
            all_targets.extend(data['targets'])

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

            if len(all_preds_t) >= 2:
                r2_val = r2(all_preds_t, all_targets_t)
            else:
                r2_val = torch.tensor(float('nan'))

            results['nrmse_overall'] = nrmse_val.item()
            results['mape_overall'] = mape_val.item()
            results['r2_overall'] = r2_val.item()
            results['rmse_overall'] = rmse_val.item()
            results['mae_overall'] = mae_val.item()
            results['mse_overall'] = mse_val.item()
            results['smape_overall'] = smape_val.item()

        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        if self.config.lr_scheduler_lambda is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self.config.lr_scheduler_lambda
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer


def create_cnn_lstm_model(config: TSTModelConfig) -> CNNLSTMYieldModel:
    """
    Factory function to create a CNN-LSTM yield model.

    Args:
        config: TSTModelConfig with model configuration

    Returns:
        CNNLSTMYieldModel instance
    """
    return CNNLSTMYieldModel(config)
