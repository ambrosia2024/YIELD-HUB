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

NOTE (parity refactor): CNNLSTMYieldModel now subclasses BaseTimeSeriesModel
(see tstLayer.py), so it shares the EXACT training / evaluation / MQL / uncertainty
/ recursive-lag / evaluation-protocol code paths used by the linear and transformer
baselines. Only the architecture (_build_model) and forward pass are CNN-LSTM
specific; all step/metric/protocol logic is inherited. This guarantees the model
is directly comparable with the other baselines.
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
# Shared base model: provides all training/eval/MQL/uncertainty/protocol logic.
from tstLayer import BaseTimeSeriesModel

import sys
sys.path.append('../process/')
from metrics import ModelMetrics
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


class CNNLSTMYieldModel(BaseTimeSeriesModel):
    """
    CNN-LSTM yield model (Khaki et al. 2020), implemented as a BaseTimeSeriesModel
    subclass so it shares the exact training / evaluation / MQL / uncertainty /
    recursive-lag / evaluation-protocol machinery used by the linear and
    transformer baselines. Only the architecture (`_build_model`) and the forward
    pass are CNN-LSTM-specific; everything else (normalisation, residual trend,
    weighted loss, MQL loss + uncertainty metrics, per-year metrics, recursive
    lags, bias correction, the evaluation protocol hooks) is inherited.

    Architecture:
      1. Each weather variable -> WeatherCNN; each RS/domain variable -> SoilCNN
      2. Concatenate per-branch CNN outputs, project (weather -> 60, soil -> 40)
      3. Concatenate [weather_proj | soil_proj | pooled SOTA | static features]
      4. Feed as a length-1 sequence to an LSTM; take the final hidden state
      5. Regression head -> point value (MSE) or quantiles (MQL)

    Supports MQL automatically: `_create_regression_head` sizes the output to
    `self.n_quantiles` (1 for MSE, len(quantiles) for MQL) and the base step
    handles the quantile loss + uncertainty metrics.
    """

    def _build_model(self) -> nn.Module:
        config = self.config

        # Feature-group counts. Channel order of x_ts MUST match
        # BaseTimeSeriesModel._normalize_time_series: weather, domain
        # (gdd/rue/rtai), remote-sensing, sota.
        self.weather_features = config.weather_features
        self.n_weather = len(self.weather_features)
        self.n_rs = len(REMOTE_SENSING_FEATURES)
        self.n_domain_ts = sum([config.use_gdd, config.use_rue, config.use_rtai])
        self.n_sota = len(SOTA_TEMPORAL_VARS_LIST) if config.use_sota_features else 0

        # CNN output dimensions are deterministic given the sequence length.
        seq_len = config.seq_len

        def weather_cnn_output_dim(t: int) -> int:
            t = (t - 8) // 2   # conv1(k=9) + pool
            t = (t - 2) // 2   # conv2(k=3) + pool
            t = (t - 2) // 2   # conv3(k=3) + pool
            t = (t - 2) // 2   # conv4(k=3) + pool
            return t * 20      # 20 channels * remaining timesteps

        def soil_cnn_output_dim(t: int) -> int:
            t = (t - 2) // 2   # conv1(k=3) + pool
            t = (t - 2) // 2   # conv2(k=3) + pool
            t = t - 1          # conv3(k=2), no pool
            return t * 12      # 12 channels * remaining timesteps

        weather_per_var = weather_cnn_output_dim(seq_len)
        soil_per_var = soil_cnn_output_dim(seq_len)
        self.weather_cnn_out_dim = self.n_weather * weather_per_var
        self.soil_cnn_out_dim = (self.n_rs + self.n_domain_ts) * soil_per_var

        # CNN branches (shared weights across variables of the same type).
        self.weather_cnn = WeatherCNN()
        self.soil_cnn = SoilCNN()
        # Fusion projections (Khaki et al.: weather -> 60, soil -> 40).
        self.weather_fc = nn.Linear(self.weather_cnn_out_dim, 60)
        self.soil_fc = nn.Linear(self.soil_cnn_out_dim, 40)
        self.cnn_output_dim = 60 + 40

        # LSTM hyperparameters (dedicated cnnlstm_* config fields; previously the
        # patchtst_* fields were reused as a hack).
        self.lstm_hidden_size = getattr(config, 'cnnlstm_hidden_size', 64)
        self.lstm_num_layers = getattr(config, 'cnnlstm_num_layers', 1)
        self.lstm_dropout = getattr(config, 'cnnlstm_dropout', 0.0)

        # LSTM input = CNN projections + pooled SOTA + static features.
        # (SOTA was previously omitted from the size, which broke
        # --use_sota_features because forward concatenates pooled SOTA.)
        lstm_input_size = self.cnn_output_dim + self.n_sota + self.n_static_features

        lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout if self.lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Regression head: base sizes the output to n_quantiles automatically.
        self.regression_head = self._create_regression_head(
            self.lstm_hidden_size, self.lstm_hidden_size // 2
        )

        logging.info(
            f"[CNNLSTM] seq_len={seq_len}, weather={self.n_weather}x{weather_per_var}, "
            f"soil={(self.n_rs + self.n_domain_ts)}x{soil_per_var}, sota={self.n_sota}, "
            f"static={self.n_static_features}, lstm_in={lstm_input_size}, "
            f"hidden={self.lstm_hidden_size}, layers={self.lstm_num_layers}, "
            f"n_quantiles={self.n_quantiles}"
        )

        self._model_ready = True
        # Returned module is stored as self.base_model by the base __init__.
        return lstm

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None,
                return_embeddings: bool = False):
        """
        Forward pass. Inputs are already normalised by the base step.

        Args:
            x_ts: Time series features (batch, seq_len, n_ts_features). Channel
                order is weather, domain, remote-sensing, sota.
            x_static: Static features (batch, n_static_features).
            observed_mask: Unused here (the CNN path aggregates the full season);
                kept for interface compatibility with the base step.
            return_embeddings: If True, also return the pre-head representation
                (for auxiliary heads / multitask parity).

        Returns:
            (batch,) for MSE loss, or (batch, n_quantiles) for MQL.
        """
        n_weather, n_domain, n_rs, n_sota = (
            self.n_weather, self.n_domain_ts, self.n_rs, self.n_sota
        )

        # ISSUE-2 FIX (mask-aware aggregation). The CNN branches below convolve
        # and average-pool over the FULL padded sequence, and the SOTA / static
        # branches mean-pool over time. Padded (unobserved) positions are ~0 after
        # normalisation, so every pooled feature is biased toward 0 in proportion
        # to how much of the season is padding -- and padding grows as the forecast
        # origin moves earlier in the season, confounding cross-origin comparisons.
        # Mean-impute padded positions per sample & channel with the observed-mean
        # (B1): the convolutions then see a flat continuation instead of a zero
        # cliff, and a plain temporal mean equals the masked (observed-only) mean.
        # When no mask is provided this is a no-op (previous behaviour).
        if observed_mask is not None:
            m = observed_mask.unsqueeze(-1).to(x_ts.dtype)          # (B, T, 1)
            valid = m.sum(dim=1, keepdim=True).clamp(min=1.0)       # (B, 1, 1)
            ch_mean = (x_ts * m).sum(dim=1, keepdim=True) / valid   # (B, 1, C)
            x_ts = torch.where(observed_mask.unsqueeze(-1), x_ts, ch_mean)

        # Split channels in the true order: weather, domain, rs, sota.
        idx = 0
        weather_feats = [x_ts[:, :, i:i + 1] for i in range(idx, idx + n_weather)]
        idx += n_weather
        domain_feats = [x_ts[:, :, i:i + 1] for i in range(idx, idx + n_domain)]
        idx += n_domain
        rs_feats = [x_ts[:, :, i:i + 1] for i in range(idx, idx + n_rs)]
        idx += n_rs
        sota_features = x_ts[:, :, idx:idx + n_sota] if n_sota > 0 else None

        # Weather branch -> project to 60.
        weather_out = F.relu(self.weather_fc(
            torch.cat([self.weather_cnn(f) for f in weather_feats], dim=1)
        ))

        # Soil branch: domain + remote-sensing variables through SoilCNN -> 40.
        soil_inputs = [self.soil_cnn(f) for f in domain_feats] + \
                      [self.soil_cnn(f) for f in rs_feats]
        soil_out = F.relu(self.soil_fc(torch.cat(soil_inputs, dim=1)))

        cnn_concat = torch.cat([weather_out, soil_out], dim=1)
        if sota_features is not None:
            # SOTA features skip the CNN; mean-pool across time. Because x_ts was
            # mean-imputed above, this plain mean already equals the masked
            # (observed-only) mean of the SOTA channels.
            cnn_concat = torch.cat([cnn_concat, sota_features.mean(dim=1)], dim=1)

        # CY-BENCH may pass static features as a (B, T, n_static) sequence. When it
        # does, mean-pool over observed timesteps only (ISSUE-2 fix); padded
        # positions must not drag the pooled static features toward 0.
        if x_static.dim() == 3:
            if observed_mask is not None:
                ms = observed_mask.unsqueeze(-1).to(x_static.dtype)        # (B, T, 1)
                x_static_pooled = (x_static * ms).sum(dim=1) / ms.sum(dim=1).clamp(min=1.0)
            else:
                x_static_pooled = x_static.mean(dim=1)
        else:
            x_static_pooled = x_static
        combined = torch.cat([cnn_concat, x_static_pooled], dim=1)

        # Feed as a length-1 sequence and take the final LSTM state.
        lstm_out, _ = self.base_model(combined.unsqueeze(1))  # base_model == LSTM
        lstm_final = lstm_out[:, -1, :]

        predictions = self._reshape_output(self.regression_head(lstm_final))

        if return_embeddings:
            return predictions, torch.cat([lstm_final, x_static_pooled], dim=1)
        return predictions


def create_cnn_lstm_model(config: TSTModelConfig) -> CNNLSTMYieldModel:
    """
    Factory function to create a CNN-LSTM yield model.

    Passes lr / weight_decay from the config so training settings match the other
    baselines (previously these were left at the base __init__ defaults).

    Args:
        config: TSTModelConfig with model configuration

    Returns:
        CNNLSTMYieldModel instance
    """
    return CNNLSTMYieldModel(config, lr=config.lr, weight_decay=config.weight_decay)
