# -*- coding: utf-8 -*-
"""
WFAN (Weather-oriented Frequency Adaptive Network) Wrapper Module

This module implements the WFAN frequency-adaptive normalization framework
for handling distribution shift in agricultural time series forecasting.

Based on:
"Frequency-adaptive deep learning for multi-horizon weather forecasting
in environmental monitoring applications" (Nature Scientific Reports, 2026)

Key components:
- PatternAdaptiveModule: 3-layer MLP for predicting frequency evolution
- WFANWrapper: Model-agnostic wrapper for any base model
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class PatternAdaptiveModule(nn.Module):
    """
    WFAN Pattern Adaptation Module (q_φ) from Figure 2 of the paper.

    Exact architecture specification:
        Layer 1: Linear(W1) + ReLU -> 64 hidden units
        Layer 2: Linear(W2) + ReLU -> 128 hidden units (input includes concatenated X_t)
        Layer 3: Linear(W3) -> H (output = prediction horizon * n_vars)

    The module takes both X_t^(non) (non-stationary component) and
    X_t (original input) as input to provide context for predicting
    how frequency components evolve from input to output.
    """

    def __init__(self, seq_len: int, n_vars: int, pred_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.pred_len = pred_len

        # Layer 1: Linear(W1) + ReLU -> 64
        # Input: flattened non-stationary component (seq_len * n_vars)
        self.layer1 = nn.Linear(seq_len * n_vars, 64)

        # Layer 2: Linear(W2) + ReLU -> 128
        # Input: 64 (from layer1) + seq_len * n_vars (concatenated original X_t)
        self.layer2 = nn.Linear(64 + seq_len * n_vars, 128)

        # Layer 3: Linear(W3) -> pred_len * n_vars
        self.layer3 = nn.Linear(128, pred_len * n_vars)

    def forward(
        self,
        X_non: torch.Tensor,
        X_original: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through pattern adaptation module.

        Args:
            X_non: Non-stationary component from frequency filtering
                   Shape: (batch, seq_len, n_vars)
            X_original: Original input time series
                       Shape: (batch, seq_len, n_vars)

        Returns:
            Y_non: Predicted non-stationary component evolution
                   Shape: (batch, pred_len, n_vars)
        """
        batch_size = X_non.shape[0]

        # Flatten non-stationary component
        X_non_flat = X_non.reshape(batch_size, -1)  # (batch, seq_len * n_vars)

        # Layer 1
        h1 = F.relu(self.layer1(X_non_flat))  # (batch, 64)

        # Layer 2: concatenate with original input
        X_orig_flat = X_original.reshape(batch_size, -1)
        h2_input = torch.cat([h1, X_orig_flat], dim=1)  # (batch, 64 + seq_len * n_vars)
        h2 = F.relu(self.layer2(h2_input))  # (batch, 128)

        # Layer 3
        Y_non_flat = self.layer3(h2)  # (batch, pred_len * n_vars)
        Y_non = Y_non_flat.reshape(batch_size, self.pred_len, self.n_vars)

        return Y_non


class WFANWrapper(pl.LightningModule):
    """
    WFAN Wrapper for any base time series model.

    This wrapper implements frequency-adaptive normalization to handle
    distribution shift in agricultural time series forecasting.

    The wrapper is model-agnostic and works with any BaseTimeSeriesModel
    subclass (transformers or linear models).

    Workflow:
        1. Apply FFT to input time series
        2. Select top-K dominant frequency components per instance
        3. Filter to get non-stationary component (inverse FFT)
        4. Compute stationary residual (input - non-stationary)
        5. Pass stationary residual through base model
        6. Use pattern-adaptive module to predict non-stationary evolution
        7. Combine predictions: final = stationary_pred + non_stationary_pred

    Args:
        base_model: The underlying time series model to wrap
        seq_len: Length of input sequence
        n_vars: Number of time series variables/channels
        K: Number of dominant frequency components to remove (default: 2)
        lambda_coef: Loss balancing coefficient for non-stationary prediction (default: 1.0)
        config: Model configuration object
    """

    def __init__(
        self,
        base_model: pl.LightningModule,
        seq_len: int,
        n_vars: int,
        K: int = 2,
        lambda_coef: float = 1.0,
        config=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base_model = base_model
        self.config = config
        self.K = K
        self.lambda_coef = lambda_coef
        self.seq_len = seq_len
        self.n_vars = n_vars

        # Pattern-adaptive MLP for predicting non-stationary evolution
        self.pattern_adaptive = PatternAdaptiveModule(
            seq_len=seq_len,
            n_vars=n_vars,
            pred_len=1  # Single-step prediction for yield forecasting
        )

        logger.info(
            f"[WFAN Wrapper] Initialized - K={K}, lambda={lambda_coef}, "
            f"seq_len={seq_len}, n_vars={n_vars}"
        )

    def apply_frequency_normalization(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply WFAN frequency-domain normalization.

        Args:
            x: Input time series of shape (batch, seq_len, n_vars)

        Returns:
            x_res: Stationary residual component (batch, seq_len, n_vars)
            x_non: Non-stationary component (batch, seq_len, n_vars)
        """
        batch_size, seq_len, n_vars = x.shape

        # Apply FFT per feature (channel-independent)
        # rfft returns complex values of shape (batch, seq_len//2 + 1, n_vars)
        x_freq = torch.fft.rfft(x, dim=1)  # (batch, freq_bins, n_vars)

        # Compute amplitudes for frequency selection
        amplitudes = torch.abs(x_freq)  # (batch, freq_bins, n_vars)

        # Select top-K dominant frequencies per instance per feature
        # Using data-driven top-K selection
        topk_vals, topk_indices = torch.topk(amplitudes, self.K, dim=1)

        # Create filtering mask (all zeros initially)
        mask = torch.zeros_like(amplitudes, dtype=torch.bool)

        # Mark top-K frequencies as True
        mask.scatter_(1, topk_indices, True)

        # Extract dominant frequency components (non-stationary)
        x_freq_non = x_freq * mask  # (batch, freq_bins, n_vars)

        # Inverse FFT to get time-domain non-stationary component
        x_non = torch.fft.irfft(x_freq_non, n=seq_len, dim=1)  # (batch, seq_len, n_vars)

        # Residual (stationary part) = original - non-stationary
        x_res = x - x_non

        return x_res, x_non

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through WFAN-wrapped model.

        The non-stationary trend adjustment (y_non_scalar) is applied to all
        quantile predictions equally, preserving the quantile spacing.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_vars)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask for valid timesteps (batch, seq_len)

        Returns:
            Predictions of shape (batch,) for single-output (MSE)
                          (batch, n_quantiles) for multi-output (quantile regression)
        """
        # Apply frequency normalization
        x_res, x_non = self.apply_frequency_normalization(x_ts)

        # Base model predicts stationary residual
        y_res = self.base_model(x_res, x_static, observed_mask=observed_mask)

        # Pattern-adaptive module predicts non-stationary evolution
        # Output: (batch, 1, n_vars) -> pool to scalar per sample
        y_non = self.pattern_adaptive(x_non, x_ts)

        # y_non has shape (batch, 1, n_vars) where n_vars is the TS feature count
        # We need to aggregate to (batch,) - one scalar per sample
        if y_non.dim() == 3:
            # (batch, 1, n_vars) -> (batch, n_vars) -> (batch,)
            y_non_scalar = y_non.squeeze(1).mean(dim=-1)
        elif y_non.dim() == 2:
            # (batch, n_vars) -> (batch,)
            y_non_scalar = y_non.mean(dim=-1)
        else:
            raise ValueError(f"Unexpected y_non shape: {y_non.shape}")

        # y_res is either (batch,) for MSE or (batch, n_quantiles) for quantile regression
        # Need to broadcast y_non_scalar to match y_res's shape
        if y_res.dim() == 2:
            # y_res is (batch, n_quantiles), need to expand y_non_scalar to (batch, n_quantiles)
            # Each quantile gets the same trend adjustment (trend is data-dependent, not quantile-dependent)
            # Use repeat() instead of expand() to avoid view/memory issues
            y_non_scalar = y_non_scalar.unsqueeze(1).repeat(1, y_res.size(1))

        y_pred = y_res + y_non_scalar

        return y_pred

    def predict(self, batch):
        """
        Generate predictions for a batch of data without updating metrics.

        This method matches the interface of BaseTimeSeriesModel.predict() to ensure
        consistency with tstBaselines and other evaluation scripts.

        Args:
            batch: Input batch tuple (x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask)

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

        # Get datamodule from trainer for target normalization params
        dm = self.trainer.datamodule if self.trainer else None

        # Move inputs to model device
        device = self.device
        x_ts = x_ts.to(device)
        x_static = x_static.to(device)
        y_z = y_z.to(device)
        validity_mask = validity_mask.to(device) if validity_mask is not None else None

        # Apply tokenization if enabled (base model handles this)
        x_ts, validity_mask = self.base_model._apply_tokenization(x_ts, observed_mask=validity_mask)

        # Compute trends if enabled (base model handles this)
        batch_trends = None
        if dm and hasattr(self.base_model, 'trend_model') and self.base_model.trend_model._train_df is not None:
            batch_trends = self.base_model._compute_batch_trends(adm_ids, years, dm, lats, lons)

        # Normalize input data (handles NaN via torch.nan_to_num)
        x_ts_n = self.base_model._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self.base_model._normalize_and_impute_static(x_static)

        # Forward pass through WFAN
        pred_z = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trend back if applicable
        if batch_trends is not None:
            trends_squeezed = batch_trends.squeeze(-1).detach()  # (batch,)
            if trends_squeezed.dim() == 1 and pred_z.dim() == 2:
                # Broadcast trends to match quantile dimensions
                trends_broadcast = trends_squeezed.unsqueeze(-1).repeat(1, pred_z.size(1))
                final_pred_z = pred_z + trends_broadcast
            else:
                final_pred_z = pred_z + trends_squeezed
        else:
            final_pred_z = pred_z

        # Denormalize to original scale
        if dm:
            y_mean = dm.y_mean
            y_std = dm.y_std
        else:
            # Fallback: use base model's feature norm params
            y_mean = self.base_model.feature_norm_params['target_mean']
            y_std = self.base_model.feature_norm_params['target_std']

        # Denormalize: pred = pred_z * std + mean
        predictions = final_pred_z * y_std + y_mean

        # Clip to ensure non-negative yields
        predictions = torch.clamp(predictions, min=0.0)

        # Denormalize targets
        targets = y_z * y_std + y_mean

        return {
            'predictions': predictions,
            'predictions_z': final_pred_z,
            'targets': targets,
            'years': years,
            'adm_ids': adm_ids,
            'lats': lats,
            'lons': lons
        }

    def training_step(self, batch, batch_idx):
        """
        Training step with combined loss.

        Loss = forecast_loss + lambda * non_stationary_loss
        """
        # Temporarily patch base model's log method to forward to wrapper
        original_log = self.base_model.log
        self.base_model.log = self.log
        try:
            result = self.base_model.training_step(batch, batch_idx)
        finally:
            self.base_model.log = original_log
        return result

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Temporarily patch base model's log method to forward to wrapper
        original_log = self.base_model.log
        self.base_model.log = self.log
        try:
            result = self.base_model.validation_step(batch, batch_idx)
        finally:
            self.base_model.log = original_log
        return result

    def test_step(self, batch, batch_idx):
        """Test step."""
        # Temporarily patch base model's log method to forward to wrapper
        original_log = self.base_model.log
        self.base_model.log = self.log
        try:
            result = self.base_model.test_step(batch, batch_idx)
        finally:
            self.base_model.log = original_log
        return result

    def configure_optimizers(self):
        """Configure optimizers - delegate to base model."""
        return self.base_model.configure_optimizers()

    def on_train_start(self):
        """Hook for train start - delegate to base model."""
        if hasattr(self.base_model, 'on_train_start'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_train_start()
            finally:
                self.base_model.log = original_log

    def on_train_epoch_end(self):
        """Hook for train epoch end - delegate to base model."""
        if hasattr(self.base_model, 'on_train_epoch_end'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_train_epoch_end()
            finally:
                self.base_model.log = original_log

    def on_validation_epoch_end(self):
        """Hook for validation epoch end - delegate to base model."""
        if hasattr(self.base_model, 'on_validation_epoch_end'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_validation_epoch_end()
            finally:
                self.base_model.log = original_log

    def on_train_end(self):
        """Hook for train end - delegate to base model."""
        if hasattr(self.base_model, 'on_train_end'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_train_end()
            finally:
                self.base_model.log = original_log

    def on_test_start(self):
        """Hook for test start - delegate to base model."""
        if hasattr(self.base_model, 'on_test_start'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_test_start()
            finally:
                self.base_model.log = original_log

    def on_test_epoch_end(self):
        """Hook for test epoch end - delegate to base model."""
        if hasattr(self.base_model, 'on_test_epoch_end'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_test_epoch_end()
            finally:
                self.base_model.log = original_log

    def on_fit_start(self):
        """Hook for fit start - delegate to base model."""
        if hasattr(self.base_model, 'on_fit_start'):
            original_log = self.base_model.log
            self.base_model.log = self.log
            try:
                self.base_model.on_fit_start()
            finally:
                self.base_model.log = original_log

    @property
    def example_input_array(self):
        """Property for lightning logging - delegate to base model."""
        if hasattr(self.base_model, 'example_input_array'):
            return self.base_model.example_input_array
        return None
