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

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_vars)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask for valid timesteps (batch, seq_len)

        Returns:
            Predictions of shape (batch,)
        """
        # Apply frequency normalization
        x_res, x_non = self.apply_frequency_normalization(x_ts)

        # Base model predicts stationary residual
        # Note: base_model expects (x_ts, x_static, observed_mask)
        y_res = self.base_model(x_res, x_static, observed_mask=observed_mask)

        # Pattern-adaptive module predicts non-stationary evolution
        # Takes both non-stationary component and original input
        y_non = self.pattern_adaptive(x_non, x_ts)

        # For yield forecasting, we need a scalar prediction per sample
        # The pattern_adaptive output is (batch, 1, n_vars) - pool over vars
        y_non_scalar = y_non.squeeze(1).mean(dim=-1)  # (batch,)

        # Combine predictions
        y_pred = y_res + y_non_scalar

        return y_pred

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


class MultitaskWFANWrapper(pl.LightningModule):
    """
    WFAN Wrapper for multitask models with MQL (Multi-Quantile Loss) support.

    This is a variant of WFANWrapper that handles multi-dimensional outputs from
    MQL-enabled models. When the base model returns (batch, n_quantiles) instead
    of (batch,), this wrapper broadcasts the non-stationary correction to all
    quantiles.

    This wrapper is designed for multitask learning where the base model may
    output quantiles (for MQL) or scalar predictions (for MSE).

    Key difference from WFANWrapper:
        - Handles (batch, n_quantiles) regression outputs
        - Broadcasts non-stationary correction to all quantiles

    Args:
        base_model: The underlying time series model to wrap
        seq_len: Length of input sequence
        n_vars: Number of time series variables/channels
        K: Number of dominant frequency components to remove (default: 2)
        lambda_coef: Loss balancing coefficient for non-stationary prediction (default: 1.0)
        config: Model configuration object
        debug_mode: Enable extensive debugging output
    """

    def __init__(
        self,
        base_model: pl.LightningModule,
        seq_len: int,
        n_vars: int,
        K: int = 2,
        lambda_coef: float = 1.0,
        config=None,
        debug_mode: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.base_model = base_model
        self.config = config
        self.K = K
        self.lambda_coef = lambda_coef
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.debug_mode = debug_mode

        # Pattern-adaptive MLP for predicting non-stationary evolution
        # Reuse the same PatternAdaptiveModule as WFANWrapper
        self.pattern_adaptive = PatternAdaptiveModule(
            seq_len=seq_len,
            n_vars=n_vars,
            pred_len=1  # Single-step prediction for yield forecasting
        )

        # Copy MQL attributes from base model
        self.loss_type = getattr(base_model, 'loss_type', 'mse')
        self.quantiles = getattr(base_model, 'quantiles', None)
        self.mq_loss_fn = getattr(base_model, 'mq_loss_fn', None)

        # Detect if using MQL
        self.use_mql = (
            hasattr(base_model, 'loss_type') and
            base_model.loss_type == "mql"
        )
        self.n_quantiles = getattr(base_model, 'quantiles', [0.1, 0.5, 0.9])
        if isinstance(self.n_quantiles, list):
            self.n_quantiles = len(self.n_quantiles)

        logger.info(
            f"[MultitaskWFANWrapper] Initialized - K={K}, lambda={lambda_coef}, "
            f"seq_len={seq_len}, n_vars={n_vars}, MQL={self.use_mql}, "
            f"loss_type={self.loss_type}, "
            f"n_quantiles={self.n_quantiles if self.use_mql else 'N/A'}, "
            f"debug={debug_mode}"
        )

        if self.debug_mode:
            logger.info(f"[MultitaskWFANWrapper DEBUG] Wrapping chain analysis:")
            current = base_model
            depth = 0
            while current is not None and depth < 5:
                logger.info(f"[MultitaskWFANWrapper DEBUG]   Layer {depth}: {type(current).__name__}")
                if hasattr(current, 'base_model'):
                    current = current.base_model
                elif hasattr(current, 'inner_model'):
                    current = current.inner_model
                else:
                    current = None
                depth += 1

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
        x_freq = torch.fft.rfft(x, dim=1)  # (batch, freq_bins, n_vars)

        # Compute amplitudes for frequency selection
        amplitudes = torch.abs(x_freq)

        # Select top-K dominant frequencies per instance per feature
        topk_vals, topk_indices = torch.topk(amplitudes, self.K, dim=1)

        # Create filtering mask
        mask = torch.zeros_like(amplitudes, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # Extract dominant frequency components (non-stationary)
        x_freq_non = x_freq * mask

        # Inverse FFT to get time-domain non-stationary component
        x_non = torch.fft.irfft(x_freq_non, n=seq_len, dim=1)

        # Residual (stationary part) = original - non-stationary
        x_res = x - x_non

        return x_res, x_non

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through MultitaskWFAN-wrapped model.

        Handles both scalar outputs (MSE) and multi-quantile outputs (MQL).
        Optionally returns embeddings when return_embeddings=True.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_vars)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask for valid timesteps (batch, seq_len)
            return_embeddings: If True, return (predictions, embeddings) tuple

        Returns:
            predictions: Predictions of shape (batch,) for MSE or (batch, n_quantiles) for MQL
            embeddings (optional): When return_embeddings=True
        """
        if self.debug_mode and not hasattr(self, '_wfan_forward_logged'):
            logger.info(f"[MultitaskWFANWrapper DEBUG] === FORWARD PASS ===")
            logger.info(f"[MultitaskWFANWrapper DEBUG] x_ts shape: {x_ts.shape}, mean: {x_ts.mean():.6f}, std: {x_ts.std():.6f}")
            logger.info(f"[MultitaskWFANWrapper DEBUG] x_static shape: {x_static.shape}, mean: {x_static.mean():.6f}, std: {x_static.std():.6f}")
            self._wfan_forward_logged = True

        # Apply frequency normalization
        x_res, x_non = self.apply_frequency_normalization(x_ts)

        if self.debug_mode and not hasattr(self, '_wfan_freq_norm_logged'):
            logger.info(f"[MultitaskWFANWrapper DEBUG] After frequency normalization:")
            logger.info(f"[MultitaskWFANWrapper DEBUG] x_res (stationary) shape: {x_res.shape}, mean: {x_res.mean():.6f}, std: {x_res.std():.6f}")
            logger.info(f"[MultitaskWFANWrapper DEBUG] x_non (non-stationary) shape: {x_non.shape}, mean: {x_non.mean():.6f}, std: {x_non.std():.6f}")
            self._wfan_freq_norm_logged = True

        # Base model predicts stationary residual
        # Request embeddings if needed
        base_output = self.base_model(x_res, x_static, observed_mask=observed_mask, return_embeddings=return_embeddings)

        # Extract predictions and optional embeddings
        if return_embeddings and isinstance(base_output, tuple) and len(base_output) == 2:
            y_res, embeddings = base_output
        else:
            y_res = base_output
            embeddings = None

        if self.debug_mode and not hasattr(self, '_wfan_base_output_logged'):
            logger.info(f"[MultitaskWFANWrapper DEBUG] Base model output shape: {y_res.shape if hasattr(y_res, 'shape') else 'N/A'}")
            if hasattr(y_res, 'shape'):
                logger.info(f"[MultitaskWFANWrapper DEBUG] Base model output stats: mean: {y_res.mean():.6f}, std: {y_res.std():.6f}")
            self._wfan_base_output_logged = True

        # Pattern-adaptive module predicts non-stationary evolution
        y_non = self.pattern_adaptive(x_non, x_ts)

        # For yield forecasting, we need a scalar prediction per sample
        # The pattern_adaptive output is (batch, 1, n_vars) - pool over vars
        y_non_scalar = y_non.squeeze(1).mean(dim=-1)  # (batch,)

        if self.debug_mode and not hasattr(self, '_wfan_non_stationary_logged'):
            logger.info(f"[MultitaskWFANWrapper DEBUG] Non-stationary prediction shape: {y_non_scalar.shape}, mean: {y_non_scalar.mean():.6f}, std: {y_non_scalar.std():.6f}")
            self._wfan_non_stationary_logged = True

        # Combine predictions - handle both MSE and MQL cases
        if y_res.dim() > 1:
            # MQL case: y_res is (batch, n_quantiles)
            # Broadcast y_non_scalar to match quantiles dimension
            y_non_expanded = y_non_scalar.unsqueeze(-1).expand_as(y_res)
            y_pred = y_res + y_non_expanded
        else:
            # MSE case: y_res is (batch,)
            y_pred = y_res + y_non_scalar

        if self.debug_mode and not hasattr(self, '_wfan_final_logged'):
            logger.info(f"[MultitaskWFANWrapper DEBUG] Final prediction shape: {y_pred.shape}, mean: {y_pred.mean():.6f}, std: {y_pred.std():.6f}")
            self._wfan_final_logged = True

        # Return predictions with embeddings if requested
        if return_embeddings:
            return y_pred, embeddings
        return y_pred

    def training_step(self, batch, batch_idx):
        """Training step - delegate to base model."""
        original_log = self.base_model.log
        self.base_model.log = self.log
        try:
            result = self.base_model.training_step(batch, batch_idx)
        finally:
            self.base_model.log = original_log
        return result

    def validation_step(self, batch, batch_idx):
        """Validation step - delegate to base model."""
        original_log = self.base_model.log
        self.base_model.log = self.log
        try:
            result = self.base_model.validation_step(batch, batch_idx)
        finally:
            self.base_model.log = original_log
        return result

    def test_step(self, batch, batch_idx):
        """Test step - delegate to base model."""
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
