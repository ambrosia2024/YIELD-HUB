# -*- coding: utf-8 -*-
"""
Multitask-specific model wrappers that extract embeddings for classification.

This module provides multitask extensions for base models without modifying
the original implementations. It intercepts the forward pass to extract
encoder embeddings for the classification head.

Python version: 3.12.0
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class MultitaskTSTWrapper(pl.LightningModule):
    """
    Wrapper for TST models (PatchTST, Autoformer, etc.) that enables multitask learning.

    This wrapper intercepts the base model's forward pass to extract encoder embeddings
    before the regression head, making them available for the classification head.

    Architecture:
        x_ts, x_static → base_model.encoder → embeddings → classification_head → class_logits
                                                ↓
                                         regression_head → reg_output

    The wrapper maintains full compatibility with the base model while adding
    the ability to use shared embeddings for classification.
    """

    def __init__(self, base_model: pl.LightningModule, debug_mode: bool = False):
        """
        Args:
            base_model: The base TST model (PatchTST, Autoformer, etc.)
                       May be wrapped with WFAN or other wrappers.
            debug_mode: Enable extensive debugging output
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])

        self.base_model = base_model
        self.debug_mode = debug_mode

        # Detect nested wrapping and find the actual TST model
        # Chain could be: MultitaskWFANWrapper → MultitaskTSTWrapper → PatchTST
        # Or: MultitaskTSTWrapper → PatchTST
        self.inner_model = base_model
        self.is_wrapped = False

        # Try to unwrap to find the actual TST model with encoder/regression_head
        current = base_model
        depth = 0
        max_depth = 3  # Prevent infinite loops

        while depth < max_depth:
            # Check if current has encoder (this is a TST model)
            if hasattr(current, 'encoder') and hasattr(current, 'regression_head'):
                self.inner_model = current
                break

            # Check if current is a wrapper (has base_model attribute)
            if hasattr(current, 'base_model'):
                self.is_wrapped = True
                current = current.base_model
                depth += 1
            else:
                # Not a wrapper and not a TST model
                break

        if self.is_wrapped or self.debug_mode:
            logger.info(f"[MultitaskTSTWrapper] Detected wrapped base model, unwrapped to depth {depth}")
            if self.debug_mode:
                logger.info(f"[MultitaskTSTWrapper DEBUG] Wrapping chain analysis:")
                current = base_model
                d = 0
                while current is not None and d < 5:
                    logger.info(f"[MultitaskTSTWrapper DEBUG]   Layer {d}: {type(current).__name__}")
                    if hasattr(current, 'base_model'):
                        current = current.base_model
                    elif hasattr(current, 'inner_model'):
                        current = current.inner_model
                    else:
                        current = None
                    d += 1

        # Copy training settings from base model (not inner model)
        self.lr = getattr(base_model, 'lr', 1e-4)
        self.weight_decay = getattr(base_model, 'weight_decay', 1e-5)
        self.config = getattr(base_model, 'config', None)
        self.loss_type = getattr(base_model, 'loss_type', 'mse')
        self.quantiles = getattr(base_model, 'quantiles', None)
        self.mq_loss_fn = getattr(base_model, 'mq_loss_fn', None)

        # Store encoder and regression head references from inner model
        # For PatchTST and similar models:
        # - inner_model.base_model is the HuggingFace model (encoder)
        # - inner_model.regression_head is the final prediction layer
        self.encoder = getattr(self.inner_model, 'base_model', None)
        self.regression_head = getattr(self.inner_model, 'regression_head', None)

        # Store methods needed for embedding extraction
        self._extract_hidden_state = getattr(self.inner_model, '_extract_hidden_state', None)
        self._pool_hidden_state = getattr(self.inner_model, '_pool_hidden_state', None)
        self._apply_positional_encoding = getattr(self.inner_model, '_apply_positional_encoding', None)
        self._actual_context_length = getattr(self.inner_model, '_actual_context_length', None)
        self._reshape_output = getattr(self.inner_model, '_reshape_output', None)

        # Store additional attributes that might be needed
        self.n_ts_features = getattr(self.inner_model, 'n_ts_features', None)
        self.n_static_features = getattr(self.inner_model, 'n_static_features', None)
        self.use_positional_encoding = getattr(self.inner_model, 'use_positional_encoding', False)
        self.positional_encoding = getattr(self.inner_model, 'positional_encoding', None)

        if self.encoder is None or self.regression_head is None:
            logger.warning(
                "[MultitaskTSTWrapper] Could not find encoder or regression_head in base model. "
                "This wrapper is designed for TST-style models with explicit encoder/head separation."
            )

        logger.info(
            f"[MultitaskTSTWrapper] Initialized - loss_type={self.loss_type}, "
            f"encoder={'found' if self.encoder else 'not found'}, "
            f"regression_head={'found' if self.regression_head else 'not found'}"
        )

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional embedding extraction.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask for valid timesteps
            return_embeddings: If True, return (predictions, embeddings)

        Returns:
            predictions: Regression predictions
            embeddings (optional): Pooled encoder embeddings (when return_embeddings=True)
        """
        if self.debug_mode and not hasattr(self, '_tst_forward_logged'):
            logger.info(f"[MultitaskTSTWrapper DEBUG] === FORWARD PASS ===")
            logger.info(f"[MultitaskTSTWrapper DEBUG] x_ts shape: {x_ts.shape}, mean: {x_ts.mean():.6f}, std: {x_ts.std():.6f}")
            logger.info(f"[MultitaskTSTWrapper DEBUG] x_static shape: {x_static.shape}, mean: {x_static.mean():.6f}, std: {x_static.std():.6f}")
            logger.info(f"[MultitaskTSTWrapper DEBUG] return_embeddings: {return_embeddings}")
            logger.info(f"[MultitaskTSTWrapper DEBUG] Have encoder: {self.encoder is not None}")
            logger.info(f"[MultitaskTSTWrapper DEBUG] Have regression_head: {self.regression_head is not None}")
            self._tst_forward_logged = True

        # If we can't extract embeddings, just call base model directly
        if (self.encoder is None or self.regression_head is None or
            self._extract_hidden_state is None or self._pool_hidden_state is None):
            # Fallback: just use base model as-is
            if self.debug_mode and not hasattr(self, '_tst_fallback_logged'):
                logger.warning(f"[MultitaskTSTWrapper DEBUG] Cannot extract embeddings - missing components, falling back to base model")
                self._tst_fallback_logged = True
            output = self.base_model(x_ts, x_static, observed_mask)
            if return_embeddings:
                # Can't extract embeddings, return None
                return output, None
            return output

        # Truncate input to match context_length
        B, T, C = x_ts.shape
        if self._actual_context_length and T > self._actual_context_length:
            if self.debug_mode and not hasattr(self, '_tst_truncate_logged'):
                logger.info(f"[MultitaskTSTWrapper DEBUG] Truncating x_ts from {T} to {self._actual_context_length}")
                self._tst_truncate_logged = True
            x_ts = x_ts[:, :self._actual_context_length, :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, :self._actual_context_length]
            T = self._actual_context_length

        # Process temporal features through encoder
        # This mimics what the base model does in its forward pass
        try:
            # For HuggingFace models (PatchTST, Autoformer, etc.)
            past_observed_mask = (
                observed_mask.unsqueeze(-1).expand(-1, -1, C).float()
                if observed_mask is not None else None
            )

            # Call the encoder (base_model.base_model for TST models)
            outputs = self.encoder(
                past_values=x_ts,
                past_observed_mask=past_observed_mask,
                future_values=None
            )

            # Extract hidden state
            h = self._extract_hidden_state(outputs)

            if self.debug_mode and not hasattr(self, '_tst_hidden_logged'):
                logger.info(f"[MultitaskTSTWrapper DEBUG] Hidden state shape: {h.shape}, mean: {h.mean():.6f}, std: {h.std():.6f}")
                self._tst_hidden_logged = True

            # Apply positional encoding if enabled
            if self._apply_positional_encoding:
                h = self._apply_positional_encoding(h, observed_mask)

            # Pool hidden state to get embeddings
            embeddings = self._pool_hidden_state(h)  # (B, pooled_dim)

            if self.debug_mode and not hasattr(self, '_tst_embeddings_logged'):
                logger.info(f"[MultitaskTSTWrapper DEBUG] Extracted embeddings shape: {embeddings.shape}, mean: {embeddings.mean():.6f}, std: {embeddings.std():.6f}")
                self._tst_embeddings_logged = True

            # Concatenate with static features and pass through regression head
            combined = torch.cat([embeddings, x_static], dim=-1)
            predictions = self._reshape_output(self.regression_head(combined))

            if self.debug_mode and not hasattr(self, '_tst_predictions_logged'):
                logger.info(f"[MultitaskTSTWrapper DEBUG] Combined shape: {combined.shape}, predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                self._tst_predictions_logged = True

            if return_embeddings:
                return predictions, embeddings
            return predictions

        except Exception as e:
            logger.error(f"[MultitaskTSTWrapper] Error extracting embeddings: {e}")
            logger.error(f"[MultitaskTSTWrapper] Falling back to base model forward pass")
            # Fallback to base model
            output = self.base_model(x_ts, x_static, observed_mask)
            if return_embeddings:
                return output, None
            return output

    def training_step(self, batch, batch_idx):
        """Delegate to base model."""
        return self.base_model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Delegate to base model."""
        return self.base_model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Delegate to base model."""
        return self.base_model.test_step(batch, batch_idx)

    def configure_optimizers(self):
        """Delegate to base model."""
        return self.base_model.configure_optimizers()

    def on_train_start(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_train_start'):
            self.base_model.on_train_start()

    def on_train_epoch_end(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_train_epoch_end'):
            self.base_model.on_train_epoch_end()

    def on_validation_epoch_end(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_validation_epoch_end'):
            self.base_model.on_validation_epoch_end()

    def on_train_end(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_train_end'):
            self.base_model.on_train_end()

    def on_test_start(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_test_start'):
            self.base_model.on_test_start()

    def on_test_epoch_end(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_test_epoch_end'):
            self.base_model.on_test_epoch_end()

    def on_fit_start(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'on_fit_start'):
            self.base_model.on_fit_start()

    @property
    def example_input_array(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'example_input_array'):
            return self.base_model.example_input_array
        return None


def create_multitask_tst_model(base_model):
    """
    Create a multitask-enabled TST model.

    This function wraps a base TST model with MultitaskTSTWrapper to enable
    embedding extraction for classification.

    Args:
        base_model: The base TST model (e.g., PatchTST, Autoformer)

    Returns:
        MultitaskTSTWrapper: Wrapped model with embedding extraction capability
    """
    return MultitaskTSTWrapper(base_model)
