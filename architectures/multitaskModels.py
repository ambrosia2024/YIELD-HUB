# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Multitask model wrappers for dual-head yield forecasting.
Combines regression (with uncertainty metrics) and ordinal classification heads.

This module is standalone and wraps existing model architectures
without modifying them.

Features:
- Regression head: Point prediction (MSE) or quantile regression (MQL)
- Classification head: 5-class ordinal regression for yield deviation
- Uncertainty metrics: CRPS, PICP, PINAW, Winkler Score (when using MQL)
- Spatiotemporal metrics: Spatial/temporal/anomaly correlations

Python version: 3.12.0
--------------------
"""

import logging
from typing import Optional, Dict, List, Tuple, Union
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import lightning.pytorch as pl

from cybench.config import KEY_LOC, KEY_YEAR

# Import ordinal loss
try:
    from ordinalLoss import get_ordinal_loss
except ImportError:
    # Try relative import
    from .ordinalLoss import get_ordinal_loss

# Import existing metrics
import sys
sys.path.append('../process/')
try:
    from metrics import ModelMetrics
except ImportError:
    # For standalone testing, create a minimal ModelMetrics
    class ModelMetrics(pl.LightningModule):
        def __init__(self, prefix="test", include_nrmse=True):
            super().__init__()
            from torchmetrics import MetricCollection, MeanSquaredError
            self.metrics = MetricCollection({'mse': MeanSquaredError()})

        def update(self, preds, targets):
            self.metrics.update(preds, targets)

        def compute(self):
            return self.metrics.compute()

        def reset(self):
            self.metrics.reset()

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

class MultitaskModelConfig:
    """
    Configuration for multitask model training.

    Attributes:
        classification_alpha: Weight for classification loss (total_loss = regression_loss + alpha * classification_loss)
        num_classes: Number of ordinal classes
        ordinal_loss_type: Type of ordinal loss ('corn' or 'ce')
        classification_dropout: Dropout rate for classification head
    """

    def __init__(
        self,
        classification_alpha: float = 0.3,
        num_classes: int = 5,
        ordinal_loss_type: str = "corn",
        classification_dropout: float = 0.1,
    ):
        self.classification_alpha = classification_alpha
        self.num_classes = num_classes
        self.ordinal_loss_type = ordinal_loss_type
        self.classification_dropout = classification_dropout


# ============================================================================
# Classification Head
# ============================================================================

class MultitaskHead(nn.Module):
    """
    Classification head for ordinal regression.

    Architecture: Linear -> LayerNorm -> ReLU -> Dropout -> Linear
    Output dimension: num_classes - 1 for CORN loss, num_classes for CE loss
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        ordinal_loss_type: str = "corn",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of ordinal classes
            ordinal_loss_type: Type of ordinal loss ('corn' or 'ce')
            hidden_dim: Hidden dimension (defaults to input_dim // 2)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_loss_type = ordinal_loss_type

        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)

        # Output dimension: K-1 for CORN, K for CE
        output_dim = num_classes - 1 if ordinal_loss_type == "corn" else num_classes

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        logger.info(
            f"[MultitaskHead] Input dim: {input_dim}, Hidden: {hidden_dim}, "
            f"Output: {output_dim}, Loss: {ordinal_loss_type}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        return self.head(x)


# ============================================================================
# Uncertainty Metrics (for MQL Loss)
# ============================================================================

class UncertaintyMetrics(nn.Module):
    """
    Uncertainty metrics for quantile regression predictions.

    Computes:
    - CRPS (Continuous Ranked Probability Score)
    - PICP (Prediction Interval Coverage Probability)
    - PINAW (Prediction Interval Normalized Average Width)
    - Winkler Score
    - Mean Interval Width
    - Pinball Loss per quantile
    """

    def __init__(self, quantiles: List[float] = None):
        """
        Args:
            quantiles: List of quantiles to evaluate (default: [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_quantiles = len(self.quantiles)

        # Storage for metrics computation
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets."""
        self.all_predictions = []
        self.all_targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update stored predictions and targets.

        Args:
            predictions: (batch, n_quantiles) predicted quantiles
            targets: (batch,) true values
        """
        self.all_predictions.append(predictions.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute all uncertainty metrics."""
        if len(self.all_predictions) == 0:
            return {}

        # Concatenate all batches
        all_preds = torch.cat(self.all_predictions, dim=0).numpy()
        all_targets = torch.cat(self.all_targets, dim=0).numpy()

        metrics = {}

        # Pinball loss per quantile
        pinball_losses = []
        for i, q in enumerate(self.quantiles):
            errors = all_targets - all_preds[:, i]
            q_loss = np.maximum(q * errors, (q - 1) * errors).mean()
            metrics[f'pinball_q{int(q*100)}'] = float(q_loss)
            pinball_losses.append(q_loss)

        metrics['pinball_avg'] = float(np.mean(pinball_losses))

        # CRPS (continuous ranked probability score)
        # Approximated using the quantile predictions
        metrics['crps'] = float(np.mean(pinball_losses))

        # Prediction interval metrics (using 0.1 and 0.9 quantiles)
        lower_idx = 0  # 0.1 quantile
        upper_idx = -1  # 0.9 quantile

        lower_bound = all_preds[:, lower_idx]
        upper_bound = all_preds[:, upper_idx]

        # PICP (Prediction Interval Coverage Probability)
        # Proportion of true values within the prediction interval
        within_interval = (all_targets >= lower_bound) & (all_targets <= upper_bound)
        metrics['picp'] = float(np.mean(within_interval))

        # PINAW (Prediction Interval Normalized Average Width)
        # Normalized by the range of true values
        interval_width = upper_bound - lower_bound
        target_range = all_targets.max() - all_targets.min()
        if target_range > 0:
            metrics['pinaw'] = float(np.mean(interval_width) / target_range)
        else:
            metrics['pinaw'] = float(np.mean(interval_width))

        # Mean interval width
        metrics['mean_interval_width'] = float(np.mean(interval_width))

        # Winkler Score
        # Combines coverage penalty with interval width
        alpha = 0.1  # For 90% interval
        penalty = 2 / alpha

        winkler_scores = []
        for i in range(len(all_targets)):
            if all_targets[i] < lower_bound[i]:
                # Lower breach
                score = upper_bound[i] - lower_bound[i] + penalty * (lower_bound[i] - all_targets[i])
            elif all_targets[i] > upper_bound[i]:
                # Upper breach
                score = upper_bound[i] - lower_bound[i] + penalty * (all_targets[i] - upper_bound[i])
            else:
                # Within interval
                score = upper_bound[i] - lower_bound[i]
            winkler_scores.append(score)

        metrics['winkler_score'] = float(np.mean(winkler_scores))

        return metrics


# ============================================================================
# Combined Metrics (Regression + Classification + Uncertainty)
# ============================================================================

class MultitaskMetrics(nn.Module):
    """
    Comprehensive metrics for multitask model evaluation.

    Tracks:
    - Regression metrics (MSE, MAE, RMSE, R², MAPE, SMAPE, NRMSE)
    - Classification metrics (Accuracy, F1, QWK)
    - Uncertainty metrics (CRPS, PICP, PINAW, Winkler Score) - when using MQL
    """

    def __init__(
        self,
        prefix: str = "test",
        include_nrmse: bool = True,
        num_classes: int = 5,
        use_uncertainty_metrics: bool = False,
        quantiles: List[float] = None
    ):
        """
        Args:
            prefix: Metric name prefix (train/val/test)
            include_nrmse: Whether to include NRMSE
            num_classes: Number of classes for classification metrics
            use_uncertainty_metrics: Whether to compute uncertainty metrics (for MQL)
            quantiles: Quantiles for uncertainty metrics
        """
        super().__init__()
        self.prefix = prefix
        self.num_classes = num_classes
        self.use_uncertainty_metrics = use_uncertainty_metrics

        # Regression metrics
        metrics_dict = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'mape': torchmetrics.MeanAbsolutePercentageError(),
            'smape': torchmetrics.SymmetricMeanAbsolutePercentageError(),
        }
        if include_nrmse:
            metrics_dict['nrmse'] = torchmetrics.NormalizedRootMeanSquaredError(normalization='mean')

        self.regression_metrics = torchmetrics.MetricCollection(metrics_dict)

        # Classification metrics
        class_metrics_dict = {
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            'f1_macro': torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'f1_micro': torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='micro'),
            'f1_weighted': torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='weighted'),
        }
        self.classification_metrics = torchmetrics.MetricCollection(class_metrics_dict)

        # Uncertainty metrics (for MQL)
        self.uncertainty_metrics = None
        if use_uncertainty_metrics:
            self.uncertainty_metrics = UncertaintyMetrics(quantiles=quantiles)

        # Additional metrics storage
        self.qwk_scores = []
        self.class_mql_errors = []

    def update(
        self,
        reg_preds: torch.Tensor,
        reg_targets: torch.Tensor,
        class_preds: torch.Tensor,
        class_targets: torch.Tensor,
        reg_full_output: Optional[torch.Tensor] = None
    ):
        """
        Update all metrics.

        Args:
            reg_preds: Regression predictions (point estimates or median quantile)
            reg_targets: Regression targets
            class_preds: Classification predictions
            class_targets: Classification targets
            reg_full_output: Full regression output (all quantiles) for uncertainty metrics
        """
        # Update regression metrics
        self.regression_metrics.update(reg_preds, reg_targets)

        # Update classification metrics
        self.classification_metrics.update(class_preds, class_targets)

        # Update uncertainty metrics if enabled and full output provided
        if self.uncertainty_metrics is not None and reg_full_output is not None:
            self.uncertainty_metrics.update(reg_full_output, reg_targets)

        # Store for QWK calculation
        if class_preds.device.type == 'cpu':
            self.class_preds_cpu = class_preds
            self.class_targets_cpu = class_targets

    def reset(self):
        """Reset all metrics."""
        self.regression_metrics.reset()
        self.classification_metrics.reset()
        if self.uncertainty_metrics is not None:
            self.uncertainty_metrics.reset()
        self.qwk_scores = []
        self.class_mql_errors = []

    def compute(self) -> Dict:
        """Compute all metrics."""
        results = {}

        # Regression metrics - align with baseline scripts for comparison
        reg_results = self.regression_metrics.compute()
        for key, val in reg_results.items():
            results[f'regression_{key}'] = val

        # Add RMSE (sqrt of MSE) - tracked by baseline scripts
        if 'mse' in reg_results:
            results['regression_rmse'] = torch.sqrt(reg_results['mse'])

        # Classification metrics
        class_results = self.classification_metrics.compute()
        for key, val in class_results.items():
            results[f'classification_{key}'] = val

        # QWK (computed separately)
        if hasattr(self, 'class_preds_cpu') and hasattr(self, 'class_targets_cpu'):
            qwk = self._compute_qwk(self.class_preds_cpu, self.class_targets_cpu)
            results['classification_qwk'] = qwk

        # Class MQL (treats classes as 0-4 values)
        if hasattr(self, 'class_preds_cpu') and hasattr(self, 'class_targets_cpu'):
            mql = self._compute_class_mql(self.class_preds_cpu, self.class_targets_cpu)
            results['classification_mql'] = mql

        # Uncertainty metrics (logged without prefix for wandb comparison)
        if self.uncertainty_metrics is not None:
            unc_results = self.uncertainty_metrics.compute()
            for key, val in unc_results.items():
                results[key] = val

        return results

    @staticmethod
    def _compute_qwk(preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Quadratic Weighted Kappa."""
        try:
            from sklearn.metrics import cohen_kappa_score
            preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
            targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
            return float(cohen_kappa_score(targets_np, preds_np, weights='quadratic'))
        except Exception as e:
            logger.warning(f"Could not compute QWK: {e}")
            return 0.0

    @staticmethod
    def _compute_class_mql(preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute mean quantile loss treating classes as 0-4 values."""
        preds_np = preds.float().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        targets_np = targets.float().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        return float(np.mean(np.abs(preds_np - targets_np)))


# ============================================================================
# Multitask Wrapper
# ============================================================================

class MultitaskWrapper(pl.LightningModule):
    """
    Wrapper for multitask learning with existing base models.

    Wraps any time series model to add:
    - Classification head
    - Dual loss computation
    - Comprehensive metrics (regression + classification + uncertainty)
    - Spatiotemporal metrics (optional)

    Usage:
        # Create base model
        base_model = create_model(config)
        # Wrap with multitask
        multitask_model = MultitaskWrapper(base_model, multitask_config)
    """

    def __init__(
        self,
        base_model: pl.LightningModule,
        multitask_config: MultitaskModelConfig,
        classification_input_dim: Optional[int] = None,
        use_spatiotemporal_metrics: bool = False,
        debug_mode: bool = False
    ):
        """
        Args:
            base_model: Existing time series model (Linear or TST)
            multitask_config: Multitask configuration
            classification_input_dim: Input dimension for classification head
                                    (defaults to base model's output dimension)
            use_spatiotemporal_metrics: Whether to compute spatiotemporal metrics
            debug_mode: Enable extensive debugging output
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])

        self.base_model = base_model
        self.multitask_config = multitask_config
        self.use_spatiotemporal_metrics = use_spatiotemporal_metrics
        self.debug_mode = debug_mode

        # Feature normalization parameters (will be set from datamodule)
        self.feature_norm_params = None

        # Get input dimension for classification head
        if classification_input_dim is None:
            # Try to infer from base model
            if hasattr(base_model, 'n_static_features'):
                classification_input_dim = base_model.n_static_features + 64  # Heuristic
            else:
                classification_input_dim = 128  # Default

        # Create classification head (will be rebuilt on first forward if dimension mismatch)
        self.classification_input_dim = classification_input_dim
        self.classification_head = MultitaskHead(
            input_dim=classification_input_dim,
            num_classes=multitask_config.num_classes,
            ordinal_loss_type=multitask_config.ordinal_loss_type,
            dropout=multitask_config.classification_dropout
        )
        self._classification_head_built = False  # Track if head has been built with correct dims

        # Create ordinal loss function
        self.classification_loss_fn = get_ordinal_loss(
            num_classes=multitask_config.num_classes,
            loss_type=multitask_config.ordinal_loss_type
        )

        # Check if using MQL for uncertainty metrics
        self.use_mql = (
            hasattr(base_model, 'loss_type') and
            base_model.loss_type == "mql"
        )

        # Get quantiles if using MQL
        self.quantiles = None
        if self.use_mql and hasattr(base_model, 'quantiles'):
            self.quantiles = base_model.quantiles

        # Metrics
        self.train_metrics = MultitaskMetrics(
            prefix="train",
            include_nrmse=False,
            num_classes=multitask_config.num_classes,
            use_uncertainty_metrics=self.use_mql,
            quantiles=self.quantiles
        )
        self.val_metrics = MultitaskMetrics(
            prefix="val",
            include_nrmse=True,
            num_classes=multitask_config.num_classes,
            use_uncertainty_metrics=self.use_mql,
            quantiles=self.quantiles
        )
        self.test_metrics = MultitaskMetrics(
            prefix="test",
            include_nrmse=True,
            num_classes=multitask_config.num_classes,
            use_uncertainty_metrics=self.use_mql,
            quantiles=self.quantiles
        )

        # Store predictions for analysis
        self._test_results_per_year = {}
        self._spatiotemporal_metrics = {}

        # Storage for spatiotemporal computation
        self._test_y_true = []
        self._test_y_pred = []
        self._test_years = []
        self._test_regions = []

        # Copy training settings from base model
        self.lr = getattr(base_model, 'lr', 1e-4)
        self.weight_decay = getattr(base_model, 'weight_decay', 1e-5)
        self.config = getattr(base_model, 'config', None)

        logger.info(
            f"[MultitaskWrapper] Initialized with alpha={multitask_config.classification_alpha}, "
            f"classes={multitask_config.num_classes}, loss={multitask_config.ordinal_loss_type}, "
            f"MQL={self.use_mql}, spatiotemporal={use_spatiotemporal_metrics}, debug={debug_mode}"
        )

        # Debug: Log wrapping chain
        if debug_mode:
            logger.info(f"[MultitaskWrapper DEBUG] Wrapping chain analysis:")
            current = self.base_model
            depth = 0
            while current is not None and depth < 10:
                logger.info(f"[MultitaskWrapper DEBUG]   Layer {depth}: {type(current).__name__}")
                if hasattr(current, 'base_model'):
                    current = current.base_model
                elif hasattr(current, 'inner_model'):
                    current = current.inner_model
                else:
                    current = None
                depth += 1

        # CRITICAL FIX: Detect if base model handles its own normalization
        # Most TST and XLinear models have internal normalization.
        # We should NOT normalize inputs in MultitaskWrapper if the base model does it.
        self._base_model_handles_normalization = self._detect_base_model_normalization()
        if debug_mode:
            logger.info(f"[MultitaskWrapper DEBUG] Base model handles normalization: {self._base_model_handles_normalization}")
        # DON'T set feature_norm_params here - let it be fetched from datamodule when needed
        # The detection result is used as a hint, but actual normalization depends on norm_params availability

    def setup(self, stage: str):
        """
        Setup method called by Lightning Trainer.
        Propagates setup to base model so it can get feature_norm_params from datamodule.
        """
        # Call base model's setup if it exists (e.g., PatchTST needs this to get norm_params)
        current = self.base_model
        depth = 0
        max_depth = 3
        while current is not None and depth < max_depth:
            if hasattr(current, 'setup') and callable(current.setup):
                try:
                    if self.debug_mode:
                        logger.info(f"[SETUP] Calling setup() on {type(current).__name__} at depth {depth}")
                    current.setup(stage)
                    if self.debug_mode and hasattr(current, 'feature_norm_params'):
                        logger.info(f"[SETUP] {type(current).__name__}.feature_norm_params count: {len(current.feature_norm_params) if current.feature_norm_params else 0}")
                except Exception as e:
                    if self.debug_mode:
                        logger.warning(f"[SETUP] Failed to call setup() on {type(current).__name__}: {e}")
            # Navigate to next model in chain
            if hasattr(current, 'base_model'):
                current = current.base_model
                depth += 1
            elif hasattr(current, 'inner_model'):
                current = current.inner_model
                depth += 1
            else:
                break

    def _detect_base_model_normalization(self) -> bool:
        """
        Detect if the base model handles its own input normalization.

        Returns:
            True if base model has internal normalization, False otherwise
        """
        # Check through the wrapping chain
        current = self.base_model
        depth = 0
        max_depth = 5  # Don't search too deep

        # Wrapper class names to skip - these don't handle normalization themselves
        wrapper_classes = {'MultitaskTSTWrapper', 'MultitaskWFANWrapper', 'MultitaskWrapper'}

        while current is not None and depth < max_depth:
            current_class_name = type(current).__name__

            # Skip multitask wrapper classes - they don't handle normalization
            if current_class_name in wrapper_classes:
                if self.debug_mode:
                    logger.info(f"[NORM DETECTION] Skipping wrapper class at depth {depth}: {current_class_name}")
                if hasattr(current, 'base_model'):
                    current = current.base_model
                    depth += 1
                    continue
                elif hasattr(current, 'inner_model'):
                    current = current.inner_model
                    depth += 1
                    continue
                else:
                    break

            # Check for common indicators of internal normalization
            if hasattr(current, 'feature_norm_params') and current.feature_norm_params:
                if self.debug_mode:
                    logger.info(f"[NORM DETECTION] Found feature_norm_params at depth {depth} ({type(current).__name__})")
                return True

            # TST models typically have feature_norm_params
            if hasattr(current, 'n_ts_features') and hasattr(current, 'n_static_features'):
                if self.debug_mode:
                    logger.info(f"[NORM DETECTION] Found TST-style model at depth {depth} ({type(current).__name__})")
                return True

            # XLinear models also handle normalization
            if hasattr(current, 'use_revin') or hasattr(current, 'revin'):
                if self.debug_mode:
                    logger.info(f"[NORM DETECTION] Found RevIN normalization at depth {depth} ({type(current).__name__})")
                return True

            # Check for base_model attribute (wrapper pattern)
            if hasattr(current, 'base_model'):
                current = current.base_model
                depth += 1
            elif hasattr(current, 'inner_model'):
                current = current.inner_model
                depth += 1
            else:
                break

        if self.debug_mode:
            logger.info(f"[NORM DETECTION] No internal normalization detected in wrapping chain")
        return False

    def _get_feature_norm_params(self):
        """Get feature normalization parameters from datamodule or base model."""
        if self.feature_norm_params is None:
            # First, check if base model already has norm_params
            current = self.base_model
            depth = 0
            max_depth = 3
            while current is not None and depth < max_depth:
                if hasattr(current, 'feature_norm_params') and current.feature_norm_params is not None and len(current.feature_norm_params) > 0:
                    if self.debug_mode:
                        logger.info(f"[NORM DEBUG] Got norm params from base model ({type(current).__name__}), count: {len(current.feature_norm_params)}")
                    self.feature_norm_params = current.feature_norm_params
                    return self.feature_norm_params
                if hasattr(current, 'base_model'):
                    current = current.base_model
                    depth += 1
                elif hasattr(current, 'inner_model'):
                    current = current.inner_model
                    depth += 1
                else:
                    break

            # If base model doesn't have norm_params, try to get from datamodule
            if hasattr(self, 'trainer') and self.trainer is not None:
                dm = self.trainer.datamodule
                # Handle both MultitaskDataModule and regular datamodule
                if hasattr(dm, 'base_dm'):
                    # MultitaskDataModule - get from base datamodule
                    self.feature_norm_params = dm.base_dm.feature_norm_params
                    if self.debug_mode:
                        logger.info(f"[NORM DEBUG] Got norm params from MultitaskDataModule.base_dm, keys: {list(self.feature_norm_params.keys())[:5]}...")
                        # Log sample norm params
                        for key in list(self.feature_norm_params.keys())[:3]:
                            logger.info(f"[NORM DEBUG]   {key}: mean={self.feature_norm_params[key]['mean']:.4f}, std={self.feature_norm_params[key]['std']:.4f}")
                elif hasattr(dm, 'feature_norm_params'):
                    # Regular datamodule
                    self.feature_norm_params = dm.feature_norm_params
                    if self.debug_mode:
                        logger.info(f"[NORM DEBUG] Got norm params from datamodule, keys: {list(self.feature_norm_params.keys())[:5]}...")
                else:
                    if self.debug_mode:
                        logger.warning("[NORM DEBUG] No feature_norm_params found in datamodule - normalization will be skipped")
                    self.feature_norm_params = {}  # Empty dict to skip normalization
            else:
                if self.debug_mode:
                    logger.warning("[NORM DEBUG] Cannot get feature_norm_params without trainer")
                self.feature_norm_params = {}
        return self.feature_norm_params

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z-score normalize time series features using training statistics.

        CRITICAL: This method should only be called if the base model does NOT
        handle its own input normalization. Most TST and XLinear models already
        normalize internally. Calling this on already-normalized data causes
        double normalization and destroys model performance.

        Args:
            x_ts: Time series features of shape (batch, seq_len, n_ts_features)
            observed_mask: Boolean mask for valid timesteps

        Returns:
            Normalized time series features (or original if no norm params)
        """
        norm_params = self._get_feature_norm_params()

        # DEBUG: Log first call with extensive statistics
        if not hasattr(self, '_norm_ts_called'):
            if self.debug_mode:
                logger.info(f"[NORM DEBUG] === FIRST _normalize_time_series CALL ===")
                logger.info(f"[NORM DEBUG] Input x_ts shape: {x_ts.shape}")
                logger.info(f"[NORM DEBUG] Input stats - mean: {x_ts.mean():.6f}, std: {x_ts.std():.6f}, min: {x_ts.min():.6f}, max: {x_ts.max():.6f}")
                logger.info(f"[NORM DEBUG] Available norm_params count: {len(norm_params)}")
                if len(norm_params) > 0:
                    logger.info(f"[NORM DEBUG] Sample norm_params keys: {list(norm_params.keys())[:5]}")
                else:
                    logger.warning(f"[NORM DEBUG] No norm_params found - returning original x_ts (base model handles normalization)")
            self._norm_ts_called = True

        # If no norm params, base model handles its own normalization
        if len(norm_params) == 0:
            if self.debug_mode and not hasattr(self, '_norm_skip_logged'):
                logger.info(f"[NORM DEBUG] Skipping normalization - base model handles it internally")
                self._norm_skip_logged = True
            return x_ts

        # Get feature names from x_ts shape
        n_ts_features = x_ts.shape[-1]

        # Normalize each feature independently
        x_ts_norm = torch.zeros_like(x_ts)
        normalized_count = 0
        for i in range(n_ts_features):
            feature_key = f"ts_{i}"
            if feature_key in norm_params:
                mean = norm_params[feature_key]['mean']
                std = norm_params[feature_key]['std']
                x_ts_norm[..., i] = (x_ts[..., i] - mean) / std
                normalized_count += 1
            else:
                # If no norm params for this feature, leave as-is
                x_ts_norm[..., i] = x_ts[..., i]
                if self.debug_mode and not hasattr(self, '_norm_ts_missing_logged'):
                    logger.warning(f"[NORM DEBUG] Missing norm params for {feature_key}, using original values")
                    self._norm_ts_missing_logged = True

        if self.debug_mode and not hasattr(self, '_norm_ts_stats_logged'):
            logger.info(f"[NORM DEBUG] Normalized {normalized_count}/{n_ts_features} features")
            logger.info(f"[NORM DEBUG] Output stats - mean: {x_ts_norm.mean():.6f}, std: {x_ts_norm.std():.6f}, min: {x_ts_norm.min():.6f}, max: {x_ts_norm.max():.6f}")
            self._norm_ts_stats_logged = True

        # Re-zero padded positions
        if observed_mask is not None:
            # Mask should be True for valid positions, False for padded
            # We want to zero out positions where mask is False (padded)
            x_ts_norm = x_ts_norm * observed_mask.unsqueeze(-1)

        return x_ts_norm

    def _normalize_and_impute_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalize and impute static features using training statistics.

        CRITICAL: This method should only be called if the base model does NOT
        handle its own input normalization. Most TST and XLinear models already
        normalize static features internally.

        Args:
            x_static: Static features of shape (batch, n_static_features)

        Returns:
            Normalized and imputed static features (or original if no norm params)
        """
        norm_params = self._get_feature_norm_params()

        # DEBUG: Log first call
        if not hasattr(self, '_norm_static_called'):
            if self.debug_mode:
                logger.info(f"[NORM DEBUG] === FIRST _normalize_and_impute_static CALL ===")
                logger.info(f"[NORM DEBUG] Input x_static shape: {x_static.shape}")
                logger.info(f"[NORM DEBUG] Input stats - mean: {x_static.mean():.6f}, std: {x_static.std():.6f}, min: {x_static.min():.6f}, max: {x_static.max():.6f}")
            self._norm_static_called = True

        # If no norm params, base model handles its own normalization
        if len(norm_params) == 0:
            if self.debug_mode and not hasattr(self, '_norm_static_skip_logged'):
                logger.info(f"[NORM DEBUG] Skipping static normalization - base model handles it internally")
                self._norm_static_skip_logged = True
            return x_static

        n_static_features = x_static.shape[-1]
        x_static_norm = torch.zeros_like(x_static)
        normalized_count = 0

        for i in range(n_static_features):
            feature_key = f"static_{i}"
            if feature_key in norm_params:
                mean = norm_params[feature_key]['mean']
                std = norm_params[feature_key]['std']
                x_static_norm[..., i] = (x_static[..., i] - mean) / std
                normalized_count += 1
            else:
                # If no norm params, impute with 0 (normalized space)
                x_static_norm[..., i] = 0.0
                if self.debug_mode and not hasattr(self, '_norm_static_missing_logged'):
                    logger.warning(f"[NORM DEBUG] Missing norm params for {feature_key}, setting to 0")
                    self._norm_static_missing_logged = True

        if self.debug_mode and not hasattr(self, '_norm_static_stats_logged'):
            logger.info(f"[NORM DEBUG] Normalized {normalized_count}/{n_static_features} static features")
            logger.info(f"[NORM DEBUG] Output stats - mean: {x_static_norm.mean():.6f}, std: {x_static_norm.std():.6f}, min: {x_static_norm.min():.6f}, max: {x_static_norm.max():.6f}")
            self._norm_static_stats_logged = True

        return x_static_norm

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual outputs.

        Args:
            x_ts: Time series features
            x_static: Static features
            observed_mask: Optional observation mask

        Returns:
            Tuple of (regression_output, classification_logits)
        """
        # DEBUG: Log forward pass
        if self.debug_mode:
            logger.debug(f"[FORWARD DEBUG] Input shapes - x_ts: {x_ts.shape}, x_static: {x_static.shape}")
        if not self._classification_head_built and self.debug_mode:
            logger.info(f"[FORWARD DEBUG] === FIRST FORWARD PASS ===")
            logger.info(f"[FORWARD DEBUG] x_ts stats - mean: {x_ts.mean():.6f}, std: {x_ts.std():.6f}, min: {x_ts.min():.6f}, max: {x_ts.max():.6f}")
            logger.info(f"[FORWARD DEBUG] x_static stats - mean: {x_static.mean():.6f}, std: {x_static.std():.6f}, min: {x_static.min():.6f}, max: {x_static.max():.6f}")

        # Try to get embeddings from base model (for MultitaskTSTWrapper)
        embeddings = None
        reg_output = None

        # Check if base model supports return_embeddings parameter
        try:
            # Try calling with return_embeddings=True
            base_output = self.base_model(x_ts, x_static, observed_mask, return_embeddings=True)

            if not self._classification_head_built and self.debug_mode:
                logger.info(f"[FORWARD DEBUG] base_model output type: {type(base_output)}")

            if isinstance(base_output, tuple) and len(base_output) == 2:
                reg_output, embeddings = base_output
                if not self._classification_head_built and self.debug_mode:
                    logger.info(f"[FORWARD DEBUG] Got tuple - reg_output shape: {reg_output.shape if hasattr(reg_output, 'shape') else 'N/A'}")
                    logger.info(f"[FORWARD DEBUG] Got tuple - embeddings shape: {embeddings.shape if embeddings is not None and hasattr(embeddings, 'shape') else 'N/A'}")
                if embeddings is None:
                    # Base model returned None for embeddings, fall back
                    reg_output = base_output[0]
                    if not self._classification_head_built and self.debug_mode:
                        logger.info(f"[FORWARD DEBUG] embeddings is None, using reg_output")
            else:
                # Single output, no embeddings returned
                reg_output = base_output
                if not self._classification_head_built and self.debug_mode:
                    logger.info(f"[FORWARD DEBUG] Single output - shape: {reg_output.shape if hasattr(reg_output, 'shape') else 'N/A'}")
        except (TypeError, Exception) as e:
            # Base model doesn't support return_embeddings parameter
            if self.debug_mode:
                logger.info(f"[FORWARD DEBUG] Exception calling base_model: {e}")
            reg_output = self.base_model(x_ts, x_static, observed_mask)

        if not self._classification_head_built and self.debug_mode:
            logger.info(f"[FORWARD DEBUG] Final reg_output shape: {reg_output.shape if hasattr(reg_output, 'shape') else 'N/A'}")
            if hasattr(reg_output, 'shape'):
                logger.info(f"[FORWARD DEBUG] Final reg_output stats - mean: {reg_output.mean():.6f}, std: {reg_output.std():.6f}, min: {reg_output.min():.6f}, max: {reg_output.max():.6f}")

        # Determine classification input
        if embeddings is not None:
            # Use extracted encoder embeddings for classification
            class_input = embeddings
            if not self._classification_head_built and self.debug_mode:
                logger.info(f"[FORWARD DEBUG] Using embeddings for classification, shape: {embeddings.shape}")
        elif reg_output is not None:
            # Fallback: Check if reg_output could be embeddings (high-dimensional)
            # MQL outputs are (batch, 3) - too low dimensional to be embeddings
            if (hasattr(reg_output, 'shape') and len(reg_output.shape) > 1 and
                reg_output.shape[-1] > 10):  # Threshold: embeddings are >10 dims
                class_input = reg_output
                if not self._classification_head_built and self.debug_mode:
                    logger.info(f"[FORWARD DEBUG] Using reg_output as embeddings, shape: {reg_output.shape}")
            else:
                # Use static features as fallback
                class_input = x_static
                if not self._classification_head_built and self.debug_mode:
                    logger.info(f"[FORWARD DEBUG] Using x_static for classification, shape: {x_static.shape}")
        else:
            # Last resort: use static features
            class_input = x_static
            if not self._classification_head_built and self.debug_mode:
                logger.info(f"[FORWARD DEBUG] Last resort: using x_static for classification")

        # Ensure classification input has correct dimensionality
        if len(class_input.shape) == 2:
            # (batch, features) - good
            pass
        elif len(class_input.shape) == 1:
            # (batch,) - need to expand
            class_input = class_input.unsqueeze(-1)

        # Dynamically detect actual input dimension on first forward pass
        if not self._classification_head_built:
            actual_input_dim = class_input.shape[-1]
            if self.debug_mode:
                logger.info(f"[FORWARD DEBUG] Classification head - expected dim: {self.classification_input_dim}, actual dim: {actual_input_dim}")
            if actual_input_dim != self.classification_input_dim:
                logger.warning(
                    f"[MultitaskWrapper] Classification head input dimension mismatch: "
                    f"expected {self.classification_input_dim}, got {actual_input_dim}. "
                    f"Rebuilding classification head with correct dimension."
                )
                # Rebuild classification head with correct dimensions
                self.classification_head = MultitaskHead(
                    input_dim=actual_input_dim,
                    num_classes=self.multitask_config.num_classes,
                    ordinal_loss_type=self.multitask_config.ordinal_loss_type,
                    dropout=self.multitask_config.classification_dropout
                )
                # Move to the same device as the input
                self.classification_head = self.classification_head.to(class_input.device)
            self._classification_head_built = True

        # Pass through classification head
        class_logits = self.classification_head(class_input)

        if self.debug_mode and not hasattr(self, '_forward_class_logits_logged'):
            logger.info(f"[FORWARD DEBUG] class_logits shape: {class_logits.shape}, mean: {class_logits.mean():.6f}, std: {class_logits.std():.6f}")
            self._forward_class_logits_logged = True

        return reg_output, class_logits

    def training_step(self, batch, batch_idx):
        """Training step with dual loss computation."""
        # Unpack batch - handle both regular and multitask batches
        if len(batch) == 9:
            # Multitask batch: (x_ts, x_static, y, years, adm_ids, lats, lons, vm, class_labels)
            x_ts, x_static, y, years, adm_ids, lats, lons, masks, class_labels = batch
        else:
            # Regular batch: (x_ts, x_static, y, years, adm_ids, lats, lons, vm)
            x_ts, x_static, y, years, adm_ids, lats, lons, masks = batch
            class_labels = None

        y_norm = y  # Already normalized in datamodule

        # DEBUG: Print input/target stats for first batch
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TRAINING DEBUG] === BATCH 0 START ===")
            logger.info(f"[TRAINING DEBUG] x_ts shape: {x_ts.shape}, mean: {x_ts.mean():.6f}, std: {x_ts.std():.6f}, min: {x_ts.min():.6f}, max: {x_ts.max():.6f}")
            logger.info(f"[TRAINING DEBUG] x_static shape: {x_static.shape}, mean: {x_static.mean():.6f}, std: {x_static.std():.6f}, min: {x_static.min():.6f}, max: {x_static.max():.6f}")
            logger.info(f"[TRAINING DEBUG] y_norm shape: {y_norm.shape}, mean: {y_norm.mean():.6f}, std: {y_norm.std():.6f}, min: {y_norm.min():.6f}, max: {y_norm.max():.6f}")
            if masks is not None:
                logger.info(f"[TRAINING DEBUG] masks shape: {masks.shape}, valid ratio: {masks.float().mean():.4f}")

        # Replace NaN with 0 (same as base models do)
        # This handles missing remote sensing data (FPAR, NDVI, etc.)
        x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)
        x_static = torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize inputs (CRITICAL: only if base model doesn't handle it)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=masks)
        x_static_n = self._normalize_and_impute_static(x_static)

        # DEBUG: Print normalized input stats for first batch
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TRAINING DEBUG] After normalization:")
            logger.info(f"[TRAINING DEBUG] x_ts_n shape: {x_ts_n.shape}, mean: {x_ts_n.mean():.6f}, std: {x_ts_n.std():.6f}, min: {x_ts_n.min():.6f}, max: {x_ts_n.max():.6f}")
            logger.info(f"[TRAINING DEBUG] x_static_n shape: {x_static_n.shape}, mean: {x_static_n.mean():.6f}, std: {x_static_n.std():.6f}, min: {x_static_n.min():.6f}, max: {x_static_n.max():.6f}")

        # Get dual outputs
        reg_output, class_logits = self(x_ts_n, x_static_n, masks)

        # DEBUG: Print model output stats for first batch
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TRAINING DEBUG] After forward pass:")
            logger.info(f"[TRAINING DEBUG] reg_output shape: {reg_output.shape}, mean: {reg_output.mean():.6f}, std: {reg_output.std():.6f}, min: {reg_output.min():.6f}, max: {reg_output.max():.6f}")
            logger.info(f"[TRAINING DEBUG] class_logits shape: {class_logits.shape}, mean: {class_logits.mean():.6f}, std: {class_logits.std():.6f}, min: {class_logits.min():.6f}, max: {class_logits.max():.6f}")

        # Debug: Check for NaN in model outputs
        if torch.isnan(reg_output).any():
            logger.error(f"[NaN DEBUG] reg_output contains NaN at batch {batch_idx}")
            logger.error(f"[NaN DEBUG] reg_output shape: {reg_output.shape}, min: {reg_output.min()}, max: {reg_output.max()}")
            raise ValueError("Model output (regression) contains NaN")
        if torch.isnan(class_logits).any():
            logger.error(f"[NaN DEBUG] class_logits contains NaN at batch {batch_idx}")
            raise ValueError("Model output (classification) contains NaN")

        # Compute regression loss
        reg_loss = self._compute_regression_loss(reg_output, y_norm)

        # Debug: Check for NaN in regression loss
        if torch.isnan(reg_loss):
            logger.error(f"[NaN DEBUG] reg_loss is NaN at batch {batch_idx}")
            logger.error(f"[NaN DEBUG] reg_output stats: mean={reg_output.mean()}, std={reg_output.std()}")
            logger.error(f"[NaN DEBUG] y_norm stats: mean={y_norm.mean()}, std={y_norm.std()}")
            raise ValueError("Regression loss is NaN")

        # Compute classification loss
        class_targets = self._get_classification_targets(y_norm, years, adm_ids, class_labels)
        class_loss = self._compute_classification_loss(class_logits, class_targets)

        # Debug: Check for NaN in classification loss
        if torch.isnan(class_loss):
            logger.error(f"[NaN DEBUG] class_loss is NaN at batch {batch_idx}")
            logger.error(f"[NaN DEBUG] class_logits stats: mean={class_logits.mean()}, std={class_logits.std()}")
            logger.error(f"[NaN DEBUG] class_targets: {class_targets[:5]}")
            raise ValueError("Classification loss is NaN")

        # Combined loss
        total_loss = reg_loss + self.multitask_config.classification_alpha * class_loss

        # DEBUG: Log loss components for first batch
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TRAINING DEBUG] Loss components:")
            logger.info(f"[TRAINING DEBUG]   reg_loss: {reg_loss:.6f}")
            logger.info(f"[TRAINING DEBUG]   class_loss: {class_loss:.6f}")
            logger.info(f"[TRAINING DEBUG]   alpha: {self.multitask_config.classification_alpha}")
            logger.info(f"[TRAINING DEBUG]   total_loss: {total_loss:.6f}")

        # Debug: Check for NaN in total loss
        if torch.isnan(total_loss):
            logger.error(f"[NaN DEBUG] total_loss is NaN at batch {batch_idx}")
            logger.error(f"[NaN DEBUG] reg_loss={reg_loss}, class_loss={class_loss}, alpha={self.multitask_config.classification_alpha}")
            raise ValueError("Total loss is NaN")

        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train/reg_loss', reg_loss)
        self.log('train/class_loss', class_loss)

        # Update metrics
        reg_preds = self._get_regression_predictions(reg_output)
        class_preds = self._get_classification_predictions(class_logits)

        self.train_metrics.update(
            reg_preds, y_norm, class_preds, class_targets,
            reg_full_output=reg_output if self.use_mql else None
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch - handle both regular and multitask batches
        if len(batch) == 9:
            x_ts, x_static, y, years, adm_ids, lats, lons, masks, class_labels = batch
        else:
            x_ts, x_static, y, years, adm_ids, lats, lons, masks = batch
            class_labels = None

        y_norm = y

        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[VAL DEBUG] === VALIDATION BATCH 0 ===")
            logger.info(f"[VAL DEBUG] y_norm shape: {y_norm.shape}, mean: {y_norm.mean():.6f}, std: {y_norm.std():.6f}, min: {y_norm.min():.6f}, max: {y_norm.max():.6f}")

        # Replace NaN with 0 (same as training and base models)
        x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)
        x_static = torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize inputs (CRITICAL: only if base model doesn't handle it)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=masks)
        x_static_n = self._normalize_and_impute_static(x_static)

        reg_output, class_logits = self(x_ts_n, x_static_n, masks)

        reg_loss = self._compute_regression_loss(reg_output, y_norm)
        class_targets = self._get_classification_targets(y_norm, years, adm_ids, class_labels)
        class_loss = self._compute_classification_loss(class_logits, class_targets)

        total_loss = reg_loss + self.multitask_config.classification_alpha * class_loss

        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val/reg_loss', reg_loss)
        self.log('val/class_loss', class_loss)

        reg_preds = self._get_regression_predictions(reg_output)
        class_preds = self._get_classification_predictions(class_logits)

        # Denormalize predictions and targets to original scale for metrics computation
        # (matching baseline behavior)
        dm = self.trainer.datamodule
        device = reg_preds.device

        # Handle both MultitaskDataModule (denorm_params dict) and regular datamodule
        if hasattr(dm, 'denorm_params'):
            y_mean = dm.denorm_params['y_mean']
            y_std = dm.denorm_params['y_std']
        elif hasattr(dm, 'y_mean') and hasattr(dm, 'y_std'):
            y_mean = dm.y_mean
            y_std = dm.y_std
        else:
            raise AttributeError("Cannot find y_mean and y_std in datamodule")

        # DEBUG: Print denormalization params
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[VAL DEBUG] Denorm params - y_mean: {y_mean}, y_std: {y_std}")
            logger.info(f"[VAL DEBUG] reg_preds (normalized) mean: {reg_preds.mean():.6f}, std: {reg_preds.std():.6f}")

        # Convert to tensor if needed and move to device
        if not isinstance(y_std, torch.Tensor):
            y_std = torch.tensor(y_std, device=device)
        if not isinstance(y_mean, torch.Tensor):
            y_mean = torch.tensor(y_mean, device=device)

        # Denormalize predictions and targets
        reg_preds_orig = reg_preds.detach() * y_std + y_mean
        y_orig = y_norm.detach() * y_std + y_mean

        # DEBUG: Print denormalized stats
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[VAL DEBUG] After denorm:")
            logger.info(f"[VAL DEBUG] reg_preds_orig mean: {reg_preds_orig.mean():.6f}, std: {reg_preds_orig.std():.6f}, min: {reg_preds_orig.min():.6f}, max: {reg_preds_orig.max():.6f}")
            logger.info(f"[VAL DEBUG] y_orig mean: {y_orig.mean():.6f}, std: {y_orig.std():.6f}, min: {y_orig.min():.6f}, max: {y_orig.max():.6f}")
            # Check if values are reasonable
            logger.info(f"[VAL DEBUG] Are values reasonable? (mean between 5 and 15): {5 <= reg_preds_orig.mean() <= 15}")
            if reg_preds_orig.mean() < 0 or reg_preds_orig.mean() > 100:
                logger.warning(f"[VAL DEBUG] WARNING: Predicted values seem out of range for crop yields!")

        # Clip predictions to physically meaningful range (yields ≥ 0)
        reg_preds_clipped = torch.clamp(reg_preds_orig, min=0.0)

        # DEBUG: Print clipped stats
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[VAL DEBUG] After clipping - reg_preds_clipped mean: {reg_preds_clipped.mean():.6f}, std: {reg_preds_clipped.std():.6f}")
            logger.info(f"[VAL DEBUG] Clip rate: {(reg_preds_orig < 0).float().mean():.4f}")

        # Update metrics with original scale values (like baseline)
        self.val_metrics.update(
            reg_preds_clipped, y_orig, class_preds, class_targets,
            reg_full_output=None  # Uncertainty metrics only for test stage
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        """Test step with per-year metrics and spatiotemporal computation."""
        # Unpack batch - handle both regular and multitask batches
        if len(batch) == 9:
            x_ts, x_static, y, years, adm_ids, lats, lons, masks, class_labels = batch
        else:
            x_ts, x_static, y, years, adm_ids, lats, lons, masks = batch
            class_labels = None

        y_norm = y

        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] === TEST BATCH 0 ===")
            logger.info(f"[TEST DEBUG] y_norm shape: {y_norm.shape}, mean: {y_norm.mean():.6f}, std: {y_norm.std():.6f}, min: {y_norm.min():.6f}, max: {y_norm.max():.6f}")

        # Replace NaN with 0 (same as training and base models)
        x_ts = torch.nan_to_num(x_ts, nan=0.0, posinf=0.0, neginf=0.0)
        x_static = torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize inputs (CRITICAL: only if base model doesn't handle it)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=masks)
        x_static_n = self._normalize_and_impute_static(x_static)

        reg_output, class_logits = self(x_ts_n, x_static_n, masks)

        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] reg_output shape: {reg_output.shape}, mean: {reg_output.mean():.6f}, std: {reg_output.std():.6f}, min: {reg_output.min():.6f}, max: {reg_output.max():.6f}")

        reg_loss = self._compute_regression_loss(reg_output, y_norm)
        class_targets = self._get_classification_targets(y_norm, years, adm_ids, class_labels)
        class_loss = self._compute_classification_loss(class_logits, class_targets)

        total_loss = reg_loss + self.multitask_config.classification_alpha * class_loss

        reg_preds = self._get_regression_predictions(reg_output)
        class_preds = self._get_classification_predictions(class_logits)

        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] reg_preds (normalized) mean: {reg_preds.mean():.6f}, std: {reg_preds.std():.6f}, min: {reg_preds.min():.6f}, max: {reg_preds.max():.6f}")

        # Denormalize predictions and targets to original scale for metrics computation
        # (matching baseline behavior)
        dm = self.trainer.datamodule
        device = reg_preds.device

        # Handle both MultitaskDataModule (denorm_params dict) and regular datamodule
        if hasattr(dm, 'denorm_params'):
            y_mean = dm.denorm_params['y_mean']
            y_std = dm.denorm_params['y_std']
        elif hasattr(dm, 'y_mean') and hasattr(dm, 'y_std'):
            y_mean = dm.y_mean
            y_std = dm.y_std
        else:
            raise AttributeError("Cannot find y_mean and y_std in datamodule")

        # Convert to tensor if needed and move to device
        if not isinstance(y_std, torch.Tensor):
            y_std = torch.tensor(y_std, device=device)
        if not isinstance(y_mean, torch.Tensor):
            y_mean = torch.tensor(y_mean, device=device)

        # DEBUG: Print denormalization params
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] Denorm params - y_mean: {y_mean}, y_std: {y_std}")

        # Denormalize predictions and targets
        reg_preds_orig = reg_preds.detach() * y_std + y_mean
        y_orig = y_norm.detach() * y_std + y_mean

        # DEBUG: Print denormalized stats
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] After denorm:")
            logger.info(f"[TEST DEBUG] reg_preds_orig mean: {reg_preds_orig.mean():.6f}, std: {reg_preds_orig.std():.6f}, min: {reg_preds_orig.min():.6f}, max: {reg_preds_orig.max():.6f}")
            logger.info(f"[TEST DEBUG] y_orig mean: {y_orig.mean():.6f}, std: {y_orig.std():.6f}, min: {y_orig.min():.6f}, max: {y_orig.max():.6f}")
            # Check if values are reasonable
            logger.info(f"[TEST DEBUG] Are predictions reasonable? (mean between 5 and 15): {5 <= reg_preds_orig.mean() <= 15}")
            if reg_preds_orig.mean() < 0 or reg_preds_orig.mean() > 100:
                logger.warning(f"[TEST DEBUG] WARNING: Predicted values seem out of range for crop yields!")
            # Check for catastrophic errors
            if reg_preds_orig.mean() > 1000 or reg_preds_orig.mean() < -1000:
                logger.error(f"[TEST DEBUG] CATASTROPHIC ERROR: Predictions are completely wrong (mean={reg_preds_orig.mean():.2f})")
                logger.error(f"[TEST DEBUG] This indicates a fundamental problem with the model or data pipeline!")

        # Clip predictions to physically meaningful range (yields ≥ 0)
        reg_preds_clipped = torch.clamp(reg_preds_orig, min=0.0)

        # DEBUG: Print clipped stats
        if batch_idx == 0 and self.debug_mode:
            logger.info(f"[TEST DEBUG] After clipping - reg_preds_clipped mean: {reg_preds_clipped.mean():.6f}, std: {reg_preds_clipped.std():.6f}")
            logger.info(f"[TEST DEBUG] Clip rate: {(reg_preds_orig < 0).float().mean():.4f}")

        # Update metrics with original scale values (like baseline)
        # For uncertainty metrics, use the full quantile output
        reg_full_output_orig = None
        if self.use_mql and reg_output is not None:
            # Denormalize full quantile output
            reg_full_output_orig = reg_output.detach() * y_std.unsqueeze(0).unsqueeze(-1) + y_mean.unsqueeze(0).unsqueeze(-1)
            # Clip all quantiles
            reg_full_output_orig = torch.clamp(reg_full_output_orig, min=0.0)

        self.test_metrics.update(
            reg_preds_clipped, y_orig, class_preds, class_targets,
            reg_full_output=reg_full_output_orig
        )

        # Store per-year results (with denormalized values)
        self._store_per_year_results(
            years, adm_ids, reg_preds_clipped, y_orig, class_preds, class_targets
        )

        # Store for spatiotemporal metrics (with denormalized values)
        if self.use_spatiotemporal_metrics:
            self._store_for_spatiotemporal(
                years, adm_ids, reg_preds_clipped, y_orig
            )

        self.log('test_loss', total_loss)
        self.log('test/reg_loss', reg_loss)
        self.log('test/class_loss', class_loss)

        return total_loss

    def on_train_epoch_end(self):
        """Log training metrics at end of epoch."""
        metrics = self.train_metrics.compute()
        for key, val in metrics.items():
            self.log(f'train/{key}', val)
            # Also log regression metrics with baseline-compatible keys for wandb comparison
            if key.startswith('regression_'):
                baseline_key = key.replace('regression_', '')
                self.log(f'train/{baseline_key}', val)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at end of epoch."""
        metrics = self.val_metrics.compute()
        for key, val in metrics.items():
            self.log(f'val/{key}', val)
            # Also log regression metrics with baseline-compatible keys for wandb comparison
            if key.startswith('regression_'):
                baseline_key = key.replace('regression_', '')
                self.log(f'val/{baseline_key}', val)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        """Log test metrics at end of epoch."""
        metrics = self.test_metrics.compute()

        # Flatten and store per-year metrics
        self._finalize_per_year_metrics()

        # Compute spatiotemporal metrics if enabled
        if self.use_spatiotemporal_metrics:
            self._compute_spatiotemporal_metrics()

        for key, val in metrics.items():
            self.log(f'test/{key}', val)
            # Also log regression metrics with baseline-compatible keys for wandb comparison
            if key.startswith('regression_'):
                baseline_key = key.replace('regression_', '')
                self.log(f'test/{baseline_key}', val)
        self.test_metrics.reset()

    def _compute_regression_loss(self, reg_output, targets):
        """Compute regression loss based on loss type."""
        if hasattr(self.base_model, 'loss_type') and self.base_model.loss_type == "mql":
            if hasattr(self.base_model, 'mq_loss_fn') and self.base_model.mq_loss_fn is not None:
                return self.base_model.mq_loss_fn(reg_output, targets)
            else:
                # Fallback pinball loss
                return self._fallback_mql_loss(reg_output, targets)
        else:
            return nn.functional.mse_loss(reg_output.squeeze(-1), targets)

    def _fallback_mql_loss(self, predictions, targets):
        """Fallback MQL loss implementation."""
        quantiles = self.quantiles or [0.1, 0.5, 0.9]
        losses = []
        for i, q in enumerate(quantiles):
            errors = targets - predictions[:, i] if predictions.dim() > 1 else targets - predictions
            q_loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(q_loss.mean())
        return sum(losses) / len(losses)

    def _compute_classification_loss(self, class_logits, class_targets):
        """Compute classification loss."""
        return self.classification_loss_fn(class_logits, class_targets)

    def _get_regression_predictions(self, reg_output):
        """Extract regression predictions."""
        if reg_output.dim() > 1:
            # Multi-quantile output - use median (index 1)
            return reg_output[:, 1] if reg_output.shape[1] > 1 else reg_output[:, 0]
        return reg_output

    def _get_classification_predictions(self, class_logits):
        """Get classification predictions from logits."""
        if hasattr(self.classification_loss_fn, 'predict'):
            return self.classification_loss_fn.predict(class_logits)
        else:
            return torch.argmax(class_logits, dim=-1)

    def _get_classification_targets(self, y_norm, years, adm_ids, class_labels=None):
        """
        Get classification targets from batch.

        Args:
            y_norm: Normalized regression targets
            years: Year values
            adm_ids: Admin IDs
            class_labels: Pre-computed classification labels (should always be provided)

        Returns:
            Classification target tensor

        Note:
            If class_labels is None, this indicates a bug in the datamodule.
            Falls back to middle class to prevent random label corruption.
        """
        # If class_labels are provided in the batch, use them
        if class_labels is not None:
            return class_labels

        # Otherwise, this is an error state - classification labels should always
        # be pre-computed and added to the batch by the datamodule
        batch_size = y_norm.shape[0]
        logger.error(
            "Classification targets not provided in batch. "
            "This indicates a bug in the datamodule - MultitaskDatasetWrapper should "
            "always append classification labels. Using default class (middle class)."
        )
        # Use middle class as safe default (class 2 for 5 classes = "Normal")
        # This prevents random label corruption during training
        default_class = self.multitask_config.num_classes // 2
        return torch.full((batch_size,), default_class, device=y_norm.device, dtype=torch.long)

    def _store_per_year_results(self, years, adm_ids, reg_preds, reg_targets, class_preds, class_targets):
        """Store predictions and targets for per-year metrics."""
        years_np = years.cpu().numpy()
        reg_preds_np = reg_preds.detach().cpu().numpy()
        reg_targets_np = reg_targets.detach().cpu().numpy()
        class_preds_np = class_preds.detach().cpu().numpy()
        class_targets_np = class_targets.detach().cpu().numpy()

        for i, year in enumerate(years_np):
            year_int = int(year)
            if year_int not in self._test_results_per_year:
                self._test_results_per_year[year_int] = {
                    'reg_preds': [],
                    'reg_targets': [],
                    'class_preds': [],
                    'class_targets': [],
                }

            self._test_results_per_year[year_int]['reg_preds'].append(reg_preds_np[i])
            self._test_results_per_year[year_int]['reg_targets'].append(reg_targets_np[i])
            self._test_results_per_year[year_int]['class_preds'].append(class_preds_np[i])
            self._test_results_per_year[year_int]['class_targets'].append(class_targets_np[i])

    def _finalize_per_year_metrics(self):
        """Compute and store final per-year metrics."""
        # Import here to avoid circular dependency
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
        from sklearn.metrics import f1_score, cohen_kappa_score

        for year, data in self._test_results_per_year.items():
            reg_preds = np.array(data['reg_preds'])
            reg_targets = np.array(data['reg_targets'])
            class_preds = np.array(data['class_preds'])
            class_targets = np.array(data['class_targets'])

            # Filter out NaN values from predictions and targets
            valid_mask = ~(np.isnan(reg_preds) | np.isnan(reg_targets))
            n_valid = valid_mask.sum()
            n_total = len(reg_preds)

            if n_valid == 0:
                logger.warning(
                    f"[Per-Year Metrics] Year {year}: All predictions are NaN. "
                    f"Setting all metrics to NaN."
                )
                self._test_results_per_year[year]['metrics'] = {
                    'mse': float('nan'),
                    'rmse': float('nan'),
                    'mae': float('nan'),
                    'r2': float('nan'),
                    'mape': float('nan'),
                    'classification_accuracy': float('nan'),
                    'classification_f1_macro': float('nan'),
                    'classification_qwk': float('nan'),
                }
                continue

            if n_valid < n_total:
                logger.warning(
                    f"[Per-Year Metrics] Year {year}: Filtering out {n_total - n_valid} NaN "
                    f"predictions out of {n_total} total."
                )

            reg_preds_clean = reg_preds[valid_mask]
            reg_targets_clean = reg_targets[valid_mask]
            class_preds_clean = class_preds[valid_mask]
            class_targets_clean = class_targets[valid_mask]

            # Values are already denormalized (from test_step)
            # No need to denormalize again
            reg_preds_denorm = reg_preds_clean
            reg_targets_denorm = reg_targets_clean

            # Regression metrics
            mse = mean_squared_error(reg_targets_denorm, reg_preds_denorm)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(reg_targets_denorm - reg_preds_denorm))

            try:
                r2 = r2_score(reg_targets_denorm, reg_preds_denorm)
            except:
                r2 = np.nan

            try:
                mape = mean_absolute_percentage_error(reg_targets_denorm, reg_preds_denorm)
            except:
                mape = np.nan

            # Classification metrics
            acc = np.mean(class_preds_clean == class_targets_clean)
            f1_macro = f1_score(class_targets_clean, class_preds_clean, average='macro', zero_division=0)

            try:
                qwk = cohen_kappa_score(class_targets_clean, class_preds_clean, weights='quadratic')
            except:
                qwk = 0.0

            # Store
            self._test_results_per_year[year]['metrics'] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'classification_accuracy': float(acc),
                'classification_f1_macro': float(f1_macro),
                'classification_qwk': float(qwk),
            }

    def _store_for_spatiotemporal(self, years, adm_ids, reg_preds, reg_targets):
        """Store predictions and targets for spatiotemporal metrics computation."""
        years_np = years.cpu().numpy()
        adm_ids_np = adm_ids.cpu().numpy()
        reg_preds_np = reg_preds.detach().cpu().numpy()
        reg_targets_np = reg_targets.detach().cpu().numpy()

        # Values are already denormalized (from test_step)
        # No need to denormalize again

        self._test_y_true.extend(reg_targets_np)
        self._test_y_pred.extend(reg_preds_np)
        self._test_years.extend(years_np)
        self._test_regions.extend(adm_ids_np)

    def _compute_spatiotemporal_metrics(self):
        """Compute spatiotemporal metrics using the stored predictions."""
        try:
            from metrics import compute_all_spatiotemporal_metrics, flatten_spatiotemporal_metrics_simple

            y_true = np.array(self._test_y_true)
            y_pred = np.array(self._test_y_pred)
            years = np.array(self._test_years)
            regions = np.array(self._test_regions)

            # Compute all spatiotemporal metrics
            metrics = compute_all_spatiotemporal_metrics(y_true, y_pred, years, regions)

            # Store simplified version
            self._spatiotemporal_metrics = flatten_spatiotemporal_metrics_simple(metrics)

            logger.info(f"[MultitaskWrapper] Spatiotemporal metrics computed: {self._spatiotemporal_metrics}")

        except Exception as e:
            logger.warning(f"Could not compute spatiotemporal metrics: {e}")
            self._spatiotemporal_metrics = {}

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )


# ============================================================================
# Test Functions
# ============================================================================

def test_multitask_wrapper():
    """Test multitask wrapper."""
    print("[MultitaskWrapper] Running tests...")

    # Create a dummy base model
    class DummyBaseModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(64, 32)
            self.regressor = nn.Linear(32, 1)
            self.lr = 1e-4
            self.weight_decay = 1e-5
            self.loss_type = "mse"

        def forward(self, x_ts, x_static, mask=None):
            x = self.encoder(x_ts.mean(dim=1))
            return self.regressor(x)

    base_model = DummyBaseModel()
    multitask_config = MultitaskModelConfig(
        classification_alpha=0.3,
        num_classes=5,
        ordinal_loss_type="corn"
    )

    wrapped_model = MultitaskWrapper(base_model, multitask_config, classification_input_dim=32)

    # Test forward pass
    batch_size = 16
    x_ts = torch.randn(batch_size, 36, 10)
    x_static = torch.randn(batch_size, 20)

    reg_output, class_logits = wrapped_model(x_ts, x_static)

    print(f"[MultitaskWrapper] Regression output shape: {reg_output.shape}")
    print(f"[MultitaskWrapper] Classification logits shape: {class_logits.shape}")

    assert class_logits.shape[-1] == 4, "CORN should output K-1 logits"

    print("[MultitaskWrapper] All tests passed!")


if __name__ == "__main__":
    import torchmetrics
    test_multitask_wrapper()
