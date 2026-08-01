# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Uncertainty quantification metrics for quantile regression.
            Includes pinball loss, CRPS, PICP, PINAW, and Winkler score.
Python version: 3.12.0
--------------------

This module provides metrics for evaluating probabilistic predictions
from quantile regression models, commonly used for uncertainty quantification
in crop yield forecasting.

Key Metrics:
- Pinball Loss: Quantile-specific loss function
- CRPS: Continuous Ranked Probability Score (integrates over all quantiles)
- PICP: Prediction Interval Coverage Probability
- PINAW: Prediction Interval Normalized Average Width
- Winkler Score: Combines coverage and sharpness
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Union


def pinball_loss(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor],
                 quantile: float) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute pinball loss (quantile loss) for a single quantile.

    The pinball loss is used to train quantile regression models.
    It asymmetrically penalizes over-predictions and under-predictions
    based on the quantile level.

    Args:
        y_true: Ground truth values, shape (n_samples,) or (n_samples, 1)
        y_pred: Predicted values at the specified quantile, same shape as y_true
        quantile: Quantile level (0 < quantile < 1)

    Returns:
        Pinball loss values, same shape as y_true

    Formula:
        L(y, ŷ) = max(q * (y - ŷ), (q - 1) * (y - ŷ))

        - If y > ŷ (under-prediction): loss = q * (y - ŷ)
        - If y <= ŷ (over-prediction): loss = (q - 1) * (y - ŷ)
    """
    error = y_true - y_pred
    if isinstance(error, torch.Tensor):
        return torch.max(quantile * error, (quantile - 1) * error)
    else:
        return np.maximum(quantile * error, (quantile - 1) * error)


def compute_pinball_loss(y_true: Union[np.ndarray, torch.Tensor],
                        y_pred_quantiles: Union[np.ndarray, torch.Tensor],
                        quantiles: List[float],
                        aggregate: str = 'mean') -> Dict[str, float]:
    """
    Compute pinball loss for multiple quantiles.

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred_quantiles: Predicted values for each quantile, shape (n_samples, n_quantiles)
        quantiles: List of quantile levels, e.g., [0.1, 0.5, 0.9]
        aggregate: How to aggregate ('mean', 'sum', or 'none')

    Returns:
        Dictionary mapping quantile levels to their pinball loss values
    """
    if isinstance(y_pred_quantiles, torch.Tensor):
        y_pred_quantiles = y_pred_quantiles.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    results = {}
    losses = []

    for i, q in enumerate(quantiles):
        q_loss = pinball_loss(y_true, y_pred_quantiles[:, i], q)
        if aggregate == 'mean':
            results[f'pinball_q{int(q*100)}'] = float(np.mean(q_loss))
            losses.append(np.mean(q_loss))
        elif aggregate == 'sum':
            results[f'pinball_q{int(q*100)}'] = float(np.sum(q_loss))
            losses.append(np.sum(q_loss))
        else:
            results[f'pinball_q{int(q*100)}'] = q_loss

    if aggregate in ['mean', 'sum']:
        results['pinball_avg'] = float(np.mean(losses))

    return results


def compute_crps(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred_quantiles: Union[np.ndarray, torch.Tensor],
                 quantiles: List[float]) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS is the gold standard for evaluating probabilistic forecasts.
    It integrates the pinball loss across all quantiles, measuring
    the integrated squared error of the cumulative distribution function.

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred_quantiles: Predicted values for each quantile, shape (n_samples, n_quantiles)
        quantiles: List of quantile levels (must be sorted)

    Returns:
        Mean CRPS value across all samples

    Interpretation:
        - Lower CRPS is better
        - CRPS = 0 for perfect prediction (all mass at true value)
        - CRPS has the same units as the target variable
    """
    if isinstance(y_pred_quantiles, torch.Tensor):
        y_pred_quantiles = y_pred_quantiles.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    n_samples = len(y_true)
    crps_values = np.zeros(n_samples)

    # Ensure quantiles are sorted
    sorted_indices = np.argsort(quantiles)
    sorted_quantiles = np.array(quantiles)[sorted_indices]
    sorted_preds = y_pred_quantiles[:, sorted_indices]

    for i in range(n_samples):
        y_i = y_true[i]
        preds_i = sorted_preds[i]

        # Compute CRPS using trapezoidal integration
        integral = 0.0
        for j in range(len(sorted_quantiles) - 1):
            # CDF at quantile j: indicator if y_true >= predicted value
            cdf_j = 1.0 if y_i >= preds_i[j] else 0.0
            cdf_j1 = 1.0 if y_i >= preds_i[j + 1] else 0.0

            # CDF difference
            dq = sorted_quantiles[j + 1] - sorted_quantiles[j]

            # Trapezoid area: average height * width
            avg_height = ((cdf_j - sorted_quantiles[j])**2 +
                         (cdf_j1 - sorted_quantiles[j + 1])**2) / 2
            integral += avg_height * dq

        crps_values[i] = integral

    return float(np.mean(crps_values))


def compute_picp(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred_lower: Union[np.ndarray, torch.Tensor],
                 y_pred_upper: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute Prediction Interval Coverage Probability (PICP).

    PICP measures the reliability of prediction intervals - the fraction
    of true values that fall within the predicted intervals.

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred_lower: Lower bound of prediction interval, shape (n_samples,)
        y_pred_upper: Upper bound of prediction interval, shape (n_samples,)

    Returns:
        PICP value between 0 and 1

    Interpretation:
        - PICP = coverage level means well-calibrated intervals
        - PICP < coverage level: intervals are too narrow (over-confident)
        - PICP > coverage level: intervals are too wide (under-confident)
    """
    if isinstance(y_pred_lower, torch.Tensor):
        y_pred_lower = y_pred_lower.detach().cpu().numpy()
    if isinstance(y_pred_upper, torch.Tensor):
        y_pred_upper = y_pred_upper.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
    return float(np.mean(in_interval))


def compute_pinaw(y_pred_lower: Union[np.ndarray, torch.Tensor],
                  y_pred_upper: Union[np.ndarray, torch.Tensor],
                  y_true: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  y_range: Optional[tuple] = None) -> float:
    """
    Compute Prediction Interval Normalized Average Width (PINAW).

    PINAW measures the sharpness (width) of prediction intervals.
    Lower values indicate more precise forecasts.

    Args:
        y_pred_lower: Lower bound of prediction interval, shape (n_samples,)
        y_pred_upper: Upper bound of prediction interval, shape (n_samples,)
        y_true: Optional ground truth for computing range, shape (n_samples,)
        y_range: Optional tuple (min, max) for normalization

    Returns:
        PINAW value between 0 and 1

    Interpretation:
        - Lower PINAW is better (sharper intervals)
        - Must be balanced with PICP (narrow but unreliable intervals are useless)
    """
    if isinstance(y_pred_lower, torch.Tensor):
        y_pred_lower = y_pred_lower.detach().cpu().numpy()
    if isinstance(y_pred_upper, torch.Tensor):
        y_pred_upper = y_pred_upper.detach().cpu().numpy()

    interval_width = y_pred_upper - y_pred_lower
    mean_width = np.mean(interval_width)

    # Compute normalization range
    if y_range is not None:
        range_width = y_range[1] - y_range[0]
    elif y_true is not None:
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        range_width = y_true.max() - y_true.min()
    else:
        # Use prediction range if nothing else available
        range_width = y_pred_upper.max() - y_pred_lower.min()

    if range_width == 0:
        return 0.0

    return float(mean_width / range_width)


def compute_winkler_score(y_true: Union[np.ndarray, torch.Tensor],
                          y_pred_lower: Union[np.ndarray, torch.Tensor],
                          y_pred_upper: Union[np.ndarray, torch.Tensor],
                          alpha: float = 0.1) -> float:
    """
    Compute Winkler score for prediction intervals.

    The Winkler score combines coverage probability and interval width
    into a single metric. It penalizes both:
    1. Wide intervals (lack of sharpness)
    2. Missed coverage (lack of calibration)

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred_lower: Lower bound of prediction interval, shape (n_samples,)
        y_pred_upper: Upper bound of prediction interval, shape (n_samples,)
        alpha: Significance level (default: 0.1 for 90% intervals)

    Returns:
        Mean Winkler score (lower is better)

    Formula:
        If y in [lower, upper]: score = (upper - lower)
        If y < lower: score = (upper - lower) + (2/alpha) * (lower - y)
        If y > upper: score = (upper - lower) + (2/alpha) * (y - upper)

    Interpretation:
        - Lower scores are better
        - Penalty for missed coverage scales with 2/alpha
    """
    if isinstance(y_pred_lower, torch.Tensor):
        y_pred_lower = y_pred_lower.detach().cpu().numpy()
    if isinstance(y_pred_upper, torch.Tensor):
        y_pred_upper = y_pred_upper.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    interval_width = y_pred_upper - y_pred_lower
    penalty = 2.0 / alpha

    # Compute score for each sample
    scores = np.where(
        y_true < y_pred_lower,
        interval_width + penalty * (y_pred_lower - y_true),
        np.where(
            y_true > y_pred_upper,
            interval_width + penalty * (y_true - y_pred_upper),
            interval_width
        )
    )

    return float(np.mean(scores))


def compute_calibration_error(y_true: Union[np.ndarray, torch.Tensor],
                                y_pred_quantiles: Union[np.ndarray, torch.Tensor],
                                quantiles: List[float],
                                n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration error for quantile predictions.

    Calibration error measures how well the predicted quantiles match
    the empirical frequencies. Well-calibrated models have quantile
    predictions that are exceeded by the true value exactly (1-q) of the time.

    Args:
        y_true: Ground truth values, shape (n_samples,)
        y_pred_quantiles: Predicted values for each quantile, shape (n_samples, n_quantiles)
        quantiles: List of quantile levels
        n_bins: Number of bins for computing calibration

    Returns:
        Dictionary with calibration metrics

    Interpretation:
        - Lower calibration error is better
        - CE = 0 means perfectly calibrated
    """
    if isinstance(y_pred_quantiles, torch.Tensor):
        y_pred_quantiles = y_pred_quantiles.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    calibration_errors = []

    for i, q in enumerate(quantiles):
        predicted_q = y_pred_quantiles[:, i]
        empirical_freq = np.mean(y_true <= predicted_q)
        error = abs(empirical_freq - q)
        calibration_errors.append(error)

    return {
        'calibration_error_mean': float(np.mean(calibration_errors)),
        'calibration_error_max': float(np.max(calibration_errors)),
    }


def compute_all_uncertainty_metrics(y_true: Union[np.ndarray, torch.Tensor],
                                     y_pred_quantiles: Union[np.ndarray, torch.Tensor],
                                     quantiles: List[float],
                                     y_mean: Optional[float] = None,
                                     y_std: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all uncertainty metrics for quantile regression.

    This is the main function to call for comprehensive uncertainty evaluation.
    It computes pinball loss, CRPS, PICP, PINAW, Winkler score, and calibration error.

    Args:
        y_true: Ground truth values (in original scale), shape (n_samples,)
        y_pred_quantiles: Predicted values for each quantile (in original scale),
                          shape (n_samples, n_quantiles)
        quantiles: List of quantile levels
        y_mean: Mean of training targets (if y is in z-score space)
        y_std: Std of training targets (if y is in z-score space)

    Returns:
        Dictionary with all uncertainty metrics

    Example:
        >>> quantiles = [0.1, 0.5, 0.9]
        >>> y_pred = np.random.randn(100, 3)  # 100 samples, 3 quantiles
        >>> y_true = np.random.randn(100)
        >>> metrics = compute_all_uncertainty_metrics(y_true, y_pred, quantiles)
    """
    results = {}

    # 1. Pinball loss for each quantile
    pinball_results = compute_pinball_loss(y_true, y_pred_quantiles, quantiles)
    results.update(pinball_results)

    # 2. CRPS (comprehensive probabilistic metric)
    results['crps'] = compute_crps(y_true, y_pred_quantiles, quantiles)

    # 3. PICP and PINAW (use extreme quantiles for interval)
    q_lower_idx = 0
    q_upper_idx = -1
    results['picp'] = compute_picp(
        y_true,
        y_pred_quantiles[:, q_lower_idx],
        y_pred_quantiles[:, q_upper_idx]
    )
    results['pinaw'] = compute_pinaw(
        y_pred_quantiles[:, q_lower_idx],
        y_pred_quantiles[:, q_upper_idx],
        y_true
    )

    # 4. Winkler score (combines coverage and sharpness)
    coverage_level = 1 - (quantiles[q_upper_idx] - quantiles[q_lower_idx])
    results['winkler_score'] = compute_winkler_score(
        y_true,
        y_pred_quantiles[:, q_lower_idx],
        y_pred_quantiles[:, q_upper_idx],
        alpha=coverage_level
    )

    # 5. Calibration error
    cal_results = compute_calibration_error(y_true, y_pred_quantiles, quantiles)
    results.update(cal_results)

    # 6. Mean interval width
    results['mean_interval_width'] = float(
        np.mean(y_pred_quantiles[:, q_upper_idx] - y_pred_quantiles[:, q_lower_idx])
    )

    return results


class UncertaintyMetrics:
    """
    TorchMetrics-style class for uncertainty metrics.

    Compatible with PyTorch Lightning for logging during training/validation.
    """

    def __init__(self, quantiles: List[float], prefix: str = ''):
        """
        Args:
            quantiles: List of quantile levels to track
            prefix: Prefix for metric names (e.g., 'val_', 'test_')
        """
        self.quantiles = quantiles
        self.prefix = prefix
        self.reset()

    def reset(self):
        """Reset all accumulated values."""
        self.y_true = []
        self.y_pred = []

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Add predictions and targets.

        Args:
            y_true: Ground truth, shape (batch,)
            y_pred: Predictions, shape (batch, n_quantiles)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary of metric names to values
        """
        if len(self.y_true) == 0:
            return {}

        y_true_all = torch.cat(self.y_true, dim=0).numpy()
        y_pred_all = torch.cat(self.y_pred, dim=0).numpy()

        metrics = compute_all_uncertainty_metrics(
            y_true_all, y_pred_all, self.quantiles
        )

        # Add prefix to all metric names
        if self.prefix:
            metrics = {f'{self.prefix}{k}': v for k, v in metrics.items()}

        return metrics

    def compute_per_sample(self, y_true: torch.Tensor,
                           y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for a single batch (without accumulation).

        Args:
            y_true: Ground truth, shape (batch,)
            y_pred: Predictions, shape (batch, n_quantiles)

        Returns:
            Dictionary of metric names to values
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        return compute_all_uncertainty_metrics(y_true, y_pred, self.quantiles)


def pinball_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor,
                        quantiles: List[float]) -> torch.Tensor:
    """
    Compute average pinball loss across all quantiles (PyTorch version).

    This is the loss function to use during training for quantile regression.

    Args:
        y_pred: Predicted values, shape (batch, n_quantiles)
        y_true: Ground truth values, shape (batch, 1) or (batch,)
        quantiles: List of quantile levels

    Returns:
        Scalar loss value (average across all quantiles and samples)

    Example:
        >>> y_pred = model(x)  # (batch, 3) for 3 quantiles
        >>> y_true = target    # (batch,)
        >>> loss = pinball_loss_torch(y_pred, y_true, [0.1, 0.5, 0.9])
        >>> loss.backward()

    Raises:
        ValueError: If quantiles are invalid or shapes don't match
    """
    # Input validation
    if not quantiles:
        raise ValueError("quantiles list cannot be empty")
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("All quantiles must be in the range (0, 1)")
    if y_pred.dim() != 2:
        raise ValueError(f"y_pred must be 2D (batch, n_quantiles), got shape {y_pred.shape}")
    if y_pred.shape[1] != len(quantiles):
        raise ValueError(f"y_pred.shape[1] ({y_pred.shape[1]}) must match len(quantiles) ({len(quantiles)})")

    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i:i+1]
        loss = torch.max(q * error, (q - 1) * error)
        losses.append(loss.mean())

    return torch.stack(losses).mean()


def weighted_pinball_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor,
                                 quantiles: List[float],
                                 weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute weighted pinball loss (for sample weighting).

    Args:
        y_pred: Predicted values, shape (batch, n_quantiles)
        y_true: Ground truth values, shape (batch, 1) or (batch,)
        quantiles: List of quantile levels
        weights: Optional sample weights, shape (batch, 1) or (batch,)

    Returns:
        Scalar weighted loss value

    Raises:
        ValueError: If quantiles are invalid or shapes don't match
    """
    # Input validation
    if not quantiles:
        raise ValueError("quantiles list cannot be empty")
    if not all(0 < q < 1 for q in quantiles):
        raise ValueError("All quantiles must be in the range (0, 1)")
    if y_pred.dim() != 2:
        raise ValueError(f"y_pred must be 2D (batch, n_quantiles), got shape {y_pred.shape}")
    if y_pred.shape[1] != len(quantiles):
        raise ValueError(f"y_pred.shape[1] ({y_pred.shape[1]}) must match len(quantiles) ({len(quantiles)})")
    if weights is not None and weights.shape[0] != y_pred.shape[0]:
        raise ValueError(f"weights.shape[0] ({weights.shape[0]}) must match y_pred.shape[0] ({y_pred.shape[0]})")

    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    # Ensure weights has correct shape
    if weights is not None and weights.dim() == 1:
        weights = weights.unsqueeze(-1)

    batch_size = y_pred.shape[0]
    n_quantiles = len(quantiles)
    losses = []

    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i:i+1]
        loss = torch.max(q * error, (q - 1) * error)

        if weights is not None:
            loss = loss * weights

        losses.append(loss.sum())

    total_loss = torch.stack(losses).sum()

    # Normalize by (sum of weights * n_quantiles) for consistency with unweighted version
    # Unweighted: divides by (batch_size * n_quantiles)
    # Weighted: divides by (weights.sum() * n_quantiles)
    if weights is not None:
        return total_loss / (weights.sum() * n_quantiles)
    else:
        return total_loss / (batch_size * n_quantiles)
