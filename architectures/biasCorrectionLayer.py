# -*- coding: utf-8 -*-
"""
Post-hoc bias correction for crop yield predictions.

Provides standard bias correction methods used in climate and agricultural modeling:
- Mean (additive): constant offset correction
- Ratio (multiplicative): constant scaling correction
- Linear regression: handles both offset and scale errors (recommended default)
- Quantile mapping: corrects full distribution shape

IMPORTANT: Always fit the correction on one set of data (calibration/training)
and evaluate it on a separate set (validation/test). Fitting and evaluating
on the same data will make the correction look better than it actually is.

Typical usage:
    1. Fit on training/validation predictions: bc.fit(pred_calib, obs_calib)
    2. Apply to test predictions: corrected = bc.correct(pred_test)
    3. Access fitted parameters: bc.a, bc.b (for linear method)

Reference:
    Bias correction for deep learning crop yield predictions.
    Similar to methods used in climate model downscaling and crop yield forecasting.
"""

import logging
from typing import Tuple, Optional, Literal
from scipy.stats import linregress
import numpy as np

logger = logging.getLogger(__name__)


class BiasCorrection:
    """
    Post-hoc bias correction for yield predictions.

    Supports four correction methods via linear regression on calibration data:
    - 'mean': Additive correction (constant offset)
    - 'ratio': Multiplicative correction (constant scaling)
    - 'linear': Linear regression (offset + scale, recommended default)
    - 'quantile': Empirical quantile mapping (distribution correction)

    The linear method is recommended for recursive lag forecasting because:
    - It captures both offset (b) and scale errors (a)
    - It handles "regression to the mean" bias common in DL models (a < 1)
    - It works well with limited calibration data (2-3 years typical)
    - It's interpretable and standard in agricultural modeling

    Attributes:
        method: The bias correction method to use
        a: Fitted slope parameter (linear method only)
        b: Fitted intercept parameter (linear/ratio methods)
        ratio: Fitted ratio parameter (ratio method only)
        bias: Fitted bias parameter (mean method only)
        is_fitted: Whether the correction parameters have been fitted
    """

    METHODS = ['mean', 'ratio', 'linear', 'quantile']

    def __init__(self, method: Literal['mean', 'ratio', 'linear', 'quantile'] = 'linear'):
        """
        Initialize BiasCorrection.

        Args:
            method: Bias correction method. Default: 'linear' (recommended).
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.METHODS}")
        self.method = method
        self.a: Optional[float] = None  # Slope (linear)
        self.b: Optional[float] = None  # Intercept (linear/mean)
        self.ratio: Optional[float] = None  # Ratio (ratio)
        self.bias: Optional[float] = None  # Bias offset (mean)
        self._pred_calib: Optional[np.ndarray] = None  # For quantile mapping
        self._obs_calib: Optional[np.ndarray] = None  # For quantile mapping
        self.is_fitted = False

    def fit(self, pred_calib: np.ndarray, obs_calib: np.ndarray) -> 'BiasCorrection':
        """
        Fit bias correction parameters on calibration data.

        Args:
            pred_calib: Model predictions on calibration set (e.g., validation years)
            obs_calib: Corresponding observed yields on calibration set

        Returns:
            self (fitted BiasCorrection instance)

        Raises:
            ValueError: If arrays have mismatched lengths or insufficient data
        """
        pred_calib = np.asarray(pred_calib).flatten()
        obs_calib = np.asarray(obs_calib).flatten()

        if len(pred_calib) != len(obs_calib):
            raise ValueError(f"Length mismatch: pred={len(pred_calib)}, obs={len(obs_calib)}")

        if len(pred_calib) < 2:
            raise ValueError(f"Need at least 2 samples for fitting, got {len(pred_calib)}")

        if self.method == 'mean':
            # Mean (additive) bias correction
            self.bias = float(np.mean(obs_calib) - np.mean(pred_calib))
            self.b = self.bias
            logger.info(f"[BiasCorrection] Fitted mean correction: bias={self.bias:.4f} t/ha")

        elif self.method == 'ratio':
            # Ratio (multiplicative) bias correction
            self.ratio = float(np.mean(obs_calib) / np.mean(pred_calib))
            self.b = self.ratio
            logger.info(f"[BiasCorrection] Fitted ratio correction: ratio={self.ratio:.4f}")

        elif self.method == 'linear':
            # Linear regression bias correction (recommended)
            self.a, self.b, r_value, p_value, std_err = linregress(pred_calib, obs_calib)
            logger.info(
                f"[BiasCorrection] Fitted linear correction: a={self.a:.4f}, b={self.b:.4f}, "
                f"R²={r_value**2:.3f}, p={p_value:.4f}"
            )

        elif self.method == 'quantile':
            # Quantile mapping - store calibration data for interpolation
            if len(pred_calib) < 10:
                logger.warning(
                    f"[BiasCorrection] Quantile mapping recommended with 10+ samples, "
                    f"got {len(pred_calib)}. Results may be unstable."
                )
            self._pred_calib = np.sort(pred_calib)
            self._obs_calib = np.sort(obs_calib)
            logger.info(f"[BiasCorrection] Fitted quantile mapping with {len(pred_calib)} samples")

        self.is_fitted = True
        return self

    def correct(self, pred_new: np.ndarray) -> np.ndarray:
        """
        Apply bias correction to new predictions.

        Args:
            pred_new: Model predictions to correct (can be single value or array)

        Returns:
            Bias-corrected predictions in the same shape as pred_new

        Raises:
            RuntimeError: If correct() is called before fit()
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before correct()")

        pred_new = np.asarray(pred_new)

        if self.method == 'mean':
            # corrected = pred_new + bias
            return pred_new + self.bias

        elif self.method == 'ratio':
            # corrected = pred_new * ratio
            return pred_new * self.ratio

        elif self.method == 'linear':
            # corrected = a * pred_new + b
            return self.a * pred_new + self.b

        elif self.method == 'quantile':
            # Quantile mapping via interpolation
            return np.interp(pred_new, self._pred_calib, self._obs_calib)

    def fit_and_correct(self, pred_calib: np.ndarray, obs_calib: np.ndarray,
                        pred_new: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit on calibration data and correct new data in one call.

        Args:
            pred_calib: Model predictions on calibration set
            obs_calib: Corresponding observed yields on calibration set
            pred_new: New predictions to correct

        Returns:
            Bias-corrected predictions
        """
        self.fit(pred_calib, obs_calib)
        return self.correct(pred_new)

    def get_correction_params(self) -> dict:
        """
        Get fitted correction parameters for logging/saving.

        Returns:
            Dictionary with method and fitted parameters
        """
        if not self.is_fitted:
            return {'method': self.method, 'fitted': False}

        params = {'method': self.method, 'fitted': True}

        if self.method == 'mean':
            params['bias'] = self.bias
        elif self.method == 'ratio':
            params['ratio'] = self.ratio
        elif self.method == 'linear':
            params['a'] = self.a
            params['b'] = self.b
        elif self.method == 'quantile':
            params['n_samples'] = len(self._pred_calib) if self._pred_calib is not None else 0

        return params


# Convenience functions for direct use without class instantiation
def mean_bias_correction(pred_calib: np.ndarray, obs_calib: np.ndarray,
                         pred_new: np.ndarray) -> np.ndarray:
    """Additive (mean) bias correction: corrected = pred_new + mean(obs - pred)."""
    bias = np.mean(obs_calib) - np.mean(pred_calib)
    return pred_new + bias


def ratio_bias_correction(pred_calib: np.ndarray, obs_calib: np.ndarray,
                         pred_new: np.ndarray) -> np.ndarray:
    """Multiplicative (ratio) bias correction: corrected = pred_new * (mean(obs) / mean(pred))."""
    ratio = np.mean(obs_calib) / np.mean(pred_calib)
    return pred_new * ratio


def linear_bias_correction(pred_calib: np.ndarray, obs_calib: np.ndarray,
                           pred_new: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Linear regression bias correction: corrected = a * pred_new + b.

    This is the recommended default method for crop yield predictions.
    Handles both offset and scale errors, plus regression-to-the-mean bias.

    Args:
        pred_calib: Model predictions on calibration set
        obs_calib: Corresponding observed yields
        pred_new: New predictions to correct

    Returns:
        Tuple of (corrected_predictions, slope_a, intercept_b)
    """
    a, b, r_value, p_value, std_err = linregress(pred_calib, obs_calib)
    corrected = a * pred_new + b
    logger.info(f"[BiasCorrection] Linear: a={a:.4f}, b={b:.4f}, R²={r_value**2:.3f}")
    return corrected, a, b


def quantile_mapping_correction(pred_calib: np.ndarray, obs_calib: np.ndarray,
                                pred_new: np.ndarray) -> np.ndarray:
    """
    Empirical quantile mapping bias correction.

    Maps predictions onto the observed distribution via CDF matching.
    Use when error distribution is non-linear or has shape mismatch.

    Warning: Requires sufficient calibration data (10+ samples recommended).
    """
    sorted_pred = np.sort(pred_calib)
    sorted_obs = np.sort(obs_calib)
    corrected = np.interp(pred_new, sorted_pred, sorted_obs)
    return corrected
