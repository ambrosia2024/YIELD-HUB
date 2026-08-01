# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Tabular foundation model wrappers for CY-Bench yield prediction.
             Implements TabPFN, TabICL, and TabDPT with Lightning module integration
             for full coherence with TST baselines.
Python version: 3.12.0
--------------------

Architecture overview:
    This module provides tabular foundation model wrappers that:
    - Flatten time series features into tabular format while preserving temporal ordering
    - Integrate with PyTorch Lightning for consistent training workflow
    - Support the same evaluation metrics as TST models
    - Handle residual trend modeling and recursive lag prediction

Models implemented:
    • TabPFN: Tabular Prior-data Fitted Network (https://arxiv.org/abs/2206.13494)
    • TabICL: In-Context Learning for Tabular Data (https://arxiv.org/abs/2408.11586)
    • TabDPT: Diffusion-based Tabular Pre-trained Transformer (https://arxiv.org/abs/2406.03991)

--------------
Key differences from TST models:
    - No sequential processing: time series features are flattened
    - No learnable positional encoding: temporal order preserved via feature naming
    - No validation loop: tabular models fit once and predict
    - No early stopping: models don't train iteratively

--------------
Integration notes:
    - Uses same data pipeline as TST models via build_tabular_input()
    - Supports same feature engineering (GDD, RUE, Farquhar, heat stress)
    - Compatible with residual trend modeling and recursive lags
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Tuple, Any
from abc import abstractmethod

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

sys.path.append('../process/')
from validateModel import ModelMetrics

logger = logging.getLogger(__name__)


class BaseTabularModel(pl.LightningModule):
    """
    Base Lightning module for tabular foundation models.

    Provides:
    - Feature normalization (z-score, from training statistics)
    - Residual trend learning (per-location OLS, fitted in on_train_start)
    - Shared train/val/test step logic
    - Metric computation (MSE, MAE, RMSE, R², MAPE, SMAPE, NRMSE)

    Subclasses must implement:
    - _fit_tabular_model(X, y) -> None
    - _predict_tabular_model(X) -> np.ndarray
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['dataset'])

        # Normalization statistics (fit on training data)
        self.X_mean: Optional[torch.Tensor] = None
        self.X_std: Optional[torch.Tensor] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        # Residual trend model
        self.trend_models: Optional[Dict[str, Any]] = None
        self.use_trend = config.use_residual_trend

        # Metrics
        self.metrics = ModelMetrics()

        # Test results storage
        self._test_results_per_year = {}
        self._test_uncertainty_metrics = {}

        # Track training data for tabular models that need refitting
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

        # Feature names for interpretability
        self.feature_names: Optional[List[str]] = None

    def on_train_start(self) -> None:
        """Fit normalization statistics and residual trends on training data."""
        logger.info("[BaseTabularModel] Fitting normalization and trends...")

        # Get training data from datamodule
        dm = self.trainer.datamodule
        X_train, y_train, train_indices = dm.get_tabular_data(split='train')

        # Store training data for tabular models
        self._train_X = X_train
        self._train_y = y_train

        # Fit normalization
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.X_mean = X_train_tensor.mean(dim=0)
        self.X_std = X_train_tensor.std(dim=0) + 1e-6

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.y_mean = y_train_tensor.mean().item()
        self.y_std = y_train_tensor.std().item() + 1e-6

        logger.info(f"[BaseTabularModel] X_mean={self.X_mean.shape}, X_std={self.X_std.shape}")
        logger.info(f"[BaseTabularModel] y_mean={self.y_mean:.4f}, y_std={self.y_std:.4f}")

        # Fit residual trend if enabled
        if self.use_trend:
            self._fit_residual_trend(train_indices, y_train)

        # Fit the tabular model
        logger.info(f"[{self.config.model_type}] Fitting tabular model...")
        X_normalized = self._normalize_X(X_train)
        y_normalized = self._normalize_y(y_train)
        self._fit_tabular_model(X_normalized, y_normalized)
        logger.info(f"[{self.config.model_type}] Model fitting complete")

    def _fit_residual_trend(self, indices: List[Tuple[str, int]], y: np.ndarray) -> None:
        """Fit linear trend per location for residual modeling."""
        from pymannkendall import original_test
        from sklearn.linear_model import LinearRegression

        self.trend_models = {}

        for i, (adm_id, year) in enumerate(indices):
            if adm_id not in self.trend_models:
                # Get all historical yields for this location
                location_data = [(idx, j) for j, idx in enumerate(indices) if idx[0] == adm_id]
                if len(location_data) < 3:  # Need at least 3 points for trend
                    self.trend_models[adm_id] = None
                    continue

                # Extract years and yields
                loc_years = [idx[1] for idx, _ in location_data]
                loc_yields = [y[j] for _, j in location_data]

                years_arr = np.array(loc_years).reshape(-1, 1)
                yields_arr = np.array(loc_yields)

                # Mann-Kendall test for trend significance
                try:
                    mk_result = original_test(yields_arr)
                    has_trend = mk_result.p < 0.05
                except:
                    has_trend = False

                if has_trend:
                    # Fit linear trend
                    lr = LinearRegression()
                    lr.fit(years_arr, yields_arr)
                    self.trend_models[adm_id] = lr
                else:
                    self.trend_models[adm_id] = None

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean, unit variance."""
        if self.X_mean is None or self.X_std is None:
            return X
        return (X - self.X_mean.cpu().numpy()) / self.X_std.cpu().numpy()

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize targets to zero mean, unit variance."""
        return (y - self.y_mean) / self.y_std

    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return y_norm * self.y_std + self.y_mean

    def _get_trend_residual(self, y: float, adm_id: str, year: int) -> float:
        """Get residual after removing linear trend."""
        if not self.use_trend or self.trend_models is None:
            return y

        model = self.trend_models.get(adm_id)
        if model is None:
            return y

        trend_pred = model.predict([[year]])[0]
        return y - trend_pred

    def _add_trend_back(self, y_pred: float, adm_id: str, year: int) -> float:
        """Add linear trend back to prediction."""
        if not self.use_trend or self.trend_models is None:
            return y_pred

        model = self.trend_models.get(adm_id)
        if model is None:
            return y_pred

        trend_pred = model.predict([[year]])[0]
        return y_pred + trend_pred

    @abstractmethod
    def _fit_tabular_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying tabular model."""
        pass

    @abstractmethod
    def _predict_tabular_model(self, X: np.ndarray) -> np.ndarray:
        """Predict using the underlying tabular model."""
        pass

    def forward(self, X: np.ndarray, adm_ids: List[str], years: List[int]) -> torch.Tensor:
        """Forward pass: normalize -> predict -> denormalize."""
        X_norm = self._normalize_X(X)
        y_pred_norm = self._predict_tabular_model(X_norm)
        y_pred = self._denormalize_y(y_pred_norm)

        # Add trend back if using residual modeling
        if self.use_trend:
            for i, (adm_id, year) in enumerate(zip(adm_ids, years)):
                y_pred[i] = self._add_trend_back(y_pred[i], adm_id, year)

        return torch.tensor(y_pred, dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        """Tabular models don't have iterative training - this is a no-op."""
        # Tabular models fit once in on_train_start
        return None

    def configure_optimizers(self):
        """Tabular models don't use optimizers - they fit once and predict."""
        return None

    def validation_step(self, batch, batch_idx):
        """Validation step - compute metrics on validation set."""
        X, y, years, adm_ids, _, _, _ = batch
        y_pred = self.forward(X.cpu().numpy(), list(adm_ids), list(years))

        # Convert to tensors for metrics
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_pred_tensor = y_pred

        # Log metrics
        for name, metric_func in self.metrics.get_metrics().items():
            metric_val = metric_func(y_pred_tensor, y_tensor)
            self.log(f"val/{name}", metric_val, batch_size=len(y))

        return {'val_loss': torch.nn.functional.mse_loss(y_pred_tensor, y_tensor)}

    def test_step(self, batch, batch_idx):
        """Test step with per-year metrics."""
        X, y, years, adm_ids, _, _, _ = batch
        y = y.cpu().numpy()
        years = years.cpu().numpy() if hasattr(years, 'cpu') else years
        adm_ids = list(adm_ids) if isinstance(adm_ids, (list, tuple)) else list(adm_ids.cpu().numpy())

        y_pred = self.forward(X.cpu().numpy(), adm_ids, list(years))
        y_pred_np = y_pred.cpu().numpy()

        # Overall metrics
        y_tensor = torch.tensor(y, dtype=torch.float32)
        y_pred_tensor = y_pred

        metrics_dict = {
            'test/mse': torch.nn.functional.mse_loss(y_pred_tensor, y_tensor),
            'test/mae': torch.nn.functional.l1_loss(y_pred_tensor, y_tensor),
            'test/rmse': torch.sqrt(torch.nn.functional.mse_loss(y_pred_tensor, y_tensor)),
        }

        # R²
        r2 = R2Score()
        metrics_dict['test/r2'] = r2(y_pred_tensor, y_tensor)

        # MAPE
        mape = MeanAbsolutePercentageError()
        metrics_dict['test/mape'] = mape(y_pred_tensor, y_tensor)

        # SMAPE
        y_abs = np.abs(y)
        y_pred_abs = np.abs(y_pred_np)
        smape = np.mean(2.0 * np.abs(y - y_pred_np) / (y_abs + y_pred_abs + 1e-6))
        metrics_dict['test/smape'] = torch.tensor(smape, dtype=torch.float32)

        # NRMSE (RMSE / mean of actuals)
        nrmse = np.sqrt(np.mean((y - y_pred_np) ** 2)) / (np.mean(y_abs) + 1e-6)
        metrics_dict['test/nrmse'] = torch.tensor(nrmse, dtype=torch.float32)

        # Log all metrics
        for k, v in metrics_dict.items():
            self.log(k, v, batch_size=len(y))

        # Per-year metrics
        for year_val in np.unique(years):
            year_mask = years == year_val
            if year_mask.sum() > 0:
                y_year = y[year_mask]
                y_pred_year = y_pred_np[year_mask]

                for metric_name in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']:
                    if metric_name == 'mse':
                        val = np.mean((y_year - y_pred_year) ** 2)
                    elif metric_name == 'mae':
                        val = np.mean(np.abs(y_year - y_pred_year))
                    elif metric_name == 'rmse':
                        val = np.sqrt(np.mean((y_year - y_pred_year) ** 2))
                    elif metric_name == 'r2':
                        ss_res = np.sum((y_year - y_pred_year) ** 2)
                        ss_tot = np.sum((y_year - np.mean(y_year)) ** 2)
                        val = 1 - (ss_res / (ss_tot + 1e-6))
                    elif metric_name == 'mape':
                        val = np.mean(np.abs(y_year - y_pred_year) / (np.abs(y_year) + 1e-6)) * 100
                    elif metric_name == 'smape':
                        val = np.mean(2.0 * np.abs(y_year - y_pred_year) /
                                     (np.abs(y_year) + np.abs(y_pred_year) + 1e-6)) * 100
                    elif metric_name == 'nrmse':
                        rmse = np.sqrt(np.mean((y_year - y_pred_year) ** 2))
                        val = rmse / (np.mean(np.abs(y_year)) + 1e-6) * 100

                    self._test_results_per_year[f'{metric_name}_{int(year_val)}'] = float(val)

        # Overall metrics for CSV export
        self._test_results_per_year['mse_overall'] = float(metrics_dict['test/mse'])
        self._test_results_per_year['mae_overall'] = float(metrics_dict['test/mae'])
        self._test_results_per_year['rmse_overall'] = float(metrics_dict['test/rmse'])
        self._test_results_per_year['r2_overall'] = float(metrics_dict['test/r2'])
        self._test_results_per_year['mape_overall'] = float(metrics_dict['test/mape'])
        self._test_results_per_year['smape_overall'] = float(metrics_dict['test/smape'])
        self._test_results_per_year['nrmse_overall'] = float(metrics_dict['test/nrmse'])

        return metrics_dict

    def predict(self, batch) -> Dict[str, torch.Tensor]:
        """Prediction interface for spatiotemporal metrics."""
        X, y, years, adm_ids, _, _, _ = batch
        y_pred = self.forward(X.cpu().numpy(), list(adm_ids), list(years))

        return {
            'predictions': y_pred,
            'targets': torch.tensor(y, dtype=torch.float32),
        }


class TabPFNModel(BaseTabularModel):
    """TabPFN: Prior-data Fitted Network for tabular data."""

    def __init__(self, config):
        super().__init__(config)
        self.tabpfn_model = None

    def _fit_tabular_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabPFN model."""
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tabpfn_model = TabPFNRegressor(device=device)

        # Apply max_train_samples subsampling if specified
        if self.config.max_train_samples is not None and len(X) > self.config.max_train_samples:
            rng = np.random.default_rng(self.config.seed)
            if self.config.subsample == 'random':
                idx = rng.choice(len(X), self.config.max_train_samples, replace=False)
            else:  # quantile subsampling
                order = np.argsort(y)
                n_bins = self.config.subsample_bins
                chunks = np.array_split(order, max(2, n_bins))
                selected = []
                per_chunk = max(1, self.config.max_train_samples // len(chunks))
                for chunk in chunks:
                    if len(chunk) > 0:
                        take = min(per_chunk, len(chunk))
                        selected.extend(rng.choice(chunk, size=take, replace=False))
                idx = np.array(selected[:self.config.max_train_samples])
            X = X[idx]
            y = y[idx]
            logger.info(f"[TabPFN] Subsampled from {len(self._train_X)} to {len(X)} samples")

        self.tabpfn_model.fit(X, y)
        logger.info(f"[TabPFN] Model fitted on {len(X)} samples")

    def _predict_tabular_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with TabPFN."""
        if self.tabpfn_model is None:
            raise RuntimeError("TabPFN model not fitted. Call fit() first.")

        return self.tabpfn_model.predict(X)


class TabICLModel(BaseTabularModel):
    """TabICL: In-Context Learning for Tabular Data."""

    def __init__(self, config):
        super().__init__(config)
        self.tabicl_model = None

    def _fit_tabular_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabICL model."""
        try:
            from tabicl import TabICLRegressor
        except ImportError:
            raise ImportError("TabICL is not installed. Install with: pip install tabicl")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tabicl_model = TabICLRegressor(device=device)

        # Apply max_train_samples subsampling if specified
        if self.config.max_train_samples is not None and len(X) > self.config.max_train_samples:
            rng = np.random.default_rng(self.config.seed)
            if self.config.subsample == 'random':
                idx = rng.choice(len(X), self.config.max_train_samples, replace=False)
            else:  # quantile subsampling
                order = np.argsort(y)
                n_bins = self.config.subsample_bins
                chunks = np.array_split(order, max(2, n_bins))
                selected = []
                per_chunk = max(1, self.config.max_train_samples // len(chunks))
                for chunk in chunks:
                    if len(chunk) > 0:
                        take = min(per_chunk, len(chunk))
                        selected.extend(rng.choice(chunk, size=take, replace=False))
                idx = np.array(selected[:self.config.max_train_samples])
            X = X[idx]
            y = y[idx]
            logger.info(f"[TabICL] Subsampled from {len(self._train_X)} to {len(X)} samples")

        self.tabicl_model.fit(X, y)
        logger.info(f"[TabICL] Model fitted on {len(X)} samples")

    def _predict_tabular_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with TabICL."""
        if self.tabicl_model is None:
            raise RuntimeError("TabICL model not fitted. Call fit() first.")

        return self.tabicl_model.predict(X)


class TabDPTModel(BaseTabularModel):
    """TabDPT: Diffusion-based Tabular Pre-trained Transformer."""

    def __init__(self, config):
        super().__init__(config)
        self.tabdpt_model = None
        self._context_size = None

    def _fit_tabular_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit TabDPT model."""
        try:
            from tabdpt import TabDPTRegressor
        except ImportError:
            raise ImportError("TabDPT is not installed. Install with: pip install tabdpt")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Pad training data if below minimum samples
        if len(X) < self.config.min_train_samples:
            n_orig = len(X)
            n_reps = (self.config.min_train_samples // len(X)) + 1
            X = np.tile(X, (n_reps, 1))[:self.config.min_train_samples]
            y = np.tile(y, n_reps)[:self.config.min_train_samples]
            logger.info(f"[TabDPT] Padded training data from {n_orig} to {len(X)} samples")

        self._context_size = len(y)

        # Apply max_train_samples subsampling if specified
        if self.config.max_train_samples is not None and len(X) > self.config.max_train_samples:
            rng = np.random.default_rng(self.config.seed)
            if self.config.subsample == 'random':
                idx = rng.choice(len(X), self.config.max_train_samples, replace=False)
            else:  # quantile subsampling
                order = np.argsort(y)
                n_bins = self.config.subsample_bins
                chunks = np.array_split(order, max(2, n_bins))
                selected = []
                per_chunk = max(1, self.config.max_train_samples // len(chunks))
                for chunk in chunks:
                    if len(chunk) > 0:
                        take = min(per_chunk, len(chunk))
                        selected.extend(rng.choice(chunk, size=take, replace=False))
                idx = np.array(selected[:self.config.max_train_samples])
            X = X[idx]
            y = y[idx]
            logger.info(f"[TabDPT] Subsampled from {len(self._train_X)} to {len(X)} samples")

        self.tabdpt_model = TabDPTRegressor(device=device)
        self.tabdpt_model.fit(X, y)
        logger.info(f"[TabDPT] Model fitted on {len(X)} samples (context_size={self._context_size})")

    def _predict_tabular_model(self, X: np.ndarray) -> np.ndarray:
        """Predict with TabDPT."""
        if self.tabdpt_model is None:
            raise RuntimeError("TabDPT model not fitted. Call fit() first.")

        # TabDPT requires context_size parameter
        preds = self.tabdpt_model.predict(
            X,
            context_size=self._context_size or 100,
            seed=self.config.seed,
        )
        return preds.reshape(-1)


def create_model(config):
    """
    Factory function to create tabular foundation model from config.

    Args:
        config: TabModelConfig instance

    Returns:
        Initialized tabular model (TabPFNModel, TabICLModel, or TabDPTModel)
    """
    model_type = config.model_type.lower()

    if model_type in ['tabpfn', 'tab-pfn', 'tab_pfn']:
        return TabPFNModel(config)
    elif model_type in ['tabicl', 'tab-icl', 'tab_icl']:
        return TabICLModel(config)
    elif model_type in ['tabdpt', 'tab-dpt', 'tab_dpt']:
        return TabDPTModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}. "
                        f"Choose from: tabpfn, tabicl, tabdpt")
