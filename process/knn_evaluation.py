"""
--------------------
Author: XYZ
Description: k-NN evaluation module for non-parametric assessment of learned representations.
             Builds FAISS index on train embeddings and retrieves nearest neighbors
             for test samples to compute distance-weighted yield predictions.

Features:
- Multiple distance metrics (Euclidean, cosine)
- Distance-weighted prediction (inverse, Gaussian, uniform)
- Comprehensive evaluation metrics (MSE, R², MAE, etc.)

Usage:
    from knn_evaluation import KNNEvaluator

    evaluator = KNNEvaluator(
        k=5,
        distance_metric='euclidean',
        weighting='inverse'
    )

    # Build index on train embeddings
    evaluator.fit(train_embeddings, train_yields)

    # Evaluate on test set
    metrics = evaluator.evaluate(test_embeddings, test_yields)

Python version: 3.12.0
--------------------
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. k-NN evaluation will use slower sklearn implementation.")

import torch
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Distance metric options."""
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'
    MANHATTAN = 'manhattan'


class WeightingScheme(Enum):
    """Weighting scheme for k-NN prediction."""
    UNIFORM = 'uniform'  # All neighbors have equal weight
    INVERSE = 'inverse'  # Weight = 1 / (distance + epsilon)
    GAUSSIAN = 'gaussian'  # Weight = exp(-distance² / 2σ²)


class KNNEvaluator:
    """
    k-NN evaluator for non-parametric assessment of learned representations.

    Evaluates whether learned embeddings capture meaningful structure for
    yield prediction by comparing k-NN performance to baseline metrics.
    """

    def __init__(
        self,
        k: int = 5,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        weighting: Union[str, WeightingScheme] = WeightingScheme.INVERSE,
        sigma: Optional[float] = None,
        epsilon: float = 1e-6
    ):
        """
        Args:
            k: Number of nearest neighbors
            distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            weighting: Weighting scheme ('uniform', 'inverse', 'gaussian')
            sigma: Bandwidth for Gaussian weighting (default: k/10)
            epsilon: Small constant to prevent division by zero
        """
        self.k = k
        self.distance_metric = DistanceMetric(distance_metric) if isinstance(distance_metric, str) else distance_metric
        self.weighting = WeightingScheme(weighting) if isinstance(weighting, str) else weighting
        self.sigma = sigma if sigma is not None else max(k / 10.0, 1.0)
        self.epsilon = epsilon

        # Storage
        self.train_embeddings = None
        self.train_yields = None
        self.index = None

        logger.info(f"KNNEvaluator initialized: k={k}, metric={self.distance_metric.value}, "
                   f"weighting={self.weighting.value}")

    def fit(
        self,
        train_embeddings: Union[np.ndarray, torch.Tensor],
        train_yields: Union[np.ndarray, torch.Tensor]
    ):
        """
        Build k-NN index on training embeddings.

        Args:
            train_embeddings: Training embeddings (n_samples, embedding_dim)
            train_yields: Training yield values (n_samples,)
        """
        # Convert to numpy if needed
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings = train_embeddings.detach().cpu().numpy()
        if isinstance(train_yields, torch.Tensor):
            train_yields = train_yields.detach().cpu().numpy()

        self.train_embeddings = np.ascontiguousarray(train_embeddings, dtype=np.float32)
        self.train_yields = np.ascontiguousarray(train_yields, dtype=np.float32)

        logger.info(f"Building k-NN index on {len(self.train_embeddings)} training samples...")

        # Build FAISS index if available
        if FAISS_AVAILABLE:
            self._build_faiss_index()
        else:
            self._build_sklearn_index()

        logger.info("k-NN index built successfully")

    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search."""
        embedding_dim = self.train_embeddings.shape[1]

        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            # L2 distance (Euclidean)
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.Index(quantizer)
        elif self.distance_metric == DistanceMetric.COSINE:
            # For cosine similarity, use L2 on normalized vectors
            faiss.normalize_L2(self.train_embeddings)
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.Index(quantizer)
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            # FAISS doesn't have direct L1 support, use sklearn
            self._build_sklearn_index()
            return
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        # Add vectors to index
        self.index.add(self.train_embeddings)

    def _build_sklearn_index(self):
        """Build sklearn NearestNeighbors index."""
        metric = self.distance_metric.value
        self.index = NearestNeighbors(
            n_neighbors=self.k,
            metric=metric,
            algorithm='auto' if self.distance_metric != DistanceMetric.MANHATTAN else 'brute'
        )
        self.index.fit(self.train_embeddings)

    def predict(
        self,
        test_embeddings: Union[np.ndarray, torch.Tensor],
        return_neighbors: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict yields for test embeddings using k-NN.

        Args:
            test_embeddings: Test embeddings (n_samples, embedding_dim)
            return_neighbors: If True, also return neighbor indices and distances

        Returns:
            Predictions (n_samples,) or (predictions, indices, distances)
        """
        if self.index is None:
            raise RuntimeError("Must call fit() before predict()")

        # Convert to numpy if needed
        if isinstance(test_embeddings, torch.Tensor):
            test_embeddings = test_embeddings.detach().cpu().numpy()

        test_embeddings = np.ascontiguousarray(test_embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        if self.distance_metric == DistanceMetric.COSINE and FAISS_AVAILABLE:
            faiss.normalize_L2(test_embeddings)

        # Search for k nearest neighbors
        if FAISS_AVAILABLE and self.distance_metric != DistanceMetric.MANHATTAN:
            distances, indices = self.index.search(test_embeddings, self.k)
            # FAISS returns squared L2 distance for IndexFlatL2
            if self.distance_metric == DistanceMetric.EUCLIDEAN:
                distances = np.sqrt(distances)
        else:
            distances, indices = self.index.kneighbors(test_embeddings)

        # Compute distance-weighted predictions
        predictions = self._compute_weighted_predictions(distances, indices)

        if return_neighbors:
            return predictions, indices, distances
        return predictions

    def _compute_weighted_predictions(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance-weighted predictions.

        Args:
            distances: Neighbor distances (n_samples, k)
            indices: Neighbor indices (n_samples, k)

        Returns:
            Weighted predictions (n_samples,)
        """
        n_samples = distances.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            neighbor_dists = distances[i]
            neighbor_indices = indices[i]
            neighbor_yields = self.train_yields[neighbor_indices]

            # Compute weights
            if self.weighting == WeightingScheme.UNIFORM:
                weights = np.ones(self.k) / self.k
            elif self.weighting == WeightingScheme.INVERSE:
                weights = 1.0 / (neighbor_dists + self.epsilon)
                weights = weights / np.sum(weights)
            elif self.weighting == WeightingScheme.GAUSSIAN:
                weights = np.exp(-neighbor_dists ** 2 / (2 * self.sigma ** 2))
                weights = weights / np.sum(weights)
            else:
                raise ValueError(f"Unknown weighting scheme: {self.weighting}")

            # Weighted average
            predictions[i] = np.sum(weights * neighbor_yields)

        return predictions

    def evaluate(
        self,
        test_embeddings: Union[np.ndarray, torch.Tensor],
        test_yields: Union[np.ndarray, torch.Tensor],
        denormalize_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        Evaluate k-NN predictions on test set.

        Args:
            test_embeddings: Test embeddings
            test_yields: True yield values
            denormalize_fn: Optional function to denormalize predictions/targets

        Returns:
            Dict of metrics
        """
        # Get predictions
        predictions = self.predict(test_embeddings)

        # Convert to numpy if needed
        if isinstance(test_yields, torch.Tensor):
            test_yields = test_yields.detach().cpu().numpy()

        # Denormalize if function provided
        if denormalize_fn is not None:
            predictions = denormalize_fn(predictions)
            test_yields = denormalize_fn(test_yields)

        # Compute metrics
        mse = mean_squared_error(test_yields, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_yields, predictions)
        r2 = r2_score(test_yields, predictions)

        # Percentage errors
        mape = np.mean(np.abs((test_yields - predictions) / (test_yields + 1e-6))) * 100
        smape = np.mean(2.0 * np.abs(test_yields - predictions) /
                       (np.abs(test_yields) + np.abs(predictions) + 1e-6)) * 100

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'smape': float(smape),
        }

        logger.info(f"k-NN Evaluation Results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  SMAPE: {smape:.2f}%")

        return metrics

    def compare_weighting_schemes(
        self,
        test_embeddings: Union[np.ndarray, torch.Tensor],
        test_yields: Union[np.ndarray, torch.Tensor],
        denormalize_fn: Optional[callable] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different weighting schemes.

        Args:
            test_embeddings: Test embeddings
            test_yields: True yield values
            denormalize_fn: Optional denormalization function

        Returns:
            Dict mapping weighting scheme to metrics
        """
        results = {}

        original_weighting = self.weighting

        for scheme in WeightingScheme:
            self.weighting = scheme
            metrics = self.evaluate(test_embeddings, test_yields, denormalize_fn)
            results[scheme.value] = metrics

        # Restore original weighting
        self.weighting = original_weighting

        return results


def evaluate_knn_with_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    k: int = 5,
    device: str = 'cuda',
    denormalize_fn: Optional[callable] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model using k-NN on learned representations.

    Args:
        model: Trained model with get_embeddings() method
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        k: Number of neighbors
        device: Device to run on
        denormalize_fn: Optional denormalization function

    Returns:
        Dict of metrics for different distance metrics and weighting schemes
    """
    model.eval()
    model = model.to(device)

    # Extract embeddings and yields
    logger.info("Extracting training embeddings...")
    train_embeddings = []
    train_yields = []

    with torch.no_grad():
        for batch in train_dataloader:
            # Get features based on data format
            if isinstance(batch, dict):
                if 'anchor_X_ts' in batch:  # Contrastive format
                    x_ts = batch['anchor_X_ts'].to(device)
                    x_static = batch['anchor_X_static'].to(device)
                else:  # Standard format
                    x_ts = batch['X_ts'].to(device)
                    x_static = batch['X_static'].to(device)
            else:
                x_ts, x_static, y, *_ = batch
                x_ts = x_ts.to(device)
                x_static = x_static.to(device)

            # Get embeddings
            embeddings = model.get_embeddings(x_ts, x_static)
            train_embeddings.append(embeddings.cpu())

    train_embeddings = torch.cat(train_embeddings, dim=0)

    logger.info("Extracting test embeddings...")
    test_embeddings = []
    test_yields = []

    with torch.no_grad():
        for batch in test_dataloader:
            if isinstance(batch, dict):
                if 'anchor_X_ts' in batch:
                    x_ts = batch['anchor_X_ts'].to(device)
                    x_static = batch['anchor_X_static'].to(device)
                else:
                    x_ts = batch['X_ts'].to(device)
                    x_static = batch['X_static'].to(device)
                    y = batch['y']
                    test_yields.append(y.cpu())
            else:
                x_ts, x_static, y, *_ = batch
                x_ts = x_ts.to(device)
                x_static = x_static.to(device)
                test_yields.append(y)

            embeddings = model.get_embeddings(x_ts, x_static)
            test_embeddings.append(embeddings.cpu())

    test_embeddings = torch.cat(test_embeddings, dim=0)

    if test_yields:
        test_yields = torch.cat(test_yields, dim=0)

    # Try to get train yields from datataloader if not directly available
    if not hasattr(train_dataloader.dataset, 'y'):
        logger.warning("Could not extract train yields. Using dummy values.")
        train_yields = torch.zeros(len(train_embeddings))
    else:
        train_yields = torch.tensor(train_dataloader.dataset.y)

    # Evaluate with different configurations
    results = {}

    for metric in [DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE]:
        for weighting in [WeightingScheme.UNIFORM, WeightingScheme.INVERSE]:
            key = f"{metric.value}_{weighting.value}"

            evaluator = KNNEvaluator(k=k, distance_metric=metric, weighting=weighting)
            evaluator.fit(train_embeddings, train_yields)
            metrics = evaluator.evaluate(test_embeddings, test_yields, denormalize_fn)
            results[key] = metrics

    return results
