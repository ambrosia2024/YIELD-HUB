# -*- coding: utf-8 -*-
"""
Unit tests for uncertainty quantification metrics.

Tests pinball loss implementation, gradient flow, and edge cases.
"""

import pytest
import torch
import numpy as np
from typing import List

from uncertainty_metrics import (
    pinball_loss,
    pinball_loss_torch,
    weighted_pinball_loss_torch,
    compute_pinball_loss,
    compute_crps,
    compute_picp,
    compute_pinaw,
    compute_winkler_score,
    compute_calibration_error,
    compute_all_uncertainty_metrics,
    UncertaintyMetrics
)


class TestPinballLoss:
    """Test suite for pinball loss implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        batch_size = 32
        quantiles = [0.1, 0.5, 0.9]

        # Generate predictions and targets
        y_pred = torch.randn(batch_size, len(quantiles))
        y_true = torch.randn(batch_size, 1)

        return y_pred, y_true, quantiles

    def test_pinball_loss_formula(self):
        """Test that pinball loss formula is correctly implemented."""
        # Test case: perfect prediction (zero error)
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        quantile = 0.5

        loss = pinball_loss(y_true, y_pred, quantile)
        assert torch.allclose(loss, torch.zeros_like(loss)), "Perfect prediction should have zero loss"

        # Test case: under-prediction (y_true > y_pred)
        y_true = torch.tensor([2.0])
        y_pred = torch.tensor([1.0])
        quantile = 0.5

        loss = pinball_loss(y_true, y_pred, quantile)
        expected = quantile * (y_true - y_pred)
        assert torch.allclose(loss, expected), "Under-prediction loss formula incorrect";

        # Test case: over-prediction (y_true < y_pred)
        y_true = torch.tensor([1.0])
        y_pred = torch.tensor([2.0])
        quantile = 0.5

        loss = pinball_loss(y_true, y_pred, quantile)
        expected = (quantile - 1) * (y_true - y_pred)
        assert torch.allclose(loss, expected), "Over-prediction loss formula incorrect"

    def test_pinball_loss_asymmetry(self):
        """Test that pinball loss is asymmetric for different quantiles."""
        y_true = torch.tensor([1.0])
        y_pred = torch.tensor([0.0])  # Under-prediction

        # Lower quantile penalizes under-prediction less
        loss_q10 = pinball_loss(y_true, y_pred, 0.1)
        # Median quantile penalizes equally
        loss_q50 = pinball_loss(y_true, y_pred, 0.5)
        # Upper quantile penalizes under-prediction more
        loss_q90 = pinball_loss(y_true, y_pred, 0.9)

        assert loss_q10 < loss_q50 < loss_q90, \
            "Pinball loss should be asymmetric based on quantile level"

    def test_pinball_loss_torch_shape(self, sample_data):
        """Test that pinball_loss_torch handles shapes correctly."""
        y_pred, y_true, quantiles = sample_data

        # Test with 2D predictions
        loss = pinball_loss_torch(y_pred, y_true, quantiles)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive for random data"

        # Test with 1D targets
        y_true_1d = y_true.squeeze(-1)
        loss_1d = pinball_loss_torch(y_pred, y_true_1d, quantiles)
        assert torch.allclose(loss, loss_1d), "1D and 2D targets should give same loss"

    def test_pinball_loss_torch_gradients(self, sample_data):
        """Test that gradients flow correctly through pinball loss."""
        y_pred, y_true, quantiles = sample_data

        # Enable gradient computation
        y_pred.requires_grad_(True)

        # Compute loss
        loss = pinball_loss_torch(y_pred, y_true, quantiles)
        loss.backward()

        # Check that gradients exist and are not all zeros
        assert y_pred.grad is not None, "Gradients should be computed"
        assert not torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad)), \
            "Gradients should not be all zeros"

        # Check that gradients are finite
        assert torch.all(torch.isfinite(y_pred.grad)), "Gradients should be finite"

    def test_weighted_pinball_loss_consistency(self, sample_data):
        """Test that weighted and unweighted losses are consistent when weights are uniform."""
        y_pred, y_true, quantiles = sample_data
        batch_size = y_pred.shape[0]

        # Create uniform weights (all ones)
        uniform_weights = torch.ones(batch_size, 1)

        # Compute both losses
        unweighted_loss = pinball_loss_torch(y_pred, y_true, quantiles)
        weighted_loss = weighted_pinball_loss_torch(y_pred, y_true, quantiles, uniform_weights)

        # With uniform weights, losses should be identical
        assert torch.allclose(unweighted_loss, weighted_loss, atol=1e-6), \
            "Uniform weights should give same loss as unweighted"

    def test_weighted_pinball_loss_scaling(self, sample_data):
        """Test that weighted loss correctly scales with different weights."""
        y_pred, y_true, quantiles = sample_data
        batch_size = y_pred.shape[0]

        # Create weights that heavily favor first sample
        weighted_weights = torch.zeros(batch_size, 1)
        weighted_weights[0, 0] = 1.0  # Only first sample contributes

        # Create weights that heavily favor last sample
        weighted_weights_alt = torch.zeros(batch_size, 1)
        weighted_weights_alt[-1, 0] = 1.0  # Only last sample contributes

        # The two weighted losses should be different
        loss_1 = weighted_pinball_loss_torch(y_pred, y_true, quantiles, weighted_weights)
        loss_2 = weighted_pinball_loss_torch(y_pred, y_true, quantiles, weighted_weights_alt)

        assert not torch.allclose(loss_1, loss_2), \
            "Different weight distributions should produce different losses"

    def test_pinball_loss_input_validation(self):
        """Test that input validation works correctly."""
        y_pred = torch.randn(10, 3)
        y_true = torch.randn(10, 1)
        quantiles = [0.1, 0.5, 0.9]

        # Test invalid quantiles
        with pytest.raises(ValueError, match="quantiles list cannot be empty"):
            pinball_loss_torch(y_pred, y_true, [])

        with pytest.raises(ValueError, match="All quantiles must be in the range"):
            pinball_loss_torch(y_pred, y_true, [0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="All quantiles must be in the range"):
            pinball_loss_torch(y_pred, y_true, [-0.1, 0.5, 1.1])

        # Test shape mismatch
        with pytest.raises(ValueError, match="must match len"):
            pinball_loss_torch(y_pred, y_true, [0.1, 0.5])

        # Test invalid y_pred dimensions
        with pytest.raises(ValueError, match="must be 2D"):
            pinball_loss_torch(y_pred[:, 0], y_true, quantiles)

    def test_weighted_pinball_loss_validation(self):
        """Test that weighted pinball loss validates weights correctly."""
        y_pred = torch.randn(10, 3)
        y_true = torch.randn(10, 1)
        quantiles = [0.1, 0.5, 0.9]

        # Test weight shape mismatch
        wrong_weights = torch.randn(5, 1)  # Wrong batch size
        with pytest.raises(ValueError, match="weights.shape"):
            weighted_pinball_loss_torch(y_pred, y_true, quantiles, wrong_weights)

    def test_pinball_loss_median_properties(self):
        """Test that 0.5 quantile (median) has expected properties."""
        torch.manual_seed(42)
        n_samples = 1000

        # For median, over-prediction and under-prediction of same magnitude
        # should have equal loss
        y_true = torch.tensor([0.0])
        y_pred_over = torch.tensor([1.0])  # Over-predict by 1
        y_pred_under = torch.tensor([-1.0])  # Under-predict by 1

        loss_over = pinball_loss(y_true, y_pred_over, 0.5)
        loss_under = pinball_loss(y_true, y_pred_under, 0.5)

        assert torch.allclose(loss_over, loss_under), \
            "Median quantile should symmetrically penalize over/under prediction"

    def test_compute_pinball_loss_dict(self):
        """Test that compute_pinball_loss returns correct dictionary."""
        y_true = torch.randn(100)
        y_pred = torch.randn(100, 3)
        quantiles = [0.1, 0.5, 0.9]

        result = compute_pinball_loss(y_true, y_pred, quantiles, aggregate='mean')

        # Check that all expected keys are present
        assert 'pinball_q10' in result
        assert 'pinball_q50' in result
        assert 'pinball_q90' in result
        assert 'pinball_avg' in result

        # Check that average is mean of individual quantile losses
        expected_avg = (result['pinball_q10'] + result['pinball_q50'] + result['pinball_q90']) / 3
        assert abs(result['pinball_avg'] - expected_avg) < 1e-6


class TestUncertaintyMetricsIntegration:
    """Integration tests for uncertainty metrics."""

    def test_uncertainty_metrics_class(self):
        """Test UncertaintyMetrics class with quantile regression."""
        torch.manual_seed(42)
        quantiles = [0.1, 0.5, 0.9]
        metrics = UncertaintyMetrics(quantiles, prefix='test_')

        # Simulate multiple batches
        for _ in range(5):
            batch_size = 16
            y_pred = torch.randn(batch_size, len(quantiles))
            y_true = torch.randn(batch_size)
            metrics.update(y_true, y_pred)

        # Compute metrics
        results = metrics.compute()

        # Check that expected metrics are present
        assert 'test_crps' in results
        assert 'test_picp' in results
        assert 'test_pinaw' in results
        assert 'test_winkler_score' in results

        # Check that metrics are reasonable
        assert results['test_picp'] >= 0 and results['test_picp'] <= 1, \
            "PICP should be in [0, 1]"
        assert results['test_pinaw'] >= 0, "PINAW should be non-negative"

    def test_compute_all_uncertainty_metrics(self):
        """Test compute_all_uncertainty_metrics function."""
        torch.manual_seed(42)
        n_samples = 100
        quantiles = [0.1, 0.5, 0.9]

        y_true = torch.randn(n_samples)
        y_pred = torch.randn(n_samples, len(quantiles))

        results = compute_all_uncertainty_metrics(y_true, y_pred, quantiles)

        # Check that all expected metrics are present
        expected_keys = [
            'pinball_q10', 'pinball_q50', 'pinball_q90', 'pinball_avg',
            'crps', 'picp', 'pinaw', 'winkler_score',
            'calibration_error_mean', 'calibration_error_max',
            'mean_interval_width'
        ]

        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"

    def test_crps_properties(self):
        """Test that CRPS has expected properties."""
        # Perfect prediction should have CRPS = 0
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred_perfect = torch.tensor([[1.0, 1.0, 1.0],
                                       [2.0, 2.0, 2.0],
                                       [3.0, 3.0, 3.0]])
        quantiles = [0.1, 0.5, 0.9]

        crps = compute_crps(y_true, y_pred_perfect, quantiles)
        assert crps == 0.0, "Perfect prediction should have CRPS = 0"

    def test_picp_properties(self):
        """Test that PICP has expected properties."""
        # Create predictions where 90% of samples are within interval
        n = 100
        y_true = torch.zeros(n)
        y_pred_lower = torch.tensor([-1.0] * n)
        y_pred_upper = torch.tensor([1.0] * n)

        # All samples within interval
        picp = compute_picp(y_true, y_pred_lower, y_pred_upper)
        assert picp == 1.0, "All samples within interval should give PICP = 1"

        # No samples within interval
        picp_zero = compute_picp(y_true + 10, y_pred_lower, y_pred_upper)
        assert picp_zero == 0.0, "No samples within interval should give PICP = 0"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_quantile(self):
        """Test with single quantile."""
        y_pred = torch.randn(10, 1)
        y_true = torch.randn(10)

        loss = pinball_loss_torch(y_pred, y_true, [0.5])
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_many_quantiles(self):
        """Test with many quantiles."""
        n_quantiles = 19
        quantiles = [i / 20 for i in range(1, n_quantiles + 1)]  # 0.05, 0.1, ..., 0.95

        y_pred = torch.randn(10, n_quantiles)
        y_true = torch.randn(10)

        loss = pinball_loss_torch(y_pred, y_true, quantiles)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_extreme_quantiles(self):
        """Test with extreme quantile values."""
        y_pred = torch.randn(10, 2)
        y_true = torch.randn(10)

        # Very low quantile
        loss_01 = pinball_loss_torch(y_pred, y_true, [0.01])
        assert not torch.isnan(loss_01)

        # Very high quantile
        loss_99 = pinball_loss_torch(y_pred, y_true, [0.99])
        assert not torch.isnan(loss_99)

    def test_identical_predictions(self):
        """Test when all predictions are identical."""
        y_true = torch.randn(10)
        y_pred = torch.ones(10, 3) * 5.0  # All predictions are 5.0

        loss = pinball_loss_torch(y_pred, y_true, [0.1, 0.5, 0.9])
        assert not torch.isnan(loss), "Loss should handle identical predictions"

    def test_constant_targets(self):
        """Test when all targets are constant."""
        y_true = torch.ones(10) * 3.0  # All targets are 3.0
        y_pred = torch.randn(10, 3)

        loss = pinball_loss_torch(y_pred, y_true, [0.1, 0.5, 0.9])
        assert not torch.isnan(loss), "Loss should handle constant targets"


def test_gradient_flow_through_model():
    """Test that gradients flow through a simple model with pinball loss."""
    # Create a simple linear model
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 3)  # 10 inputs, 3 quantiles outputs
    quantiles = [0.1, 0.5, 0.9]

    # Create sample data
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 1)

    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = pinball_loss_torch(y_pred, y_true, quantiles)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Check that gradients exist
    assert model.weight.grad is not None, "Model should have gradients"
    assert not torch.allclose(model.weight.grad, torch.zeros_like(model.weight.grad)), \
        "Gradients should not be all zeros"

    # Check that bias also has gradients
    assert model.bias.grad is not None, "Bias should have gradients"


if __name__ == '__main__':
    # Run tests if pytest is not available
    print("Running pinball loss tests...")

    test_instance = TestPinballLoss()

    # Test 1: Formula correctness
    print("✓ Testing pinball loss formula...")
    test_instance.test_pinball_loss_formula()

    # Test 2: Asymmetry
    print("✓ Testing pinball loss asymmetry...")
    test_instance.test_pinball_loss_asymmetry()

    # Test 3: Gradient flow
    print("✓ Testing gradient flow...")
    torch.manual_seed(42)
    y_pred, y_true, quantiles = torch.randn(32, 3), torch.randn(32, 1), [0.1, 0.5, 0.9]
    y_pred.requires_grad_(True)
    loss = pinball_loss_torch(y_pred, y_true, quantiles)
    loss.backward()
    assert y_pred.grad is not None and not torch.allclose(y_pred.grad, torch.zeros_like(y_pred.grad))

    # Test 4: Weighted/unweighted consistency
    print("✓ Testing weighted/unweighted consistency...")
    test_instance.test_weighted_pinball_loss_consistency((y_pred.detach(), y_true, quantiles))

    # Test 5: Median properties
    print("✓ Testing median properties...")
    test_instance.test_pinball_loss_median_properties()

    print("\nAll tests passed! ✓")
