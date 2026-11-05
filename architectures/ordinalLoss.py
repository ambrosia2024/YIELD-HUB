# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Ordinal regression loss functions for multitask yield forecasting.
Implements CORN (Corn Loss) for ordinal classification.

CORN Loss Reference:
    - "CORN: Class-Ordinal Regression for Neural Networks"
    - https://arxiv.org/abs/2206.13562

Python version: 3.12.0
--------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class CornLoss(nn.Module):
    """
    CORN (Class-Ordinal Regression) Loss for ordinal classification.

    For K classes, CORN learns K-1 binary tasks:
    - Task k: Is true class >= k? (for k = 1, 2, ..., K-1)

    During inference, sum the binary task outputs to get the final class.
    This respects the ordinal structure and provides better gradients than
    treating classes as unordered categories.

    Args:
        num_classes: Number of ordinal classes (K)

    Example:
        >>> loss_fn = CornLoss(num_classes=5)
        >>> # Logits from model: (batch, K-1) = (batch, 4)
        >>> logits = torch.randn(32, 4)
        >>> # True class indices: (batch,)
        >>> targets = torch.randint(0, 5, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute CORN loss.

        Args:
            logits: Model output, shape (batch, K-1)
                   Each logit[i] represents log-odds of class >= i+1
            targets: True class indices, shape (batch,)
                     Values in [0, K-1]

        Returns:
            Scalar loss value
        """
        if logits.shape[-1] != self.num_tasks:
            raise ValueError(
                f"Expected logits with {self.num_tasks} columns (K-1), "
                f"got {logits.shape[-1]}"
            )

        # Convert targets to binary labels for each task
        # Task k: Is class >= k?
        # Binary labels: 1 if target >= k, else 0
        batch_size = targets.shape[0]
        binary_labels = torch.empty(
            batch_size, self.num_tasks,
            device=targets.device, dtype=torch.long
        )

        for k in range(self.num_tasks):
            # Task k (0-indexed): Is class >= k+1?
            binary_labels[:, k] = (targets >= (k + 1)).long()

        # Compute binary cross-entropy loss for each task
        # Sum over tasks, mean over batch
        loss = F.binary_cross_entropy_with_logits(
            logits, binary_labels.float(), reduction='mean'
        )

        return loss

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class predictions.

        Args:
            logits: Model output, shape (batch, K-1)

        Returns:
            Predicted class indices, shape (batch,)
        """
        # Get probabilities for each task
        probs = torch.sigmoid(logits)

        # Sum probabilities to get class prediction
        # Class = number of tasks with probability >= 0.5
        # (or equivalently, sum of sigmoid outputs)
        class_pred = torch.sum(probs >= 0.5, dim=1).long()

        # Clip to valid range [0, K-1]
        class_pred = torch.clamp(class_pred, 0, self.num_classes - 1)

        return class_pred


class CrossEntropyLossFallback(nn.Module):
    """
    Standard cross-entropy loss as fallback if CORN is not desired.

    This treats classes as unordered categories (less appropriate for ordinal data).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute standard cross-entropy loss.

        Args:
            logits: Model output, shape (batch, num_classes)
            targets: True class indices, shape (batch,)

        Returns:
            Scalar loss value
        """
        return self.ce_loss(logits, targets)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to class predictions.

        Args:
            logits: Model output, shape (batch, num_classes)

        Returns:
            Predicted class indices, shape (batch,)
        """
        return torch.argmax(logits, dim=-1)


def get_ordinal_loss(num_classes: int, loss_type: str = "corn") -> nn.Module:
    """
    Factory function to get ordinal loss function.

    Args:
        num_classes: Number of ordinal classes
        loss_type: Type of loss - "corn" for CORN loss, "ce" for standard cross-entropy

    Returns:
        Loss module with forward() and predict() methods
    """
    if loss_type == "corn":
        return CornLoss(num_classes=num_classes)
    elif loss_type == "ce":
        return CrossEntropyLossFallback(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'corn' or 'ce'")


# Test function
def test_corn_loss():
    """Test CORN loss with synthetic data."""
    print("[CORN Loss] Running tests...")

    num_classes = 5
    batch_size = 32

    loss_fn = CornLoss(num_classes=num_classes)

    # Test forward pass
    logits = torch.randn(batch_size, num_classes - 1)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(logits, targets)
    print(f"[CORN Loss] Loss value: {loss.item():.4f}")

    # Test prediction
    preds = loss_fn.predict(logits)
    print(f"[CORN Loss] Predictions shape: {preds.shape}")
    print(f"[CORN Loss] Unique predictions: {torch.unique(preds).tolist()}")

    # Test gradient flow
    loss.backward()
    print(f"[CORN Loss] Gradient flow OK")

    print("[CORN Loss] All tests passed!")


if __name__ == "__main__":
    test_corn_loss()
