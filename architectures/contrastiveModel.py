"""
--------------------
Author: XYZ
Description: Contrastive pre-training model with AEZ auxiliary classification.
             Wraps XLinear or PatchTST as encoder backbone and adds:
             - InfoNCE contrastive loss for representation learning
             - AEZ classification as auxiliary task
             - Adapter architecture for extended features (fine-tuning)
             - k-NN evaluation support

Architecture:
    Core features → Encoder (XLinear/PatchTST) → Embedding
                                             → AEZ Classifier (auxiliary)
                                             → Projection Head (for contrastive loss)

    Extended features → Adapter → Combined Embedding → Yield Predictor (fine-tuning)

Python version: 3.12.0
--------------------
"""

import sys
import logging
from typing import Dict, List, Optional, Tuple, Union
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import Accuracy, R2Score, MeanSquaredError

# CY-BENCH dependencies
from cybench.config import LOCATION_PROPERTIES, SOIL_PROPERTIES

# Model architectures
from linearLayer import XLinearModel
from tstLayer import PatchTSTModel

# Data config
sys.path.append('../process/')
from contrastiveDataModule import ContrastiveModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Projects embeddings to a lower-dimensional space where
    contrastive loss is computed. This follows SimCLR framework.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
            num_layers: Number of MLP layers
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input tensor."""
        return self.projection(x)


class AEZClassifier(nn.Module):
    """
    AEZ classification auxiliary head.

    Predicts agro-ecological zone from embedding to guide
    the encoder to learn agro-ecologically meaningful patterns.
    """

    def __init__(
        self,
        input_dim: int,
        num_aez_classes: int,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Input embedding dimension
            num_aez_classes: Number of unique AEZ codes
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()

        if hidden_dim is None:
            self.classifier = nn.Linear(input_dim, num_aez_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_aez_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input tensor."""
        return self.classifier(x)


class ExtendedFeatureAdapter(nn.Module):
    """
    Adapter for extended features during fine-tuning.

    Projects extended features (multi-year summaries, exponential tau, etc.)
    into the embedding space to combine with core encoder outputs.
    """

    def __init__(
        self,
        num_extended_features: int,
        embedding_dim: int,
        hidden_dim: int = 64
    ):
        """
        Args:
            num_extended_features: Number of extended feature dimensions
            embedding_dim: Target embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        if num_extended_features == 0:
            # No extended features - pass through zero
            self.adapter = nn.Identity()
            self.has_extended = False
        else:
            self.adapter = nn.Sequential(
                nn.Linear(num_extended_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.has_extended = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project extended features to embedding space."""
        if self.has_extended:
            return self.adapter(x)
        return torch.zeros(x.shape[0], self.adapter[-1].out_features).to(x.device)


class ContrastiveEncoder(nn.Module):
    """
    Wrapper for encoder backbone (XLinear or PatchTST).

    Extracts the encoder part of the model and provides
    embeddings instead of predictions.
    """

    def __init__(
        self,
        model_type: str,
        config,
        embedding_dim: Optional[int] = None
    ):
        """
        Args:
            model_type: 'xlinear' or 'patchtst'
            config: Model configuration
            embedding_dim: Target embedding dimension (uses model output if None)
        """
        super().__init__()

        self.model_type = model_type.lower()
        self.config = config
        self.embedding_dim = embedding_dim

        # Create the base model
        if self.model_type == 'xlinear':
            # Import and create XLinear
            self.base_model = XLinearModel(config)
        elif self.model_type == 'patchtst':
            # Import and create PatchTST
            self.base_model = PatchTSTModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Get the encoder output dimension
        encoder_out_dim = self._get_encoder_output_dim()

        # Optional projection to target embedding dimension
        if embedding_dim is not None and encoder_out_dim != embedding_dim:
            self.proj_to_embedding = nn.Linear(encoder_out_dim, embedding_dim)
        else:
            self.proj_to_embedding = nn.Identity()

    def _get_encoder_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        # This depends on the model architecture
        # For now, use a reasonable default based on model type
        if self.model_type == 'xlinear':
            # XLinear hidden size
            return getattr(self.config, 'xlinear_hidden_size', 64)
        elif self.model_type == 'patchtst':
            # PatchTST d_model
            return getattr(self.config, 'patchtst_d_model', 64)
        return 64

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Encode input and return embedding.

        Args:
            x_ts: Time series features (batch, seq_len, n_features)
            x_static: Static features (batch, n_static)

        Returns:
            Embedding tensor (batch, embedding_dim)
        """
        # Get the model's internal representation
        # For now, we'll use the model's output before the final prediction layer
        # This is a simplification - in practice, you'd want to access the actual encoder output

        # Call the base model forward pass
        # Note: This is a placeholder - actual implementation depends on model internals
        if self.model_type == 'xlinear':
            # For XLinear, access the encoder output
            # This would require modifying the XLinearModel to expose encoder outputs
            # For now, we'll use a simplified approach
            output = self.base_model.base_model(x_ts, x_static)  # Get internal representation
        elif self.model_type == 'patchtst':
            # For PatchTST
            output = self.base_model.base_model(x_ts, x_static)

        # Project to target embedding dimension
        embedding = self.proj_to_embedding(output)

        return embedding


class ContrastivePretrainModel(pl.LightningModule):
    """
    Contrastive pre-training model with AEZ auxiliary classification.

    Training flow:
    1. Encode anchor samples → embeddings
    2. L2-normalize embeddings
    3. Compute similarity to positives/negatives
    4. InfoNCE loss on similarities
    5. AEZ classification loss (auxiliary)
    """

    def __init__(
        self,
        config,
        model_type: str = 'xlinear',
        embedding_dim: int = 128,
        projection_dim: int = 128,
        num_aez_classes: int = 50,
        contrastive_temp: float = 0.07,
        aez_loss_weight: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            config: Model configuration
            model_type: 'xlinear' or 'patchtst'
            embedding_dim: Embedding dimension
            projection_dim: Projection head output dimension
            num_aez_classes: Number of unique AEZ codes
            contrastive_temp: Temperature for InfoNCE loss
            aez_loss_weight: Weight for AEZ auxiliary loss
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.contrastive_temp = contrastive_temp
        self.aez_loss_weight = aez_loss_weight

        # Core encoder
        self.encoder = ContrastiveEncoder(
            model_type=model_type,
            config=config,
            embedding_dim=embedding_dim
        )

        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=256,
            output_dim=projection_dim
        )

        # AEZ classifier (auxiliary task)
        self.aez_classifier = AEZClassifier(
            input_dim=embedding_dim,
            num_aez_classes=num_aez_classes,
            hidden_dim=128
        )

        # Metrics
        self.train_aez_acc = Accuracy(task="multiclass", num_classes=num_aez_classes)
        self.val_aez_acc = Accuracy(task="multiclass", num_classes=num_aez_classes)

        # For tracking
        self.train_contrastive_loss = 0.0
        self.train_aez_loss_val = 0.0

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: return embeddings.

        Args:
            x_ts: Time series features (batch, seq_len, n_features)
            x_static: Static features (batch, n_static)

        Returns:
            Embedding tensor (batch, embedding_dim)
        """
        return self.encoder(x_ts, x_static)

    def compute_contrastive_loss(
        self,
        anchor_embeddings: torch.Tensor,
        positive_features: List,
        negative_features: List,
        positive_static_features: List,
        negative_static_features: List
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss with actual positives and negatives.

        Args:
            anchor_embeddings: L2-normalized anchor embeddings (batch, dim)
            positive_features: List of positive time series features (batch, variable, seq, n_feat)
            negative_features: List of negative time series features (batch, variable, seq, n_feat)
            positive_static_features: List of positive static features (batch, variable, n_static)
            negative_static_features: List of negative static features (batch, variable, n_static)

        Returns:
            Contrastive loss scalar
        """
        batch_size = anchor_embeddings.shape[0]
        device = anchor_embeddings.device

        total_loss = 0.0
        valid_anchors = 0

        for i in range(batch_size):
            anchor = anchor_embeddings[i:i+1]  # (1, dim)

            # Get positives for this anchor
            pos_features = positive_features[i]  # (n_pos, seq, n_feat) or empty
            pos_static = positive_static_features[i]  # (n_pos, n_static) or empty

            # Get negatives for this anchor
            neg_features = negative_features[i]  # (n_neg, seq, n_feat) or empty
            neg_static = negative_static_features[i]  # (n_neg, n_static) or empty

            # Skip if no valid pairs
            if len(pos_features) == 0 or len(neg_features) == 0:
                continue

            valid_anchors += 1

            # Encode positives (batch encode all at once for efficiency)
            n_pos = len(pos_features)
            if n_pos > 0:
                pos_X_ts = torch.tensor(pos_features, dtype=torch.float32).to(device)
                pos_X_static = torch.tensor(pos_static, dtype=torch.float32).to(device)
                pos_embeddings = self.encoder(pos_X_ts, pos_X_static)
                pos_embeddings = F.normalize(pos_embeddings, p=2, dim=1)
            else:
                pos_embeddings = torch.zeros((0, anchor_embeddings.shape[1]), device=device)

            # Encode negatives
            n_neg = len(neg_features)
            if n_neg > 0:
                neg_X_ts = torch.tensor(neg_features, dtype=torch.float32).to(device)
                neg_X_static = torch.tensor(neg_static, dtype=torch.float32).to(device)
                neg_embeddings = self.encoder(neg_X_ts, neg_X_static)
                neg_embeddings = F.normalize(neg_embeddings, p=2, dim=1)
            else:
                neg_embeddings = torch.zeros((0, anchor_embeddings.shape[1]), device=device)

            # Compute similarities
            if pos_embeddings.shape[0] > 0:
                pos_sim = torch.matmul(anchor, pos_embeddings.T) / self.contrastive_temp  # (1, n_pos)
                pos_score = torch.logsumexp(pos_sim, dim=1)  # (1,)
            else:
                pos_score = torch.tensor([0.0], device=device)

            if neg_embeddings.shape[0] > 0:
                neg_sim = torch.matmul(anchor, neg_embeddings.T) / self.contrastive_temp  # (1, n_neg)
                neg_score = torch.logsumexp(neg_sim, dim=1)  # (1,)
            else:
                neg_score = torch.tensor([0.0], device=device)

            # InfoNCE loss: -log(exp(pos_score) / (exp(pos_score) + exp(neg_score)))
            # Which simplifies to: -(pos_score - logsumexp([pos_score, neg_score]))

            all_scores = torch.cat([pos_sim, neg_sim], dim=1) if pos_embeddings.shape[0] > 0 and neg_embeddings.shape[0] > 0 else (
                pos_sim if neg_embeddings.shape[0] == 0 else neg_sim
            )

            if all_scores.shape[1] > 0:
                all_score = torch.logsumexp(all_scores, dim=1)  # (1,)
                loss = -pos_score + all_score if pos_embeddings.shape[0] > 0 else -neg_score
                total_loss += loss

        return total_loss / max(valid_anchors, 1)

    def compute_aez_loss(
        self,
        embeddings: torch.Tensor,
        aez_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AEZ classification loss.

        Args:
            embeddings: Encoder embeddings (batch, dim)
            aez_labels: AEZ class labels (batch,)

        Returns:
            Classification loss scalar
        """
        logits = self.aez_classifier(embeddings)
        loss = F.cross_entropy(logits, aez_labels)
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        # Extract anchor data
        anchor_X_ts = batch['anchor_X_ts']
        anchor_X_static = batch['anchor_X_static']
        anchor_aez_codes = batch['anchor_aez_codes']

        # Extract positive and negative features
        positives_X_ts = batch['positives_X_ts']
        positives_X_static = batch['positives_X_static']
        negatives_X_ts = batch['negatives_X_ts']
        negatives_X_static = batch['negatives_X_static']

        # Encode anchors
        anchor_embeddings = self.encoder(anchor_X_ts, anchor_X_static)

        # Project for contrastive loss
        projected = self.projection_head(anchor_embeddings)
        projected = F.normalize(projected, p=2, dim=1)

        # Contrastive loss with actual positives/negatives
        contrastive_loss = self.compute_contrastive_loss(
            anchor_embeddings=projected,
            positive_features=positives_X_ts,
            negative_features=negatives_X_ts,
            positive_static_features=positives_X_static,
            negative_static_features=negatives_X_static,
        )

        # AEZ auxiliary loss
        # Convert AEZ codes to class indices
        aez_indices = self._aez_codes_to_indices(anchor_aez_codes)
        aez_loss = self.compute_aez_loss(anchor_embeddings, aez_indices)

        # Combined loss
        total_loss = contrastive_loss + self.aez_loss_weight * aez_loss

        # Log metrics
        self.log('train/contrastive_loss', contrastive_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/aez_loss', aez_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        # AEZ accuracy
        preds = self.aez_classifier(anchor_embeddings)
        self.train_aez_acc(preds, aez_indices)
        self.log('train/aez_acc', self.train_aez_acc, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        anchor_X_ts = batch['anchor_X_ts']
        anchor_X_static = batch['anchor_X_static']
        anchor_aez_codes = batch['anchor_aez_codes']

        # Extract positive and negative features
        positives_X_ts = batch['positives_X_ts']
        positives_X_static = batch['positives_X_static']
        negatives_X_ts = batch['negatives_X_ts']
        negatives_X_static = batch['negatives_X_static']

        anchor_embeddings = self.encoder(anchor_X_ts, anchor_X_static)
        projected = self.projection_head(anchor_embeddings)
        projected = F.normalize(projected, p=2, dim=1)

        contrastive_loss = self.compute_contrastive_loss(
            anchor_embeddings=projected,
            positive_features=positives_X_ts,
            negative_features=negatives_X_ts,
            positive_static_features=positives_X_static,
            negative_static_features=negatives_X_static,
        )

        aez_indices = self._aez_codes_to_indices(anchor_aez_codes)
        aez_loss = self.compute_aez_loss(anchor_embeddings, aez_indices)

        total_loss = contrastive_loss + self.aez_loss_weight * aez_loss

        self.log('val/contrastive_loss', contrastive_loss, on_step=False, on_epoch=True)
        self.log('val/aez_loss', aez_loss, on_step=False, on_epoch=True)
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True)

        preds = self.aez_classifier(anchor_embeddings)
        self.val_aez_acc(preds, aez_indices)
        self.log('val/aez_acc', self.val_aez_acc, on_step=False, on_epoch=True)

        return total_loss

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        anchor_X_ts = batch['anchor_X_ts']
        anchor_X_static = batch['anchor_X_static']
        anchor_aez_codes = batch['anchor_aez_codes']

        anchor_embeddings = self.encoder(anchor_X_ts, anchor_X_static)

        aez_indices = self._aez_codes_to_indices(anchor_aez_codes)
        aez_loss = self.compute_aez_loss(anchor_embeddings, aez_indices)

        self.log('test/aez_loss', aez_loss, on_step=False, on_epoch=True)

        preds = self.aez_classifier(anchor_embeddings)
        self.val_aez_acc(preds, aez_indices)
        self.log('test/aez_acc', self.val_aez_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _aez_codes_to_indices(self, aez_codes: List[str]) -> torch.Tensor:
        """
        Convert AEZ code strings to class indices.

        This is a simplified version - in practice, you'd have a
        pre-built mapping from AEZ codes to class indices.
        """
        # TODO: Build proper AEZ code to index mapping
        # For now, use a simple hash-based mapping
        indices = []
        for code in aez_codes:
            # Simple hash to convert string to integer
            idx = hash(code) % self.hparams.num_aez_classes
            indices.append(idx)

        return torch.tensor(indices, device=self.device)

    def get_embeddings(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Get L2-normalized embeddings for k-NN evaluation.

        Args:
            x_ts: Time series features
            x_static: Static features

        Returns:
            L2-normalized embeddings
        """
        with torch.no_grad():
            embeddings = self.encoder(x_ts, x_static)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ContrastiveFinetuneModel(pl.LightningModule):
    """
    Fine-tuning model with adapter for extended features.

    Architecture:
        Core features → Frozen encoder → core_embedding
        Extended features → Adapter → extended_embedding
        Combined → Prediction head → yield

    The encoder is frozen during fine-tuning to preserve
    pre-trained representations.
    """

    def __init__(
        self,
        pretrained_encoder: ContrastiveEncoder,
        num_extended_features: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            pretrained_encoder: Pre-trained contrastive encoder
            num_extended_features: Number of extended feature dimensions
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        # Freeze the encoder
        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get embedding dimension
        embedding_dim = self.encoder.embedding_dim or 64

        # Adapter for extended features
        self.extended_adapter = ExtendedFeatureAdapter(
            num_extended_features=num_extended_features,
            embedding_dim=embedding_dim
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # Metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        x_extended: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: predict yield.

        Args:
            x_ts: Core time series features
            x_static: Core static features
            x_extended: Extended features (optional)

        Returns:
            Yield predictions (batch, 1)
        """
        # Get core embedding from frozen encoder
        core_embedding = self.encoder(x_ts, x_static)

        # Get extended embedding from adapter
        if x_extended is not None and self.extended_adapter.has_extended:
            extended_embedding = self.extended_adapter(x_extended)
            # Combine: add embeddings
            combined_embedding = core_embedding + extended_embedding
        else:
            combined_embedding = core_embedding

        # Predict yield
        yield_pred = self.prediction_head(combined_embedding)

        return yield_pred

    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        x_ts = batch['X_ts']
        x_static = batch['X_static']
        y = batch['y']

        # Forward pass (no extended features for now)
        y_pred = self.forward(x_ts, x_static).squeeze(-1)

        # Compute loss
        loss = F.mse_loss(y_pred, y)

        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_mse(y_pred, y)
        self.log('train/mse', self.train_mse, on_step=False, on_epoch=True)
        self.train_r2(y_pred, y)
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        x_ts = batch['X_ts']
        x_static = batch['X_static']
        y = batch['y']

        y_pred = self.forward(x_ts, x_static).squeeze(-1)
        loss = F.mse_loss(y_pred, y)

        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.val_mse(y_pred, y)
        self.log('val/mse', self.val_mse, on_step=False, on_epoch=True)
        self.val_r2(y_pred, y)
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        x_ts = batch['X_ts']
        x_static = batch['X_static']
        y = batch['y']

        y_pred = self.forward(x_ts, x_static).squeeze(-1)
        loss = F.mse_loss(y_pred, y)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.test_mse(y_pred, y)
        self.log('test/mse', self.test_mse, on_step=False, on_epoch=True)
        self.test_r2(y_pred, y)
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizers."""
        # Only optimize adapter and prediction head (encoder is frozen)
        params = list(self.extended_adapter.parameters()) + list(self.prediction_head.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
