# -*- coding: utf-8 -*-
"""
Monte Carlo Dropout Inference Layer for Crop Yield Prediction Models

This module provides functions for generating predictions with MC dropout uncertainty quantification.
It works with trained models and their existing data pipelines without requiring additional data processing.

Supports both MSE (point prediction) and Pinball (quantile regression) loss types.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


def enable_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    Sets model to train mode (enables dropout) but keeps batch norm in eval mode.
    """
    model.train()  # Set to train mode to enable dropout
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()  # Keep batch norm in eval mode


def disable_dropout(model: torch.nn.Module) -> None:
    """
    Disable dropout for deterministic inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.eval()


def get_quantiles_from_model(model: torch.nn.Module) -> Optional[List[float]]:
    """Get quantiles list from model config if using pinball loss."""
    if hasattr(model, 'config') and hasattr(model.config, 'loss_type'):
        if model.config.loss_type == 'pinball':
            return getattr(model.config, 'quantiles', [0.1, 0.5, 0.9])
    return None


def get_quantile_column_names(quantiles: List[float], prefix: str = '') -> List[str]:
    """Generate column names for quantile predictions."""
    if prefix:
        return [f'{prefix}q{int(q*100) if q*100 == int(q*100) else q}' for q in quantiles]
    return [f'q{int(q*100) if q*100 == int(q*100) else q}' for q in quantiles]


def run_single_forward_pass(model: torch.nn.Module, batch: torch.Tensor,
                             device: str = 'cuda', datamodule=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass through the model.

    Args:
        model: The trained model with predict() method
        batch: Tuple of (x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask)
        device: Device to run inference on
        datamodule: Optional DataModule for trend computation and denormalization

    Returns:
        predictions: Model predictions (may be multi-quantile)
        targets: Ground truth targets (denormalized)
    """
    x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch

    # Move to device
    x_ts = x_ts.to(device)
    x_static = x_static.to(device)
    y = y.to(device)
    validity_mask = validity_mask.to(device) if validity_mask is not None else None
    batch = (x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask)

    # Call predict() if available (handles all preprocessing)
    if hasattr(model, 'predict'):
        result = model.predict(batch, datamodule=datamodule)
        predictions = result['predictions']
        targets = result['targets']
    else:
        raise RuntimeError(f"Model {type(model).__name__} does not have predict() method")

    return predictions, targets


def deterministic_inference_with_metadata(model: torch.nn.Module, dataloader: DataLoader,
                                          device: str = 'cuda',
                                          data_type: str = 'test',
                                          datamodule=None) -> pd.DataFrame:
    """
    Run deterministic inference (dropout disabled) and collect predictions with metadata.

    Args:
        model: The trained model
        dataloader: DataLoader for the split (train/val/test)
        device: Device to run inference on
        data_type: Type of data ('train', 'val', or 'test')
        datamodule: Optional DataModule for trend computation and denormalization

    Returns:
        DataFrame with columns based on loss type:
        - MSE: country, adm_id, year, true_value, pred_value, data_type
        - Pinball: country, adm_id, year, true_value, q0.1, q0.5, q0.9, ..., data_type
    """
    disable_dropout(model)

    all_metadata = []  # List of dicts for each sample

    # Check if using quantile regression
    quantiles = get_quantiles_from_model(model)
    is_quantile = quantiles is not None

    if is_quantile:
        print(f"[Deterministic Inference - {data_type.upper()}] Quantile regression mode: {quantiles}")
    else:
        print(f"[Deterministic Inference - {data_type.upper()}] Point prediction mode")

    print(f"[Deterministic Inference - {data_type.upper()}] Running with dropout DISABLED...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch

            preds, targets = run_single_forward_pass(model, batch, device, datamodule=datamodule)

            preds_tensor = preds.detach().cpu()
            targets = targets.detach().cpu().numpy().flatten()

            # Extract predictions based on loss type
            if is_quantile:
                # Extract all quantiles: (batch_size, n_quantiles)
                if preds_tensor.dim() > 1 and preds_tensor.shape[1] == len(quantiles):
                    preds_array = preds_tensor.numpy()  # (batch_size, n_quantiles)
                else:
                    # Fallback: treat as point prediction
                    preds_array = preds_tensor.numpy().flatten().reshape(-1, 1)
            else:
                # Point prediction: single value
                if preds_tensor.dim() > 1 and preds_tensor.shape[1] == 1:
                    preds_array = preds_tensor[:, 0].numpy().flatten()
                else:
                    preds_array = preds_tensor.numpy().flatten()

            # Extract metadata
            batch_years = years.numpy().flatten() if hasattr(years, 'numpy') else years
            if hasattr(adm_ids, 'numpy'):
                batch_adm_ids = adm_ids.numpy().flatten()
            elif isinstance(adm_ids, np.ndarray):
                batch_adm_ids = adm_ids.flatten()
            else:
                batch_adm_ids = np.array(list(adm_ids))

            country = getattr(model.config, 'country', 'UNKNOWN') if hasattr(model, 'config') else 'UNKNOWN'

            # Store results
            for i in range(len(targets)):
                row = {
                    'country': country,
                    'adm_id': batch_adm_ids[i],
                    'year': batch_years[i],
                    'true_value': targets[i],
                    'data_type': data_type
                }

                if is_quantile:
                    # Add each quantile as a separate column
                    for j, q in enumerate(quantiles):
                        col_name = f'q{int(q*100) if q*100 == int(q*100) else q}'
                        row[col_name] = preds_array[i, j]
                else:
                    # Single prediction value
                    row['pred_value'] = preds_array[i] if not is_quantile else preds_array[i, 0]

                all_metadata.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_metadata)

    print(f"[Deterministic Inference - {data_type.upper()}] Collected {len(df)} samples")
    return df


def mc_dropout_inference_with_metadata(model: torch.nn.Module, dataloader: DataLoader,
                                        n_passes: int = 30, device: str = 'cuda',
                                        data_type: str = 'test',
                                        datamodule=None) -> pd.DataFrame:
    """
    Run MC Dropout inference and collect predictions with metadata.

    Args:
        model: The trained model
        dataloader: DataLoader for the split (train/val/test)
        n_passes: Number of MC dropout forward passes
        device: Device to run inference on
        data_type: Type of data ('train', 'val', or 'test')
        datamodule: Optional DataModule for trend computation and denormalization

    Returns:
        DataFrame with columns based on loss type:
        - MSE: country, adm_id, year, true_value, pred_value (list of MC samples), data_type
        - Pinball: country, adm_id, year, true_value, mc_q0.1 (list), mc_q0.5 (list), mc_q0.9 (list), ..., data_type
    """
    # Enable dropout but keep batch norms in eval mode
    model.eval()
    enable_dropout(model)

    all_metadata = []  # List of dicts for each sample

    # Check if using quantile regression
    quantiles = get_quantiles_from_model(model)
    is_quantile = quantiles is not None

    if is_quantile:
        print(f"[MC Dropout Inference - {data_type.upper()}] Quantile regression mode: {quantiles}")
    else:
        print(f"[MC Dropout Inference - {data_type.upper()}] Point prediction mode")

    print(f"[MC Dropout Inference - {data_type.upper()}] Running {n_passes} forward passes with dropout...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch

            # Extract metadata
            batch_years = years.numpy().flatten() if hasattr(years, 'numpy') else years
            if hasattr(adm_ids, 'numpy'):
                batch_adm_ids = adm_ids.numpy().flatten()
            elif isinstance(adm_ids, np.ndarray):
                batch_adm_ids = adm_ids.flatten()
            else:
                batch_adm_ids = np.array(list(adm_ids))

            country = getattr(model.config, 'country', 'UNKNOWN') if hasattr(model, 'config') else 'UNKNOWN'

            # Run multiple forward passes
            if is_quantile:
                # For quantile regression, collect (n_passes, batch_size, n_quantiles)
                batch_preds = []
                for _ in range(n_passes):
                    preds, targets = run_single_forward_pass(model, batch, device, datamodule=datamodule)
                    preds_tensor = preds.detach().cpu()

                    # Expected shape: (batch_size, n_quantiles)
                    if preds_tensor.dim() > 1 and preds_tensor.shape[1] == len(quantiles):
                        batch_preds.append(preds_tensor.numpy())
                    else:
                        # Fallback
                        batch_preds.append(preds_tensor.numpy().reshape(-1, 1))

                # Stack: (n_passes, batch_size, n_quantiles)
                batch_preds = np.stack(batch_preds, axis=0)
                # Permute to (batch_size, n_quantiles, n_passes)
                batch_preds = batch_preds.transpose(1, 2, 0)

            else:
                # For point prediction, collect (n_passes, batch_size)
                batch_preds = []
                for _ in range(n_passes):
                    preds, targets = run_single_forward_pass(model, batch, device, datamodule=datamodule)
                    preds_tensor = preds.detach().cpu()

                    if preds_tensor.dim() > 1 and preds_tensor.shape[1] == 1:
                        preds = preds_tensor[:, 0].numpy().flatten()
                    else:
                        preds = preds_tensor.numpy().flatten()
                    batch_preds.append(preds)

                # Stack: (n_passes, batch_size)
                batch_preds = np.stack(batch_preds, axis=0)
                # Transpose to (batch_size, n_passes)
                batch_preds = batch_preds.transpose(1, 0)

            targets = targets.detach().cpu().numpy().flatten()

            # Store results
            for i in range(len(targets)):
                row = {
                    'country': country,
                    'adm_id': batch_adm_ids[i],
                    'year': batch_years[i],
                    'true_value': targets[i],
                    'data_type': data_type
                }

                if is_quantile:
                    # Add each quantile's MC samples as a separate list column
                    for j, q in enumerate(quantiles):
                        col_name = f'mc_q{int(q*100) if q*100 == int(q*100) else q}'
                        row[col_name] = batch_preds[i, j, :].tolist()  # List of n_passes samples
                else:
                    # Single list of MC predictions
                    row['pred_value'] = batch_preds[i, :].tolist()

                all_metadata.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_metadata)

    n_samples = len(df)
    if is_quantile:
        print(f"[MC Dropout Inference - {data_type.upper()}] Collected {n_samples} samples with {n_passes} MC samples for each of {len(quantiles)} quantiles")
    else:
        print(f"[MC Dropout Inference - {data_type.upper()}] Collected {n_samples} samples with {n_passes} MC predictions each")
    return df


def generate_predictions_for_all_splits(model: torch.nn.Module, datamodule,
                                         device: str = 'cuda',
                                         k_mc_dropouts: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Generate predictions for train, val, and test splits.

    Args:
        model: The trained model
        datamodule: DataModule with train/val/test dataloaders
        device: Device to run inference on
        k_mc_dropouts: Number of MC dropout passes (if None, skip MC dropout)

    Returns:
        predictions_df: DataFrame with deterministic predictions for all splits
        mc_predictions_df: DataFrame with MC dropout predictions for all splits (or None)
    """
    # Get loss type info
    quantiles = get_quantiles_from_model(model)
    if quantiles:
        print(f"\n[Prediction Mode] Quantile Regression (pinball loss) with quantiles: {quantiles}")
    else:
        print(f"\n[Prediction Mode] Point Prediction (MSE loss)")

    all_dfs = []

    # Generate deterministic predictions for all splits
    for split_name, dataloader in [('train', datamodule.train_dataloader()),
                                     ('val', datamodule.val_dataloader()),
                                     ('test', datamodule.test_dataloader())]:
        if dataloader is not None:
            df = deterministic_inference_with_metadata(model, dataloader, device, split_name, datamodule=datamodule)
            all_dfs.append(df)

    predictions_df = pd.concat(all_dfs, ignore_index=True)

    mc_predictions_df = None
    if k_mc_dropouts and k_mc_dropouts > 0:
        all_mc_dfs = []

        # Generate MC dropout predictions for all splits
        for split_name, dataloader in [('train', datamodule.train_dataloader()),
                                         ('val', datamodule.val_dataloader()),
                                         ('test', datamodule.test_dataloader())]:
            if dataloader is not None:
                df = mc_dropout_inference_with_metadata(model, dataloader, k_mc_dropouts, device, split_name, datamodule=datamodule)
                all_mc_dfs.append(df)

        mc_predictions_df = pd.concat(all_mc_dfs, ignore_index=True)

    return predictions_df, mc_predictions_df


def save_predictions_to_csv(predictions_df: pd.DataFrame,
                             mc_predictions_df: Optional[pd.DataFrame],
                             checkpoint_dir: str) -> None:
    """
    Save predictions to CSV files in the checkpoint directory.

    Args:
        predictions_df: DataFrame with deterministic predictions
        mc_predictions_df: DataFrame with MC dropout predictions (optional)
        checkpoint_dir: Directory where checkpoint is saved
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save deterministic predictions
    predictions_path = os.path.join(checkpoint_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[Predictions Saved] {predictions_path}")

    # Save MC dropout predictions if available
    if mc_predictions_df is not None:
        mc_predictions_path = os.path.join(checkpoint_dir, 'mc_dropout_predictions.csv')
        mc_predictions_df.to_csv(mc_predictions_path, index=False)
        print(f"[MC Dropout Predictions Saved] {mc_predictions_path}")
