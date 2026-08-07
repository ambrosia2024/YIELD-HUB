# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Post-hoc prediction generation and model evaluation script.
             Supports deterministic predictions and MC Dropout uncertainty quantification.
Python version: 3.12.0
--------------------

This script provides functionality for:
1. Loading trained model checkpoints (XLinear, PatchTST, with/without WFAN wrapper)
2. Evaluating models on test sets with proper train/val/test splits
3. Generating deterministic predictions
4. Generating MC Dropout predictions for uncertainty quantification

The script automatically loads all hyperparameters from the checkpoint and uses
the same data splitting strategy that was used during training.

Output Files:
- {crop}_{country}_predictions.csv: Deterministic predictions with scalar values
- mc_predictions.csv: MC Dropout predictions with lists of K samples per quantile

Usage:
    # Evaluation only
    python generateMCPredictions.py --checkpoint_dir <path> --crop wheat --country BG --model_name xlinear

    # Evaluation + Deterministic predictions
    python generateMCPredictions.py --checkpoint_dir <path> --crop wheat --country BG --model_name xlinear --generate_prediction

    # Evaluation + Both deterministic and MC Dropout predictions
    python generateMCPredictions.py --checkpoint_dir <path> --crop wheat --country BG --model_name xlinear --generate_prediction --k_mc_dropout 30

    # Evaluation + MC Dropout predictions only
    python generateMCPredictions.py --checkpoint_dir <path> --crop wheat --country BG --model_name xlinear --k_mc_dropout 30

Examples:
    python generateMCPredictions.py \\
        --checkpoint_dir ../setups/temporal/modelCheckpoints-eos-final \\
        --crop wheat --country BG --model_name xlinear \\
        --generate_prediction --k_mc_dropout 30
"""

import sys
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer

# CY-BENCH Dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_TYPE, set_forecast_type, KEY_LOC, KEY_YEAR, KEY_TARGET,
    CROP_CALENDAR_DATES
)
cybench.config.FORECAST_LEAD_TIME = "0-days"

from cybench.process.alignment_patch import patch_alignment
patch_alignment()

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

# Add paths for custom modules
sys.path.append('../process/')
sys.path.append('../architectures/')

from loadData import calculate_fixed_split, DailyCYBenchSeqDataModule
from validateModel import print_metrics_table

# Import all model classes so Lightning can find them when loading checkpoint
from linearLayer import XLinearYieldModel
from tstLayer import PatchTSTModel
from wfan_layer import WFANWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MC Dropout Helper Functions
# =============================================================================

def enable_dropout(model: torch.nn.Module) -> torch.nn.Module:
    """
    Enable dropout layers during inference while keeping batch norm in eval mode.

    This is critical for MC Dropout: we want stochastic forward passes (dropout active)
    but don't want to update batch normalization running statistics.

    Args:
        model: PyTorch model with dropout layers

    Returns:
        Model with dropout enabled but batch norm frozen
    """
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            module.eval()
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)):
            module.training = True
    return model


def disable_dropout(model: torch.nn.Module) -> torch.nn.Module:
    """Disable dropout and set model to eval mode."""
    model.eval()
    return model


# =============================================================================
# Checkpoint Management Functions
# =============================================================================

def find_checkpoint_file(checkpoint_dir: str) -> Optional[str]:
    """
    Find the checkpoint file in the given directory.

    Looks for .ckpt files and returns the best one (based on validation loss).
    If multiple checkpoints exist, returns the one with lowest val_loss.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the checkpoint file or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    # Find all .ckpt files
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        # Check for subdirectories with checkpoints
        for subdir in checkpoint_dir.iterdir():
            if subdir.is_dir():
                ckpt_files.extend(list(subdir.glob("*.ckpt")))

    if not ckpt_files:
        logger.error(f"No checkpoint files found in {checkpoint_dir}")
        return None

    if len(ckpt_files) == 1:
        logger.info(f"Found single checkpoint: {ckpt_files[0]}")
        return str(ckpt_files[0])

    # Multiple checkpoints - find the best one based on val_loss in filename
    # Expected format: {model_name}_..._val_{loss:.4f}.ckpt
    best_ckpt = None
    best_val_loss = float('inf')

    for ckpt in ckpt_files:
        # Try to extract val_loss from filename
        if 'val_' in ckpt.name:
            parts = ckpt.name.split('val_')
            if len(parts) > 1:
                loss_str = parts[1].split('.ckpt')[0].split('_')[0]
                try:
                    val_loss = float(loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt = ckpt
                except ValueError:
                    pass

    if best_ckpt is None:
        # Fall back to the first checkpoint if no val_loss found
        best_ckpt = ckpt_files[0]

    logger.info(f"Found multiple checkpoints, selected: {best_ckpt}")
    return str(best_ckpt)


def get_checkpoint_dir(base_dir: str, model_name: str, country: str, crop: str) -> str:
    """Construct the checkpoint directory path based on model name, country, and crop."""
    model_dir = f"yield-{model_name.lower()}"
    return str(Path(base_dir) / model_dir / country / crop)


def load_model_from_checkpoint(checkpoint_path: str, model_name: str):
    """Load model from checkpoint with fallback logic."""
    try:
        model = WFANWrapper.load_from_checkpoint(checkpoint_path, map_location='cpu')
        print(f"[Model] Successfully loaded model: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"[Model] Failed to load as WFANWrapper: {e}")
        print(f"[Model] Trying to load as base model...")
        try:
            if model_name.lower() in ['xlinear', 'nlinear', 'dlinear', 'rlinear', 'olinear']:
                model = XLinearYieldModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
            else:
                model = PatchTSTModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
            print(f"[Model] Successfully loaded model: {type(model).__name__}")
            return model
        except Exception as e2:
            print(f"[Error] Failed to load model: {e2}")
            raise


def setup_model_and_data(
    checkpoint_dir: str,
    crop: str,
    country: str,
    model_name: str,
    test_years: Optional[int] = None,
    val_years: int = 2,
    verbose: bool = True
) -> Tuple:
    """
    Common setup for model loading and data preparation.

    Returns:
        Tuple containing: (model, dm_config, fixed_splits, checkpoint_path)
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SETUP: {crop.upper()} - {country} - {model_name.upper()}")
        print(f"{'=' * 70}\n")

    set_forecast_type("0-days")

    # Find checkpoint
    full_checkpoint_dir = get_checkpoint_dir(checkpoint_dir, model_name, country, crop)
    if verbose:
        print(f"[Path] Searching for checkpoints in: {full_checkpoint_dir}")

    checkpoint_path = find_checkpoint_file(full_checkpoint_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint file found in {full_checkpoint_dir}")

    if verbose:
        print(f"[Checkpoint] Found: {checkpoint_path}")

    # Load model
    if verbose:
        print(f"\n[Model] Loading model from checkpoint...")
    model = load_model_from_checkpoint(checkpoint_path, model_name)

    # Extract config
    hparams = model.hparams
    dm_config = hparams['config']

    if test_years is not None:
        dm_config.test_years = test_years

    if verbose:
        print(f"\n[Key Configuration]")
        print(f"  Crop: {dm_config.crop}, Country: {dm_config.country}")
        print(f"  Model: {dm_config.model_type}, Loss: {dm_config.loss_type}")
        print(f"  Test years: {dm_config.test_years}")

        # Check for WFAN
        if hasattr(model, 'K'):
            print(f"  WFAN: K={model.K}, lambda={model.lambda_coef}")

    # Setup data splits
    df_y, dfs_x = load_dfs_crop(dm_config.crop, [dm_config.country])
    if df_y is None or len(df_y) == 0:
        raise ValueError(f"No data for {dm_config.crop}-{dm_config.country}")

    ds = CYDataset(dm_config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))

    if verbose:
        print(f"\n[Data] Available years: {all_years}")

    fixed_splits = calculate_fixed_split(
        all_years, test_years=dm_config.test_years, val_years=val_years
    )

    if verbose:
        print(f"[Split] Train: {len(fixed_splits['train_years'])}, Val: {len(fixed_splits['val_years'])}, Test: {len(fixed_splits['test_years'])}")

    return model, dm_config, fixed_splits, checkpoint_path


def load_model_and_evaluate(
    checkpoint_dir: str,
    crop: str,
    country: str,
    model_name: str,
    test_years: Optional[int] = None,
    val_years: int = 2
) -> Dict:
    """Load a trained model and evaluate on test set."""
    model, dm_config, fixed_splits, _ = setup_model_and_data(
        checkpoint_dir, crop, country, model_name, test_years, val_years
    )

    # Create datamodule
    print(f"\n[DataModule] Creating datamodule...")
    dm = DailyCYBenchSeqDataModule(dm_config)
    dm.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    # Create trainer and evaluate
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, enable_progress_bar=True, enable_model_summary=False, logger=False,
    )

    print(f"\n{'=' * 70}")
    print(f"EVALUATION ON TEST SET")
    print(f"{'=' * 70}\n")

    test_results = trainer.test(model, dm)
    if not test_results:
        print("[Error] Test evaluation failed")
        return {}

    r = test_results[0]
    metrics = {
        'mse': r.get('test/mse'), 'mae': r.get('test/mae'),
        'rmse': r.get('test/rmse'), 'r2': r.get('test/r2'),
        'mape': r.get('test/mape'), 'smape': r.get('test/smape'),
        'nrmse': r.get('test/nrmse'),
    }

    if dm_config.loss_type == "mql":
        metrics.update({
            'crps': r.get('test/crps'), 'picp': r.get('test/picp'),
            'pinaw': r.get('test/pinaw'), 'pinball_avg': r.get('test/pinball_avg'),
            'winkler_score': r.get('test/winkler_score'),
            'mean_interval_width': r.get('test/mean_interval_width'),
        })

    if hasattr(model, '_spatiotemporal_metrics') and model._spatiotemporal_metrics:
        metrics.update(model._spatiotemporal_metrics)

    print_metrics_table(
        f"TEST RESULTS: {dm_config.crop}-{dm_config.country} ({dm_config.model_type.upper()})",
        metrics
    )

    if hasattr(model, '_test_results_per_year') and model._test_results_per_year:
        print(f"\n[Per-Year Metrics]")
        for year in sorted(fixed_splits['test_years']):
            print(f"Year {year}:", end="")
            for metric in ['nrmse', 'mape', 'r2']:
                key = f'{metric}_{year}'
                if key in model._test_results_per_year:
                    print(f" {metric.upper()}={model._test_results_per_year[key]:.4f}", end="")
            print()

    return metrics


# =============================================================================
# Prediction Generation Functions
# =============================================================================

def generate_and_save_predictions(
    model, dm, country: str, data_type: str, device: str
) -> List[Dict]:
    """Generate predictions for a given data split."""
    model.eval().to(device)

    # Get appropriate dataloader
    dataloader = (
        dm.train_dataloader() if data_type == 'train' else
        dm.val_dataloader() if data_type == 'val' else
        dm.test_dataloader()
    )

    y_mean, y_std = dm.y_mean, dm.y_std
    predictions = []

    # For WFAN + MQL, use base_model directly (wrapper adds scalar incompatible with quantiles)
    is_wfan = isinstance(model, WFANWrapper)
    has_mql_config = hasattr(model, 'config') and hasattr(model.config, 'loss_type')
    is_mql = has_mql_config and model.config.loss_type == 'mql'
    has_base_model = hasattr(model, 'base_model')

    if is_wfan and is_mql and has_base_model:
        model_to_use = model.base_model
        print(f"[Model] Using base_model (WFANWrapper + MQL)")
    else:
        model_to_use = model

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                x_ts = batch[0].to(device)
                x_static = batch[1].to(device)
                y_z = batch[2]
                adm_ids = batch[4]

                # Get validity mask safely
                validity_mask = batch[7].to(device) if len(batch) > 7 and batch[7] is not None else None

                # CRITICAL: Replace NaN with zeros before forward pass
                x_ts = torch.nan_to_num(x_ts, nan=0.0)
                x_static = torch.nan_to_num(x_static, nan=0.0)

                outputs_z = model_to_use(x_ts, x_static, observed_mask=validity_mask)

                # Handle 1D output (MSE model)
                if outputs_z.dim() == 1:
                    outputs_z = outputs_z.unsqueeze(-1)

                # Denormalize
                outputs_orig = outputs_z * y_std + y_mean
                y_orig = y_z * y_std + y_mean

                # Extract quantiles
                if outputs_orig.shape[1] >= 3:
                    pred_q50 = outputs_orig[:, 0].cpu().numpy()
                    pred_q10 = outputs_orig[:, 1].cpu().numpy()
                    pred_q90 = outputs_orig[:, 2].cpu().numpy()
                else:
                    pred_val = outputs_orig[:, 0].cpu().numpy()
                    pred_q10 = pred_q50 = pred_q90 = pred_val

            except Exception as e:
                print(f"[Error] Batch {batch_idx} failed: {e}")
                traceback.print_exc()
                continue

            actual_yields = y_orig.cpu().numpy()
            batch_amd_ids = adm_ids.cpu().numpy() if isinstance(adm_ids, torch.Tensor) else np.array(adm_ids)

            for i in range(len(batch_amd_ids)):
                predictions.append({
                    'country': country,
                    'amd_id': str(batch_amd_ids[i]),
                    'act_yield': float(actual_yields[i]),
                    'pred_yield_q0.1': float(pred_q10[i]),
                    'pred_yield_q0.5': float(pred_q50[i]),
                    'pred_yield_q0.9': float(pred_q90[i]),
                    'data_type': data_type
                })

    print(f"[Generated] {len(predictions)} predictions for {data_type}")
    return predictions


def generate_mc_dropout_predictions(
    model, dm, country: str, data_type: str, device: str, k_mc_samples: int
) -> List[Dict]:
    """
    Generate MC Dropout predictions using native implementation.

    Args:
        model: The trained model
        dm: DataModule with the dataloader
        country: Country code
        data_type: One of 'train', 'val', 'test'
        device: Device to run predictions on
        k_mc_samples: Number of MC dropout samples

    Returns:
        List of dictionaries containing MC dropout prediction data
    """
    # Get appropriate dataloader
    dataloader = (
        dm.train_dataloader() if data_type == 'train' else
        dm.val_dataloader() if data_type == 'val' else
        dm.test_dataloader()
    )

    y_mean, y_std = dm.y_mean, dm.y_std

    # For WFAN + MQL, use base_model directly
    is_wfan = isinstance(model, WFANWrapper)
    has_mql_config = hasattr(model, 'config') and hasattr(model.config, 'loss_type')
    is_mql = has_mql_config and model.config.loss_type == 'mql'
    has_base_model = hasattr(model, 'base_model')

    if is_wfan and is_mql and has_base_model:
        model_to_use = model.base_model
    else:
        model_to_use = model

    predictions = []
    print(f"[MC Dropout] Using native implementation with {k_mc_samples} samples")

    # Move model to device first, then enable dropout
    model_to_use = model_to_use.to(device)
    model_to_use = enable_dropout(model_to_use)

    for batch_idx, batch in enumerate(dataloader):
        try:
            x_ts = batch[0].to(device)
            x_static = batch[1].to(device)
            y_z = batch[2]
            adm_ids = batch[4]

            validity_mask = batch[7].to(device) if len(batch) > 7 and batch[7] is not None else None

            x_ts = torch.nan_to_num(x_ts, nan=0.0)
            x_static = torch.nan_to_num(x_static, nan=0.0)

            mc_samples = []

            for _ in range(k_mc_samples):
                outputs_z = model_to_use(x_ts, x_static, observed_mask=validity_mask)
                if outputs_z.dim() == 1:
                    outputs_z = outputs_z.unsqueeze(-1)

                pred_z = outputs_z[:, 0] if outputs_z.shape[1] >= 1 else outputs_z.squeeze()
                pred_orig = pred_z * y_std + y_mean
                mc_samples.append(pred_orig.detach().cpu().numpy())

            mc_samples = np.stack(mc_samples, axis=0)
            mc_samples_list = mc_samples.tolist()

            y_orig = y_z * y_std + y_mean
            actual_yields = y_orig.cpu().numpy()
            batch_amd_ids = adm_ids.cpu().numpy() if isinstance(adm_ids, torch.Tensor) else np.array(adm_ids)

            for i in range(len(batch_amd_ids)):
                predictions.append({
                    'country': country,
                    'amd_id': str(batch_amd_ids[i]),
                    'act_yield': float(actual_yields[i]),
                    'pred_yield_q0.1': [sample[i] for sample in mc_samples_list],
                    'pred_yield_q0.5': [sample[i] for sample in mc_samples_list],
                    'pred_yield_q0.9': [sample[i] for sample in mc_samples_list],
                    'data_type': data_type
                })
        except Exception as e:
            print(f"[Error] Batch {batch_idx} failed: {e}")
            traceback.print_exc()
            continue

    # Disable dropout after MC sampling
    model_to_use = disable_dropout(model_to_use)

    print(f"[MC Dropout] Generated {len(predictions)} predictions for {data_type}")
    return predictions


def generate_predictions(
    checkpoint_dir: str,
    crop: str,
    country: str,
    model_name: str,
    test_years: Optional[int] = None,
    val_years: int = 2,
    save_to_checkpoint_dir: bool = False,
    k_mc_dropout: Optional[int] = None
):
    """
    Generate and save predictions for train, val, and test sets.

    Args:
        k_mc_dropout: If provided (int), performs MC Dropout with K samples.
                      Otherwise, performs standard deterministic prediction.
    """
    print(f"\n{'=' * 70}")
    if k_mc_dropout:
        print(f"GENERATING MC DROPOUT PREDICTIONS (K={k_mc_dropout})")
    else:
        print(f"GENERATING PREDICTIONS")
    print(f"{'=' * 70}\n")

    model, dm_config, fixed_splits, checkpoint_path = setup_model_and_data(
        checkpoint_dir, crop, country, model_name, test_years, val_years, verbose=True
    )

    # Create datamodule
    dm = DailyCYBenchSeqDataModule(dm_config)
    dm.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using {device}\n")

    # Generate predictions for each split
    all_predictions = []

    # Choose prediction function based on MC Dropout
    pred_func = (
        generate_mc_dropout_predictions if k_mc_dropout
        else generate_and_save_predictions
    )

    for data_type in ['train', 'val', 'test']:
        if k_mc_dropout:
            split_predictions = pred_func(
                model=model, dm=dm, country=country, data_type=data_type,
                device=device, k_mc_samples=k_mc_dropout
            )
        else:
            split_predictions = pred_func(
                model=model, dm=dm, country=country, data_type=data_type, device=device
            )
        all_predictions.extend(split_predictions)

    # Save predictions
    if save_to_checkpoint_dir:
        output_dir = Path(checkpoint_path).parent / "predictions"
    else:
        output_dir = Path.cwd() / "predictions"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Different filename for MC Dropout
    if k_mc_dropout:
        output_path = output_dir / f"mc_predictions.csv"
    else:
        output_path = output_dir / f"{crop}_{country}_predictions.csv"

    df = pd.DataFrame(all_predictions)
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[Saved] {len(df)} predictions to: {output_path}")
    if k_mc_dropout:
        print(f"[MC Dropout] Each pred_yield column contains {k_mc_dropout} samples")
    print(f"{'=' * 70}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Load trained model and evaluate on test set or generate predictions")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Base directory containing model checkpoints')
    parser.add_argument('--crop', type=str, required=True, choices=['maize', 'wheat'],
                        help='Crop type')
    parser.add_argument('--country', type=str, required=True,
                        help='Country code (e.g., NL, DE, FR)')
    parser.add_argument('--model_name', type=str, required=True, choices=['xlinear', 'patchtst'],
                        help='Model architecture')
    parser.add_argument('--test_years', type=int, default=None,
                        help='Number of years for test set')
    parser.add_argument('--val_years', type=int, default=2,
                        help='Number of years for validation set')
    parser.add_argument('--generate_prediction', action='store_true',
                        help='Generate deterministic predictions file')
    parser.add_argument('--k_mc_dropout', type=int, default=None,
                        help='Number of MC Dropout samples. Generates mc_predictions.csv.')

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # Always evaluate first when generating predictions
    if args.generate_prediction or args.k_mc_dropout:
        print(f"\n{'=' * 70}")
        print("STEP 1: MODEL EVALUATION")
        print(f"{'=' * 70}\n")
        load_model_and_evaluate(
            checkpoint_dir=args.checkpoint_dir,
            crop=args.crop,
            country=args.country,
            model_name=args.model_name,
            test_years=args.test_years,
            val_years=args.val_years
        )

        print(f"\n{'=' * 70}")
        print("STEP 2: GENERATING PREDICTIONS")
        print(f"{'=' * 70}\n")

        # Generate deterministic predictions if requested
        if args.generate_prediction:
            generate_predictions(
                checkpoint_dir=args.checkpoint_dir,
                crop=args.crop,
                country=args.country,
                model_name=args.model_name,
                test_years=args.test_years,
                val_years=args.val_years,
                save_to_checkpoint_dir=True,
                k_mc_dropout=None  # Deterministic
            )

        # Generate MC Dropout predictions if requested
        if args.k_mc_dropout:
            generate_predictions(
                checkpoint_dir=args.checkpoint_dir,
                crop=args.crop,
                country=args.country,
                model_name=args.model_name,
                test_years=args.test_years,
                val_years=args.val_years,
                save_to_checkpoint_dir=True,
                k_mc_dropout=args.k_mc_dropout
            )

        print(f"\n{'=' * 70}")
        print("PREDICTION GENERATION COMPLETE")
        generated = []
        if args.generate_prediction:
            generated.append(f"- {args.crop}_{args.country}_predictions.csv (deterministic)")
        if args.k_mc_dropout:
            generated.append(f"- mc_predictions.csv (MC Dropout, K={args.k_mc_dropout})")
        for g in generated:
            print(g)
        print(f"{'=' * 70}\n")
    else:
        # Just evaluate
        load_model_and_evaluate(
            checkpoint_dir=args.checkpoint_dir,
            crop=args.crop,
            country=args.country,
            model_name=args.model_name,
            test_years=args.test_years,
            val_years=args.val_years
        )
        print(f"{'=' * 70}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
