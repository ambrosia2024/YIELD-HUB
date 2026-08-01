# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: Contrastive pre-training for crop yield prediction with multi-country data.
             Implements self-supervised contrastive learning with hierarchical positive
             sampling and AEZ auxiliary classification.

Features:
- Multi-country data loading with per-country temporal splits
- Hierarchical positive sampling (hard/soft/weak positives)
- InfoNCE contrastive loss with temperature scaling
- AEZ auxiliary classification
- XLinear and PatchTST encoder backends
- k-NN evaluation for non-parametric assessment
- Adapter architecture ready for fine-tuning
- Same evaluation metrics and logging as baseline scripts

Usage:
    # Pre-train with XLinear on all countries
    python pretrainBaselines.py --crop maize --countries all --model_type xlinear \\
        --epochs 100 --batch_size 32 --contrastive_temp 0.07

    # Pre-train with PatchTST on specific countries
    python pretrainBaselines.py --crop maize --countries BE NL DE --model_type patchtst \\
        --epochs 50 --batch_size 16 --aez_loss_weight 0.1

    # Quick test run
    python pretrainBaselines.py --crop maize --countries NL --model_type xlinear \\
        --epochs 2 --test_years 2

Python version: 3.12.0
--------------------
"""

import os
import sys
import random
import argparse
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Union

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# CY-BENCH dependencies
import cybench.config
from cybench.config import (
    LOCATION_PROPERTIES, SOIL_PROPERTIES,
    KEY_LOC, KEY_YEAR, KEY_TARGET
)

# Custom functions and classes
sys.path.append('../../process/')
sys.path.append('../../architectures/')
sys.path.append('../../models/')

from helpers import generate_checkpoint_name
from validateModel import print_metrics_table
from loadData import calculate_fixed_split
from contrastiveDataModule import (
    ContrastiveModelConfig,
    MultiCountryContrastiveDataModule
)
from knn_evaluation import evaluate_knn_with_model, KNNEvaluator
from contrastiveModel import (
    ContrastivePretrainModel,
    ContrastiveFinetuneModel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set precision
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision('high')
        logger.info(f"Enabled high matmul precision (GPU capability {capability})")
    else:
        logger.info(f"Keeping default matmul precision (GPU capability {capability} < 8.0)")
else:
    logger.info("Running on CPU, matmul precision setting has no effect")


def get_available_countries(crop: str) -> List[str]:
    """Get list of available countries for a crop."""
    if crop == "maize":
        return ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR',
                'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE']
    else:  # wheat
        return ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
                'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL',
                'PL', 'PT', 'RO', 'SE']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Contrastive Pre-training for Crop Yield Prediction"
    )

    # Basic arguments
    parser.add_argument('--crop', default="maize", choices=['maize', 'wheat'])
    parser.add_argument('--countries', nargs='+', default=['all'],
                        help='Countries to process (default: all). '
                             'Specify space-separated codes: BE NL DE or use "all"')
    parser.add_argument('--model_type', default='xlinear', choices=['xlinear', 'patchtst'],
                        help='Encoder backbone architecture')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=42)

    # Data arguments
    parser.add_argument('--aggregation', default='dekad',
                        choices=['daily', 'weekly', 'dekad'])
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of season data to use (1.0 = full season)')
    parser.add_argument('--lag_years', type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument('--test_years', type=int, default=3,
                        help='Number of years for test set (default: 3)')
    parser.add_argument('--val_years', type=int, default=2,
                        help='Number of years for validation set (default: 2)')

    # Feature arguments (core features for pre-training)
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--use_gdd', action='store_true')
    parser.add_argument('--use_heat_stress_days', action='store_true')
    parser.add_argument('--use_rue', action='store_true')
    parser.add_argument('--use_farquhar', action='store_true')
    parser.add_argument('--use_cwb_feature', action='store_true')
    parser.add_argument('--drop_tavg', action='store_true')

    # Multi-year summaries (extended features, for fine-tuning only)
    parser.add_argument('--multi_year_summaries', action='store_true',
                        help='Enable multi-year summaries (extended feature, fine-tuning only)')
    parser.add_argument('--multi_year_window', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--multi_year_features', nargs='+', default=['weather'],
                        choices=['weather', 'remote_sensing', 'phenology', 'all'])

    # Exponential weighting (extended feature, fine-tuning only)
    parser.add_argument('--use_exponential_weighting', action='store_true')
    parser.add_argument('--exponential_tau', type=float, default=10.0)

    # Contrastive learning arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension (default: 128)')
    parser.add_argument('--contrastive_temp', type=float, default=0.07,
                        help='Temperature for InfoNCE loss (default: 0.07)')
    parser.add_argument('--aez_loss_weight', type=float, default=0.1,
                        help='Weight for AEZ auxiliary loss (default: 0.1)')
    parser.add_argument('--num_aez_classes', type=int, default=50,
                        help='Number of AEZ classes (default: 50, auto-detected if available)')

    # Pair sampling arguments (for pre-computation)
    parser.add_argument('--num_hard_positives', type=int, default=2)
    parser.add_argument('--num_soft_positives', type=int, default=2)
    parser.add_argument('--num_weak_positives', type=int, default=2)
    parser.add_argument('--num_negatives', type=int, default=4)

    # Pre-computed file paths
    parser.add_argument('--aez_lookup_path', default=None,
                        help='Path to pre-computed AEZ lookup CSV')
    parser.add_argument('--pairs_cache_path', default=None,
                        help='Path to pre-computed contrastive pairs (base path without _split_suffix)')

    # k-NN evaluation arguments
    parser.add_argument('--run_knn_eval', action='store_true',
                        help='Run k-NN evaluation after pre-training')
    parser.add_argument('--knn_k', type=int, default=5,
                        help='Number of neighbors for k-NN evaluation (default: 5)')
    parser.add_argument('--knn_metrics', nargs='+', default=['euclidean_inverse', 'cosine_inverse'],
                        choices=['euclidean_uniform', 'euclidean_inverse', 'euclidean_gaussian',
                                'cosine_uniform', 'cosine_inverse', 'cosine_gaussian'])

    # Checkpointing and logging
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-contrastive',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', default='checkpoints-contrastive/results',
                        help='Directory to save CSV results')
    parser.add_argument('--wandb_project', default=None,
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', default=None,
                        help='WandB run name')
    parser.add_argument('--run_id', default=None,
                        help='Custom run ID for tracking')

    # Worker settings
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers (default: auto-calculated)')

    # Mode
    parser.add_argument('--mode', default='pretrain', choices=['pretrain', 'finetune'],
                        help='Training mode (default: pretrain)')

    return parser.parse_args()


def find_latest_cache_file(cache_dir: str, pattern: str) -> Optional[str]:
    """Find the latest file matching a pattern in a directory."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    files = list(cache_path.glob(pattern))
    if not files:
        return None

    return str(max(files, key=os.path.getctime))


def main():
    args = parse_args()

    # Handle 'all' countries
    if 'all' in args.countries:
        args.countries = get_available_countries(args.crop)

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id if args.run_id else str(uuid.uuid4())[:8]

    # Set num_workers
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 1
        args.num_workers = min(cpu_count // 4, 8)

    print(f"\n{'=' * 70}")
    print(f"CONTRASTIVE PRE-TRAINING | {args.crop} | {args.model_type.upper()}")
    print(f"{'=' * 70}")
    print(f"Countries ({len(args.countries)}): {args.countries}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Contrastive temp: {args.contrastive_temp}")
    print(f"AEZ loss weight: {args.aez_loss_weight}")
    print(f"Test years: {args.test_years}, Val years: {args.val_years}")
    print(f"Run ID: {run_id}")
    print(f"{'=' * 70}\n")

    # Find pre-computed files if not specified
    if args.aez_lookup_path is None:
        aez_cache_dir = Path(__file__).parent.parent.parent / 'pre-compute' / 'aez_cache'
        args.aez_lookup_path = find_latest_cache_file(str(aez_cache_dir), f'{args.crop}_aez_*_lookup.csv')
        if args.aez_lookup_path:
            print(f"[Auto-detected] AEZ lookup: {args.aez_lookup_path}")
        else:
            print(f"[Warning] No AEZ lookup found. Please run precompute_aez.py first.")

    if args.pairs_cache_path is None and args.mode == 'pretrain':
        pairs_cache_dir = Path(__file__).parent.parent.parent / 'pre-compute' / 'pairs_cache'
        args.pairs_cache_path = find_latest_cache_file(str(pairs_cache_dir), f'{args.crop}_pairs_*')
        if args.pairs_cache_path:
            print(f"[Auto-detected] Pairs cache: {args.pairs_cache_path}")
        else:
            print(f"[Warning] No pairs cache found. Please run precompute_contrastive_pairs.py first.")

    # Create configuration
    config = ContrastiveModelConfig(
        crop=args.crop,
        countries=args.countries,
        aggregation=args.aggregation,
        data_fraction=args.data_fraction,
        lag_years=args.lag_years,
        test_years=args.test_years,
        val_years=args.val_years,
        use_gdd=args.use_gdd,
        use_heat_stress_days=args.use_heat_stress_days,
        use_rue=args.use_rue,
        use_farquhar=args.use_farquhar,
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        include_spatial_features=args.include_spatial_features,
        multi_year_summaries=args.multi_year_summaries,
        multi_year_window=args.multi_year_window,
        multi_year_features=args.multi_year_features,
        use_exponential_weighting=args.use_exponential_weighting,
        exponential_tau=args.exponential_tau,
        use_sota_features=args.use_sota_features,
        aez_lookup_path=args.aez_lookup_path,
        pairs_cache_path=args.pairs_cache_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mode=args.mode,
        num_hard_positives=args.num_hard_positives,
        num_soft_positives=args.num_soft_positives,
        num_weak_positives=args.num_weak_positives,
        num_negatives=args.num_negatives,
    )

    # Create data module
    print("\n[Data Module] Setting up data...")
    dm = MultiCountryContrastiveDataModule(config)
    dm.setup()

    # Create model
    print(f"\n[Model] Creating {args.model_type.upper()} encoder model...")
    model = ContrastivePretrainModel(
        config=config,
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        num_aez_classes=args.num_aez_classes,
        contrastive_temp=args.contrastive_temp,
        aez_loss_weight=args.aez_loss_weight,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # Setup logging
    try:
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-Contrastive"
        base_run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model_type}-{args.crop}-contrastive"
        wandb_run_name = f"{base_run_name}-{run_id}"
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args),
            group=f"{args.crop}-contrastive"
        )
        loggers = [wandb_logger]
    except Exception as e:
        print(f"[WandB Warning] Could not initialize WandB logger: {e}")
        loggers = [CSVLogger("logs/", name="cybench-contrastive")]

    # Setup callbacks
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val/total_loss', patience=10, mode='min', verbose=True),
        ModelCheckpoint(
            monitor='val/total_loss',
            save_top_k=1,
            mode='min',
            dirpath=args.save_checkpoint_dir,
            filename=f'{args.model_type}_{args.crop}_contrastive_{{epoch:02d}}_{{val_total_loss:.4f}}_runid:{run_id}',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Train model
    print("\n[Training] Starting contrastive pre-training...")
    trainer.fit(model, dm)

    # Evaluate
    print("\n[Evaluation] Evaluating on validation set...")
    val_results = trainer.validate(model, dm.val_dataloader())

    print("\n[Evaluation] Evaluating on test set...")
    test_results = trainer.test(model, dm.test_dataloader())

    # Print final results
    if test_results:
        r = test_results[0]
        final_metrics = {
            'test_aez_loss': r.get('test/aez_loss'),
            'test_aez_acc': r.get('test/aez_acc'),
        }
        print_metrics_table("FINAL RESULTS", final_metrics)

    # k-NN evaluation (optional)
    if args.run_knn_eval:
        print("\n[k-NN Evaluation] Running k-NN evaluation...")
        knn_results = evaluate_knn_with_model(
            model=model,
            train_dataloader=dm.train_dataloader(),
            test_dataloader=dm.test_dataloader(),
            k=args.knn_k,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            denormalize_fn=lambda x: x * dm.y_std + dm.y_mean  # Denormalize
        )

        print("\n[k-NN Results]")
        for key, metrics in knn_results.items():
            print(f"  {key}:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    R²: {metrics['r2']:.4f}")

        # Log k-NN results to WandB
        for key, metrics in knn_results.items():
            for metric_name, metric_value in metrics.items():
                if hasattr(logger, 'log_metrics'):
                    logger.log_metrics({f'knn_{key}_{metric_name}': metric_value})

    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop} - {args.model_type}")
    print(f"Run ID: {run_id}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
