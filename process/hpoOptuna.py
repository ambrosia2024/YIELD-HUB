"""
--------------------
Author: XYZ
Description: Shared hyperparameter optimization utilities using Optuna. This script provides helper functions for hyperparameter optimization
that are used across different baseline training scripts
 
Python version: 3.12.0
--------------------.
"""

import csv
import os
from typing import Callable, Dict, Any, Optional
import optuna


def create_study(
    study_name: str,
    objective: str,
    storage: Optional[str] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
) -> optuna.Study:
    """Create an Optuna study with appropriate direction based on objective.

    Args:
        study_name: Name of the study
        objective: Optimization objective ('nrmse', 'r2', or 'multi')
        storage: Optuna storage URL for distributed optimization
        pruner: Optuna pruner instance

    Returns:
        Configured Optuna study
    """
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

    if objective == 'multi':
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            directions=['minimize', 'maximize'],  # Minimize nrmse, maximize r2
            pruner=pruner
        )
    elif objective == 'r2':
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='maximize',
            pruner=pruner
        )
    else:  # nrmse
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            pruner=pruner
        )


def print_best_results(study: optuna.Study, objective: str) -> None:
    """Print the best trial(s) from an Optuna study.

    Args:
        study: Completed Optuna study
        objective: Optimization objective ('nrmse', 'r2', or 'multi')
    """
    if objective == 'multi':
        if not study.best_trials:
            print("No completed trials found.")
            return
        print("Best trials (multi-objective):")
        best_nrmse_trial = min(study.best_trials, key=lambda t: t.values[0])
        best_r2_trial = max(study.best_trials, key=lambda t: t.values[1])

        print(f"\nBest NRMSE (Trial {best_nrmse_trial.number}):")
        print(f"  NRMSE: {best_nrmse_trial.values[0]:.4f}")
        print(f"  R²: {best_nrmse_trial.values[1]:.4f}")
        print(f"  Params: {best_nrmse_trial.params}")

        print(f"\nBest R² (Trial {best_r2_trial.number}):")
        print(f"  NRMSE: {best_r2_trial.values[0]:.4f}")
        print(f"  R²: {best_r2_trial.values[1]:.4f}")
        print(f"  Params: {best_r2_trial.params}")
    else:
        if not study.best_trial:
            print("No completed trials found.")
            return
        print(f"Best trial (Trial {study.best_trial.number}):")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")


def save_results_to_file(
    study: optuna.Study,
    hpo_results_file: str,
    model_type: str,
    crop: str,
    country: str,
    study_name: str,
    objective: str,
    timestamp: str,
) -> None:
    """Save Optuna study results to a text file.

    Args:
        study: Completed Optuna study
        hpo_results_file: Path to save the results
        model_type: Type of model being optimized
        crop: Crop name
        country: Country name
        study_name: Name of the study
        objective: Optimization objective ('nrmse', 'r2', or 'multi')
        timestamp: Timestamp string for the results
    """
    with open(hpo_results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"OPTUNA HPO RESULTS - {model_type.upper()} | {crop}-{country}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Study Name: {study_name}\n")
        f.write(f"Objective: {objective}\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        if objective == 'multi':
            f.write("BEST TRIALS (Multi-Objective):\n")
            f.write("-" * 70 + "\n")

            if study.best_trials:
                best_nrmse_trial = min(study.best_trials, key=lambda t: t.values[0])
                f.write(f"\nBest NRMSE (Trial {best_nrmse_trial.number}):\n")
                f.write(f"  NRMSE: {best_nrmse_trial.values[0]:.6f}\n")
                f.write(f"  R²: {best_nrmse_trial.values[1]:.6f}\n")
                f.write(f"  Hyperparameters:\n")
                for param, value in best_nrmse_trial.params.items():
                    f.write(f"    {param}: {value}\n")

                best_r2_trial = max(study.best_trials, key=lambda t: t.values[1])
                f.write(f"\nBest R² (Trial {best_r2_trial.number}):\n")
                f.write(f"  NRMSE: {best_r2_trial.values[0]:.6f}\n")
                f.write(f"  R²: {best_r2_trial.values[1]:.6f}\n")
                f.write(f"  Hyperparameters:\n")
                for param, value in best_r2_trial.params.items():
                    f.write(f"    {param}: {value}\n")
            else:
                f.write("No completed trials found.\n")
        else:
            f.write("BEST TRIAL:\n")
            f.write("-" * 70 + "\n")
            if study.best_trial:
                f.write(f"Trial Number: {study.best_trial.number}\n")
                f.write(f"Objective Value: {study.best_value:.6f}\n")
                f.write(f"Hyperparameters:\n")
                for param, value in study.best_params.items():
                    f.write(f"    {param}: {value}\n")
            else:
                f.write("No completed trials found.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL TRIALS\n")
        f.write("=" * 70 + "\n\n")

        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if objective == 'multi':
                    f.write(f"Trial {trial.number}: NRMSE={trial.values[0]:.6f}, R²={trial.values[1]:.6f}\n")
                else:
                    f.write(f"Trial {trial.number}: Value={trial.value:.6f}\n")
                f.write(f"  Params: {trial.params}\n\n")


def save_best_params_to_csv(
    study: optuna.Study,
    save_dir: str,
    objective: str,
) -> None:
    """Save best hyperparameters to CSV files.

    Args:
        study: Completed Optuna study
        save_dir: Directory to save CSV files
        objective: Optimization objective ('nrmse', 'r2', or 'multi')
    """
    if objective == 'multi':
        if not study.best_trials:
            print("[HPO] No completed trials found, skipping CSV export.")
            return
        csv_rmse_path = os.path.join(save_dir, 'optuna_rmse.csv')
        csv_r2_path = os.path.join(save_dir, 'optuna_r2.csv')

        best_rmse_trial = min(study.best_trials, key=lambda t: t.values[0])
        best_r2_trial = max(study.best_trials, key=lambda t: t.values[1])

        # Save RMSE-optimized hyperparameters
        with open(csv_rmse_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['hyperparameter', 'value'])
            for param, value in best_rmse_trial.params.items():
                writer.writerow([param, value])
            writer.writerow(['nrmse', best_rmse_trial.values[0]])

        # Save R2-optimized hyperparameters
        with open(csv_r2_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['hyperparameter', 'value'])
            for param, value in best_r2_trial.params.items():
                writer.writerow([param, value])
            writer.writerow(['r2', best_r2_trial.values[1]])
    else:
        # Single-objective optimization (nrmse or r2) - do not save CSV
        return

def run_hpo(
    objective: Callable[[optuna.Trial], Any],
    study_name: str,
    hpo_objective: str,
    n_trials: int,
    storage: Optional[str] = None,
    baseline_params: Optional[Dict[str, Any]] = None,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """Run Optuna hyperparameter optimization.

    Args:
        objective: Objective function that takes a trial and returns metric(s)
        study_name: Name of the study
        hpo_objective: Optimization objective ('nrmse', 'r2', or 'multi')
        n_trials: Number of trials to run
        storage: Optuna storage URL for distributed optimization
        baseline_params: Optional baseline configuration to enqueue
        show_progress_bar: Whether to show progress bar

    Returns:
        Completed Optuna study
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = create_study(study_name, hpo_objective, storage, pruner)

    print(f"\n{'=' * 70}")
    print(f"Starting Optuna optimization with {n_trials} trials...")
    print(f"{'=' * 70}\n")

    if baseline_params is not None:
        study.enqueue_trial(baseline_params)
        print(f"[HPO] Enqueued baseline configuration: {baseline_params}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    print(f"\n{'=' * 70}")
    print(f"HPO COMPLETED")
    print(f"{'=' * 70}\n")

    return study
