"""Hyperparameter sweep utilities for ALPIN experiments.

This module provides functions to perform grid searches over hyperparameters
and analyze robustness to signal properties (e.g., noise levels).
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from alpin.partition import solve_optimal_partition
from alpin.data.synthetic import generate_signal
from alpin.metrics import evaluate_all


def sweep_beta(
    signals: List[np.ndarray],
    ground_truths: List[List[int]],
    beta_range: List[float],
    n_splits: int = 3,
) -> pd.DataFrame:
    """Grid search over beta values with K-fold cross-validation.

    For each beta value, evaluates changepoint detection performance using
    solve_optimal_partition directly (without model training). Returns metrics
    computed on test folds only.

    Parameters
    ----------
    signals : List[np.ndarray]
        List of signal arrays, each shape (n_samples,)
    ground_truths : List[List[int]]
        List of ground truth changepoint lists (0-indexed sample positions)
    beta_range : List[float]
        List of beta values to sweep over
    n_splits : int, default=3
        Number of folds for cross-validation

    Returns
    -------
    pd.DataFrame
        Results with columns: beta, fold, precision, recall, hausdorff_distance,
        annotation_error, rand_index. Shape: (n_betas * n_splits, 7)

    Examples
    --------
    >>> signals = [np.array([0,0,0,5,5,5]) for _ in range(10)]
    >>> truths = [[3] for _ in range(10)]
    >>> beta_range = [1.0, 10.0, 100.0]
    >>> results = sweep_beta(signals, truths, beta_range, n_splits=2)
    >>> results.shape[0]  # 3 betas * 2 splits = 6 rows
    6
    >>> set(results.columns)
    {'beta', 'fold', 'precision', 'recall', 'hausdorff_distance',
     'annotation_error', 'rand_index'}
    """
    indices = np.arange(len(signals))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for beta in beta_range:
        for fold_idx, (_, test_indices) in enumerate(kfold.split(indices)):
            # Evaluate on test fold
            for test_signal, test_truth in zip(
                [signals[i] for i in test_indices],
                [ground_truths[i] for i in test_indices],
            ):
                predicted = solve_optimal_partition(test_signal, beta)
                metrics = evaluate_all(
                    predicted,
                    test_truth,
                    len(test_signal),
                    tolerance=10,
                )

                results.append(
                    {
                        "beta": beta,
                        "fold": fold_idx,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "hausdorff_distance": metrics["hausdorff_distance"],
                        "annotation_error": metrics["annotation_error"],
                        "rand_index": metrics["rand_index"],
                    }
                )

    return pd.DataFrame(results)


def sweep_noise(
    n_signals: int = 20,
    n_samples: int = 100,
    noise_levels: List[float] = [0.5, 1.0, 2.0, 5.0],
    n_splits: int = 3,
    beta: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Study robustness to noise levels with fixed beta.

    Generates synthetic signals at different noise levels and evaluates
    changepoint detection using solve_optimal_partition with fixed beta.

    Parameters
    ----------
    n_signals : int, default=20
        Number of signals to generate per noise level
    n_samples : int, default=100
        Length of each signal
    noise_levels : List[float], default=[0.5, 1.0, 2.0, 5.0]
        List of noise standard deviations to test
    n_splits : int, default=3
        Number of CV folds
    beta : float, default=10.0
        Fixed penalty parameter for partition optimization
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Results with columns: noise_std, fold, precision, recall,
        hausdorff_distance, annotation_error, rand_index.
        Shape: (len(noise_levels) * n_splits, 7)

    Examples
    --------
    >>> results = sweep_noise(n_signals=10, n_samples=50,
    ...                      noise_levels=[0.5, 1.0], n_splits=2)
    >>> results.shape[0]  # 2 noise levels * 2 splits
    4
    >>> sorted(results['noise_std'].unique())
    [0.5, 1.0]
    """
    results = []

    for noise_std in noise_levels:
        # Generate all signals for this noise level
        signals = []
        truths = []

        for i in range(n_signals):
            signal, truth = generate_signal(
                n_samples=n_samples,
                noise_std=noise_std,
                seed=seed + i if seed is not None else None,
            )
            signals.append(signal)
            truths.append(truth)

        # K-fold evaluation
        indices = np.arange(len(signals))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (_, test_indices) in enumerate(kfold.split(indices)):
            for test_signal, test_truth in zip(
                [signals[i] for i in test_indices],
                [truths[i] for i in test_indices],
            ):
                predicted = solve_optimal_partition(test_signal, beta)
                metrics = evaluate_all(
                    predicted,
                    test_truth,
                    len(test_signal),
                    tolerance=10,
                )

                results.append(
                    {
                        "noise_std": noise_std,
                        "fold": fold_idx,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "hausdorff_distance": metrics["hausdorff_distance"],
                        "annotation_error": metrics["annotation_error"],
                        "rand_index": metrics["rand_index"],
                    }
                )

    return pd.DataFrame(results)
