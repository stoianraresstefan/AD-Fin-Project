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
    """
    grid search over beta values using K-fold cross-validation to evaluate changepoint detection performance.

    Output: pd.DataFrame - results with columns beta, fold, precision, recall, hausdorff_distance, annotation_error, rand_index
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
    """
    checks robustness to noise by generating synthetic signals at different noise levels and evaluating detection with fixed beta.
   
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
