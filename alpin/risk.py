"""Risk calculation functions for ALPIN algorithm.

This module implements three core risk functions:
1. Empirical quadratic risk R(y,A) - sum of squared residuals
2. Penalized risk R_beta(y,A) = R(y,A) + beta|A|
3. Excess risk E(y,beta) - difference between ground truth and optimal risk

These functions are used in the ALPIN algorithm to learn the optimal penalty
parameter beta for change point detection.
"""

import numpy as np
from alpin.partition import solve_optimal_partition


def empirical_risk(signal: np.ndarray, partition: list[int]) -> float:
    """Calculates empirical quadratic risk as the sum of squared residuals across all segments in the partition.

    Input: signal (np.ndarray) - 1D signal array, partition (list of int) - changepoint indices (empty list means one segment)
    Output: float - empirical risk value (0.0 if signal perfectly matches partition)
    """
    n = len(signal)

    # Create segment boundaries: [0, cp1, cp2, ..., n]
    boundaries = [0] + partition + [n]

    # Calculate risk as sum of squared residuals within each segment
    total_risk = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Extract segment data
        segment = signal[start:end]

        # Compute mean and sum of squared deviations
        segment_mean = np.mean(segment)
        squared_residuals = np.sum((segment - segment_mean) ** 2)

        total_risk += squared_residuals

    return float(total_risk)


def penalized_risk(signal: np.ndarray, partition: list[int], beta: float) -> float:
    """Combines empirical risk with a linear penalty on the number of segments to balance accuracy and complexity.

    Input: signal (np.ndarray) - 1D signal array, partition (list of int) - changepoint indices, beta (float) - penalty parameter (beta >= 0)
    Output: float - penalized risk value (sum of empirical risk + beta * number of segments)
    """
    # Number of segments = number of changepoints + 1
    n_segments = len(partition) + 1

    # Compute empirical risk + penalty term
    emp_risk = empirical_risk(signal, partition)
    penalty_term = beta * n_segments

    return emp_risk + penalty_term


def excess_risk(signal: np.ndarray, ground_truth: list[int], beta: float) -> float:
    """Measures how much worse the ground truth partition is compared to the optimal partition for a given penalty parameter.

    Input: signal (np.ndarray) - 1D signal array, ground_truth (list of int) - true changepoint indices, beta (float) - penalty parameter (beta >= 0)
    Output: float - excess risk value (0.0 if ground truth is optimal for this beta, positive otherwise)
    """
    # Compute penalized risk of ground truth partition
    ground_truth_risk = penalized_risk(signal, ground_truth, beta)

    # Find optimal partition for this beta
    optimal_partition = solve_optimal_partition(signal, beta)

    # Compute penalized risk of optimal partition
    optimal_risk = penalized_risk(signal, optimal_partition, beta)

    # Excess risk is the difference
    excess = ground_truth_risk - optimal_risk

    return float(excess)
