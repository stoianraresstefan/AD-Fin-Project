"""Risk calculation functions for ALPIN algorithm.

This module implements three core risk functions:
1. Empirical quadratic risk R(y,A) - sum of squared residuals
2. Penalized risk R_β(y,A) = R(y,A) + β|A|
3. Excess risk E(y,β) - difference between ground truth and optimal risk

These functions are used in the ALPIN algorithm to learn the optimal penalty
parameter β for change point detection.
"""

import numpy as np
from alpin.partition import solve_optimal_partition


def empirical_risk(signal: np.ndarray, partition: list[int]) -> float:
    """Calculate empirical quadratic risk R(y,A) for a given partition.

    Computes the sum of squared residuals across all segments:
        R(y,A) = Σ_{a∈A} Σ_{i∈a} (y_i - ȳ_a)²

    where:
    - y is the signal
    - A is the partition (set of segments)
    - ȳ_a is the mean of signal values in segment a

    Args:
        signal: Input signal as 1D numpy array of shape (n,).
        partition: List of changepoint indices (0-indexed).
                  Empty list means one segment covering entire signal.
                  Example: [50, 100] for signal of length 150 creates 3 segments:
                  [0:50], [50:100], [100:150].

    Returns:
        Empirical risk value (float >= 0). Returns 0.0 if signal is perfectly
        piecewise constant with the given partition.

    Examples:
        >>> import numpy as np
        >>> # Perfect piecewise constant signal
        >>> signal = np.array([0, 0, 0, 5, 5, 5])
        >>> partition = [3]  # changepoint at index 3
        >>> empirical_risk(signal, partition)
        0.0

        >>> # Noisy signal
        >>> signal = np.array([0, 1, 0, 5, 6, 5])
        >>> empirical_risk(signal, partition)
        2.0

        >>> # No changepoints (entire signal as one segment)
        >>> signal = np.array([1, 2, 3, 4, 5])
        >>> empirical_risk(signal, [])
        10.0
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
    """Calculate penalized risk R_β(y,A) = R(y,A) + β|A|.

    Combines empirical risk with a linear penalty on the number of segments:
        R_β(y,A) = R(y,A) + β|A|

    where:
    - R(y,A) is the empirical quadratic risk
    - β is the penalty parameter
    - |A| is the number of segments in partition A

    Args:
        signal: Input signal as 1D numpy array of shape (n,).
        partition: List of changepoint indices (0-indexed).
        beta: Penalty parameter (β >= 0). Higher values favor fewer segments.

    Returns:
        Penalized risk value (float >= 0).

    Examples:
        >>> import numpy as np
        >>> signal = np.array([0, 0, 0, 5, 5, 5])
        >>> partition = [3]  # 2 segments
        >>> penalized_risk(signal, partition, beta=10.0)
        20.0

        >>> # Higher penalty discourages more segments
        >>> penalized_risk(signal, [2, 3, 4], beta=10.0)  # 4 segments
        40.0
    """
    # Number of segments = number of changepoints + 1
    n_segments = len(partition) + 1

    # Compute empirical risk + penalty term
    emp_risk = empirical_risk(signal, partition)
    penalty_term = beta * n_segments

    return emp_risk + penalty_term


def excess_risk(signal: np.ndarray, ground_truth: list[int], beta: float) -> float:
    """Calculate excess penalized risk E(y,β).

    Measures how much worse the ground truth partition is compared to the
    optimal partition for a given β:
        E(y,β) = R_β(y,A^lab) - min_A R_β(y,A)

    where:
    - A^lab is the ground truth (labeled) partition
    - min_A R_β(y,A) is the penalized risk of the β-optimal partition

    This is the loss function minimized by the ALPIN algorithm to learn β.
    Lower values indicate better alignment between ground truth and optimal
    partition.

    Args:
        signal: Input signal as 1D numpy array of shape (n,).
        ground_truth: Ground truth partition (list of changepoint indices).
        beta: Penalty parameter (β >= 0).

    Returns:
        Excess risk value (float >= 0). Returns 0.0 if ground truth is
        optimal for the given β.

    Raises:
        ValueError: If signal is too short (< 20 samples) for partition solver.

    Examples:
        >>> import numpy as np
        >>> # Ground truth matches optimal for this β
        >>> signal = np.array([0]*50 + [5]*50)
        >>> ground_truth = [50]
        >>> excess_risk(signal, ground_truth, beta=10.0)
        0.0

        >>> # Ground truth has extra changepoints (suboptimal for high β)
        >>> ground_truth = [25, 50, 75]  # 4 segments
        >>> excess_risk(signal, ground_truth, beta=100.0) > 0
        True
    """
    # Compute penalized risk of ground truth partition
    ground_truth_risk = penalized_risk(signal, ground_truth, beta)

    # Find optimal partition for this β
    optimal_partition = solve_optimal_partition(signal, beta)

    # Compute penalized risk of optimal partition
    optimal_risk = penalized_risk(signal, optimal_partition, beta)

    # Excess risk is the difference
    excess = ground_truth_risk - optimal_risk

    return float(excess)
