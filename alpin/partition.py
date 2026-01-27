"""Optimal partition solver using ruptures Pelt algorithm.

This module provides the optimal partition function that solves:
    argmin_A [R(y,A) + β|A|]

where:
- R(y,A) is the empirical quadratic risk (l2 cost)
- β is the penalty parameter controlling the number of segments
- |A| is the number of segments in partition A
"""

import numpy as np
import ruptures as rpt


def solve_optimal_partition(signal: np.ndarray, beta: float) -> list[int]:
    """Solve optimal partition problem using Pelt algorithm with l2 cost.

    Finds the optimal segmentation of a signal by minimizing the penalized
    empirical quadratic risk:
        argmin_A [R(y,A) + β|A|]

    Uses the Pelt algorithm from ruptures library with:
    - model="l2": Quadratic loss (sum of squared residuals)
    - penalty=beta: Linear penalty on number of segments
    - min_size=2: Minimum segment length
    - jump=1: Check every index as potential changepoint

    Args:
        signal: Input signal as 1D numpy array. Must have length >= 20.
        beta: Penalty parameter (β) controlling segmentation granularity.
              Higher values produce fewer segments.

    Returns:
        List of changepoint indices (0-indexed positions where segments change).
        Empty list if no changepoints are detected (signal is homogeneous).
        Example: [50, 100] means changepoints at indices 50 and 100.

    Raises:
        ValueError: If signal length is less than 20.

    Examples:
        >>> import numpy as np
        >>> # 3-segment signal: zeros, ones*5, zeros
        >>> signal = np.concatenate([np.zeros(50), np.ones(50)*5, np.zeros(50)])
        >>> cps = solve_optimal_partition(signal, beta=10.0)
        >>> len(cps)
        2
        >>> 45 < cps[0] < 55  # First changepoint near index 50
        True
        >>> 95 < cps[1] < 105  # Second changepoint near index 100
        True

        >>> # Constant signal with high penalty: no changepoints
        >>> signal = np.ones(100) * 3.0
        >>> solve_optimal_partition(signal, beta=100.0)
        []
    """
    if len(signal) < 20:
        raise ValueError(
            f"Signal too short: length {len(signal)} < 20. "
            f"Minimum signal length is 20 samples."
        )

    # Initialize Pelt algorithm with l2 cost model
    # min_size=2: prevents trivial single-sample segments
    # jump=1: checks every index (no subsampling)
    algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)

    # Predict changepoints with penalty parameter
    # Result format: [cp1, cp2, ..., signal_length]
    result = algo.predict(pen=beta)

    # Remove last element (signal length) to get actual changepoints
    # Empty case: result=[n] -> changepoints=[]
    changepoints = result[:-1]

    return changepoints
