"""Optimal partition solver using ruptures Pelt algorithm.

This module provides the optimal partition function that solves:
    argmin_A [R(y,A) + beta|A|]

where:
- R(y,A) is the empirical quadratic risk (l2 cost)
- beta is the penalty parameter controlling the number of segments
- |A| is the number of segments in partition A
"""

import numpy as np
import ruptures as rpt


def solve_optimal_partition(signal: np.ndarray, beta: float) -> list[int]:
    """
    Finds the optimal segmentation by minimizing penalized quadratic risk using the Pelt algorithm.
    Computes argmin_A [R(y,A) + beta|A|] where R(y,A) is l2 cost and beta controls the number of segments.

    Input: signal (np.ndarray) - 1D signal with minimum length 20, beta (float) - penalty parameter controlling granularity
    Output: list of int - changepoint indices where segments change, empty list if no changepoints detected
    """
    if len(signal) < 20:
        raise ValueError(
            f"Signal too short: length {len(signal)} < 20. "
            f"Minimum signal length is 20 samples."
        )

    # Initialize Pelt algorithm with l2 cost model
    # min_size=2: prevents trivial single-sample segments
    # jump=3: checks every 3rd index for 3x speedup
    algo = rpt.Pelt(model="l2", min_size=2, jump=3).fit(signal)

    # Predict changepoints with penalty parameter
    # Result format: [cp1, cp2, ..., signal_length]
    result = algo.predict(pen=beta)

    # Remove last element (signal length) to get actual changepoints
    # Empty case: result=[n] -> changepoints=[]
    changepoints = result[:-1]

    return changepoints
