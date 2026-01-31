"""T-test baseline changepoint detector for comparison with ALPIN."""

import numpy as np
from scipy.stats import ttest_ind
from typing import List


class TTestBaseline:

    def __init__(self, window_fraction: float = 0.05, confidence: float = 0.95):
        self.window_fraction = window_fraction
        self.confidence = confidence

    def detect(self, signal: np.ndarray) -> List[int]:
        """
        Detects changepoints using t-test comparison on sliding windows across the signal.
        Performs independent t-tests at each position and merges nearby candidates within window distance.

        Input: signal (np.ndarray) - 1D signal array
        Output: list of int - sorted list of detected changepoint indices
        """
        signal = np.asarray(signal, dtype=float)
        n = len(signal)

        # calculate window size
        window_size = max(1, int(self.window_fraction * n))
        alpha = 1.0 - self.confidence  # significance level


        candidates = []

        for i in range(window_size, n - window_size):
            left_window = signal[i - window_size : i]
            right_window = signal[i : i + window_size]

            # actual t-test
            result = ttest_ind(left_window, right_window)

            # mark changepoint if p-value is significant
            if result.pvalue < alpha:
                candidates.append(i)

        # Mergeing of nearby changepoints (within window_size distance)
        if not candidates:
            return []

        merged = []
        current_group = [candidates[0]]

        for cp in candidates[1:]:
            if cp - current_group[-1] <= window_size:
                current_group.append(cp)
            else:
                # mean of the group as the changepoint
                merged.append(int(np.mean(current_group)))
                current_group = [cp]

        if current_group:
            merged.append(int(np.mean(current_group)))

        return sorted(list(set(merged)))
