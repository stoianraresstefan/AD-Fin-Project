"""T-test baseline changepoint detector for comparison with ALPIN."""

import numpy as np
from scipy.stats import ttest_ind
from typing import List


class TTestBaseline:
    """
    T-test baseline for changepoint detection.

    Uses a sliding window approach to detect changepoints by comparing
    the statistical difference between signal segments before and after
    each position using independent t-tests.

    Attributes:
        window_fraction: Fraction of signal length to use as window size (default: 0.05)
        confidence: Confidence level for significance testing (default: 0.95)
    """

    def __init__(self, window_fraction: float = 0.05, confidence: float = 0.95):
        """
        Initialize TTestBaseline detector.

        Args:
            window_fraction: Fraction of signal length to use as window size.
                           Controls sensitivity and computation (default: 0.05).
            confidence: Confidence level for t-test significance (default: 0.95).
                       Significance level = 1 - confidence.
        """
        self.window_fraction = window_fraction
        self.confidence = confidence

    def detect(self, signal: np.ndarray) -> List[int]:
        """
        Detect changepoints in the signal using sliding window t-test.

        Algorithm:
        1. Compute window size as window_fraction * signal length
        2. Slide a window across the signal
        3. At each position, perform t-test comparing left and right windows
        4. Mark changepoint if p-value < (1 - confidence)
        5. Merge nearby detections within window_size distance

        Args:
            signal: Input signal as 1D numpy array.

        Returns:
            Sorted list of detected changepoint indices.
        """
        signal = np.asarray(signal, dtype=float)
        n = len(signal)

        # Calculate window size
        window_size = max(1, int(self.window_fraction * n))
        alpha = 1.0 - self.confidence  # Significance level

        # Detect candidate changepoints
        candidates = []

        for i in range(window_size, n - window_size):
            left_window = signal[i - window_size : i]
            right_window = signal[i : i + window_size]

            # Perform t-test
            t_stat, p_val = ttest_ind(left_window, right_window)

            # Mark as changepoint if p-value is significant
            if p_val < alpha:
                candidates.append(i)

        # Merge nearby changepoints (within window_size distance)
        if not candidates:
            return []

        merged = []
        current_group = [candidates[0]]

        for cp in candidates[1:]:
            if cp - current_group[-1] <= window_size:
                current_group.append(cp)
            else:
                # Take the mean of the group as the changepoint
                merged.append(int(np.mean(current_group)))
                current_group = [cp]

        # Don't forget the last group
        if current_group:
            merged.append(int(np.mean(current_group)))

        # Return sorted, unique changepoints
        return sorted(list(set(merged)))
