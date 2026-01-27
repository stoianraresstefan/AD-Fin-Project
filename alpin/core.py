"""Core ALPIN algorithm for learning optimal penalty parameter.

This module implements the ALPIN (Adaptive Learning of Penalty for INference)
algorithm that learns the optimal penalty parameter β from labeled training data
using L-BFGS optimization.

The algorithm follows the paper's approach (Section 3):
1. Warm start: Optimize β on a single randomly selected signal
2. Global optimization: Minimize average excess risk across all signals

Reference:
    T. Truong, C. Oriot, V. Lecomte. "Automatic labeling of
    piecewise-constant signals". EUSIPCO 2017.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from alpin.partition import solve_optimal_partition
from alpin.risk import excess_risk


class ALPIN:
    """ALPIN algorithm for learning optimal penalty parameter.

    ALPIN learns the optimal penalty parameter β from labeled training data
    by minimizing the average excess risk over a set of signals with known
    ground truth changepoints.

    The learning process consists of two phases:
    1. **Warm Start**: A random signal is selected and β is optimized for
       that single signal to get a good initial value.
    2. **Global Optimization**: Starting from the warm-start β, L-BFGS-B
       minimizes the average excess risk over all training signals.

    Attributes:
        beta_bounds: Tuple (min, max) for valid β values.
        beta_opt: Learned optimal β after fit(). None before fitting.

    Examples:
        >>> import numpy as np
        >>> from alpin import ALPIN
        >>>
        >>> # Create simple training data
        >>> def make_signal():
        ...     return np.concatenate([
        ...         np.zeros(50) + np.random.randn(50) * 0.5,
        ...         np.ones(50) * 5 + np.random.randn(50) * 0.5,
        ...         np.zeros(50) + np.random.randn(50) * 0.5
        ...     ])
        >>>
        >>> signals = [make_signal() for _ in range(10)]
        >>> truths = [[50, 100] for _ in range(10)]
        >>>
        >>> # Fit model
        >>> model = ALPIN()
        >>> model.fit(signals, truths)
        >>> print(f"Learned beta: {model.beta_opt:.4f}")
        >>>
        >>> # Predict on new signal
        >>> test_signal = make_signal()
        >>> changepoints = model.predict(test_signal)
        >>> print(f"Detected changepoints: {changepoints}")
    """

    def __init__(self, beta_bounds: tuple[float, float] = (1e-6, 1e6)) -> None:
        """Initialize ALPIN model.

        Args:
            beta_bounds: Tuple (min_beta, max_beta) defining valid range for β.
                        Defaults to (1e-6, 1e6) as suggested in the paper.
        """
        self.beta_bounds = beta_bounds
        self.beta_opt: Optional[float] = None

    def fit(
        self,
        signals: list[np.ndarray],
        ground_truths: list[list[int]],
    ) -> "ALPIN":
        """Learn optimal beta from labeled training data.

        Implements the ALPIN learning algorithm:
        1. Randomly select one signal for warm start
        2. Optimize β on that single signal
        3. Use warm-start β as initial value for global optimization
        4. Minimize average excess risk across all signals using L-BFGS-B

        Args:
            signals: List of 1D numpy arrays (training signals).
                    Each signal must have length >= 20.
            ground_truths: List of ground truth partitions. Each partition is
                          a list of changepoint indices for the corresponding
                          signal.

        Returns:
            self: Fitted ALPIN instance with beta_opt set.

        Raises:
            ValueError: If signals and ground_truths have different lengths,
                       or if any signal is too short.

        Examples:
            >>> import numpy as np
            >>> model = ALPIN()
            >>> signals = [np.concatenate([np.zeros(50), np.ones(50) * 5])
            ...            for _ in range(5)]
            >>> truths = [[50] for _ in range(5)]
            >>> model.fit(signals, truths)
            >>> model.beta_opt > 0
            True
        """
        if len(signals) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(signals)} signals vs "
                f"{len(ground_truths)} ground truths"
            )

        if len(signals) == 0:
            raise ValueError("Cannot fit on empty dataset")

        for i, signal in enumerate(signals):
            if len(signal) < 20:
                raise ValueError(f"Signal {i} too short: length {len(signal)} < 20")

        # PHASE 1: WARM START - optimize β on single random signal
        warm_idx = np.random.choice(len(signals))
        warm_signal = signals[warm_idx]
        warm_truth = ground_truths[warm_idx]

        def warm_objective(log_beta: np.ndarray) -> float:
            beta = np.exp(log_beta[0])
            return excess_risk(warm_signal, warm_truth, beta)

        initial_log_beta = np.log(np.sqrt(self.beta_bounds[0] * self.beta_bounds[1]))
        log_bounds = [(np.log(self.beta_bounds[0]), np.log(self.beta_bounds[1]))]

        warm_result = minimize(
            fun=warm_objective,
            x0=np.array([initial_log_beta]),
            method="L-BFGS-B",
            bounds=log_bounds,
        )
        warm_start_beta = np.exp(warm_result.x[0])

        # PHASE 2: GLOBAL OPTIMIZATION - minimize average excess risk
        def global_objective(log_beta: np.ndarray) -> float:
            beta = np.exp(log_beta[0])
            total_risk = 0.0
            for signal, truth in zip(signals, ground_truths):
                total_risk += excess_risk(signal, truth, beta)
            return total_risk / len(signals)

        global_result = minimize(
            fun=global_objective,
            x0=np.array([np.log(warm_start_beta)]),
            method="L-BFGS-B",
            bounds=log_bounds,
        )

        self.beta_opt = float(np.exp(global_result.x[0]))

        return self

    def predict(self, signal: np.ndarray) -> list[int]:
        """Detect changepoints using learned beta.

        Uses the β-optimal partition solver with the learned penalty parameter
        to detect changepoints in a new signal.

        Args:
            signal: Input signal as 1D numpy array. Must have length >= 20.

        Returns:
            List of detected changepoint indices (0-indexed).
            Empty list if no changepoints detected.

        Raises:
            RuntimeError: If model has not been fitted (beta_opt is None).
            ValueError: If signal is too short (< 20 samples).

        Examples:
            >>> import numpy as np
            >>> model = ALPIN()
            >>> signals = [np.concatenate([np.zeros(50), np.ones(50) * 5])
            ...            for _ in range(5)]
            >>> truths = [[50] for _ in range(5)]
            >>> model.fit(signals, truths)
            >>>
            >>> test = np.concatenate([np.zeros(50), np.ones(50) * 5])
            >>> cps = model.predict(test)
            >>> len(cps)
            1
        """
        if self.beta_opt is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

        return solve_optimal_partition(signal, self.beta_opt)

    def fit_predict(
        self,
        signals: list[np.ndarray],
        ground_truths: list[list[int]],
        signal: np.ndarray,
    ) -> list[int]:
        """Fit model and predict on a single signal.

        Convenience method that combines fit() and predict() in one call.

        Args:
            signals: List of training signals.
            ground_truths: List of ground truth partitions.
            signal: Signal to predict on after fitting.

        Returns:
            List of detected changepoint indices for the input signal.

        Examples:
            >>> import numpy as np
            >>> model = ALPIN()
            >>> signals = [np.concatenate([np.zeros(50), np.ones(50) * 5])
            ...            for _ in range(5)]
            >>> truths = [[50] for _ in range(5)]
            >>> test = np.concatenate([np.zeros(50), np.ones(50) * 5])
            >>> cps = model.fit_predict(signals, truths, test)
            >>> isinstance(cps, list)
            True
        """
        self.fit(signals, ground_truths)
        return self.predict(signal)
