from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from alpin.partition import solve_optimal_partition
from alpin.risk import excess_risk


class ALPIN:
    """
    Learns the optimal penalty parameter beta from labeled training data by minimizing average excess risk over signals.
    Uses two-phase optimization: warm start on a random signal, then global optimization with L-BFGS-B across all data.

    Input: beta_bounds (tuple of float) - valid range for beta values (default: 1e-6 to 1e6)
    Output: fitted ALPIN instance with beta_opt attribute, list of changepoints when predicting
    """

    def __init__(self, beta_bounds: tuple[float, float] = (1e-6, 1e6)) -> None:
        self.beta_bounds = beta_bounds
        self.beta_opt: Optional[float] = None

    def fit(
        self,
        signals: list[np.ndarray],
        ground_truths: list[list[int]],
    ) -> "ALPIN":
        """
        Learns optimal beta using two-phase optimization: first warm start on a random signal, then global L-BFGS-B minimization of average excess risk.
        Returns self for method chaining, setting beta_opt attribute to the learned value.

        Input: signals (list of np.ndarray) - training signals (each >= 20 samples), ground_truths (list of list of int) - changepoint indices
        Output: self - fitted ALPIN instance with beta_opt set
        """
        if len(signals) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(signals)} signals vs "
                f"{len(ground_truths)} ground truths"
            )

        if len(signals) == 0:
            raise ValueError("empty dataset")

        for i, signal in enumerate(signals):
            if len(signal) < 20:
                raise ValueError(f"Signal {i} too short, length {len(signal)} < 20")

        #  WARM START - optimize beta on single random signal
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

        # -- minimize average excess risk
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
        """
        Detects changepoints in a new signal using the learned penalty parameter
        
        Output: list of int - detected changepoint indices, empty list if no changepoints found
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
        self.fit(signals, ground_truths)
        return self.predict(signal)
