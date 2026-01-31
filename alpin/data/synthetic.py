"""
Synthetic signal generation for ALPIN experiments.

Implements synthetic signal generation per TOV-EUSIPCO-17 paper (Section 5.1).
Supports both Protocol I (all changepoints) and Protocol II (amplitude filtering).
"""

from typing import Literal
import numpy as np
import pandas as pd
from datetime import timedelta


def generate_signal(
    n_samples: int,
    n_changepoints: int | None = None,
    noise_std: float = 1.0,
    seed: int | None = None,
    protocol: Literal["I", "II"] = "I",
) -> tuple[np.ndarray, list[int]]:
    """
    Generates a piecewise constant signal with random changepoints, jump amplitudes, and Gaussian noise per paper specifications.
    Supports Protocol I (all changepoints) and Protocol II (amplitude filtering > 3).

    Input: n_samples (int) - signal length, n_changepoints (int or None) - number of changepoints, noise_std (float) - Gaussian noise std, seed (int or None) - reproducibility, protocol (str) - "I" or "II"
    Output: tuple of (np.ndarray signal, list of int changepoint indices)
    """
    if seed is not None:
        np.random.seed(seed)

    # ------- num of changepoints
    if n_changepoints is None:
        n_changepoints = np.random.randint(3, 8)  # 3-7 inclusive

    # ------- Generate regime lengths (as fractions of total, sum to 1)
    n_regimes = n_changepoints + 1
    regime_fractions = np.random.uniform(0.05, 0.30, n_regimes)
    regime_fractions = regime_fractions / regime_fractions.sum()  # normalize

    # ------- Convert fractions to sample counts and compute changepoint indices
    regime_samples = (regime_fractions * n_samples).astype(int)
    # Adjust last regime to ensure exact sum
    regime_samples[-1] = n_samples - regime_samples[:-1].sum()

    # Compute changepoint indices (cumulative positions)
    changepoint_indices = np.cumsum(regime_samples)[:-1].tolist()

    # ------- Generate jump amplitudes for each regime
    # Start at level 0, apply jumps
    current_level = 0.0
    regime_levels = [current_level]
    jump_amplitudes = []

    for _ in range(n_changepoints):
        jump = np.random.uniform(1, 5) * np.random.choice([-1, 1])
        current_level += jump
        regime_levels.append(current_level)
        jump_amplitudes.append(jump)

    # ------- Create piecewise constant signal (clean)
    y_clean = np.zeros(n_samples)
    idx = 0
    for regime_idx, regime_size in enumerate(regime_samples):
        y_clean[idx : idx + regime_size] = regime_levels[regime_idx]
        idx += regime_size

    # ------- Add Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    signal = y_clean + noise

    # ------- Apply protocol filtering if Protocol II
    if protocol == "II":
        # Filter changepoints: keep only those with |amplitude| > 3
        filtered_cps = []
        for cp_idx, cp in enumerate(changepoint_indices):
            # Jump amplitude is difference between regime levels
            jump = jump_amplitudes[cp_idx]
            if abs(jump) > 3:
                filtered_cps.append(cp)
        changepoints = filtered_cps
    else:
        # Protocol I: all changepoints
        changepoints = changepoint_indices

    return signal, changepoints


def generate_synthetic_signals(
    n_signals: int = 100,
    n_samples: int = 500,
    noise_std: float = 1.0,
    seed: int | None = None,
    protocol: Literal["I", "II"] = "I",
) -> tuple[list[np.ndarray], list[list[int]]]:
    """
    Generates a batch of synthetic signals with ground truth changepoints, each created using generate_signal with incremented seed for variation.

    Input: n_signals (int) - number of signals, n_samples (int) - signal length, noise_std (float) - noise std, seed (int or None) - base seed, protocol (str) - "I" or "II"
    Output: tuple of (list of np.ndarray signals, list of list of int changepoints)
    """
    signals = []
    truths = []

    for i in range(n_signals):
        signal_seed = None if seed is None else seed + i
        signal, changepoints = generate_signal(
            n_samples=n_samples,
            n_changepoints=None,
            noise_std=noise_std,
            seed=signal_seed,
            protocol=protocol,
        )
        signals.append(signal)
        truths.append(changepoints)

    return signals, truths


def alpin_signals_to_deepar_df(
    signals: list[np.ndarray],
    changepoints_list: list[list[int]],
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Converts ALPIN synthetic signals into a pandas DataFrame compatible with pytorch-forecasting's TimeSeriesDataSet.
    Combines multiple signals as separate time series with daily timestamps and series_id labels.

    Input: signals (list of np.ndarray) - list of 1D signals, changepoints_list (list of list of int) - changepoints per signal, start_date (str) - starting date YYYY-MM-DD
    Output: pd.DataFrame with columns time_idx, date, value, series_id
    """
    # Parse start date
    base_date = pd.to_datetime(start_date)

    # Collect data from all signals
    rows = []
    for series_idx, signal in enumerate(signals):
        series_id = f"series_{series_idx}"

        for time_idx, value in enumerate(signal):
            # Generate date for this time step
            date = base_date + timedelta(days=time_idx)

            rows.append(
                {
                    "time_idx": time_idx,
                    "date": date,
                    "value": float(value),
                    "series_id": series_id,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure correct dtypes
    df["time_idx"] = df["time_idx"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].astype(float)
    df["series_id"] = df["series_id"].astype(str)

    return df
