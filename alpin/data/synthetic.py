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
    Generate a piecewise constant signal with noise per paper specifications.

    Creates a synthetic signal with random changepoints, jump amplitudes, and
    additive Gaussian noise. Supports two annotation protocols:
    - Protocol I: All changepoints included (default)
    - Protocol II: Only changepoints with |amplitude| > 3

    Parameters
    ----------
    n_samples : int
        Number of samples in the signal
    n_changepoints : int, optional
        Number of changepoints. If None, randomly chosen from {3, 4, 5, 6, 7}
    noise_std : float, optional
        Standard deviation of Gaussian noise (default: 1.0)
    seed : int, optional
        Random seed for reproducibility
    protocol : {"I", "II"}, optional
        Labeling protocol:
        - "I": All changepoints included
        - "II": Only changepoints with |amplitude| > 3
        Default: "I"

    Returns
    -------
    signal : np.ndarray
        Generated noisy piecewise constant signal of shape (n_samples,)
    changepoints : list[int]
        List of changepoint indices (sample positions where regime changes)
        Indices are in range [0, n_samples)

    Notes
    -----
    Paper specifications (Section 5.1):
    - Changepoints: 3-7 per signal
    - Regime lengths: 5%-30% of signal (uniform distribution)
    - Jump amplitudes: 1-5 (uniform), ± (random sign)
    - Noise: Gaussian N(0, σ²)
    - Protocol II filters changepoints where |amplitude| <= 3
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Determine number of changepoints
    if n_changepoints is None:
        n_changepoints = np.random.randint(3, 8)  # 3-7 inclusive

    # Step 2: Generate regime lengths (as fractions of total, sum to 1)
    n_regimes = n_changepoints + 1
    regime_fractions = np.random.uniform(0.05, 0.30, n_regimes)
    regime_fractions = regime_fractions / regime_fractions.sum()  # Normalize

    # Step 3: Convert fractions to sample counts and compute changepoint indices
    regime_samples = (regime_fractions * n_samples).astype(int)
    # Adjust last regime to ensure exact sum
    regime_samples[-1] = n_samples - regime_samples[:-1].sum()

    # Compute changepoint indices (cumulative positions)
    changepoint_indices = np.cumsum(regime_samples)[:-1].tolist()

    # Step 4: Generate jump amplitudes for each regime
    # Start at level 0, apply jumps
    current_level = 0.0
    regime_levels = [current_level]
    jump_amplitudes = []

    for _ in range(n_changepoints):
        jump = np.random.uniform(1, 5) * np.random.choice([-1, 1])
        current_level += jump
        regime_levels.append(current_level)
        jump_amplitudes.append(jump)

    # Step 5: Create piecewise constant signal (clean)
    y_clean = np.zeros(n_samples)
    idx = 0
    for regime_idx, regime_size in enumerate(regime_samples):
        y_clean[idx : idx + regime_size] = regime_levels[regime_idx]
        idx += regime_size

    # Step 6: Add Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    signal = y_clean + noise

    # Step 7: Apply protocol filtering if Protocol II
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
    Generate a batch of synthetic signals with ground truth changepoints.

    Parameters
    ----------
    n_signals : int, optional
        Number of signals to generate (default: 100)
    n_samples : int, optional
        Number of samples per signal (default: 500)
    noise_std : float, optional
        Standard deviation of Gaussian noise (default: 1.0)
    seed : int, optional
        Random seed for reproducibility. If provided, each signal uses
        seed + signal_index for deterministic generation
    protocol : {"I", "II"}, optional
        Labeling protocol for all signals (default: "I")

    Returns
    -------
    signals : list[np.ndarray]
        List of generated signals, each of shape (n_samples,)
    truths : list[list[int]]
        List of ground truth changepoint lists, one per signal

    Notes
    -----
    If seed is provided, each signal is generated with seed+index to ensure
    reproducibility while creating different signals.
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
    Convert ALPIN synthetic signals into pandas DataFrame for pytorch-forecasting.

    Transforms a list of 1D ALPIN signals and their changepoints into a single
    pandas DataFrame compatible with pytorch-forecasting's TimeSeriesDataSet.
    Combines multiple signals as separate time series with series_id labels.

    Parameters
    ----------
    signals : list[np.ndarray]
        List of 1D numpy arrays representing ALPIN signals
    changepoints_list : list[list[int]]
        List of changepoint lists, one per signal. Each inner list contains
        indices where regime changes occur.
    start_date : str, optional
        Starting date in 'YYYY-MM-DD' format (default: "2020-01-01")
        Dates increment daily for each time step.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        - time_idx : int - Sequential time index (resets per series)
        - date : datetime - Timestamps starting from start_date
        - value : float - Signal values
        - series_id : str - Series identifier ("series_0", "series_1", etc.)

    Examples
    --------
    >>> import numpy as np
    >>> from alpin.data.synthetic import alpin_signals_to_deepar_df
    >>>
    >>> # Create two sample signals
    >>> signals = [np.array([1.0, 1.5, 2.0, 1.8]), np.array([0.5, 0.3, 0.1])]
    >>> changepoints = [[1, 2], [1]]
    >>>
    >>> df = alpin_signals_to_deepar_df(signals, changepoints)
    >>> print(df)
       time_idx       date  value  series_id
    0         0 2020-01-01    1.0   series_0
    1         1 2020-01-02    1.5   series_0
    2         2 2020-01-03    2.0   series_0
    3         3 2020-01-04    1.8   series_0
    4         0 2020-01-01    0.5   series_1
    5         1 2020-01-02    0.3   series_1
    6         2 2020-01-03    0.1   series_1
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
