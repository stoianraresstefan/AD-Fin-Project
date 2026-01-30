---
title: Synthetic Data Module
tags: [module, data, synthetic]
---

# Synthetic Data Module (`alpin/data/synthetic.py`)

The `synthetic.py` module handles the generation of artificial piecewise-constant signals for testing and training.

## Signal Generation

### `generate_signal(n_samples, n_changepoints=None, noise_std=1.0, protocol="I", ...)`
Generates a single noisy piecewise-constant signal.

**Parameters:**
- `protocol`: 
    - `"I"`: All generated changepoints are included in the labels.
    - `"II"`: Only changepoints with a jump amplitude $> 3$ are included.

### `generate_synthetic_signals(n_signals, ...)`
Generates a batch of signals with their corresponding ground truth labels.

## Forecasting Integration

### `alpin_signals_to_deepar_df(signals, changepoints_list, ...)`
Converts a list of ALPIN signals into a `pandas.DataFrame` format compatible with `pytorch-forecasting`'s `TimeSeriesDataSet`.

**Columns created:**
- `time_idx`: Sequential index.
- `date`: Daily timestamps.
- `value`: Signal amplitude.
- `series_id`: Identifier for the time series (e.g., "series_0").

## References
- [[Concepts/Changepoint-Detection|Changepoint Detection Concepts]]
- [[Architecture/DeepCAR-Architecture|DeepCAR Architecture]]
