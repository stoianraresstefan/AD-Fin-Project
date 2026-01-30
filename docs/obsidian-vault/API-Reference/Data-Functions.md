---
title: Data Functions API
tags: [api, data, synthetic]
---

# Data Functions API

Functions for generating synthetic data and loading CSV files.

## Synthetic Data (`alpin.data.synthetic`)

### `generate_signal(...)`
```python
generate_signal(n_samples, n_changepoints=None, noise_std=1.0, seed=None, protocol="I")
```
Generates a single piecewise-constant signal.

### `generate_synthetic_signals(...)`
```python
generate_synthetic_signals(n_signals=100, n_samples=500, noise_std=1.0, seed=None, protocol="I")
```
Generates a batch of signals and labels.

### `alpin_signals_to_deepar_df(...)`
```python
alpin_signals_to_deepar_df(signals, changepoints_list, start_date="2020-01-01")
```
Converts ALPIN data to a `pandas.DataFrame` for forecasting.

## Data Loading (`alpin.data.loader`)

### `CSVLoader.load(path)`
Loads a signal and optional labels from a CSV file.

### `load_directory(path, loader)`
Loads all CSV files in a directory.

## References
- [[Modules/Data-Synthetic|Synthetic Data Module]]
- [[Modules/Data-Loader|Data Loader Module]]
