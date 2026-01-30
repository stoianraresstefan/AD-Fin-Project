---
title: Experiments Sweep Module
tags: [module, experiments, sweep]
---

# Experiments Sweep Module (`alpin/experiments/sweep.py`)

The `sweep.py` module contains utilities for performing systematic experiments and hyperparameter searches.

## Functions

### `sweep_beta(signals, ground_truths, beta_range, n_splits=3)`
Performs a grid search over a range of $\beta$ values using K-fold cross-validation.

**Returns:**
- A `pandas.DataFrame` containing metrics (Precision, Recall, etc.) for each $\beta$ and each fold.

### `sweep_noise(n_signals, n_samples, noise_levels, beta, ...)`
Evaluates the robustness of a fixed $\beta$ (or the ALPIN algorithm) across different noise standard deviations.

**Returns:**
- A `pandas.DataFrame` indexed by noise level.

## Usage in Analysis

These functions are extensively used in [[Notebooks/03-Analysis-Guide|Notebook 03]] to generate performance curves and robustness plots.

```python
from alpin.experiments.sweep import sweep_beta

beta_range = [1.0, 10.0, 100.0, 1000.0]
results = sweep_beta(signals, truths, beta_range)
```

## References
- [[Modules/Visualization|Visualization Module]]
- [[Notebooks/02-Training-Guide|Notebook 02: Training and Sweeps]]
