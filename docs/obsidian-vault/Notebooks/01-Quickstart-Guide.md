---
title: 01 Quickstart Guide
tags: [notebook, tutorial, quickstart]
---

# 01 Quickstart Guide

This guide walkthrough the `notebooks/01_quickstart.ipynb` notebook, which provides a 5-minute introduction to the ALPIN algorithm.

## Overview

The quickstart covers:
1. Generating a small set of synthetic signals.
2. Training the ALPIN model to learn the optimal penalty $\beta$.
3. Predicting changepoints on a new, unseen signal.
4. Visualizing the results.

## Key Steps

### 1. Data Generation
We generate 5 synthetic signals with 500 samples each.
```python
from alpin.data import generate_synthetic_signals
signals, truths = generate_synthetic_signals(n_signals=5, n_samples=500)
```

### 2. Model Training
The `ALPIN` class is initialized and fitted to the signals.
```python
from alpin import ALPIN
model = ALPIN()
model.fit(signals, truths)
print(f"Learned beta: {model.beta_opt}")
```

### 3. Prediction
We use the learned model to predict changepoints on a test signal.
```python
predictions = model.predict(test_signal)
```

## Visual Results

The notebook uses `alpin.visualization.plot_signal` to show the detected changepoints (red lines) against the ground truth (green dashed lines).

## Summary
By the end of this notebook, you will have a working ALPIN model and understand the basic `fit`/`predict` workflow.

## References
- [[Modules/Core-ALPIN|ALPIN Class]]
- [[Tutorials/Getting-Started|Getting Started Tutorial]]
