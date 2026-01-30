---
title: ALPIN Class API
tags: [api, core, class]
---

# `ALPIN` Class API

The main class for learning penalty parameters and detecting changepoints.

## Constructor

```python
ALPIN(beta_bounds=(1e-6, 1e6))
```

- **`beta_bounds`** (tuple): The range `(min, max)` for the optimal $\beta$ search.

## Methods

### `fit(signals, ground_truths)`
Learns the optimal $\beta$ from labeled data.

- **`signals`** (list of `np.ndarray`): Training signals.
- **`ground_truths`** (list of `list[int]`): True changepoint indices for each signal.
- **Returns**: `self`

### `predict(signal)`
Detects changepoints in a new signal.

- **`signal`** (np.ndarray): Input signal.
- **Returns**: `list[int]` - Detected changepoint indices.

### `fit_predict(signals, ground_truths, signal)`
Fits the model and predicts on a target signal in one step.

## Attributes

- **`beta_opt`** (float): The learned optimal penalty parameter.

## References
- [[Modules/Core-ALPIN|Core ALPIN Module]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm Flow]]
