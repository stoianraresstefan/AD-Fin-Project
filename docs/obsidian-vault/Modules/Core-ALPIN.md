---
title: Core ALPIN Module
tags: [module, core, implementation]
---

# Core ALPIN Module (`alpin/core.py`)

The `core.py` module contains the primary `ALPIN` class, which implements the supervised learning logic for penalty parameter estimation.

## The `ALPIN` Class

The `ALPIN` class follows a scikit-learn-like API with `fit` and `predict` methods.

### Key Attributes

- `beta_bounds`: A tuple `(min, max)` defining the search space for $\beta$. Defaults to `(1e-6, 1e6)`.
- `beta_opt`: The learned optimal penalty parameter. Available after calling `fit()`.

### Main Methods

#### `fit(signals, ground_truths)`
Learns the optimal $\beta$ from a list of training signals and their corresponding ground truth changepoints.

**Process:**
1. **Warm Start**: Selects a random signal and optimizes $\beta$ on it to find a good starting point.
2. **Global Optimization**: Minimizes the average excess risk across all training signals using the `L-BFGS-B` algorithm.

#### `predict(signal)`
Detects changepoints in a new signal using the learned `beta_opt`.

#### `fit_predict(signals, ground_truths, signal)`
A convenience method that fits the model and then predicts on a target signal.

## Code Example

```python
from alpin import ALPIN

# Initialize
model = ALPIN(beta_bounds=(1e-2, 1e4))

# Train
model.fit(train_signals, train_labels)
print(f"Optimal beta: {model.beta_opt}")

# Predict
detected_cps = model.predict(test_signal)
```

## Implementation Details

- **Optimization Domain**: The algorithm optimizes $\log \beta$ to ensure $\beta > 0$ and to handle the wide range of possible values efficiently.
- **Risk Function**: Relies on `alpin.risk.excess_risk` for the objective function.
- **Solver**: Relies on `alpin.partition.solve_optimal_partition` for finding the optimal segmentation at each step.

## References
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm Explanation]]
- [[Algorithms/Risk-Minimization|Risk Minimization Details]]
- [[API-Reference/ALPIN-Class|ALPIN API Reference]]
