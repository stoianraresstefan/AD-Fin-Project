---
title: Risk Module
tags: [module, risk, loss]
---

# Risk Module (`alpin/risk.py`)

The `risk.py` module implements the various risk functions used to define the ALPIN objective function.

## Functions

### `empirical_risk(signal, partition)`
Calculates the sum of squared residuals for a given partition.
$$R(y, A) = \sum_{a \in A} \sum_{i \in a} (y_i - \bar{y}_a)^2$$

### `penalized_risk(signal, partition, beta)`
Adds the complexity penalty to the empirical risk.
$$R_\beta(y, A) = R(y, A) + \beta |A|$$

### `excess_risk(signal, ground_truth, beta)`
Calculates the difference between the risk of the ground truth and the risk of the $\beta$-optimal partition.
$$E(y, \beta) = R_\beta(y, A^{lab}) - \min_A R_\beta(y, A)$$
This is the core loss function for ALPIN.

## Usage in Optimization

During the `fit` process, ALPIN calls `excess_risk` repeatedly:

```python
def global_objective(log_beta):
    beta = np.exp(log_beta[0])
    total_risk = sum(excess_risk(s, t, beta) for s, t in zip(signals, truths))
    return total_risk / len(signals)
```

## References
- [[Algorithms/Risk-Minimization|Risk Minimization Theory]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm Flow]]
