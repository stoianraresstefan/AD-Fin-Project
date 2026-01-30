---
title: Risk Minimization
tags: [algorithm, risk, loss-function]
---

# Risk Minimization

ALPIN learns the optimal penalty parameter $\beta$ by minimizing a specific loss function called **Excess Risk**.

## Types of Risk

### 1. Empirical Risk $R(y, A)$
The sum of squared residuals for a given partition $A$:
$$R(y, A) = \sum_{a \in A} \sum_{i \in a} (y_i - \bar{y}_a)^2$$
where $\bar{y}_a$ is the mean of the signal in segment $a$.

### 2. Penalized Risk $R_\beta(y, A)$
Adds a penalty for the number of segments:
$$R_\beta(y, A) = R(y, A) + \beta |A|$$

### 3. Excess Risk $E(y, \beta)$
The difference between the penalized risk of the ground truth partition $A^{lab}$ and the penalized risk of the $\beta$-optimal partition $\hat{A}(\beta)$:
$$E(y, \beta) = R_\beta(y, A^{lab}) - \min_A R_\beta(y, A)$$

## Why Excess Risk?

- **Alignment**: If $E(y, \beta) = 0$, it means the ground truth partition is one of the optimal partitions for that $\beta$.
- **Differentiability**: While the optimal partition $\hat{A}(\beta)$ is a discrete object, the excess risk function is piecewise linear and continuous in $\beta$, making it suitable for optimization.
- **Supervision**: It directly uses the labels ($A^{lab}$) to guide the learning of $\beta$.

## Optimization Process

ALPIN minimizes the average excess risk over the training set:
$$\mathcal{L}(\beta) = \frac{1}{N} \sum_{i=1}^N E(y_i, \beta)$$

Since $\beta$ must be positive, the optimization is performed over $\theta = \log \beta$:
$$\min_\theta \mathcal{L}(e^\theta)$$

## Implementation

The risk functions are implemented in `alpin/risk.py`:
- `empirical_risk(signal, partition)`
- `penalized_risk(signal, partition, beta)`
- `excess_risk(signal, ground_truth, beta)`

## References
- [[Modules/Risk|Risk Module]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm Flow]]
