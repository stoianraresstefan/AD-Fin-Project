---
title: Excess Risk
tags: [concept, loss-function, optimization]
---

# Excess Risk

**Excess Risk** is the loss function used by ALPIN to learn the optimal penalty parameter.

## Definition

For a signal $y$ and a penalty $\beta$, the excess risk $E(y, \beta)$ is:
$$E(y, \beta) = R_\beta(y, A^{lab}) - \min_A R_\beta(y, A)$$

Where:
- $R_\beta(y, A^{lab})$ is the penalized risk of the **labeled** (ground truth) partition.
- $\min_A R_\beta(y, A)$ is the penalized risk of the **optimal** partition for that $\beta$.

## Intuition

- If the ground truth partition is the optimal one for a given $\beta$, then $E(y, \beta) = 0$.
- If the ground truth is not optimal, $E(y, \beta) > 0$ measures "how far" the ground truth is from being optimal.

By minimizing the average excess risk over many signals, ALPIN finds a $\beta$ that makes the ground truth partitions "as optimal as possible" across the entire dataset.

## References
- [[Algorithms/Risk-Minimization|Risk Minimization Details]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm Flow]]
