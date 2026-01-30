---
title: Penalty Parameter (beta)
tags: [concept, optimization, complexity]
---

# Penalty Parameter ($\beta$)

The penalty parameter $\beta$ (often called the "smoothing parameter" or "regularization weight") is the most critical hyperparameter in penalized changepoint detection.

## Role of $\beta$

In the cost function:
$$\min_A \sum_{j=1}^k [c(y_{t_{j-1}:t_j}) + \beta]$$

- **Small $\beta$**: The cost of adding a changepoint is low. The algorithm will detect many changepoints, potentially fitting to noise (**Over-segmentation**).
- **Large $\beta$**: The cost of adding a changepoint is high. The algorithm will detect few changepoints, potentially missing real shifts (**Under-segmentation**).

## Common Heuristics

In unsupervised settings, $\beta$ is often chosen using heuristics like:
- **BIC (Bayesian Information Criterion)**: $\beta = \sigma^2 \log n$
- **AIC (Akaike Information Criterion)**: $\beta = 2\sigma^2$

However, these heuristics often fail on real-world data where the noise is not perfectly Gaussian or the "importance" of a changepoint is subjective.

## ALPIN's Approach

ALPIN treats $\beta$ as a parameter to be learned from data. By providing examples of what a "correct" segmentation looks like, ALPIN finds the $\beta$ that best reproduces those results.

## References
- [[Algorithms/Optimal-Partition|Optimal Partition (Pelt)]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm]]
