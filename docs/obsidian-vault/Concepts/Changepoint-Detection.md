---
title: Changepoint Detection
tags: [concept, segmentation, time-series]
---

# Changepoint Detection

Changepoint detection is the task of identifying the times when the probability distribution of a stochastic process or time series changes.

## Piecewise-Constant Signals

In the context of ALPIN, we focus on **piecewise-constant signals**. These are signals that stay at a constant mean level for a period of time and then "jump" to a new level.

$$y_t = \mu_k + \epsilon_t \quad \text{for } t \in [t_{k-1}, t_k)$$

where:
- $\mu_k$ is the mean of the $k$-th segment.
- $\epsilon_t$ is white noise (usually Gaussian).
- $t_k$ are the changepoint locations.

## The Detection Problem

The goal is to estimate the number and locations of the changepoints $t_k$ given the noisy observations $y_t$.

## Supervised vs Unsupervised

- **Unsupervised**: Methods like Pelt or Binary Segmentation require the user to provide a penalty parameter $\beta$ manually.
- **Supervised (ALPIN)**: Uses labeled examples to learn the optimal $\beta$ that matches human intuition or ground truth.

## References
- [[Algorithms/Optimal-Partition|Optimal Partition (Pelt)]]
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm]]
