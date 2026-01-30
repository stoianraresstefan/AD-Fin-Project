---
title: Optimal Partition (Pelt)
tags: [algorithm, dynamic-programming, segmentation]
---

# Optimal Partition (Pelt)

The optimal partition problem involves finding the best segmentation of a signal by minimizing a penalized cost function. ALPIN uses the **Pelt** (Pruned Exact Linear Time) algorithm for this purpose.

## Mathematical Definition

For a signal $y$ of length $n$, we seek a partition $A = \{t_0, t_1, \dots, t_k\}$ where $0 = t_0 < t_1 < \dots < t_k = n$, that minimizes:
$$\mathcal{C}(A) = \sum_{j=1}^k [c(y_{t_{j-1}:t_j}) + \beta]$$
where:
- $c(\cdot)$ is a cost function (ALPIN uses the $L_2$ cost).
- $\beta$ is the penalty for adding a new changepoint.

## Pelt Algorithm

Pelt is an exact search algorithm that uses dynamic programming with a pruning rule to achieve linear time complexity $O(n)$ under certain conditions.

### Pruning Rule
A potential changepoint $t$ can be pruned if:
$$[\min_A \mathcal{C}(y_{1:t}, A)] + c(y_{t:T}) > [\min_A \mathcal{C}(y_{1:T}, A)]$$
This ensures that $t$ will never be part of an optimal partition for any signal ending after $T$.

## Implementation in ALPIN

ALPIN wraps the `ruptures.Pelt` implementation:

```python
import ruptures as rpt

def solve_optimal_partition(signal, beta):
    # model="l2": Quadratic loss (sum of squared residuals)
    # min_size=2: Minimum segment length
    # jump=1: Check every index
    algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)
    result = algo.predict(pen=beta)
    return result[:-1] # Remove signal length from end
```

## Parameters

- **Cost Model**: `l2` (sum of squared residuals from the mean).
- **Penalty ($\beta$)**: The parameter learned by ALPIN.
- **Min Size**: 2 (prevents trivial single-sample segments).
- **Jump**: 1 (ensures exact search by checking every possible position).

## References
- [[Modules/Partition|Partition Module]]
- [[Concepts/Penalty-Parameter|What is beta?]]
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of changepoints with a linear computational cost".
