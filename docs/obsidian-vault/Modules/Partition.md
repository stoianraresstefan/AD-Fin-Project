---
title: Partition Module
tags: [module, partition, pelt]
---

# Partition Module (`alpin/partition.py`)

The `partition.py` module provides the interface to the optimal partition solver.

## Functions

### `solve_optimal_partition(signal, beta)`

This is the core function used throughout the project to find the best segmentation for a given penalty $\beta$.

**Arguments:**
- `signal`: 1D `numpy.ndarray` of signal values.
- `beta`: The penalty parameter (float).

**Returns:**
- `list[int]`: A list of changepoint indices (0-indexed).

## Implementation

The module uses the `ruptures` library, specifically the `Pelt` algorithm with an $L_2$ cost model.

```python
import ruptures as rpt

def solve_optimal_partition(signal: np.ndarray, beta: float) -> list[int]:
    # Initialize Pelt with L2 cost (sum of squared residuals)
    algo = rpt.Pelt(model="l2", min_size=2, jump=1).fit(signal)
    
    # Predict with penalty beta
    result = algo.predict(pen=beta)
    
    # result includes the signal length at the end, which we remove
    return result[:-1]
```

## Key Considerations

- **Minimum Segment Size**: Set to 2 to avoid trivial segments consisting of a single point.
- **Jump**: Set to 1 to ensure an exact search by checking every possible index.
- **Cost Model**: `l2` is appropriate for piecewise-constant signals with Gaussian noise.

## References
- [[Algorithms/Optimal-Partition|Optimal Partition (Pelt) Algorithm]]
- [[Concepts/Penalty-Parameter|What is beta?]]
