---
title: Metrics Module
tags: [module, metrics, evaluation]
---

# Metrics Module (`alpin/metrics.py`)

The `metrics.py` module provides functions to evaluate the quality of detected changepoints against ground truth.

## Implemented Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of predicted changepoints that are "close" to a ground truth point. |
| **Recall** | Fraction of ground truth changepoints that were successfully detected. |
| **Hausdorff Distance** | The maximum distance between a predicted point and its nearest ground truth (and vice versa). |
| **Annotation Error** | The absolute difference between the number of predicted and true changepoints. |
| **Rand Index** | Measures the similarity between the predicted and true segmentations as a clustering problem. |

## Key Functions

### `evaluate_all(predicted, ground_truth, signal_length, tolerance=10)`
A convenience function that returns a dictionary containing all the above metrics.

### `precision(predicted, ground_truth, signal_length, tolerance=10)`
Uses a margin-based approach. A prediction is considered correct if it falls within `tolerance` samples of a true changepoint.

### `rand_index(predicted, ground_truth, signal_length)`
Uses `sklearn.metrics.adjusted_rand_score` to compare the segment labels assigned to each time point.

## Code Example

```python
from alpin.metrics import evaluate_all

results = evaluate_all(detected_cps, true_cps, len(signal), tolerance=5)
print(f"Precision: {results['precision']:.2f}")
print(f"Recall: {results['recall']:.2f}")
```

## References
- [[Concepts/Evaluation-Metrics|Evaluation Metrics Explained]]
