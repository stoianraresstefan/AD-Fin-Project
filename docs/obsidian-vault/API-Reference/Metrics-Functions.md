---
title: Metrics Functions API
tags: [api, metrics, evaluation]
---

# Metrics Functions API

Functions for evaluating changepoint detection performance.

## Core Functions (`alpin.metrics`)

### `evaluate_all(...)`
```python
evaluate_all(predicted, ground_truth, signal_length, tolerance=10)
```
Returns a dictionary with all metrics.

### `precision(...)`
```python
precision(predicted, ground_truth, signal_length, tolerance=10)
```
Calculates margin-based precision.

### `recall(...)`
```python
recall(predicted, ground_truth, signal_length, tolerance=10)
```
Calculates margin-based recall.

### `hausdorff_distance(...)`
```python
hausdorff_distance(predicted, ground_truth, signal_length)
```
Calculates the Hausdorff distance between sets of points.

### `rand_index(...)`
```python
rand_index(predicted, ground_truth, signal_length)
```
Calculates the Adjusted Rand Index for the segmentations.

## References
- [[Modules/Metrics|Metrics Module]]
- [[Concepts/Evaluation-Metrics|Evaluation Metrics Concepts]]
