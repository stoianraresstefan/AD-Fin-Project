---
title: Baselines Module
tags: [module, baselines, ttest]
---

# Baselines Module (`alpin/baselines/ttest.py`)

The `ttest.py` module implements a classical statistical baseline for changepoint detection.

## `TTestBaseline` Class

This detector uses a sliding window approach to identify mean shifts.

### Algorithm
1. Slide a window of size $W$ across the signal.
2. At each point $t$, perform an independent T-test comparing the segment $[t-W, t]$ with $[t, t+W]$.
3. If the p-value is below the significance threshold (e.g., 0.05), mark $t$ as a candidate changepoint.
4. Merge nearby candidates within $W$ distance.

### Parameters
- `window_fraction`: The size of the window as a fraction of the total signal length (default: 0.05).
- `confidence`: The confidence level for the T-test (default: 0.95).

## Usage Example

```python
from alpin.baselines import TTestBaseline

detector = TTestBaseline(window_fraction=0.05)
detected_cps = detector.detect(signal)
```

## References
- [[Notebooks/03-Analysis-Guide|Notebook 03: Analysis and Comparison]]
