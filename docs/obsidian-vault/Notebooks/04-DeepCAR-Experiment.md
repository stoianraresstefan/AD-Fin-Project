---
title: 04 DeepCAR Experiment
tags: [notebook, forecasting, deepcar, deepar]
---

# 04 DeepCAR Experiment

This guide walkthrough the `notebooks/04_deepcar_experiment.ipynb` notebook, which implements the **DeepCAR** (Changepoint-Aware DeepAR) method.

## Overview

The experiment compares two forecasting approaches:
1. **Baseline DeepAR**: A standard DeepAR model trained on all available data.
2. **ALPIN-Enhanced DeepAR**: A DeepAR model trained using **BatchCP** filtering.

## The BatchCP Method

The core of the experiment is the `ChangePointAwareDataLoader`. It wraps the training data and skips any batch where the **encoder window** (context) contains a changepoint detected by ALPIN.

## Results

The experiment demonstrates a significant improvement in forecast accuracy when using ALPIN-based filtering:

| Metric | Baseline DeepAR | ALPIN-Enhanced | Improvement |
|--------|-----------------|----------------|-------------|
| **MAE** | 1.3815          | 1.2614         | **+8.69%**  |
| **RMSE**| 1.7519          | 1.5961         | **+8.90%**  |

## Key Components

- **`ChangePointAwareDataLoader`**: Implements the filtering logic.
- **`alpin_signals_to_deepar_df`**: Converts synthetic signals to the required long-format DataFrame.
- **`pytorch-forecasting`**: Provides the DeepAR implementation.

## Summary
This notebook provides empirical evidence that accurate changepoint detection with ALPIN can directly improve downstream tasks like time series forecasting.

## References
- [[Architecture/DeepCAR-Architecture|DeepCAR Architecture]]
- [[Algorithms/BatchCP-Filtering|BatchCP Filtering Algorithm]]
- [[Tutorials/Running-DeepCAR|Running DeepCAR Tutorial]]
