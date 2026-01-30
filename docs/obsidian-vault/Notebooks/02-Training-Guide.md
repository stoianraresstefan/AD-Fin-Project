---
title: 02 Training Guide
tags: [notebook, training, cross-validation]
---

# 02 Training Guide

This guide walkthrough the `notebooks/02_training.ipynb` notebook, which demonstrates a robust training workflow for ALPIN.

## Overview

The training notebook focuses on:
1. **Cross-Validation**: Using 10-fold CV to evaluate the stability of the learned $\beta$.
2. **Protocol Comparison**: Comparing the learned $\beta$ for Protocol I (all jumps) vs Protocol II (large jumps only).
3. **Hyperparameter Sweeps**: Visualizing the precision-recall tradeoff across different $\beta$ values.

## Key Insights

### Cross-Validation
The notebook shows that the learned $\beta$ is relatively stable across different folds of synthetic data, typically yielding high precision and recall (>0.95).

### Protocol I vs Protocol II
- **Protocol I**: Learns a smaller $\beta$ (e.g., ~12) to capture all mean shifts.
- **Protocol II**: Learns a larger $\beta$ (e.g., ~50) to ignore small jumps and only detect shifts with amplitude $> 3$.

### Precision-Recall Tradeoff
The grid search visualization clearly shows:
- **Low $\beta$**: High recall, low precision (over-segmentation).
- **High $\beta$**: Low recall, high precision (under-segmentation).
- **ALPIN**: Automatically finds the "elbow" of the curve that minimizes excess risk.

## Summary
This notebook proves that ALPIN generalizes well and adapts its sensitivity based on the provided annotation protocol.

## References
- [[Algorithms/ALPIN-Algorithm|ALPIN Algorithm]]
- [[Modules/Experiments-Sweep|Sweep Module]]
- [[Tutorials/Training-ALPIN|Training Tutorial]]
