---
title: 03 Analysis Guide
tags: [notebook, analysis, comparison]
---

# 03 Analysis Guide

This guide walkthrough the `notebooks/03_analysis.ipynb` notebook, which provides a deep dive into ALPIN's performance and robustness.

## Overview

The analysis notebook covers:
1. **ALPIN vs T-Test**: A head-to-head comparison with a classical statistical baseline.
2. **Noise Robustness**: Evaluating how performance degrades as the signal-to-noise ratio decreases.
3. **Publication Figures**: Generating high-quality plots for reports and papers.

## Key Findings

### ALPIN vs T-Test
ALPIN consistently outperforms the T-test baseline in:
- **Localization Accuracy**: Lower Hausdorff distance.
- **Balance**: Better F1-score (harmonic mean of precision and recall).
- **Segmentation Similarity**: Higher Rand Index.

### Noise Robustness
The study shows that ALPIN is robust up to $\sigma \approx 2.0$. Beyond this point, the noise starts to mask the jumps, leading to a drop in precision (false positives).

| Noise ($\sigma$) | Precision | Recall |
|-----------------|-----------|--------|
| 0.5             | 1.00      | 0.99   |
| 1.0             | 0.98      | 0.97   |
| 2.0             | 0.54      | 0.99   |
| 5.0             | 0.45      | 1.00   |

## Summary
This notebook is essential for understanding the operational limits of ALPIN and its advantages over non-adaptive methods.

## References
- [[Modules/Baselines-TTest|T-Test Baseline]]
- [[Modules/Visualization|Visualization Module]]
- [[Concepts/Evaluation-Metrics|Evaluation Metrics]]
