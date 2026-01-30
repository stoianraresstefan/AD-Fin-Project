---
title: Evaluation Metrics
tags: [concept, metrics, evaluation]
---

# Evaluation Metrics

Evaluating changepoint detection is challenging because it involves both the number of points and their locations.

## Margin-Based Metrics

ALPIN uses a **tolerance margin** $M$ (default 10 samples) to define "correct" detections.

### Precision
The fraction of predicted changepoints that fall within $M$ samples of a true changepoint.
$$\text{Precision} = \frac{|\{\hat{t}_j : \exists t_i, |\hat{t}_j - t_i| \le M\}|}{|\{\hat{t}_j\}|}$$

### Recall
The fraction of true changepoints that have at least one predicted point within $M$ samples.
$$\text{Recall} = \frac{|\{t_i : \exists \hat{t}_j, |\hat{t}_j - t_i| \le M\}|}{|\{t_i\}|}$$

## Distance-Based Metrics

### Hausdorff Distance
Measures the "worst-case" error. It is the maximum distance from any point in one set to the nearest point in the other set.
$$d_H(T, \hat{T}) = \max \{ \max_i \min_j |t_i - \hat{t}_j|, \max_j \min_i |\hat{t}_j - t_i| \}$$

## Segmentation-Based Metrics

### Adjusted Rand Index (ARI)
Treats segmentation as a clustering problem. It measures the similarity between the predicted segments and the true segments, corrected for chance.
- **1.0**: Perfect match.
- **0.0**: Random labeling.

## References
- [[Modules/Metrics|Metrics Module]]
- [[API-Reference/Metrics-Functions|Metrics API]]
