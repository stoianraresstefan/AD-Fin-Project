---
title: Visualization Functions API
tags: [api, visualization, plotting]
---

# Visualization Functions API

Functions for plotting signals and results.

## Matplotlib API (`alpin.visualization`)

### `plot_signal(...)`
```python
plot_signal(signal, true_changepoints=None, pred_changepoints=None, title="Signal", figsize=(12, 6), show=True)
```
Plots a signal with vertical lines for changepoints.

### `plot_metrics_comparison(...)`
```python
plot_metrics_comparison(metrics_dict, title="Method Comparison", metric_keys=None, figsize=(10, 6), show=True)
```
Grouped bar chart for metrics.

### `plot_sweep_results(...)`
```python
plot_sweep_results(results_df, x_col, y_col, group_col=None, title="Sweep Results", figsize=(10, 6), show=True)
```
Line plot for parameter sweeps with error bands.

## Plotly API (`alpin.visualization`)

### `plot_signal_interactive(...)`
```python
plot_signal_interactive(signal, true_changepoints=None, pred_changepoints=None, title="Interactive View")
```
Returns a `go.Figure` for interactive exploration.

## References
- [[Modules/Visualization|Visualization Module]]
