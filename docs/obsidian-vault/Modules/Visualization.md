---
title: Visualization Module
tags: [module, visualization, plotting]
---

# Visualization Module (`alpin/visualization.py`)

The `visualization.py` module provides tools for plotting signals, changepoints, and experiment results. It supports both static (Matplotlib) and interactive (Plotly) plots.

## Static Plotting (Matplotlib)

### `plot_signal(signal, true_changepoints=None, pred_changepoints=None, ...)`
Plots the 1D signal with vertical lines for ground truth (dashed green) and predicted (solid red) changepoints.

### `plot_metrics_comparison(metrics_dict, ...)`
Creates a grouped bar chart comparing metrics (Precision, Recall, etc.) across different methods.

### `plot_sweep_results(results_df, x_col, y_col, ...)`
Plots the results of a parameter sweep, including error bands if multiple runs/folds are present.

## Interactive Plotting (Plotly)

### `plot_signal_interactive(signal, true_changepoints=None, pred_changepoints=None, ...)`
Creates a Plotly figure that allows zooming and hovering over the signal and changepoints.

## Usage Example

```python
from alpin.visualization import plot_signal

plot_signal(
    signal, 
    true_changepoints=[100, 200], 
    pred_changepoints=[105, 198],
    title="ALPIN Detection Result"
)
```

## References
- [[Modules/Experiments-Sweep|Experiments Sweep Module]]
- [[Notebooks/03-Analysis-Guide|Notebook 03: Analysis]]
