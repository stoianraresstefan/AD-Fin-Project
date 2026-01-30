---
title: Making Predictions
tags: [tutorial, inference, visualization]
---

# Making Predictions with ALPIN

Once you have a trained ALPIN model, you can use it to detect changepoints in new signals.

## Basic Prediction

```python
# Assuming 'model' is already fitted
detected_cps = model.predict(new_signal)
```

## Visualizing Results

Visualizing the detections is crucial for qualitative assessment.

```python
from alpin.visualization import plot_signal

plot_signal(
    new_signal, 
    pred_changepoints=detected_cps,
    title="ALPIN Detection"
)
```

## Interactive Exploration

For long signals, use the interactive Plotly view:

```python
from alpin.visualization import plot_signal_interactive

fig = plot_signal_interactive(new_signal, pred_changepoints=detected_cps)
fig.show()
```

## Evaluating Performance

If you have ground truth for your test signal, you can compute metrics:

```python
from alpin.metrics import evaluate_all

metrics = evaluate_all(detected_cps, true_cps, len(new_signal))
print(metrics)
```

## References
- [[Modules/Visualization|Visualization Module]]
- [[Modules/Metrics|Metrics Module]]
