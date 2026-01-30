---
title: Running DeepCAR
tags: [tutorial, forecasting, deepcar]
---

# Running the DeepCAR Experiment

The DeepCAR experiment demonstrates how ALPIN improves forecasting by filtering training data.

## Prerequisites

Ensure you have the forecasting dependencies installed:
```bash
uv sync
```

## Step 1: Prepare the Forecasting Dataset

Convert your signals into the long-format DataFrame required by `pytorch-forecasting`.

```python
from alpin.data.synthetic import alpin_signals_to_deepar_df

df = alpin_signals_to_deepar_df(signals, truths)
```

## Step 2: Detect Changepoints

Use ALPIN to find changepoints in all training signals.

```python
detected_cps = {f"series_{i}": model.predict(s) for i, s in enumerate(signals)}
```

## Step 3: Create the Filtered DataLoader

Use the `ChangePointAwareDataLoader` to skip contaminated batches.

```python
# This class is defined in notebooks/04_deepcar_experiment.ipynb
train_dataloader = ChangePointAwareDataLoader(
    dataloader=original_dataloader,
    changepoints_dict=detected_cps,
    encoder_length=60
)
```

## Step 4: Train and Evaluate

Train your DeepAR model using the filtered dataloader and compare the MAE/RMSE against a baseline model trained on the full dataset.

## References
- [[Architecture/DeepCAR-Architecture|DeepCAR Architecture]]
- [[Notebooks/04-DeepCAR-Experiment|Notebook 04 Walkthrough]]
