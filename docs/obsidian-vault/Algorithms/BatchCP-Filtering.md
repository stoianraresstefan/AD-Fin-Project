---
title: BatchCP Filtering
tags: [algorithm, forecasting, deepcar]
---

# BatchCP Filtering

**BatchCP** is a data filtering technique used in the **DeepCAR** experiment to improve the training of time series forecasting models (like DeepAR).

## The Concept

In time series forecasting with RNNs, the model is trained on many small windows (batches) extracted from the long signal. 
- **Encoder Window**: The context window used by the model to understand the history.
- **Prediction Window**: The future window the model tries to predict.

A batch is "contaminated" if its **encoder window** contains a changepoint. Training on such batches forces the model to learn patterns that span across different regimes, which can lead to inaccurate internal states.

## The BatchCP Rule

> **Skip any training batch where the encoder window overlaps with a detected changepoint.**

## Implementation Logic

The filtering is implemented as a wrapper around a standard PyTorch `DataLoader`:

```python
def _batch_contains_changepoint(self, batch):
    x_dict, y = batch
    encoder_times = x_dict["encoder_time_idx"]
    groups = x_dict["groups"]
    
    for i in range(batch_size):
        start_time = encoder_times[i].min()
        end_time = encoder_times[i].max()
        series_id = f"series_{groups[i, 0]}"
        
        for cp in self.changepoints_dict[series_id]:
            if (start_time - tolerance) <= cp <= (end_time + tolerance):
                return True # Contaminated
    return False
```

## Workflow in DeepCAR

1. **Train ALPIN**: Learn $\beta$ from a small labeled subset or synthetic data.
2. **Detect CPs**: Run `ALPIN.predict()` on the entire training dataset.
3. **Filter Batches**: Use `ChangePointAwareDataLoader` to yield only "clean" batches.
4. **Train Forecaster**: Train DeepAR on the filtered data.

## Benefits

- **Cleaner Gradients**: The model only sees consistent temporal patterns.
- **Better State Initialization**: The LSTM hidden state at the end of the encoder window is more representative of the current regime.
- **Improved Accuracy**: Typically results in ~9% reduction in MAE/RMSE.

## References
- [[Architecture/DeepCAR-Architecture|DeepCAR Architecture]]
- [[Notebooks/04-DeepCAR-Experiment|Notebook 04: DeepCAR Walkthrough]]
