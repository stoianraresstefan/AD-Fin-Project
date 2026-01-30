---
title: Time Series Forecasting
tags: [concept, forecasting, deepar]
---

# Time Series Forecasting

Time series forecasting is the use of a model to predict future values based on previously observed values.

## DeepAR

**DeepAR** is a probabilistic forecasting algorithm based on Recurrent Neural Networks (RNNs), specifically LSTMs.

### Key Features
- **Probabilistic**: Instead of a single point estimate, it predicts a probability distribution (e.g., Gaussian or Negative Binomial).
- **Global Model**: It learns from many related time series simultaneously, allowing it to capture complex patterns and handle cold-start problems.

## The Impact of Changepoints

Standard forecasting models assume that the underlying process is stationary or follows a consistent trend/seasonality. **Changepoints** violate this assumption.

If a training window contains a changepoint, the LSTM hidden state will be a mixture of two different regimes, which can degrade the forecast quality for the future window.

## References
- [[Architecture/DeepCAR-Architecture|DeepCAR Architecture]]
- [[Algorithms/BatchCP-Filtering|BatchCP Filtering]]
