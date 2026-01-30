---
title: Data Flow
tags: [architecture, data-flow]
---

# Data Flow

This document illustrates how data moves through the ALPIN system, from generation to evaluation.

## Training Data Flow

The training process learns the optimal penalty parameter $\beta$ from labeled signals.

```mermaid
sequenceDiagram
    participant G as Data Generator
    participant A as ALPIN.fit()
    participant O as Optimizer (L-BFGS-B)
    participant R as Risk Module
    participant P as Partition Solver

    G->>A: Training Signals + Ground Truth CPs
    A->>O: Initial log(beta)
    loop Optimization Iterations
        O->>R: Current beta
        R->>P: Signal + beta
        P->>R: Optimal Partition
        R->>A: Excess Risk (Loss)
        A->>O: Gradient/Loss
    end
    O->>A: Optimal Beta (beta_opt)
```

## Inference Data Flow

Once trained, ALPIN can detect changepoints in new, unlabeled signals.

```mermaid
graph LR
    S[New Signal] --> P[ALPIN.predict]
    B[beta_opt] --> P
    P --> PS[Partition Solver]
    PS --> CP[Detected Changepoints]
    CP --> V[Visualization]
    CP --> E[Evaluation Metrics]
```

## DeepCAR Forecasting Data Flow

In the DeepCAR experiment, ALPIN is used as a preprocessing step to filter training data for a DeepAR model.

```mermaid
graph TD
    subgraph Preprocessing
        DS[Raw Dataset] --> ALPIN[ALPIN Detector]
        ALPIN --> CP[Detected CPs]
    end

    subgraph Filtering
        DS --> BCP[BatchCP Filter]
        CP --> BCP
        BCP --> Clean[Clean Batches]
    end

    subgraph Training
        Clean --> DeepAR[DeepAR Model]
    end

    subgraph Prediction
        DeepAR --> FC[Forecasts]
        DS_test[Test Data] --> DeepAR
    end
```

## Data Formats

- **Signals**: 1D `numpy.ndarray` of floats.
- **Changepoints**: `list[int]` containing 0-indexed sample positions.
- **DeepAR Data**: `pandas.DataFrame` with `time_idx`, `series_id`, and `value` columns.
