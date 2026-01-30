---
title: Training ALPIN
tags: [tutorial, training, data-loading]
---

# Training ALPIN on Your Data

To train ALPIN on your own dataset, you need to provide a list of signals and their corresponding ground truth changepoints.

## Preparing Data

Your data should be in the following format:
- **Signals**: A list of 1D `numpy` arrays.
- **Ground Truths**: A list of lists, where each inner list contains the indices (integers) of the changepoints.

### Loading from CSV

If your data is in CSV files, you can use the `CSVLoader`:

```python
from alpin.data.loader import CSVLoader, load_directory

# Assume CSVs have 'price' column and 'changepoints' column (comma-separated string)
loader = CSVLoader(signal_column="price", label_column="changepoints")
signals, truths = load_directory("path/to/your/csv/folder/", loader)
```

## Training with Cross-Validation

It is recommended to use cross-validation to ensure your learned $\beta$ is robust.

```python
from sklearn.model_selection import KFold
from alpin import ALPIN

kf = KFold(n_splits=5)
betas = []

for train_idx, test_idx in kf.split(signals):
    train_signals = [signals[i] for i in train_idx]
    train_truths = [truths[i] for i in train_idx]
    
    model = ALPIN()
    model.fit(train_signals, train_truths)
    betas.append(model.beta_opt)

print(f"Average beta: {np.mean(betas)}")
```

## References
- [[Modules/Data-Loader|Data Loader Module]]
- [[API-Reference/ALPIN-Class|ALPIN API]]
