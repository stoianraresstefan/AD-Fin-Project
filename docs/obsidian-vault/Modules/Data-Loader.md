---
title: Data Loader Module
tags: [module, data, loader]
---

# Data Loader Module (`alpin/data/loader.py`)

The `loader.py` module provides a flexible interface for loading time series data from external files.

## Protocols

### `DataLoader` (Protocol)
Defines the interface for any data loader implementation.
- `load(path)`: Should return `(signal, labels)`.

## Implementations

### `CSVLoader`
Loads signals from CSV files using `pandas`.

**Features:**
- Configurable `signal_column` and `label_column`.
- Labels are expected as a comma-separated string of integers in the first row of the label column.

## Utility Functions

### `load_directory(path, loader)`
Scans a directory for `.csv` files and loads them all using the provided `loader` implementation.

## Usage Example

```python
from alpin.data.loader import CSVLoader, load_directory

loader = CSVLoader(signal_column="price", label_column="events")
signals, labels = load_directory("data/raw/", loader)
```

## References
- [[Tutorials/Getting-Started|Getting Started Tutorial]]
