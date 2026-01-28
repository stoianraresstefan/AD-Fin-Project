# ALPIN - Adaptive Learning of Penalty for INference

Implementation of the ALPIN algorithm for automatic learning of optimal penalty parameters in changepoint detection, based on the paper:

> T. Truong, C. Oriot, V. Lecomte. "Automatic labeling of piecewise-constant signals". EUSIPCO 2017.

## What is ALPIN?

ALPIN learns the optimal penalty parameter β from labeled training data for changepoint detection in piecewise-constant signals. Instead of manually tuning β, ALPIN:

1. **Warm Start**: Optimizes β on a randomly selected signal
2. **Global Optimization**: Minimizes average excess risk across all training signals using L-BFGS-B

The learned β can then be applied to new, unlabeled signals for automatic changepoint detection.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/AD-Fin-Project.git
cd AD-Fin-Project

# Install dependencies
uv sync
```

## Quick Start

```python
from alpin import ALPIN
from alpin.data.synthetic import generate_synthetic_signals

# Generate synthetic training data
signals, changepoints = generate_synthetic_signals(
    n_signals=100,
    n_samples=500,
    noise_std=1.0,
    seed=42
)

# Split into train/test
train_signals, train_cps = signals[:80], changepoints[:80]
test_signals, test_cps = signals[80:], changepoints[80:]

# Train ALPIN
model = ALPIN()
model.fit(train_signals, train_cps)
print(f"Learned optimal β: {model.beta_opt:.4f}")

# Detect changepoints on new signal
detected = model.predict(test_signals[0])
print(f"Detected changepoints: {detected}")
```

## Notebooks

Interactive tutorials are available in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| [01_quickstart.ipynb](notebooks/01_quickstart.ipynb) | Basic usage and quick examples |
| [02_training.ipynb](notebooks/02_training.ipynb) | Training ALPIN on synthetic and real data |
| [03_analysis.ipynb](notebooks/03_analysis.ipynb) | Performance analysis and visualizations |

## Project Structure

```
alpin/
├── __init__.py           # Package exports
├── core.py               # ALPIN algorithm implementation
├── partition.py          # Optimal partition solver (Pelt)
├── risk.py               # Risk estimation functions
├── metrics.py            # Evaluation metrics (precision, recall, Hausdorff, etc.)
├── visualization.py      # Plotting utilities (matplotlib + plotly)
├── data/
│   ├── synthetic.py      # Synthetic signal generation
│   └── loader.py         # Data loading utilities
├── baselines/
│   └── ttest.py          # T-test baseline for comparison
└── experiments/
    └── sweep.py          # Parameter sweep utilities
```

## Key Features

- **Automatic β learning**: No manual tuning of penalty parameter
- **Two annotation protocols**: 
  - Protocol I: All changepoints
  - Protocol II: Only large-amplitude changes (|jump| > 3)
- **Comprehensive metrics**: Precision, recall, Hausdorff distance, annotation error, Rand index
- **Visualization**: Publication-quality plots with matplotlib and interactive plotly figures
- **Flexible data loading**: Support for synthetic signals and CSV data

## API Reference

### ALPIN Class

```python
from alpin import ALPIN

model = ALPIN(beta_bounds=(1e-6, 1e6))
model.fit(signals, ground_truths)      # Learn optimal β
detected = model.predict(signal)       # Detect changepoints
```

### Synthetic Data Generation

```python
from alpin.data.synthetic import generate_signal, generate_synthetic_signals

# Single signal
signal, cps = generate_signal(n_samples=500, noise_std=1.0, protocol="I")

# Batch of signals
signals, all_cps = generate_synthetic_signals(n_signals=100, n_samples=500)
```

### Evaluation Metrics

```python
from alpin.metrics import evaluate_all, precision, recall, hausdorff_distance

# All metrics at once
metrics = evaluate_all(detected_cps, true_cps, signal_length=500, margin=5)

# Individual metrics
p = precision(detected_cps, true_cps, margin=5)
r = recall(detected_cps, true_cps, margin=5)
h = hausdorff_distance(detected_cps, true_cps)
```

## Dependencies

- Python ≥ 3.12
- numpy
- scipy
- ruptures (for Pelt algorithm)
- matplotlib, plotly (visualization)
- pandas (data handling)

## License

MIT License

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{truong2017automatic,
  title={Automatic labeling of piecewise-constant signals},
  author={Truong, Charles and Oriot, Christophe and Lecomte, Vincent},
  booktitle={EUSIPCO},
  year={2017}
}
```
