---
title: Getting Started
tags: [tutorial, installation, usage]
---

# Getting Started with ALPIN

This tutorial will guide you through the installation and basic usage of the ALPIN library.

## Installation

ALPIN uses `uv` for fast and reliable dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/AD-Fin-Project.git
cd AD-Fin-Project

# Install dependencies and create virtual environment
uv sync
```

## Basic Usage

The simplest way to use ALPIN is to generate some synthetic data and fit a model.

```python
import numpy as np
from alpin import ALPIN
from alpin.data.synthetic import generate_synthetic_signals

# 1. Generate data
signals, truths = generate_synthetic_signals(n_signals=10, n_samples=500)

# 2. Initialize and Train
model = ALPIN()
model.fit(signals, truths)
print(f"Learned beta: {model.beta_opt:.4f}")

# 3. Predict
new_signal, _ = generate_synthetic_signals(n_signals=1)
detected = model.predict(new_signal[0])
print(f"Detected changepoints: {detected}")
```

## Next Steps
- [[Tutorials/Training-ALPIN|Learn how to train on your own data]]
- [[Tutorials/Running-DeepCAR|Run the DeepCAR forecasting experiment]]
