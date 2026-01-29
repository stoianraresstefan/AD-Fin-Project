# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: alpin (3.12.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ALPIN Quickstart Demo
#
# This notebook provides a 5-minute introduction to the **ALPIN** (Adaptive Learning of Penalty for INference) algorithm. 
# ALPIN is a supervised approach for learning the optimal penalty parameter $beta$ in change point detection problems.
#
# Instead of manually tuning $beta$, ALPIN learns it from a set of annotated signals by minimizing the average excess penalized risk.

# %% [markdown]
# ## 1. Installation
#
# Ensure you have the dependencies installed. If you are using `uv`, you can sync the environment:

# %%
# !uv sync

# %% [markdown]
# ## 2. Generate Synthetic Data
#
# We'll generate 5 synthetic signals with 3-7 changepoints each, following the protocol described in the ALPIN paper.

# %%
import numpy as np
from alpin.data import generate_synthetic_signals
from alpin.visualization import plot_signal

# Generate 5 signals for training/demo
signals, truths = generate_synthetic_signals(n_signals=5, n_samples=500, noise_std=1.0, seed=42)

print(f"Generated {len(signals)} signals.")
plot_signal(signals[0], truths[0], title="Example Synthetic Signal with Ground Truth")

# %% [markdown]
# ## 3. Train ALPIN Model
#
# We initialize the ALPIN model and fit it to our training signals. The model will learn the optimal $beta$ that best matches the ground truth partitions.

# %%
from alpin import ALPIN

model = ALPIN()
model.fit(signals, truths)

print(f"Learned optimal beta: {model.beta_opt:.4f}")

# %% [markdown]
# ## 4. Make Predictions
#
# Now we use the learned $beta$ to predict changepoints on a new signal.

# %%
# Generate a new test signal
test_signals, test_truths = generate_synthetic_signals(n_signals=1, n_samples=500, noise_std=1.0, seed=99)
test_signal = test_signals[0]
test_truth = test_truths[0]

# Predict changepoints
predictions = model.predict(test_signal)

print(f"True changepoints: {test_truth}")
print(f"Predicted changepoints: {predictions}")

# %% [markdown]
# ## 5. Visualize Results
#
# Let's compare the predicted changepoints with the ground truth.

# %%
plot_signal(
    test_signal, 
    true_changepoints=test_truth, 
    pred_changepoints=predictions, 
    title="ALPIN Prediction vs Ground Truth"
)

# %% [markdown]
# ## 6. Show Metrics
#
# Finally, we evaluate the performance using standard metrics.

# %%
from alpin.metrics import evaluate_all

metrics = evaluate_all(predictions, test_truth, signal_length=len(test_signal))

for name, value in metrics.items():
    print(f"{name:20s}: {value:.4f}")
