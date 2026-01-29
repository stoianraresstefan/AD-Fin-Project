# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: AD-Fin-Project (3.12.6)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ALPIN Training Workflow
#
# This notebook demonstrates the comprehensive training workflow for the ALPIN (Adaptive Learning of Penalty for INference) algorithm. We will cover:
# 1. Synthetic data generation
# 2. Model training with cross-validation
# 3. Performance evaluation using multiple metrics
# 4. Comparison of labeling protocols (Protocol I vs. Protocol II)
# 5. Hyperparameter sweep visualization

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from alpin import ALPIN
from alpin.data.synthetic import generate_synthetic_signals
from alpin.metrics import evaluate_all
from alpin.experiments.sweep import sweep_beta
from alpin.visualization import plot_signal, plot_sweep_results

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 1. Load Synthetic Data
#
# We generate 100 synthetic signals following the specifications in the EUSIPCO 2017 paper. Each signal has 500 samples and between 3 to 7 changepoints.

# %%
n_signals = 100
n_samples = 500

signals, truths = generate_synthetic_signals(
    n_signals=n_signals, 
    n_samples=n_samples, 
    noise_std=1.0, 
    seed=42,
    protocol="I"
)

print(f"Generated {len(signals)} signals with Protocol I labels.")
plot_signal(signals[0], truths[0], title="Example Synthetic Signal (Protocol I)")
plt.show()

# %% [markdown]
# ## 2. Training with Cross-Validation
#
# We use 10-fold cross-validation to evaluate the stability of the learned penalty parameter $\beta$ and the model's generalization performance.
#
# ### Why Cross-Validation?
# Cross-validation ensures that our evaluation is not biased by a specific train/test split. For each fold, we:
# 1. Train ALPIN on 90% of the data to learn $\beta_{opt}$.
# 2. Evaluate the learned $\beta_{opt}$ on the remaining 10% (test set).

# %%
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(signals), total=10, desc="CV Folds")):
    # Split data
    train_signals = [signals[i] for i in train_idx]
    train_truths = [truths[i] for i in train_idx]
    test_signals = [signals[i] for i in test_idx]
    test_truths = [truths[i] for i in test_idx]
    
    # Train model
    model = ALPIN()
    model.fit(train_signals, train_truths)
    beta_fold = model.beta_opt
    
    # Evaluate on test set
    fold_metrics = []
    for s, t in zip(test_signals, test_truths):
        pred = model.predict(s)
        m = evaluate_all(pred, t, len(s), tolerance=10)
        fold_metrics.append(m)
    
    # Average metrics for this fold
    avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
    avg_metrics['beta'] = beta_fold
    avg_metrics['fold'] = fold
    cv_results.append(avg_metrics)

df_cv = pd.DataFrame(cv_results)
display(df_cv)

# %% [markdown]
# ### Summary of CV Results
# The table below shows the average performance across all 10 folds.

# %%
summary = df_cv.drop(columns=['fold']).agg(['mean', 'std']).T
display(summary)

# %% [markdown]
# ## 3. Protocol I vs. Protocol II
#
# The paper defines two labeling protocols:
# - **Protocol I**: All changepoints are labeled.
# - **Protocol II**: Only changepoints with a jump amplitude $|\Delta| > 3$ are labeled.
#
# We expect Protocol II to result in a larger $\beta$ (higher penalty) because it ignores smaller jumps.

# %%
protocols = ["I", "II"]
protocol_comparison = []

for p in protocols:
    # Generate data for this protocol
    p_signals, p_truths = generate_synthetic_signals(n_signals=50, seed=123, protocol=p)
    
    # Train on full set
    model = ALPIN()
    model.fit(p_signals, p_truths)
    
    protocol_comparison.append({
        "Protocol": p,
        "Learned Beta": model.beta_opt,
        "Avg. Changepoints": np.mean([len(t) for t in p_truths])
    })

df_proto = pd.DataFrame(protocol_comparison)
display(df_proto)

# %% [markdown]
# ## 4. Hyperparameter Sweep Visualization
#
# We can visualize how the choice of $\beta$ affects different metrics using a grid search (sweep).

# %%
beta_range = np.logspace(0, 3, 20)
sweep_results = sweep_beta(signals[:20], truths[:20], beta_range, n_splits=3)

# Aggregate results
sweep_agg = sweep_results.groupby('beta').mean().reset_index()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sweep_agg['beta'], sweep_agg['precision'], label='Precision')
plt.plot(sweep_agg['beta'], sweep_agg['recall'], label='Recall')
plt.xscale('log')
plt.xlabel('Beta')
plt.ylabel('Score')
plt.title('Precision-Recall Tradeoff')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sweep_agg['beta'], sweep_agg['rand_index'], label='Rand Index', color='green')
plt.xscale('log')
plt.xlabel('Beta')
plt.ylabel('Score')
plt.title('Rand Index vs. Beta')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Conclusion
#
# In this notebook, we demonstrated that ALPIN successfully learns an optimal penalty parameter $\beta$ that balances precision and recall. We also observed how different labeling protocols influence the learned parameter, with Protocol II leading to a more conservative model that ignores small jumps.
