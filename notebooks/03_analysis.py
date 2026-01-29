# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ALPIN Analysis and Method Comparison
#
# This notebook provides a comprehensive analysis of the ALPIN algorithm, including:
# 1. **ALPIN vs T-test Baseline Comparison** - Evaluating ALPIN against a classical statistical approach
# 2. **Noise Robustness Analysis** - Testing how performance degrades with increasing noise
# 3. **Publication-Quality Figures** - Generating figures suitable for academic papers
#
# By the end of this notebook, you will understand when ALPIN excels and where its limitations lie.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alpin import ALPIN
from alpin.baselines import TTestBaseline
from alpin.data import generate_synthetic_signals
from alpin.metrics import evaluate_all
from alpin.experiments.sweep import sweep_noise
from alpin.visualization import plot_signal, plot_metrics_comparison, plot_sweep_results

# Reproducibility
np.random.seed(42)

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100

print("Setup complete!")

# %% [markdown]
# ## 2. Data Generation & Model Training
#
# We generate 50 synthetic signals with 200 samples each. These signals contain piecewise constant segments with Gaussian noise, mimicking real-world changepoint detection scenarios.

# %%
# Generate training signals
n_signals = 50
n_samples = 200

signals, truths = generate_synthetic_signals(
    n_signals=n_signals,
    n_samples=n_samples,
    noise_std=1.0,
    seed=42
)

print(f"Generated {len(signals)} signals, each with {n_samples} samples.")
print(f"Average changepoints per signal: {np.mean([len(t) for t in truths]):.1f}")

# %%
# Train ALPIN model
model = ALPIN()
model.fit(signals, truths)

print(f"Learned optimal beta: {model.beta_opt:.4f}")

# %% [markdown]
# ### What ALPIN Learned
#
# The learned $\beta$ parameter controls the trade-off between data fidelity and model complexity:
# - **Higher $\beta$**: Fewer changepoints detected (more conservative)
# - **Lower $\beta$**: More changepoints detected (more sensitive)
#
# ALPIN automatically finds the $\beta$ that minimizes the average excess penalized risk, adapting to the signal characteristics in the training data. Typical learned $\beta$ values range from 10-200 depending on noise levels and jump amplitudes.

# %% [markdown]
# ## 3. ALPIN vs T-Test Baseline Comparison
#
# We compare ALPIN against a classical T-test baseline. The T-test detector uses a sliding window approach, comparing adjacent segments using independent t-tests to detect significant mean shifts.

# %%
# Initialize T-test baseline
ttest_baseline = TTestBaseline(window_fraction=0.05, confidence=0.95)

print(f"T-test parameters:")
print(f"  - Window fraction: {ttest_baseline.window_fraction}")
print(f"  - Confidence level: {ttest_baseline.confidence}")

# %%
# Generate test signals (separate from training)
test_signals, test_truths = generate_synthetic_signals(
    n_signals=10,
    n_samples=n_samples,
    noise_std=1.0,
    seed=999  # Different seed for test data
)

print(f"Generated {len(test_signals)} test signals for evaluation.")

# %%
# Evaluate both methods on test signals
alpin_metrics_list = []
ttest_metrics_list = []

for signal, truth in zip(test_signals, test_truths):
    # ALPIN prediction
    alpin_pred = model.predict(signal)
    alpin_m = evaluate_all(alpin_pred, truth, len(signal), tolerance=10)
    alpin_metrics_list.append(alpin_m)
    
    # T-test prediction
    ttest_pred = ttest_baseline.detect(signal)
    ttest_m = evaluate_all(ttest_pred, truth, len(signal), tolerance=10)
    ttest_metrics_list.append(ttest_m)

# Aggregate metrics
alpin_avg = pd.DataFrame(alpin_metrics_list).mean().to_dict()
ttest_avg = pd.DataFrame(ttest_metrics_list).mean().to_dict()

print("Evaluation complete!")

# %%
# Create comparison table
comparison_df = pd.DataFrame({
    'Metric': list(alpin_avg.keys()),
    'ALPIN': [f"{v:.4f}" for v in alpin_avg.values()],
    'T-Test': [f"{v:.4f}" for v in ttest_avg.values()]
})

print("\n=== Comparison Table: ALPIN vs T-Test Baseline ===")
display(comparison_df)

# %%
# Visualize comparison
metrics_comparison = {
    'ALPIN': alpin_avg,
    'T-Test Baseline': ttest_avg
}

plot_metrics_comparison(
    metrics_comparison,
    title='ALPIN vs T-Test Baseline: Metric Comparison',
    metric_keys=['precision', 'recall', 'rand_index'],
    figsize=(10, 6)
)

# %% [markdown]
# ### Discussion: Which Method Performs Better?
#
# From the comparison above, we can observe:
#
# **ALPIN Advantages:**
# - Learns the optimal penalty from data, adapting to signal characteristics
# - Generally achieves better balance between precision and recall
# - Provides theoretical guarantees based on minimizing penalized risk
#
# **T-Test Advantages:**
# - Simple and interpretable
# - No training required
# - Works well when jump sizes are consistently large
#
# The T-test baseline may struggle with:
# - Small jump sizes relative to noise
# - Variable segment lengths
# - Signals requiring different sensitivity levels

# %% [markdown]
# ## 4. Noise Robustness Analysis
#
# A critical question for any changepoint detection method: **How does performance degrade as noise increases?**
#
# We test ALPIN (with fixed $\beta$) at different noise levels: 0.5, 1.0, 2.0, and 5.0.

# %%
# Run noise sweep experiment
noise_levels = [0.5, 1.0, 2.0, 5.0]

noise_results = sweep_noise(
    n_signals=20,
    n_samples=n_samples,
    noise_levels=noise_levels,
    n_splits=3,
    beta=model.beta_opt,  # Use the learned beta
    seed=42
)

print(f"Noise sweep complete! Results shape: {noise_results.shape}")
display(noise_results.groupby('noise_std').mean())

# %%
# Plot noise sweep results - Precision
plot_sweep_results(
    noise_results,
    x_col='noise_std',
    y_col='precision',
    title='Precision vs Noise Level (ALPIN)',
    figsize=(10, 6)
)

# %%
# Plot noise sweep results - Recall
plot_sweep_results(
    noise_results,
    x_col='noise_std',
    y_col='recall',
    title='Recall vs Noise Level (ALPIN)',
    figsize=(10, 6)
)

# %% [markdown]
# ### Noise Robustness Analysis
#
# From the plots above, we observe:
#
# 1. **Low Noise (σ = 0.5)**: Excellent performance, high precision and recall.
# 2. **Moderate Noise (σ = 1.0)**: Slight degradation but still robust.
# 3. **High Noise (σ = 2.0)**: Noticeable performance drop, especially in precision.
# 4. **Very High Noise (σ = 5.0)**: Significant degradation; the noise magnitude may exceed jump sizes.
#
# **Key Insight**: The learned $\beta$ is optimized for a specific noise level (σ = 1.0 in training). When noise deviates significantly from training conditions, performance suffers. This suggests **adaptive or noise-aware $\beta$ selection** could be beneficial.

# %% [markdown]
# ## 5. Side-by-Side Prediction Example
#
# Let's visualize a single signal with predictions from both methods to understand their differences qualitatively.

# %%
# Select one test signal for detailed comparison
example_idx = 0
example_signal = test_signals[example_idx]
example_truth = test_truths[example_idx]

# Get predictions from both methods
alpin_pred = model.predict(example_signal)
ttest_pred = ttest_baseline.detect(example_signal)

print(f"Ground Truth changepoints: {example_truth}")
print(f"ALPIN predictions: {alpin_pred}")
print(f"T-Test predictions: {ttest_pred}")

# %%
# Plot ALPIN prediction
plot_signal(
    example_signal,
    true_changepoints=example_truth,
    pred_changepoints=alpin_pred,
    title='ALPIN Prediction vs Ground Truth'
)

# %%
# Plot T-Test prediction
plot_signal(
    example_signal,
    true_changepoints=example_truth,
    pred_changepoints=ttest_pred,
    title='T-Test Baseline Prediction vs Ground Truth'
)

# %% [markdown]
# ### Visual Comparison Discussion
#
# Observing the two plots above:
#
# - **ALPIN** typically provides more accurate localization due to the learned penalty that balances detection sensitivity.
# - **T-Test** may detect spurious changepoints in noisy regions or miss subtle changes.
#
# The difference is most pronounced when:
# - Jump amplitudes are small relative to noise
# - Segments have varying lengths
# - The optimal detection sensitivity varies across the signal

# %% [markdown]
# ## 6. Publication-Quality Figures
#
# We now create three polished figures suitable for academic publications.

# %%
# Figure 1: Metrics Comparison Bar Chart
fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

metrics = ['precision', 'recall', 'rand_index']
x = np.arange(len(metrics))
width = 0.35

alpin_vals = [alpin_avg[m] for m in metrics]
ttest_vals = [ttest_avg[m] for m in metrics]

bars1 = ax1.bar(x - width/2, alpin_vals, width, label='ALPIN', color='#2E86AB', edgecolor='white')
bars2 = ax1.bar(x + width/2, ttest_vals, width, label='T-Test', color='#A23B72', edgecolor='white')

ax1.set_ylabel('Score')
ax1.set_title('Method Comparison: ALPIN vs T-Test Baseline')
ax1.set_xticks(x)
ax1.set_xticklabels(['Precision', 'Recall', 'Rand Index'])
ax1.legend(loc='lower right')
ax1.set_ylim(0, 1.1)
ax1.bar_label(bars1, fmt='%.2f', padding=3, fontsize=9)
ax1.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('figure1_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 1 saved as 'figure1_metrics_comparison.png'")

# %%
# Figure 2: Noise Sweep Line Plot
fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=150)

# Aggregate results
noise_agg = noise_results.groupby('noise_std').agg(['mean', 'std']).reset_index()

# Plot precision
ax2.errorbar(
    noise_agg['noise_std'], 
    noise_agg[('precision', 'mean')],
    yerr=noise_agg[('precision', 'std')],
    marker='o', markersize=8, linewidth=2, capsize=4,
    label='Precision', color='#2E86AB'
)

# Plot recall
ax2.errorbar(
    noise_agg['noise_std'], 
    noise_agg[('recall', 'mean')],
    yerr=noise_agg[('recall', 'std')],
    marker='s', markersize=8, linewidth=2, capsize=4,
    label='Recall', color='#A23B72'
)

# Plot rand index
ax2.errorbar(
    noise_agg['noise_std'], 
    noise_agg[('rand_index', 'mean')],
    yerr=noise_agg[('rand_index', 'std')],
    marker='^', markersize=8, linewidth=2, capsize=4,
    label='Rand Index', color='#F18F01'
)

ax2.set_xlabel('Noise Standard Deviation (σ)')
ax2.set_ylabel('Score')
ax2.set_title('ALPIN Performance vs Noise Level')
ax2.legend(loc='lower left')
ax2.set_ylim(0, 1.1)
ax2.set_xlim(0, 5.5)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('figure2_noise_sweep.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 2 saved as 'figure2_noise_sweep.png'")

# %%
# Figure 3: Example Prediction Comparison (side-by-side)
fig3, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, sharex=True)

# Common x-axis
x_axis = np.arange(len(example_signal))

# Top plot: ALPIN
axes[0].plot(x_axis, example_signal, color='#2C3E50', linewidth=1.2, alpha=0.8, label='Signal')
for i, cp in enumerate(example_truth):
    label = 'Ground Truth' if i == 0 else None
    axes[0].axvline(x=cp, color='#27AE60', linestyle='--', linewidth=2, alpha=0.8, label=label)
for i, cp in enumerate(alpin_pred):
    label = 'ALPIN Prediction' if i == 0 else None
    axes[0].axvline(x=cp, color='#E74C3C', linestyle='-', linewidth=2, alpha=0.8, label=label)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('ALPIN Prediction', fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Bottom plot: T-Test
axes[1].plot(x_axis, example_signal, color='#2C3E50', linewidth=1.2, alpha=0.8, label='Signal')
for i, cp in enumerate(example_truth):
    label = 'Ground Truth' if i == 0 else None
    axes[1].axvline(x=cp, color='#27AE60', linestyle='--', linewidth=2, alpha=0.8, label=label)
for i, cp in enumerate(ttest_pred):
    label = 'T-Test Prediction' if i == 0 else None
    axes[1].axvline(x=cp, color='#9B59B6', linestyle='-', linewidth=2, alpha=0.8, label=label)
axes[1].set_xlabel('Time Index')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('T-Test Baseline Prediction', fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figure3_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure 3 saved as 'figure3_prediction_comparison.png'")

# %% [markdown]
# ## 7. Discussion & Recommendations
#
# ### When Does ALPIN Work Well?
#
# 1. **Clean to Moderate Noise**: ALPIN excels when the signal-to-noise ratio is reasonable (σ ≤ 2).
# 2. **Consistent Signal Properties**: When training and test signals share similar characteristics.
# 3. **Learned Penalty Advantage**: The learned $\beta$ adapts to the specific data distribution, outperforming fixed heuristics.
# 4. **Sufficient Training Data**: Performance improves with more diverse training examples.
#
# ### When Does ALPIN Struggle?
#
# 1. **High Noise**: When σ approaches or exceeds jump magnitudes, detection becomes unreliable.
# 2. **Few Training Samples**: With limited training data, $\beta$ may not generalize well.
# 3. **Distribution Shift**: If test signals differ significantly from training (e.g., different noise levels or segment structures).
# 4. **Very Short Signals**: The partition optimization may have limited statistical power.
#
# ### T-Test Baseline: Pros and Cons
#
# **Advantages:**
# - No training required; works out-of-the-box
# - Interpretable statistical foundation
# - Fast computation for real-time applications
#
# **Disadvantages:**
# - Fixed sensitivity; cannot adapt to data
# - Window size selection is critical and often manual
# - May produce false positives in highly variable signals
#
# ### Recommendations for Practitioners
#
# | Scenario | Recommended Method |
# |----------|-------------------|
# | Labeled training data available | **ALPIN** |
# | No training data, need quick results | T-Test Baseline |
# | High noise (σ > 3) | Consider noise-robust preprocessing |
# | Real-time detection needed | T-Test (faster) or pre-trained ALPIN |
# | Research/publication quality | **ALPIN** with cross-validation |
#
# ### Future Directions
#
# 1. **Noise-Adaptive $\beta$**: Automatically adjust $\beta$ based on estimated noise level.
# 2. **Online Learning**: Update $\beta$ as new labeled data becomes available.
# 3. **Ensemble Methods**: Combine ALPIN with other detectors for robustness.
# 4. **Multi-scale Analysis**: Apply ALPIN at multiple resolutions.

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we:
#
# 1. **Trained ALPIN** on 50 synthetic signals and learned an optimal $\beta$.
# 2. **Compared ALPIN to T-Test Baseline** using precision, recall, and Rand Index.
# 3. **Analyzed noise robustness** across noise levels from 0.5 to 5.0.
# 4. **Created publication-quality figures** for academic use.
# 5. **Discussed practical recommendations** for when to use each method.
#
# ALPIN's ability to learn from data gives it an edge over classical methods when labeled examples are available, but practitioners should be mindful of training-test distribution alignment and noise conditions.
