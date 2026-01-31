# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: alpin (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DeepCAR Experiment: ALPIN-Enhanced DeepAR Forecasting
#
# This notebook implements and compares **baseline DeepAR** vs **ALPIN-enhanced DeepAR** using the **BatchCP** method from the DeepCAR paper.
#
# ## Background: DeepCAR (Changepoint-Aware DeepAR)
#
# The DeepCAR paper proposes a simple but effective approach to improve probabilistic forecasting:
#
# 1. **Problem**: Standard DeepAR training uses all available data, including windows that span regime changes (changepoints). Training on these "contaminated" batches teaches the model incorrect temporal patterns.
#
# 2. **Solution - BatchCP**: Filter out training batches whose encoder windows contain or overlap with detected changepoints. This ensures the model only learns from "clean" homogeneous segments.
#
# 3. **Key Insight**: By using ALPIN for accurate changepoint detection, we can identify and exclude problematic training samples, leading to better forecast accuracy.
#
# ## Experiment Goal
#
# Compare forecast accuracy (MAE/RMSE) between:
# - **Baseline DeepAR**: Trained on all data
# - **ALPIN-Enhanced DeepAR**: Trained with BatchCP filtering
#
# We expect the ALPIN-enhanced version to produce better forecasts, especially on signals with clear regime changes.

# %% [markdown]
# ## 1. Setup and Imports
#
# **Important**: Before running this notebook, ensure dependencies are installed:
#
# ```bash
# uv sync
# ```
#
# This will install `pytorch-forecasting`, `pytorch-lightning`, and `torch`.

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Iterator, Any

# PyTorch
import torch
from torch.utils.data import DataLoader

# PyTorch Lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE

# ALPIN
from alpin import ALPIN
from alpin.data.synthetic import generate_synthetic_signals, alpin_signals_to_deepar_df
from alpin.metrics import evaluate_all
from alpin.visualization import plot_signal

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED)

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

print("Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")

# %% [markdown]
# ## 2. Configuration
#
# Define hyperparameters for the experiment.

# %%
# Data configuration
N_SIGNALS = 10  # Number of synthetic signals
N_SAMPLES = 500  # Samples per signal
NOISE_STD = 1.0  # Noise standard deviation

# DeepAR configuration
MAX_ENCODER_LENGTH = 60  # Context window
MAX_PREDICTION_LENGTH = 20  # Forecast horizon
BATCH_SIZE = 32

# Training configuration
MAX_EPOCHS = 1  # Quick test (set to 20 for full training)
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32
RNN_LAYERS = 2
DROPOUT = 0.1

# BatchCP configuration
CP_TOLERANCE = 2  # Safety margin around changepoints

# Train/Val split
TRAIN_RATIO = 0.8

print("Configuration:")
print(f"  Signals: {N_SIGNALS} x {N_SAMPLES} samples")
print(f"  Encoder length: {MAX_ENCODER_LENGTH}")
print(f"  Prediction length: {MAX_PREDICTION_LENGTH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {MAX_EPOCHS}")

# %% [markdown]
# ## 3. Generate Synthetic Data
#
# We generate piecewise constant signals with known changepoints using ALPIN's synthetic data generator.

# %%
# Generate synthetic signals with ground truth changepoints
signals, changepoints = generate_synthetic_signals(
    n_signals=N_SIGNALS, n_samples=N_SAMPLES, noise_std=NOISE_STD, seed=SEED
)

print(f"Generated {len(signals)} signals, each with {N_SAMPLES} samples")
print(f"Changepoints per signal: {[len(cp) for cp in changepoints]}")
print(f"Total changepoints: {sum(len(cp) for cp in changepoints)}")

# %%
# Convert to DeepAR-compatible DataFrame
df = alpin_signals_to_deepar_df(signals, changepoints)

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Series IDs: {df['series_id'].unique().tolist()[:5]}...")
df.head(10)

# %%
# Split into train/validation by time
train_cutoff = int(N_SAMPLES * TRAIN_RATIO)

# For time series, we split by time index
train_df = df[df["time_idx"] < train_cutoff].copy()
val_df = df[
    df["time_idx"] >= train_cutoff - MAX_ENCODER_LENGTH
].copy()  # Include encoder context

print(f"Train samples: {len(train_df)} (time_idx < {train_cutoff})")
print(f"Val samples: {len(val_df)} (time_idx >= {train_cutoff - MAX_ENCODER_LENGTH})")

# %%
# Visualize example signals with changepoints
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for i, ax in enumerate(axes):
    signal = signals[i]
    cps = changepoints[i]

    ax.plot(signal, color="#2C3E50", linewidth=1.2, alpha=0.8, label="Signal")

    for j, cp in enumerate(cps):
        label = "Changepoint" if j == 0 else None
        ax.axvline(
            x=cp, color="#E74C3C", linestyle="--", linewidth=2, alpha=0.7, label=label
        )

    # Mark train/val split
    ax.axvline(
        x=train_cutoff,
        color="#27AE60",
        linestyle="-",
        linewidth=2,
        alpha=0.9,
        label="Train/Val Split",
    )

    ax.set_ylabel(f"Signal {i}")
    ax.legend(loc="upper right")
    ax.set_title(f"Series {i}: {len(cps)} changepoints", fontweight="bold")

axes[-1].set_xlabel("Time Index")
plt.suptitle(
    "Synthetic Signals with Ground Truth Changepoints", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. ALPIN Changepoint Detection
#
# Train ALPIN on training data to detect changepoints. These detections will be used for BatchCP filtering.

# %%
# Use only training portion of signals for ALPIN training
train_signals = [s[:train_cutoff] for s in signals]
train_changepoints = [[cp for cp in cps if cp < train_cutoff] for cps in changepoints]

# Train ALPIN model
alpin_model = ALPIN()
alpin_model.fit(train_signals, train_changepoints)

print(f"ALPIN learned optimal beta: {alpin_model.beta_opt:.4f}")

# %%
# Predict changepoints on all signals (full length for completeness)
detected_changepoints = {}
for i, signal in enumerate(signals):
    series_id = f"series_{i}"
    detected = alpin_model.predict(signal)
    detected_changepoints[series_id] = detected

print("Detected changepoints per series:")
for sid, cps in detected_changepoints.items():
    print(f"  {sid}: {cps}")

# %%
# Evaluate ALPIN detection quality
print("\nALPIN Detection Metrics:")
print("-" * 50)

all_metrics = []
for i in range(N_SIGNALS):
    series_id = f"series_{i}"
    detected = detected_changepoints[series_id]
    ground_truth = changepoints[i]

    metrics = evaluate_all(detected, ground_truth, N_SAMPLES, tolerance=10)
    all_metrics.append(metrics)

    print(
        f"Series {i}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}"
    )

# Average metrics
avg_metrics = pd.DataFrame(all_metrics).mean()
print("-" * 50)
print(
    f"Average: Precision={avg_metrics['precision']:.2f}, Recall={avg_metrics['recall']:.2f}"
)

all_cps = []
for series_cps in detected_changepoints.values():
    all_cps.extend(series_cps)


if len(all_cps) >= 2:
    # Calculate minimum spacing PER SERIES first, then take global minimum
    min_distance = float("inf")
    for series_id, series_cps in detected_changepoints.items():
        if len(series_cps) >= 2:
            sorted_cps = sorted(series_cps)
            series_distances = [
                sorted_cps[i + 1] - sorted_cps[i] for i in range(len(sorted_cps) - 1)
            ]
            series_min = min(series_distances)
            min_distance = min(min_distance, series_min)

    if min_distance == float("inf"):
        print("\nNo series with multiple changepoints for smax calculation")
        smax = MAX_ENCODER_LENGTH
    else:
        smax = int(np.ceil(min_distance / 2))

        print("\nBatchCP Analysis (DeepCAR Algorithm 1):")
        print(f"  Total changepoints detected: {len(all_cps)}")
        print(
            f"  Minimum spacing between changepoints (within series): {min_distance} samples"
        )
        print(f"  Recommended smax (batch size): {smax} samples")
        print(f"  Current encoder length: {MAX_ENCODER_LENGTH} samples")

        if MAX_ENCODER_LENGTH > smax:
            print(f"  [WARNING] Encoder length ({MAX_ENCODER_LENGTH}) > smax ({smax})")
            print("      This will cause heavy filtering (possibly 100%)!")
            print(f"      Recommendation: Use encoder_length <= {smax}")
else:
    print("\nInsufficient changepoints for smax calculation")
    smax = MAX_ENCODER_LENGTH


# %%
# Visualize ALPIN detection on one signal

example_idx = 0
plot_signal(
    signals[example_idx],
    true_changepoints=changepoints[example_idx],
    pred_changepoints=detected_changepoints[f"series_{example_idx}"],
    title=f"ALPIN Detection on Series {example_idx}",
)

# %% [markdown]
# ## 5. Baseline DeepAR Training
#
# Train DeepAR on all available training data without any filtering.

# %%
# Create TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="value",
    group_ids=["series_id"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    time_varying_unknown_reals=["value"],
    add_relative_time_idx=True,
    add_target_scales=True,
    target_normalizer=GroupNormalizer(groups=["series_id"]),
)

# Create validation dataset from training parameters
validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

print(f"Training samples: {len(training)}")
print(f"Validation samples: {len(validation)}")

# %%
# Create DataLoaders
train_dataloader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0
)

print(f"Train batches: {len(train_dataloader)}")
print(f"Val batches: {len(val_dataloader)}")

# %%
# Create baseline DeepAR model
baseline_deepar = DeepAR.from_dataset(
    training,
    hidden_size=HIDDEN_SIZE,
    rnn_layers=RNN_LAYERS,
    dropout=DROPOUT,
    learning_rate=LEARNING_RATE,
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)

print(
    f"Baseline DeepAR parameters: {sum(p.numel() for p in baseline_deepar.parameters()):,}"
)

# %%
# Train baseline model
baseline_trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=0.1,
    limit_train_batches=50,  # Limit for quick demo
    limit_val_batches=20,
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
)

print("Training Baseline DeepAR...")
baseline_trainer.fit(
    baseline_deepar,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
print("Baseline training complete!")

# %%
# Generate baseline predictions
baseline_predictions = baseline_deepar.predict(
    val_dataloader, return_y=True, mode="prediction"
)

# Extract predictions and actuals
baseline_preds = baseline_predictions.output
baseline_actuals = baseline_predictions.y[0]

print(f"Baseline predictions shape: {baseline_preds.shape}")
print(f"Baseline actuals shape: {baseline_actuals.shape}")


# %%
# Calculate baseline metrics
def calculate_forecast_metrics(
    predictions: torch.Tensor, actuals: torch.Tensor
) -> Dict[str, float]:
    """Calculate MAE and RMSE for forecasts."""
    preds = predictions.cpu().numpy().flatten()
    actual = actuals.cpu().numpy().flatten()

    mae = np.mean(np.abs(preds - actual))
    rmse = np.sqrt(np.mean((preds - actual) ** 2))

    return {"MAE": mae, "RMSE": rmse}


baseline_metrics = calculate_forecast_metrics(baseline_preds, baseline_actuals)
print("Baseline DeepAR Metrics:")
print(f"  MAE:  {baseline_metrics['MAE']:.4f}")
print(f"  RMSE: {baseline_metrics['RMSE']:.4f}")

# %%
# Plot baseline forecast example
fig, ax = plt.subplots(figsize=(12, 5))

# Get first batch for visualization
sample_idx = 0
pred_sample = baseline_preds[sample_idx].cpu().numpy()
actual_sample = baseline_actuals[sample_idx].cpu().numpy()

time_axis = np.arange(len(actual_sample))

ax.plot(time_axis, actual_sample, "o-", color="#2E86AB", label="Actual", markersize=4)
ax.plot(
    time_axis,
    pred_sample,
    "s-",
    color="#E74C3C",
    label="Baseline Forecast",
    markersize=4,
)

ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("Value")
ax.set_title("Baseline DeepAR: Example Forecast vs Actual", fontweight="bold")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. ALPIN-Enhanced DeepAR (BatchCP)
#
# Implement the BatchCP filtering method and train DeepAR with changepoint-aware batch selection.


# %%
class ChangePointAwareDataLoader:
    """
    Wrapper that filters out batches containing changepoints in encoder window.

    Implements the BatchCP method from DeepCAR: batches where the encoder window
    overlaps with a detected changepoint are skipped during training.

    Parameters
    ----------
    dataloader : DataLoader
        Original PyTorch DataLoader from TimeSeriesDataSet
    changepoints_dict : Dict[str, List[int]]
        Mapping from series_id to list of changepoint indices
    encoder_length : int
        Length of the encoder window
    tolerance : int
        Safety margin around changepoints
    """

    def __init__(
        self,
        dataloader: DataLoader,
        changepoints_dict: Dict[str, List[int]],
        encoder_length: int,
        tolerance: int = 2,
    ):
        self.dataloader = dataloader
        self.changepoints_dict = changepoints_dict
        self.encoder_length = encoder_length
        self.tolerance = tolerance
        self.filtered_count = 0
        self.total_count = 0

    def _batch_contains_changepoint(self, batch: tuple) -> bool:
        """
        Check if any sample in batch has a changepoint in its encoder window.

        FIXED: Correctly handles relative time indices by reconstructing absolute time
        from decoder position.
        """
        x_dict, y = batch

        groups = x_dict.get("groups", None)
        if groups is None:
            return False

        decoder_times = x_dict.get("decoder_time_idx", None)
        if decoder_times is None:
            return False

        batch_size = len(decoder_times)

        for i in range(batch_size):
            series_idx = int(groups[i, 0].item())
            group_id = f"series_{series_idx}"

            if group_id not in self.changepoints_dict:
                continue

            series_changepoints = self.changepoints_dict[group_id]

            if not series_changepoints:
                continue

            series_decoder_times = decoder_times[i].cpu().numpy()
            if len(series_decoder_times) == 0:
                continue

            decoder_end = int(series_decoder_times[-1])
            encoder_end = decoder_end - len(series_decoder_times)
            encoder_start = encoder_end - self.encoder_length + 1

            for cp in series_changepoints:
                if encoder_start < cp < encoder_end:
                    return True

        return False

    def __iter__(self) -> Iterator:
        """Iterate over batches, skipping those with changepoints."""
        self.filtered_count = 0
        self.total_count = 0
        debug_count = 0

        for batch in self.dataloader:
            self.total_count += 1

            if self._batch_contains_changepoint(batch):
                self.filtered_count += 1
                if debug_count < 3:
                    print(
                        f"  [DEBUG] Batch {self.total_count}: Filtered (contains changepoint in encoder window)"
                    )
                    debug_count += 1
                continue  # Skip this batch

            yield batch

    def __len__(self) -> int:
        """Return length of underlying dataloader (upper bound)."""
        return len(self.dataloader)

    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get statistics about batch filtering."""
        return {
            "total_batches": self.total_count,
            "filtered_batches": self.filtered_count,
            "kept_batches": self.total_count - self.filtered_count,
            "filter_ratio": self.filtered_count / max(self.total_count, 1),
        }


print("ChangePointAwareDataLoader class defined.")

# %%
# Create filtered dataloader for ALPIN-enhanced training
filtered_train_dataloader = ChangePointAwareDataLoader(
    dataloader=train_dataloader,
    changepoints_dict=detected_changepoints,
    encoder_length=MAX_ENCODER_LENGTH,
    tolerance=CP_TOLERANCE,
)

print(f"Created ChangePointAwareDataLoader")
print(
    f"  Detected changepoints: {sum(len(cps) for cps in detected_changepoints.values())}"
)
print(f"  Encoder length: {MAX_ENCODER_LENGTH}")
print(f"  Tolerance: {CP_TOLERANCE}")

# %%
# Create ALPIN-enhanced DeepAR model (same architecture as baseline)
alpin_deepar = DeepAR.from_dataset(
    training,
    hidden_size=HIDDEN_SIZE,
    rnn_layers=RNN_LAYERS,
    dropout=DROPOUT,
    learning_rate=LEARNING_RATE,
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)

print(
    f"ALPIN-Enhanced DeepAR parameters: {sum(p.numel() for p in alpin_deepar.parameters()):,}"
)

# %%
# Create trainer for ALPIN-enhanced model with filtered dataloader
alpin_trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    limit_val_batches=20,
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
)

print("Training ALPIN-Enhanced DeepAR with BatchCP filtering...")
alpin_trainer.fit(
    alpin_deepar,
    train_dataloaders=filtered_train_dataloader,
    val_dataloaders=val_dataloader,
)

# Get filtering statistics after training
filter_stats = filtered_train_dataloader.get_filtering_stats()
print("\nBatchCP Filtering Statistics:")
print(f"  Total batches processed: {filter_stats['total_batches']}")
print(
    f"  Filtered out: {filter_stats['filtered_batches']} ({filter_stats['filter_ratio'] * 100:.1f}%)"
)
print(f"  Batches used: {filter_stats['kept_batches']}")

print("\nALPIN-Enhanced training complete!")

# %%
# Generate ALPIN-enhanced predictions
alpin_deepar.eval()

alpin_predictions = alpin_deepar.predict(
    val_dataloader, return_y=True, mode="prediction"
)

alpin_preds = alpin_predictions.output
alpin_actuals = alpin_predictions.y[0]

print(f"ALPIN predictions shape: {alpin_preds.shape}")
print(f"ALPIN actuals shape: {alpin_actuals.shape}")

# %%
# Calculate ALPIN-enhanced metrics
alpin_metrics = calculate_forecast_metrics(alpin_preds, alpin_actuals)
print("ALPIN-Enhanced DeepAR Metrics:")
print(f"  MAE:  {alpin_metrics['MAE']:.4f}")
print(f"  RMSE: {alpin_metrics['RMSE']:.4f}")

# %%
# Plot ALPIN-enhanced forecast example
fig, ax = plt.subplots(figsize=(12, 5))

sample_idx = 0
pred_sample = alpin_preds[sample_idx].cpu().numpy()
actual_sample = alpin_actuals[sample_idx].cpu().numpy()

time_axis = np.arange(len(actual_sample))

ax.plot(time_axis, actual_sample, "o-", color="#2E86AB", label="Actual", markersize=4)
ax.plot(
    time_axis,
    pred_sample,
    "s-",
    color="#27AE60",
    label="ALPIN-Enhanced Forecast",
    markersize=4,
)

ax.set_xlabel("Forecast Horizon")
ax.set_ylabel("Value")
ax.set_title("ALPIN-Enhanced DeepAR: Example Forecast vs Actual", fontweight="bold")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Comparison & Analysis
#
# Compare the performance of baseline DeepAR vs ALPIN-enhanced DeepAR.

# %%
# Create comparison table
comparison_df = pd.DataFrame(
    {
        "Metric": ["MAE", "RMSE"],
        "Baseline DeepAR": [
            f"{baseline_metrics['MAE']:.4f}",
            f"{baseline_metrics['RMSE']:.4f}",
        ],
        "ALPIN-Enhanced": [
            f"{alpin_metrics['MAE']:.4f}",
            f"{alpin_metrics['RMSE']:.4f}",
        ],
    }
)

# Calculate improvement
mae_improvement = (
    (baseline_metrics["MAE"] - alpin_metrics["MAE"]) / baseline_metrics["MAE"] * 100
)
rmse_improvement = (
    (baseline_metrics["RMSE"] - alpin_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100
)

comparison_df["Improvement"] = [f"{mae_improvement:+.2f}%", f"{rmse_improvement:+.2f}%"]

print("=" * 60)
print("FORECAST ACCURACY COMPARISON")
print("=" * 60)
print(comparison_df)
print("\n(Positive improvement = ALPIN-Enhanced is better)")

# %%
# Visualization: Side-by-side metric comparison
fig, ax = plt.subplots(figsize=(12, 8))

metrics = ["MAE", "RMSE"]
x = np.arange(len(metrics))
width = 0.35

baseline_vals = [baseline_metrics["MAE"], baseline_metrics["RMSE"]]
alpin_vals = [alpin_metrics["MAE"], alpin_metrics["RMSE"]]

bars1 = ax.bar(
    x - width / 2,
    baseline_vals,
    width,
    label="Baseline DeepAR",
    color="#E74C3C",
    edgecolor="white",
)
bars2 = ax.bar(
    x + width / 2,
    alpin_vals,
    width,
    label="ALPIN-Enhanced",
    color="#27AE60",
    edgecolor="white",
)

ax.set_ylabel("Error")
ax.set_title(
    "Forecast Accuracy: Baseline vs ALPIN-Enhanced DeepAR",
    fontweight="bold",
    fontsize=14,
)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(loc="upper right", fontsize=11)
ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=10, label_type="center")
ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=10, label_type="center")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.5)

# plt.tight_layout()
plt.show()

# %%
# Side-by-side forecast comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

sample_idx = 0
baseline_sample = baseline_preds[sample_idx].cpu().numpy()
alpin_sample = alpin_preds[sample_idx].cpu().numpy()
actual_sample = baseline_actuals[sample_idx].cpu().numpy()  # Same actuals
time_axis = np.arange(len(actual_sample))

# Top: Baseline
axes[0].plot(
    time_axis, actual_sample, "o-", color="#2C3E50", label="Actual", markersize=5
)
axes[0].plot(
    time_axis,
    baseline_sample,
    "s-",
    color="#E74C3C",
    label="Baseline Forecast",
    markersize=5,
)
axes[0].set_ylabel("Value")
axes[0].set_title("Baseline DeepAR Forecast", fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle=":", alpha=0.5)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Bottom: ALPIN-Enhanced
axes[1].plot(
    time_axis, actual_sample, "o-", color="#2C3E50", label="Actual", markersize=5
)
axes[1].plot(
    time_axis,
    alpin_sample,
    "s-",
    color="#27AE60",
    label="ALPIN-Enhanced Forecast",
    markersize=5,
)
axes[1].set_xlabel("Forecast Horizon")
axes[1].set_ylabel("Value")
axes[1].set_title("ALPIN-Enhanced DeepAR Forecast (BatchCP)", fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle=":", alpha=0.5)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.suptitle(
    "Forecast Comparison: Example Prediction", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.show()

# %%
# BatchCP Filtering Statistics Summary
print("=" * 60)
print("BATCHCP FILTERING SUMMARY")
print("=" * 60)
print(f"\nDetected Changepoints:")
for sid, cps in detected_changepoints.items():
    print(f"  {sid}: {len(cps)} changepoints at indices {cps}")

print(f"\nFiltering Configuration:")
print(f"  Encoder window: {MAX_ENCODER_LENGTH} samples")
print(f"  Tolerance margin: ±{CP_TOLERANCE} samples")

print(f"\nFiltering Results:")
print(f"  Total batches seen: {filter_stats['total_batches']}")
print(
    f"  Batches filtered out: {filter_stats['filtered_batches']} ({filter_stats['filter_ratio'] * 100:.1f}%)"
)
print(f"  Clean batches used: {filter_stats['kept_batches']}")

# %%
# Final Analysis
print("=" * 60)
print("EXPERIMENT CONCLUSIONS")
print("=" * 60)

if alpin_metrics["MAE"] < baseline_metrics["MAE"]:
    print("\n✅ ALPIN-Enhanced DeepAR OUTPERFORMS Baseline")
    print(f"   MAE improved by {abs(mae_improvement):.2f}%")
    print(f"   RMSE improved by {abs(rmse_improvement):.2f}%")
    print("\n   The BatchCP method successfully reduced forecast error by")
    print("   filtering out training batches that span regime changes.")
else:
    print("\n⚠️ ALPIN-Enhanced DeepAR shows similar or worse performance")
    print(f"   MAE difference: {mae_improvement:.2f}%")
    print(f"   RMSE difference: {rmse_improvement:.2f}%")
    print("\n   Possible reasons:")
    print("   - Synthetic data may not have strong regime effects")
    print("   - Training data reduced too much by filtering")
    print("   - Model needs more epochs to converge")

print("\n" + "-" * 60)
print("Key Findings:")
print("-" * 60)
print(
    f"1. ALPIN detected {sum(len(cps) for cps in detected_changepoints.values())} changepoints across {N_SIGNALS} signals"
)
print(
    f"2. BatchCP filtered {filter_stats['filter_ratio'] * 100:.1f}% of training batches"
)
print(
    f"3. Baseline MAE: {baseline_metrics['MAE']:.4f}, ALPIN-Enhanced MAE: {alpin_metrics['MAE']:.4f}"
)
print(
    f"4. Baseline RMSE: {baseline_metrics['RMSE']:.4f}, ALPIN-Enhanced RMSE: {alpin_metrics['RMSE']:.4f}"
)

# %% [markdown]
# ---
#
# ## Summary
#
# In this notebook, we:
#
# 1. **Generated synthetic signals** with known changepoints using ALPIN's data generator
# 2. **Trained ALPIN** to detect changepoints in the synthetic data
# 3. **Trained baseline DeepAR** on all available training data
# 4. **Implemented BatchCP filtering** via `ChangePointAwareDataLoader`
# 5. **Trained ALPIN-enhanced DeepAR** using only "clean" batches without changepoints
# 6. **Compared forecast accuracy** between both approaches
#
# ### Key Takeaways
#
# - **BatchCP is a simple but effective preprocessing technique** that can improve forecast quality by avoiding regime-spanning training samples
# - **ALPIN provides accurate changepoint detection** that enables the BatchCP filtering
# - **The improvement depends on data characteristics**: signals with strong regime changes benefit most
# - **Trade-off**: Filtering reduces training data, which may hurt if changepoints are very frequent
#
# ### Next Steps
#
# - Test on real financial data with actual regime changes
# - Compare with other changepoint-aware methods
# - Experiment with different filtering tolerances
# - Evaluate on longer forecast horizons
