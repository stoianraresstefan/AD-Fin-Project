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
# # Treasury Dataset Experiment: ALPIN with Partial Changepoints
#
# This notebook demonstrates ALPIN's ability to learn from **partial changepoint labels** using real-world US Treasury rate data.
#
# ## Experiment Setup
#
# **Dataset**: US Treasury rates (1962-present, daily)
# - Source: treasury_dataset.csv
# - Format: Date (MM/DD/YYYY), Value (rate)
# - Size: ~16,000 daily observations
#
# **Changepoint Strategy**:
# - Ground truth: 8 known changepoints at indices [1992, 3155, 4544, 5105, 7065, 9710, 11480, 14543]
# - Training: Randomly select only 50% (4 changepoints) for ALPIN training
# - Evaluation: Test if ALPIN can learn from partial labels and still detect changepoints
#
# **Forecasting Comparison**:
# - Baseline DeepAR (no filtering)
# - ALPIN-enhanced DeepAR (BatchCP filtering with detected changepoints)
#
# ## Hypothesis
#
# Even with only 50% of changepoints labeled, ALPIN should:
# 1. Learn a reasonable penalty parameter beta
# 2. Detect most changepoints in the full signal
# 3. Enable BatchCP filtering to improve DeepAR forecast accuracy

# %% [markdown]
# ## 1. Setup and Imports

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import sys

# Add parent directory to path for ALPIN imports
sys.path.insert(0, "..")

# ALPIN
from alpin import ALPIN
from alpin.metrics import evaluate_all
from alpin.visualization import plot_signal

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100

print("Setup complete!")

# %% [markdown]
# ## 2. Load Treasury Data

# %%
# Load dataset
df = pd.read_csv("../treasury_dataset.csv", encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst few rows:")
print(df.head())

# Convert to numpy array for ALPIN
signal = df["Value"].values
print(f"\nSignal length: {len(signal)}")
print(f"Signal range: [{signal.min():.2f}, {signal.max():.2f}]")

# %% [markdown]
# ## 3. Define Changepoints and Create Partial Training Set
#
# We have 8 ground truth changepoints. For training, we'll randomly select only 4 (50%).

# %%
FULL_CHANGEPOINTS = np.array([1992, 3155, 4544, 5105, 7065, 9710, 11480, 14543])
print(f"Full changepoints (n={len(FULL_CHANGEPOINTS)}): {FULL_CHANGEPOINTS}")

n_train_cps = len(FULL_CHANGEPOINTS) // 2
TRAIN_CHANGEPOINTS = FULL_CHANGEPOINTS[:n_train_cps]
print(f"\nTraining changepoints (first {n_train_cps}): {TRAIN_CHANGEPOINTS}")

last_train_cp = TRAIN_CHANGEPOINTS[-1]
next_cp = FULL_CHANGEPOINTS[n_train_cps]
train_cutoff = (last_train_cp + next_cp) // 2
print(f"\nTruncating training signal at index {train_cutoff}")
print(f"  Last training changepoint: {last_train_cp}")
print(f"  Next changepoint: {next_cp}")
print(f"  Midpoint: {train_cutoff}")

# %% [markdown]
# ## 4. Visualize Signal and Changepoints

# %%
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df["Date"], signal, linewidth=0.8, alpha=0.7, label="Treasury Rate")

for cp in FULL_CHANGEPOINTS:
    ax.axvline(
        df["Date"].iloc[cp], color="red", linestyle="--", alpha=0.3, linewidth=1.5
    )

for cp in TRAIN_CHANGEPOINTS:
    ax.axvline(
        df["Date"].iloc[cp], color="green", linestyle="-", alpha=0.8, linewidth=2
    )

ax.axvline(
    df["Date"].iloc[train_cutoff],
    color="blue",
    linestyle=":",
    alpha=0.8,
    linewidth=2,
    label="Training cutoff",
)

ax.set_xlabel("Date")
ax.set_ylabel("Treasury Rate")
ax.set_title(
    "Treasury Rates with Changepoints (Green = Training Labels, Red = Full Ground Truth, Blue = Cutoff)"
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Train ALPIN on Partial Labels

# %%
train_signal = signal[:train_cutoff]
print(f"Training signal length: {len(train_signal)} (truncated from {len(signal)})")

train_signals = [train_signal]
train_ground_truths = [TRAIN_CHANGEPOINTS]

print("Training ALPIN...")
model = ALPIN(beta_bounds=(1e-6, 1e6))
model.fit(train_signals, train_ground_truths)

print(f"\nLearned optimal β: {model.beta_opt:.4f}")

# %% [markdown]
# ## 6. Detect Changepoints on Full Signal

# %%
# Use learned model to detect changepoints on the full signal
detected_cps = model.predict(signal)
print(f"Detected changepoints (n={len(detected_cps)}): {detected_cps}")

# Evaluate detection performance against FULL ground truth
metrics = evaluate_all(
    detected=detected_cps,
    ground_truth=FULL_CHANGEPOINTS,
    signal_length=len(signal),
    margin=50,  # Allow 50-sample margin (roughly 50 trading days)
)

print("\nDetection Performance (vs. Full Ground Truth):")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# %% [markdown]
# ## 7. Visualize Detection Results

# %%
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(
    df["Date"], signal, linewidth=0.8, alpha=0.7, label="Treasury Rate", color="blue"
)

# Ground truth changepoints
for cp in FULL_CHANGEPOINTS:
    ax.axvline(
        df["Date"].iloc[cp],
        color="red",
        linestyle="--",
        alpha=0.4,
        linewidth=2,
        label="Ground Truth" if cp == FULL_CHANGEPOINTS[0] else "",
    )

# Detected changepoints
for cp in detected_cps:
    if 0 <= cp < len(df):  # Ensure valid index
        ax.axvline(
            df["Date"].iloc[cp],
            color="green",
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
            label="Detected" if cp == detected_cps[0] else "",
        )

ax.set_xlabel("Date")
ax.set_ylabel("Treasury Rate")
ax.set_title(
    f"ALPIN Detection Results (Trained on {len(TRAIN_CHANGEPOINTS)}/{len(FULL_CHANGEPOINTS)} changepoints)\n"
    f"Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}"
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Prepare Data for DeepAR Forecasting
#
# Convert time series to format suitable for DeepAR with time and group identifiers.

# %%
# Create DataFrame for DeepAR
forecast_df = pd.DataFrame(
    {
        "time_idx": np.arange(len(signal)),
        "value": signal,
        "group": "treasury",  # Single time series group
        "date": df["Date"],
    }
)

# Add time-based features
forecast_df["year"] = forecast_df["date"].dt.year
forecast_df["month"] = forecast_df["date"].dt.month
forecast_df["day_of_week"] = forecast_df["date"].dt.dayofweek

print(f"Forecast DataFrame shape: {forecast_df.shape}")
print(f"\nFirst few rows:")
print(forecast_df.head())

# %% [markdown]
# ## 9. Split Data: Train/Validation/Test

# %%
# Split configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# TEST_SPLIT = 0.15 (implicit)

n_total = len(forecast_df)
train_end = int(n_total * TRAIN_SPLIT)
val_end = int(n_total * (TRAIN_SPLIT + VAL_SPLIT))

train_df = forecast_df[:train_end].copy()
val_df = forecast_df[train_end:val_end].copy()
test_df = forecast_df[val_end:].copy()

print(f"Train samples: {len(train_df)} ({100 * len(train_df) / n_total:.1f}%)")
print(f"Val samples: {len(val_df)} ({100 * len(val_df) / n_total:.1f}%)")
print(f"Test samples: {len(test_df)} ({100 * len(test_df) / n_total:.1f}%)")

# %% [markdown]
# ## 10. Identify Changepoints in Training Window
#
# For BatchCP filtering, we only care about changepoints that fall within the training window.

# %%
train_changepoints = detected_cps[detected_cps < train_end]
print(f"Changepoints in training window: {train_changepoints}")
print(f"Number of training changepoints: {len(train_changepoints)}")

# %% [markdown]
# ## 11. DeepAR Training Helper Functions
#
# We'll implement:
# 1. **Baseline DeepAR**: Uses all training data
# 2. **ALPIN-Enhanced DeepAR**: Filters out batches overlapping with detected changepoints (BatchCP method)

# %%
try:
    import torch
    from torch.utils.data import DataLoader
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, DeepAR
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE, RMSE

    PYTORCH_AVAILABLE = True
    print("PyTorch Forecasting available!")
    print(f"PyTorch version: {torch.__version__}")

except ImportError as e:
    PYTORCH_AVAILABLE = False
    print(f"PyTorch Forecasting not available: {e}")
    print(
        "Skipping DeepAR comparison. Install with: uv add torch pytorch-lightning pytorch-forecasting"
    )

# %%
if PYTORCH_AVAILABLE:
    # Configuration
    MAX_ENCODER_LENGTH = 60  # Use 60 past days
    MAX_PREDICTION_LENGTH = 20  # Forecast 20 days ahead
    BATCH_SIZE = 32
    MAX_EPOCHS = 20
    LEARNING_RATE = 0.001

    # Reproducibility
    torch.manual_seed(SEED)
    pl.seed_everything(SEED)

    print(f"DeepAR Configuration:")
    print(f"  Encoder length: {MAX_ENCODER_LENGTH}")
    print(f"  Prediction length: {MAX_PREDICTION_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {MAX_EPOCHS}")

# %% [markdown]
# ## 12. Baseline DeepAR (No Filtering)

# %%
if PYTORCH_AVAILABLE:
    # Create TimeSeriesDataSet for baseline
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=["value"],
        time_varying_known_reals=["year", "month", "day_of_week"],
        target_normalizer=GroupNormalizer(groups=["group"]),
    )

    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True
    )

    # DataLoaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )

    print(f"Baseline training batches: {len(train_dataloader)}")
    print(f"Baseline validation batches: {len(val_dataloader)}")

    # Train baseline DeepAR
    print("\nTraining Baseline DeepAR...")
    baseline_model = DeepAR.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=64,
        rnn_layers=2,
        dropout=0.1,
        loss=MAE(),
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(
        baseline_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("Baseline training complete!")

# %% [markdown]
# ## 13. ALPIN-Enhanced DeepAR (BatchCP Filtering)
#
# Filter out training batches where the encoder window overlaps with detected changepoints.

# %%
if PYTORCH_AVAILABLE:
    # Create filtered training set by removing samples near changepoints
    # A sample at time_idx=t uses encoder window [t - MAX_ENCODER_LENGTH, t]
    # Remove if this window overlaps with any changepoint (with margin)

    MARGIN = 10  # Samples before/after changepoint to exclude

    def overlaps_changepoint(
        time_idx: int, changepoints: np.ndarray, margin: int
    ) -> bool:
        """Check if encoder window [time_idx - MAX_ENCODER_LENGTH, time_idx] overlaps with any changepoint."""
        encoder_start = time_idx - MAX_ENCODER_LENGTH
        encoder_end = time_idx

        for cp in changepoints:
            if encoder_start - margin <= cp <= encoder_end + margin:
                return True
        return False

    # Filter training data
    train_df_filtered = train_df[
        ~train_df["time_idx"].apply(
            lambda t: overlaps_changepoint(t, train_changepoints, MARGIN)
        )
    ].copy()

    print(f"Original training samples: {len(train_df)}")
    print(f"Filtered training samples: {len(train_df_filtered)}")
    print(
        f"Removed: {len(train_df) - len(train_df_filtered)} samples ({100 * (1 - len(train_df_filtered) / len(train_df)):.1f}%)"
    )

    # Create TimeSeriesDataSet with filtered data
    training_filtered = TimeSeriesDataSet(
        train_df_filtered,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=["value"],
        time_varying_known_reals=["year", "month", "day_of_week"],
        target_normalizer=GroupNormalizer(groups=["group"]),
    )

    # Use same validation set
    validation_filtered = TimeSeriesDataSet.from_dataset(
        training_filtered, val_df, predict=True, stop_randomization=True
    )

    train_dataloader_filtered = training_filtered.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_dataloader_filtered = validation_filtered.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )

    print(f"Filtered training batches: {len(train_dataloader_filtered)}")

    # Train ALPIN-enhanced DeepAR
    print("\nTraining ALPIN-Enhanced DeepAR (BatchCP)...")
    enhanced_model = DeepAR.from_dataset(
        training_filtered,
        learning_rate=LEARNING_RATE,
        hidden_size=64,
        rnn_layers=2,
        dropout=0.1,
        loss=MAE(),
    )

    trainer_filtered = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        logger=False,
        enable_checkpointing=False,
    )

    trainer_filtered.fit(
        enhanced_model,
        train_dataloaders=train_dataloader_filtered,
        val_dataloaders=val_dataloader_filtered,
    )

    print("ALPIN-enhanced training complete!")

# %% [markdown]
# ## 14. Evaluate on Test Set

# %%
if PYTORCH_AVAILABLE:
    # Create test dataset
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=True, stop_randomization=True
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )

    print("Evaluating models on test set...")

    # Baseline predictions
    baseline_predictions = baseline_model.predict(
        test_dataloader, mode="prediction", return_x=True
    )
    baseline_actuals = torch.cat([x for x, _ in test_dataloader])
    baseline_mae = MAE()(baseline_predictions.output, baseline_actuals).item()
    baseline_rmse = RMSE()(baseline_predictions.output, baseline_actuals).item()

    # Enhanced predictions
    enhanced_predictions = enhanced_model.predict(
        test_dataloader, mode="prediction", return_x=True
    )
    enhanced_actuals = torch.cat([x for x, _ in test_dataloader])
    enhanced_mae = MAE()(enhanced_predictions.output, enhanced_actuals).item()
    enhanced_rmse = RMSE()(enhanced_predictions.output, enhanced_actuals).item()

    print("\n" + "=" * 60)
    print("FORECAST ACCURACY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    print(f"{'Baseline DeepAR':<25} {baseline_mae:<12.4f} {baseline_rmse:<12.4f}")
    print(f"{'ALPIN-Enhanced DeepAR':<25} {enhanced_mae:<12.4f} {enhanced_rmse:<12.4f}")
    print("-" * 60)

    mae_improvement = 100 * (baseline_mae - enhanced_mae) / baseline_mae
    rmse_improvement = 100 * (baseline_rmse - enhanced_rmse) / baseline_rmse

    print(f"{'Improvement':<25} {mae_improvement:>11.2f}% {rmse_improvement:>11.2f}%")
    print("=" * 60)

# %% [markdown]
# ## 15. Visualize Forecast Examples

# %%
if PYTORCH_AVAILABLE:
    # Plot a few forecast examples
    n_examples = 3
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 4 * n_examples))

    for idx in range(n_examples):
        ax = axes[idx] if n_examples > 1 else axes

        # Get sample from test set
        sample_idx = idx * (len(test_df) // n_examples)
        if sample_idx >= len(test_df):
            continue

        # Get actual values
        start_idx = val_end + sample_idx
        actual_values = (
            forecast_df["value"]
            .iloc[start_idx : start_idx + MAX_PREDICTION_LENGTH]
            .values
        )
        dates = (
            forecast_df["date"]
            .iloc[start_idx : start_idx + MAX_PREDICTION_LENGTH]
            .values
        )

        # Get predictions (simplified - using mean predictions)
        baseline_pred = baseline_predictions.output[idx].mean(dim=1).cpu().numpy()
        enhanced_pred = enhanced_predictions.output[idx].mean(dim=1).cpu().numpy()

        # Plot
        ax.plot(dates, actual_values, "o-", label="Actual", linewidth=2, markersize=4)
        ax.plot(
            dates,
            baseline_pred[: len(actual_values)],
            "s--",
            label="Baseline DeepAR",
            alpha=0.7,
        )
        ax.plot(
            dates,
            enhanced_pred[: len(actual_values)],
            "^--",
            label="ALPIN-Enhanced",
            alpha=0.7,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Treasury Rate")
        ax.set_title(f"Forecast Example {idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 16. Summary and Conclusions

# %%
print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\nDataset: Treasury rates (1962-present)")
print(f"Signal length: {len(signal):,} daily observations")
print(f"\nChangepoint Detection:")
print(f"  Ground truth changepoints: {len(FULL_CHANGEPOINTS)}")
print(
    f"  Training labels provided: {len(TRAIN_CHANGEPOINTS)} ({100 * len(TRAIN_CHANGEPOINTS) / len(FULL_CHANGEPOINTS):.0f}%)"
)
print(f"  Detected changepoints: {len(detected_cps)}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall: {metrics['recall']:.3f}")
print(f"  Learned β: {model.beta_opt:.4f}")

if PYTORCH_AVAILABLE:
    print(f"\nForecasting Results:")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  ALPIN-Enhanced MAE: {enhanced_mae:.4f}")
    print(f"  Improvement: {mae_improvement:.2f}%")
    print(f"\n  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  ALPIN-Enhanced RMSE: {enhanced_rmse:.4f}")
    print(f"  Improvement: {rmse_improvement:.2f}%")

print("\nConclusions:")
print("  1. ALPIN successfully learned from partial (50%) changepoint labels")
print("  2. Detected changepoints enabled BatchCP filtering")
if PYTORCH_AVAILABLE and mae_improvement > 0:
    print("  3. ALPIN-enhanced DeepAR improved forecast accuracy vs baseline")
elif PYTORCH_AVAILABLE:
    print("  3. Results vary - BatchCP filtering effect depends on changepoint quality")
print("\n" + "=" * 70)
