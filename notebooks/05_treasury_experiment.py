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
from typing import List, Dict, Iterator, Any

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

# Store original signal length before resampling
signal_daily = df["Value"].values
original_length = len(signal_daily)
print(f"\nSignal length: {len(signal_daily)}")
print(f"Signal range: [{signal_daily.min():.2f}, {signal_daily.max():.2f}]")

# Define changepoints in original daily data
DAILY_CHANGEPOINTS = np.array([1992, 3155, 4544, 5105, 7065, 9710, 11480, 14543])

# Resample to weekly for faster training (16k → 2.3k samples)
df_weekly = df.set_index("Date").resample("W").mean().reset_index()
df_weekly = df_weekly.sort_values("Date").reset_index(drop=True)
signal = df_weekly["Value"].values
df = df_weekly

# Calculate scale factor and adjust changepoints programmatically
resampled_length = len(signal)
scale_factor = resampled_length / original_length
FULL_CHANGEPOINTS = np.round(DAILY_CHANGEPOINTS * scale_factor).astype(int)

# Print information
print(f"\nOriginal daily signal: {original_length} samples")
print(f"Resampled weekly signal: {resampled_length} samples")
print(f"Scale factor: {scale_factor:.4f}")
print(f"Daily changepoints: {DAILY_CHANGEPOINTS}")
print(f"Resampled changepoints: {FULL_CHANGEPOINTS}")

# %% [markdown]
# ## 3. Define Changepoints and Create Partial Training Set
#
# We have 8 ground truth changepoints. For training, we'll randomly select only 4 (50%).

# %%
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
metrics = evaluate_all(detected_cps, FULL_CHANGEPOINTS, len(signal), tolerance=50)

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
train_changepoints = detected_cps[:train_end]
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
    MAX_ENCODER_LENGTH = 30  # Reduced from 60
    MAX_PREDICTION_LENGTH = 20  # Keep same
    BATCH_SIZE = 64  # Increased from 32 (fewer batches)
    MAX_EPOCHS = 10  # Reduced from 20
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
        add_relative_time_idx=True,
        add_target_scales=True,
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
    baseline_deepar = DeepAR.from_dataset(
        training,
        hidden_size=64,
        rnn_layers=2,
        dropout=0.1,
        learning_rate=LEARNING_RATE,
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=3,
    )

    print(
        f"Baseline DeepAR parameters: {sum(p.numel() for p in baseline_deepar.parameters()):,}"
    )

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

# %% [markdown]
# ## 13. ALPIN-Enhanced DeepAR (BatchCP Filtering)
#
# Filter out training batches where the encoder window overlaps with detected changepoints.

# %%
if PYTORCH_AVAILABLE:
    # Create changepoints dictionary in format expected by ChangePointAwareDataLoader
    # Treasury has single time series with group="treasury"
    changepoints_dict = {"treasury": train_changepoints}

    print(f"Changepoints for filtering: {changepoints_dict}")

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
            Mapping from group identifier to list of changepoint indices
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

            Args:
                batch: Tuple of (x_dict, y_tuple) from TimeSeriesDataSet

            Returns:
                True if batch should be filtered out
            """
            x_dict, y = batch

            # Get encoder time indices - these tell us the time range of each sample
            # encoder_time_idx has shape (batch_size, encoder_length)
            if "encoder_time_idx" in x_dict:
                encoder_times = x_dict["encoder_time_idx"]
            else:
                # Fallback: use relative time index if available
                encoder_times = x_dict.get("time_idx", None)
                if encoder_times is None:
                    return False  # Cannot determine, don't filter

            # Get groups to identify which time series each sample belongs to
            groups = x_dict.get("groups", None)
            if groups is None:
                # For single time series (treasury), we can assume all samples are from the same group
                group_id = "treasury"
            else:
                group_id = "treasury"  # Treasury notebook uses single group

            batch_size = encoder_times.shape[0]

            for i in range(batch_size):
                # Get time range of this sample's encoder window
                sample_times = encoder_times[i].cpu().numpy()
                start_time = int(sample_times.min())
                end_time = int(sample_times.max())

                # Check if any changepoint falls within encoder window
                if group_id in self.changepoints_dict:
                    for cp in self.changepoints_dict[group_id]:
                        # Check if changepoint (with tolerance) overlaps encoder window
                        if (
                            (start_time - self.tolerance)
                            <= cp
                            <= (end_time + self.tolerance)
                        ):
                            return True

            return False

        def __iter__(self) -> Iterator:
            """Iterate over batches, skipping those with changepoints."""
            self.filtered_count = 0
            self.total_count = 0

            for batch in self.dataloader:
                self.total_count += 1

                if self._batch_contains_changepoint(batch):
                    self.filtered_count += 1
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

    # Create filtered dataloader for ALPIN-enhanced training
    filtered_train_dataloader = ChangePointAwareDataLoader(
        dataloader=train_dataloader,
        changepoints_dict=changepoints_dict,
        encoder_length=MAX_ENCODER_LENGTH,
        tolerance=10,
    )

    print(f"Created ChangePointAwareDataLoader")
    print(f"  Detected changepoints: {train_changepoints}")
    print(f"  Encoder length: {MAX_ENCODER_LENGTH}")
    print(f"  Tolerance: 10")

    # Create ALPIN-enhanced DeepAR model (same architecture as baseline)
    alpin_deepar = DeepAR.from_dataset(
        training,
        hidden_size=64,
        rnn_layers=2,
        dropout=0.1,
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
     baseline_predictions = baseline_deepar.predict(
        test_dataloader, mode="prediction", return_x=True
    )
    baseline_actuals = torch.cat([x for x, _ in test_dataloader])
    baseline_mae = MAE()(baseline_predictions.output, baseline_actuals).item()
    baseline_rmse = RMSE()(baseline_predictions.output, baseline_actuals).item()

    # Enhanced predictions
     enhanced_predictions = alpin_deepar.predict(
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
