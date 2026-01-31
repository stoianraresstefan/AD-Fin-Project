"""
Pluggable data loaders for CSV and other formats.

Provides a Protocol interface for flexible data loading and a CSV-specific
implementation using pandas.
"""

from typing import Protocol
from pathlib import Path
import numpy as np
import pandas as pd


class DataLoader(Protocol):
    """
    Protocol interface for pluggable data loaders that support flexible loading of time series data.
    Any implementing class can load signals from files and optionally return associated labels.

    Input: path (str) - file path to load
    Output: tuple of np.ndarray (signal) and list of int or None (labels)
    """

    def load(self, path: str) -> tuple[np.ndarray, list[int] | None]:
        """
        Loads signal and optional labels from a file.

        Input: path (str) - path to the data file
        Output: tuple (np.ndarray signal, list of int or None labels)
        """
        ...


class CSVLoader:
    """
    Loads time series signals from CSV files with optional changepoint labels.
    Extracts signal and label columns, parsing labels as comma-separated integer changepoint indices.

    Input: signal_column (str) - name of signal column (default: "value"), label_column (str or None) - name of label column
    Output: tuple of (np.ndarray signal, list of int or None labels)
    """

    def __init__(self, signal_column: str = "value", label_column: str | None = None):
        """
        Initializes the CSV loader with column names for signals and labels.

        Input: signal_column (str) - column name for signal data, label_column (str or None) - column name for labels
        Output: None - initializes instance attributes
        """
        self.signal_column = signal_column
        self.label_column = label_column

    def load(self, path: str) -> tuple[np.ndarray, list[int] | None]:
        """
        Loads signal and labels from a CSV file, converting signal to float64 and parsing comma-separated label integers.
        Validates that required columns exist and contain valid numeric values for the signal.

        Input: path (str) - path to the CSV file
        Output: tuple of (np.ndarray signal, list of int or None labels)
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}") from e

        # Validate signal column exists
        if self.signal_column not in df.columns:
            raise ValueError(
                f"Signal column '{self.signal_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract signal
        signal_col = df[self.signal_column]

        # Validate signal is numeric
        try:
            signal = signal_col.astype(np.float64).values
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Signal column '{self.signal_column}' contains non-numeric values: {e}"
            ) from e

        # Extract labels if specified
        labels = None
        if self.label_column is not None:
            if self.label_column not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

            label_col = df[self.label_column]

            # Handle label extraction - get first row value (assumes single row or all same)
            if len(label_col) > 0:
                label_value = label_col.iloc[0]

                # Skip if NaN or empty string
                if pd.notna(label_value) and label_value != "":
                    try:
                        # Parse comma-separated integers
                        labels = [int(x.strip()) for x in str(label_value).split(",")]
                    except ValueError as e:
                        raise ValueError(
                            f"Labels in column '{self.label_column}' could not be parsed as "
                            f"comma-separated integers: {label_value}"
                        ) from e

        return signal, labels


def load_directory(
    path: str, loader: DataLoader
) -> tuple[list[np.ndarray], list[list[int] | None]]:
    """
    Loads all CSV files from a directory in alphabetical order using the provided loader.
    Non-recursive directory search that returns lists of signals and their associated labels.

    Input: path (str) - directory path containing CSV files, loader (DataLoader) - loader implementing DataLoader protocol
    Output: tuple of (list of np.ndarray signals, list of lists of int or None labels)
    """
    dir_path = Path(path)

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    csv_files = sorted(dir_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {path}")

    signals = []
    all_labels = []

    for csv_file in csv_files:
        signal, labels = loader.load(str(csv_file))
        signals.append(signal)
        all_labels.append(labels)

    return signals, all_labels
