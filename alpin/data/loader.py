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
    """Protocol for pluggable data loaders.

    Any class implementing this protocol can load time series data from a file
    and optionally return associated labels.
    """

    def load(self, path: str) -> tuple[np.ndarray, list[int] | None]:
        """Load signal and optional labels from a file.

        Parameters
        ----------
        path : str
            Path to the data file

        Returns
        -------
        signal : np.ndarray
            1D array of signal values
        labels : list[int] | None
            List of changepoint indices, or None if no labels present

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is invalid or required columns are missing
        """
        ...


class CSVLoader:
    """Load time series signals from CSV files.

    Extracts a signal column and optional label column from CSV files.
    Labels are expected as comma-separated integer indices (changepoint positions).

    Parameters
    ----------
    signal_column : str, optional
        Name of the column containing signal values (default: "value")
    label_column : str | None, optional
        Name of the column containing labels as comma-separated integers.
        If None, no labels are returned (default: None)

    Examples
    --------
    >>> loader = CSVLoader(signal_column="signal")
    >>> signal, labels = loader.load("data.csv")
    >>> signal.shape
    (1000,)
    """

    def __init__(self, signal_column: str = "value", label_column: str | None = None):
        """Initialize the CSV loader.

        Parameters
        ----------
        signal_column : str, optional
            Column name for signal data (default: "value")
        label_column : str | None, optional
            Column name for labels (default: None)
        """
        self.signal_column = signal_column
        self.label_column = label_column

    def load(self, path: str) -> tuple[np.ndarray, list[int] | None]:
        """Load signal and labels from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file

        Returns
        -------
        signal : np.ndarray
            1D array of signal values (float64)
        labels : list[int] | None
            List of changepoint indices, or None if no label column specified
            or if the label value is empty/NaN

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If signal_column is missing or contains non-numeric values
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
    """Load all CSV files from a directory using the provided loader.

    Loads all .csv files found in the directory (non-recursive).
    Files are processed in alphabetical order.

    Parameters
    ----------
    path : str
        Path to the directory containing CSV files
    loader : DataLoader
        A data loader implementing the DataLoader protocol

    Returns
    -------
    signals : list[np.ndarray]
        List of loaded signal arrays
    labels : list[list[int] | None]
        List of label lists (or None entries)

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If no CSV files are found in the directory

    Examples
    --------
    >>> loader = CSVLoader()
    >>> signals, labels = load_directory("data/", loader)
    >>> len(signals)
    5
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
