"""
Evaluation metrics for change point detection.

This module implements five evaluation metrics from the ALPIN paper:
- Hausdorff distance: Maximum distance between predicted and ground truth changepoints
- Precision: Fraction of predicted changepoints within tolerance of ground truth
- Recall: Fraction of ground truth changepoints detected within tolerance
- Annotation Error: Absolute difference in number of changepoints
- Rand Index: Fraction of sample pairs correctly grouped
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import adjusted_rand_score


def hausdorff_distance(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> float:
    """
    Calculate Hausdorff distance between predicted and ground truth changepoints.

    The Hausdorff distance is the maximum of two directed distances:
    - Maximum distance from predicted to nearest ground truth
    - Maximum distance from ground truth to nearest predicted

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal (used for context, not in calculation).
    tolerance : int, optional
        Tolerance window (not used in Hausdorff, included for API consistency).

    Returns
    -------
    float
        Hausdorff distance. Returns np.inf if one set is empty and the other is not.
        Returns 0.0 if both sets are empty.

    Examples
    --------
    >>> hausdorff_distance([50, 100], [48, 102], 150)
    2.0
    >>> hausdorff_distance([], [50], 100)
    inf
    >>> hausdorff_distance([], [], 100)
    0.0
    """
    # Edge case: both empty
    if len(predicted) == 0 and len(ground_truth) == 0:
        return 0.0

    # Edge case: one empty, other not
    if len(predicted) == 0 or len(ground_truth) == 0:
        return np.inf

    pred_arr = np.array(predicted, dtype=float)
    truth_arr = np.array(ground_truth, dtype=float)

    # Directed Hausdorff: predicted -> ground_truth
    # For each predicted point, find min distance to any ground truth point
    dist_pred_to_truth = np.max([np.min(np.abs(p - truth_arr)) for p in pred_arr])

    # Directed Hausdorff: ground_truth -> predicted
    # For each ground truth point, find min distance to any predicted point
    dist_truth_to_pred = np.max([np.min(np.abs(t - pred_arr)) for t in truth_arr])

    # Hausdorff is the maximum of the two directed distances
    return float(max(dist_pred_to_truth, dist_truth_to_pred))


def precision(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> float:
    """
    Calculate precision: fraction of predicted changepoints within tolerance of ground truth.

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal (used for context, not in calculation).
    tolerance : int, optional
        Tolerance window in samples (default: 10).

    Returns
    -------
    float
        Precision value in [0, 1]. Returns 1.0 if no predictions (nothing to be wrong about).

    Examples
    --------
    >>> precision([50, 100], [48, 102], 150, tolerance=10)
    1.0
    >>> precision([50, 200], [48, 102], 250, tolerance=10)
    0.5
    >>> precision([], [50], 100)
    1.0
    """
    # If no predictions, precision is 1.0 (nothing to be wrong about)
    if len(predicted) == 0:
        return 1.0

    # If no ground truth but have predictions, precision is 0.0
    if len(ground_truth) == 0:
        return 0.0

    pred_arr = np.array(predicted)
    truth_arr = np.array(ground_truth)

    # Count how many predicted changepoints are within tolerance of any ground truth
    correct = 0
    for p in pred_arr:
        # Check if this predicted point is within tolerance of any ground truth
        if np.any(np.abs(p - truth_arr) < tolerance):
            correct += 1

    return correct / len(predicted)


def recall(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> float:
    """
    Calculate recall: fraction of ground truth changepoints detected within tolerance.

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal (used for context, not in calculation).
    tolerance : int, optional
        Tolerance window in samples (default: 10).

    Returns
    -------
    float
        Recall value in [0, 1]. Returns 1.0 if no ground truth (nothing to detect).

    Examples
    --------
    >>> recall([50, 100], [48, 102], 150, tolerance=10)
    1.0
    >>> recall([50], [48, 102], 150, tolerance=10)
    0.5
    >>> recall([50], [], 100)
    1.0
    """
    # If no ground truth, recall is 1.0 (nothing to detect)
    if len(ground_truth) == 0:
        return 1.0

    # If no predictions but have ground truth, recall is 0.0
    if len(predicted) == 0:
        return 0.0

    pred_arr = np.array(predicted)
    truth_arr = np.array(ground_truth)

    # Count how many ground truth changepoints are within tolerance of any prediction
    detected = 0
    for t in truth_arr:
        # Check if this ground truth point is within tolerance of any prediction
        if np.any(np.abs(t - pred_arr) < tolerance):
            detected += 1

    return detected / len(ground_truth)


def annotation_error(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> int:
    """
    Calculate annotation error: absolute difference in number of changepoints.

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal (used for context, not in calculation).
    tolerance : int, optional
        Tolerance window (not used in annotation error, included for API consistency).

    Returns
    -------
    int
        Absolute difference in number of changepoints.

    Examples
    --------
    >>> annotation_error([50, 100], [48, 102], 150)
    0
    >>> annotation_error([50], [48, 102, 200], 250)
    2
    """
    return abs(len(predicted) - len(ground_truth))


def rand_index(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> float:
    """
    Calculate Rand Index: fraction of sample pairs correctly grouped.

    The Rand Index measures the fraction of pairs (i, j) where both predicted
    and ground truth agree on whether they are in the same segment or different segments.

    This implementation uses the adjusted Rand score from scikit-learn, which is
    normalized and corrected for chance.

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal.
    tolerance : int, optional
        Tolerance window (not used in Rand Index, included for API consistency).

    Returns
    -------
    float
        Adjusted Rand Index value. Range is [-1, 1], where 1 is perfect agreement,
        0 is random labeling, and negative values indicate worse than random.

    Examples
    --------
    >>> rand_index([50, 100], [50, 100], 150)
    1.0
    >>> rand_index([75], [50, 100], 150)
    0.333...
    """

    # Convert changepoints to segment labels for each time point
    def changepoints_to_labels(changepoints: List[int], length: int) -> np.ndarray:
        """Convert changepoint indices to segment labels."""
        labels = np.zeros(length, dtype=int)
        segment_id = 0

        # Sort changepoints to ensure correct ordering
        sorted_cps = sorted(changepoints)

        # First segment: from 0 to first changepoint
        if len(sorted_cps) > 0:
            labels[: sorted_cps[0]] = segment_id
            segment_id += 1

            # Middle segments: between consecutive changepoints
            for i in range(len(sorted_cps) - 1):
                labels[sorted_cps[i] : sorted_cps[i + 1]] = segment_id
                segment_id += 1

            # Last segment: from last changepoint to end
            labels[sorted_cps[-1] :] = segment_id
        else:
            # No changepoints: entire signal is one segment
            labels[:] = 0

        return labels

    # Edge case: signal_length must be > 0
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    # Convert both to labels
    pred_labels = changepoints_to_labels(predicted, signal_length)
    truth_labels = changepoints_to_labels(ground_truth, signal_length)

    # Calculate adjusted Rand score
    ari = adjusted_rand_score(truth_labels, pred_labels)

    return float(ari)


def evaluate_all(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics at once.

    Parameters
    ----------
    predicted : List[int]
        List of predicted changepoint indices.
    ground_truth : List[int]
        List of ground truth changepoint indices.
    signal_length : int
        Total length of the signal.
    tolerance : int, optional
        Tolerance window for precision/recall (default: 10).

    Returns
    -------
    Dict[str, float]
        Dictionary containing all five metrics:
        - 'hausdorff_distance': float
        - 'precision': float
        - 'recall': float
        - 'annotation_error': int (cast to float in dict)
        - 'rand_index': float

    Examples
    --------
    >>> metrics = evaluate_all([50, 100], [48, 102], 150, tolerance=10)
    >>> metrics['precision']
    1.0
    >>> metrics['recall']
    1.0
    >>> metrics['annotation_error']
    0
    """
    return {
        "hausdorff_distance": hausdorff_distance(
            predicted, ground_truth, signal_length, tolerance
        ),
        "precision": precision(predicted, ground_truth, signal_length, tolerance),
        "recall": recall(predicted, ground_truth, signal_length, tolerance),
        "annotation_error": float(
            annotation_error(predicted, ground_truth, signal_length, tolerance)
        ),
        "rand_index": rand_index(predicted, ground_truth, signal_length, tolerance),
    }
