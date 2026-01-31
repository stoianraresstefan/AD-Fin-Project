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
    Computes the Hausdorff distance as the maximum of two directed distances: from predictions to truth and vice versa.
    Each directed distance is the maximum distance from one set to its nearest point in the other.

    Input: predicted (list of int) - predicted changepoint indices, ground_truth (list of int) - true changepoint indices, signal_length (int) - total signal length, tolerance (int) - API consistency
    Output: float - Hausdorff distance, np.inf if one set empty, 0.0 if both empty
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
    Computes precision as the fraction of predicted changepoints falling within tolerance of any ground truth changepoint.
    Returns 1.0 if no predictions, 0.0 if predictions but no ground truth.

    Input: predicted (list of int) - predicted changepoint indices, ground_truth (list of int) - true indices, signal_length (int) - length, tolerance (int) - tolerance window in samples
    Output: float - precision value in [0, 1]
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
    # no ground truth, recall is 1.0
    if len(ground_truth) == 0:
        return 1.0

    # no predictions but have ground truth, recall is 0.0
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
    Computes the absolute difference in the number of predicted versus ground truth changepoints.
    Measures over-segmentation or under-segmentation regardless of position accuracy.

    Input: predicted (list of int) - predicted changepoint indices, ground_truth (list of int) - true indices, signal_length (int) - length, tolerance (int) - unused
    Output: int - absolute difference in count of changepoints
    """
    return abs(len(predicted) - len(ground_truth))


def rand_index(
    predicted: List[int],
    ground_truth: List[int],
    signal_length: int,
    tolerance: int = 10,
) -> float:
    """
    Computes the adjusted Rand index measuring agreement on sample pair groupings between predictions and ground truth.
    Converts changepoints to segment labels then computes normalized agreement, with 1.0 = perfect agreement.

    Input: predicted (list of int) - predicted changepoint indices, ground_truth (list of int) - true indices, signal_length (int) - total length, tolerance (int) - unused
    Output: float - adjusted Rand index in [-1, 1], where 1 is perfect, 0 is random, negative is worse than random
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
    Computes all five evaluation metrics (Hausdorff distance, precision, recall, annotation error, Rand index) in one call.
    Provides a comprehensive assessment of changepoint detection performance across multiple dimensions.

    Input: predicted (list of int) - predicted changepoint indices, ground_truth (list of int) - true indices, signal_length (int) - total length, tolerance (int) - tolerance window
    Output: dict with keys hausdorff_distance, precision, recall, annotation_error, rand_index mapping to float values
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
