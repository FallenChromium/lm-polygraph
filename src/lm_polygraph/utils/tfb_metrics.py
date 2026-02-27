from typing import Dict, Iterable, List, Tuple

import numpy as np


def _validate_inputs(
    stats: Dict[str, Iterable],
    stats_key: str,
    true_labels: Iterable[int],
    expected_classes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and validate the per-sample probability tensors and the reference labels.

    Parameters
    ----------
    stats : dict
        Output dictionary produced by `TFBStatCalculator`.
    stats_key : str
        The prefix used when instantiating the calculator (e.g., "tfb_arc").
    true_labels : Iterable[int]
        Ground-truth class indices for each evaluated prompt.
    expected_classes : Optional[int]
        If provided, checks that every probability vector has this dimensionality.

    Returns
    -------
    probs : np.ndarray
        Array with shape [B, S, C] – B items, S stochastic samples, C classes.
    labels : np.ndarray
        Array with shape [B] containing the integer class labels.
    """
    key = f"{stats_key}_target_probs"
    if key not in stats:
        raise KeyError(
            f"Statistic '{key}' was not found. Did you run TFBStatCalculator in classification mode?"
        )

    raw_probs = stats[key]
    if not raw_probs:
        raise ValueError("TFB statistics contain no probability samples.")

    probs = np.asarray(raw_probs, dtype=np.float64)
    if probs.ndim != 3:
        raise ValueError(
            f"Expected '{key}' to have shape [batch, samples, classes], got {probs.shape}"
        )

    labels = np.asarray(list(true_labels), dtype=np.int64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Probability batch ({probs.shape[0]}) and labels ({labels.shape[0]}) mismatch."
        )

    if expected_classes is not None and probs.shape[-1] != expected_classes:
        raise ValueError(
            f"Probability tensors have {probs.shape[-1]} classes, "
            f"but expected {expected_classes}."
        )

    return probs, labels


def _histogram_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Standard histogram-based Expected Calibration Error.

    Parameters
    ----------
    confidences : np.ndarray
        Predicted confidence per item.
    correctness : np.ndarray
        Binary array indicating whether the corresponding prediction is correct.
    num_bins : int
        Number of equally spaced bins in [0, 1].

    Returns
    -------
    float
        Expected Calibration Error.
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == num_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = mask.sum()
        if count == 0:
            continue

        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / len(confidences)) * abs(acc - conf)

    return float(ece)


def tfb_classification_metrics(
    stats: Dict[str, Iterable],
    stats_key: str,
    true_labels: Iterable[int],
    num_bins: int = 15,
    expected_classes: int | None = None,
) -> Dict[str, np.ndarray | float]:
    """
    Compute NLL, accuracy, calibration error, and predictive distributions from TFB stats.

    Parameters
    ----------
    stats : dict
        Output dictionary from `TFBStatCalculator`.
    stats_key : str
        Prefix identifying the calculator run (e.g., "tfb_arc").
    true_labels : Iterable[int]
        Ground-truth class indices.
    num_bins : int, optional
        Number of bins used for ECE, by default 15.
    expected_classes : Optional[int]
        Enforces a specific number of classes (useful for sanity checks).

    Returns
    -------
    dict
        {
            "nll": float,
            "accuracy": float,
            "ece": float,
            "mean_probs": np.ndarray [B, C],
            "per_sample_probs": np.ndarray [B, S, C],
        }
    """
    probs, labels = _validate_inputs(stats, stats_key, true_labels, expected_classes)

    # Bayesian model averaging over TFB samples
    mean_probs = probs.mean(axis=1)
    eps = 1e-12

    # Negative log-likelihood
    true_probs = mean_probs[np.arange(len(labels)), labels]
    nll = -np.log(true_probs + eps).mean()

    # Accuracy and confidence
    predictions = mean_probs.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    confidences = mean_probs.max(axis=1)
    correctness = (predictions == labels).astype(float)

    ece = _histogram_ece(confidences, correctness, num_bins=num_bins)

    return {
        "nll": float(nll),
        "accuracy": accuracy,
        "ece": ece,
        "mean_probs": mean_probs,
        "per_sample_probs": probs,
    }
