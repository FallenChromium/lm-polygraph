"""
TFB classification metric.

A :class:`GenerationMetric` that evaluates accuracy using the Bayesian-averaged
class probabilities produced by :class:`~lm_polygraph.stat_calculators.tfb.TFBStatCalculator`.

Unlike :class:`~lm_polygraph.generation_metrics.accuracy.AccuracyMetric` (which
string-compares greedy-decoded text), this metric takes the argmax of the mean
class-probability distribution across TFB stochastic samples, then checks
whether the predicted class matches the target label string.
"""

from typing import Dict, List, Optional

import numpy as np

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric


class TFBClassificationMetric(GenerationMetric):
    """Accuracy from TFB's Bayesian-averaged class probabilities.

    Parameters
    ----------
    stats_key : str
        Must match the ``stats_key`` of the corresponding ``TFBStatCalculator``.
    labels : list[str], optional
        Ordered class labels (e.g. ``["A", "B", "C", "D"]``).  If provided,
        predicted class indices are mapped to strings and compared against
        ``target_texts`` via exact match.  If ``None``, the metric returns the
        mean-probability confidence instead (useful for UE-metric correlation
        but not for accuracy reporting).
    """

    def __init__(
        self,
        stats_key: str = "tfb",
        labels: Optional[List[str]] = None,
    ):
        self.stats_key = stats_key
        self.labels = labels
        super().__init__(
            stats_dependencies=[f"{stats_key}_target_probs"],
            level="sequence",
        )

    def __str__(self):
        return f"TFBClassificationAccuracy({self.stats_key})"

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        key = f"{self.stats_key}_target_probs"
        if key not in stats:
            raise KeyError(
                f"Statistic '{key}' not found.  Make sure TFBStatCalculator "
                f"is configured with stats_key='{self.stats_key}' in "
                f"classification mode (target_ids or target_labels set)."
            )

        # probs shape: [batch, n_samples, n_classes]
        probs = np.asarray(stats[key], dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Bayesian model averaging over stochastic samples
        mean_probs = probs.mean(axis=1)  # [batch, n_classes]
        predictions = mean_probs.argmax(axis=1)  # [batch]

        scores = []
        for idx, (pred_idx, target) in enumerate(zip(predictions, target_texts)):
            if self.labels is not None and pred_idx < len(self.labels):
                pred_str = self.labels[int(pred_idx)]
                scores.append(1.0 if pred_str.strip() == target.strip() else 0.0)
            else:
                # Fallback: return confidence as a [0, 1] quality proxy
                scores.append(float(mean_probs[idx].max()))

        return np.array(scores)
