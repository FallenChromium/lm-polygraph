"""
TFB uncertainty estimators.

These estimators consume the statistics produced by
:class:`~lm_polygraph.stat_calculators.tfb.TFBStatCalculator` and
convert them into scalar uncertainty scores.
"""

import numpy as np

from lm_polygraph.estimators.estimator import Estimator


class TFBSequenceEstimator(Estimator):
    """Predictive entropy over TFB stochastic samples (sequence level).

    For each input, the calculator stores ``n_samples`` decoded texts.  This
    estimator counts unique texts and computes the empirical entropy of the
    resulting distribution -- higher entropy means more disagreement among
    samples and thus higher uncertainty.
    """

    def __init__(self, stats_key: str = "tfb"):
        self.stats_key = stats_key
        self.target_stat = f"{stats_key}_texts"
        super().__init__([self.target_stat], "sequence")

    def __str__(self):
        return f"TFBSequenceEntropy({self.stats_key})"

    def __call__(self, stats: dict) -> np.ndarray:
        if self.target_stat not in stats:
            raise KeyError(
                f"Statistic '{self.target_stat}' not found.  Make sure "
                f"TFBStatCalculator is configured with stats_key='{self.stats_key}' "
                f"and is listed in the stat_calculators config."
            )

        batch_samples = stats[self.target_stat]
        uncertainties = []
        for samples in batch_samples:
            counts: dict = {}
            for s in samples:
                counts[s] = counts.get(s, 0) + 1
            probs = np.array(list(counts.values())) / len(samples)
            uncertainties.append(float(-np.sum(probs * np.log(probs + 1e-10))))
        return np.array(uncertainties)


class TFBTokenEstimator(Estimator):
    """Standard deviation of log-probabilities across TFB samples (token level).

    High deviation indicates that the token prediction is sensitive to weight
    noise and therefore uncertain.

    .. note::

       In MCQ / classification mode TFB generates single-token predictions,
       so this estimator returns single-element sequences per item.  It becomes
       more informative if TFB is extended to multi-token generation in the
       future.
    """

    def __init__(self, stats_key: str = "tfb"):
        self.stats_key = stats_key
        self.target_stat = f"{stats_key}_log_probs"
        super().__init__([self.target_stat], "token")

    def __str__(self):
        return f"TFBTokenVariance({self.stats_key})"

    def __call__(self, stats: dict) -> list:
        if self.target_stat not in stats:
            raise KeyError(
                f"Statistic '{self.target_stat}' not found.  Make sure "
                f"TFBStatCalculator is configured with stats_key='{self.stats_key}' "
                f"and is listed in the stat_calculators config."
            )

        batch_samples = stats[self.target_stat]
        uncertainties = []
        for samples in batch_samples:
            if not samples:
                uncertainties.append([])
                continue
            min_len = min(len(s) for s in samples)
            if min_len == 0:
                uncertainties.append([])
                continue
            aligned = np.array([s[:min_len] for s in samples])
            uncertainties.append(np.std(aligned, axis=0).tolist())
        return uncertainties
