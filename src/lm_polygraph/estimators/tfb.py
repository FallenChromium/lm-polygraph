import numpy as np
from lm_polygraph.estimators.estimator import Estimator


class TFBEstimator(Estimator):
    """
    Training-Free Bayesianization (TFB) Estimator.
    Calculates predictive entropy based on TFB samples.
    """

    def __init__(self, stats_key: str):
        # We depend on the key provided by TFBStatCalculator
        self.target_stat = stats_key
        super().__init__([self.target_stat], "sequence")

    def __str__(self):
        return f"TFB_Uncertainty({self.target_stat})"

    def __call__(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        batch_samples = stats[self.target_stat]

        uncertainties = []
        for samples in batch_samples:
            # Calculate Predictive Entropy based on unique generations
            # (Simplified metric: higher variety = higher uncertainty)
            unique_counts = {}
            for s in samples:
                unique_counts[s] = unique_counts.get(s, 0) + 1

            probs = np.array(list(unique_counts.values())) / len(samples)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            uncertainties.append(entropy)

        return np.array(uncertainties)
