import numpy as np
from lm_polygraph.estimators.estimator import Estimator


class TFBSequenceEstimator(Estimator):
    """
    TFB Sequence Estimator.
    Calculates Semantic Entropy based on unique generated texts from TFB samples.
    """

    def __init__(self, stats_key: str):
        self.stats_key = stats_key
        self.target_stat = f"{stats_key}_texts"
        super().__init__([self.target_stat], "sequence")

    def __str__(self):
        return f"TFBSequenceEntropy({self.stats_key})"

    def __call__(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        batch_samples = stats[self.target_stat]

        uncertainties = []
        for samples in batch_samples:
            # Calculate Predictive Entropy based on unique generations
            unique_counts = {}
            for s in samples:
                unique_counts[s] = unique_counts.get(s, 0) + 1

            probs = np.array(list(unique_counts.values())) / len(samples)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            uncertainties.append(entropy)

        return np.array(uncertainties)


class TFBTokenEstimator(Estimator):
    """
    TFB Token Estimator.
    Calculates the standard deviation of log-probabilities across samples for each token.
    High deviation implies the token is sensitive to weight noise (uncertain).
    """

    def __init__(self, stats_key: str):
        self.stats_key = stats_key
        self.target_stat = f"{stats_key}_log_probs"
        super().__init__([self.target_stat], "token")

    def __str__(self):
        return f"TFBTokenVariance({self.stats_key})"

    def __call__(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        # batch_samples: List[List[List[float]]] -> [Batch, Sample, SeqLen]
        batch_samples = stats[self.target_stat]
        
        uncertainties = []
        for samples in batch_samples:
            # samples: [Sample, SeqLen]
            # Note: SeqLen might differ if lengths differ, but usually in batch generation 
            # without padding removal they are padded. 
            # However, our Calculator returns unpadded lists of floats.
            
            # We need to align them. A simple heuristic is to truncate to min length
            # or work on the prefix.
            if not samples:
                uncertainties.append([])
                continue
                
            min_len = min(len(s) for s in samples)
            if min_len == 0:
                uncertainties.append([])
                continue
                
            # [Sample, MinLen]
            aligned_samples = np.array([s[:min_len] for s in samples])
            
            # Calculate std dev across samples (axis 0)
            token_std = np.std(aligned_samples, axis=0)
            uncertainties.append(token_std.tolist())

        # For token level, we return a flat list of token uncertainties if possible,
        # or list of lists. The UEManager expects list of lists for 'token' level 
        # (based on MaxTokenProbability source).
        return uncertainties
