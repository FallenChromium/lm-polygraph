"""
TFB (Training-Free Bayesianization) uncertainty estimators.

Compute uncertainty scores from TFB samples using Bayesian Model Averaging.
"""

import numpy as np
from typing import Dict
from .estimator import Estimator


class TFBMeanStd(Estimator):
    """
    Mean std of probabilities across TFB samples.
    
    Primary uncertainty metric for TFB. Higher = more uncertain.
    """

    def __init__(self):
        super().__init__(["tfb_mean_std"], "sequence")

    def __str__(self):
        return "TFBMeanStd"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array(stats["tfb_mean_std"])


class TFBBMAEntropy(Estimator):
    """
    Entropy of BMA-averaged probabilities.
    
    H = -sum(p * log(p)) on the averaged distribution.
    """

    def __init__(self):
        super().__init__(["tfb_bma_probs"], "sequence")

    def __str__(self):
        return "TFBBMAEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        bma_probs = stats["tfb_bma_probs"]
        results = []
        for probs in bma_probs:
            p = np.clip(np.array(probs), 1e-12, 1.0)
            results.append(-np.sum(p * np.log(p)))
        return np.array(results)


class TFBSampleVariance(Estimator):
    """
    Variance of log probabilities across TFB samples.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_probs"], "sequence")

    def __str__(self):
        return "TFBSampleVariance"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        log_probs = stats["tfb_sample_log_probs"]
        results = []
        for sample_lps in log_probs:
            if len(sample_lps) < 2:
                results.append(0.0)
            else:
                results.append(np.var(sample_lps))
        return np.array(results)


class TFBLexicalSimilarity(Estimator):
    """
    Lexical diversity across TFB text samples (1 - avg ROUGE).
    
    Only meaningful when generation is enabled.
    """

    def __init__(self, metric: str = "rouge1"):
        super().__init__(["tfb_sample_texts"], "sequence")
        self.metric = metric

    def __str__(self):
        return f"TFBLexicalSim_{self.metric}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer([self.metric], use_stemmer=True)
        sample_texts_batch = stats["tfb_sample_texts"]
        
        results = []
        for sample_texts in sample_texts_batch:
            if len(sample_texts) < 2:
                results.append(0.0)
                continue
            
            similarities = []
            for i in range(len(sample_texts)):
                for j in range(i + 1, len(sample_texts)):
                    if sample_texts[i] and sample_texts[j]:
                        score = scorer.score(sample_texts[i], sample_texts[j])
                        similarities.append(score[self.metric].fmeasure)
            
            if not similarities:
                results.append(0.0)
            else:
                results.append(1.0 - np.mean(similarities))
        
        return np.array(results)
