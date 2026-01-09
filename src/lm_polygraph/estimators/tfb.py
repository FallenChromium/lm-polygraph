"""
TFB (Training-Free Bayesianization) uncertainty estimators.

These estimators compute uncertainty scores from TFB samples, which capture
epistemic uncertainty in LoRA-finetuned models.
"""

import numpy as np

from typing import Dict, List

from .estimator import Estimator


class TFBPredictiveEntropy(Estimator):
    """
    Predictive entropy estimated from TFB samples.
    
    Computes the negative mean log probability across samples:
    H ≈ -E_θ[log p(y|x, θ)]
    
    Higher values indicate higher uncertainty.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_probs"], "sequence")

    def __str__(self):
        return "TFBPredictiveEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute predictive entropy from TFB samples.
        
        Args:
            stats: Dictionary containing 'tfb_sample_log_probs'
            
        Returns:
            Array of entropy values, one per input text
        """
        log_probs = stats["tfb_sample_log_probs"]
        
        results = []
        for sample_log_probs in log_probs:
            if len(sample_log_probs) == 0:
                results.append(0.0)
            else:
                # Negative mean log prob = entropy estimate
                results.append(-np.mean(sample_log_probs))
        
        return np.array(results)


class TFBMeanLogProb(Estimator):
    """
    Mean log probability across TFB samples.
    
    Lower (more negative) values indicate lower confidence / higher uncertainty.
    Note: This returns negative uncertainty (higher = more confident),
    so negate if needed for comparison with other estimators.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_probs"], "sequence")

    def __str__(self):
        return "TFBMeanLogProb"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute mean log probability from TFB samples.
        
        Args:
            stats: Dictionary containing 'tfb_sample_log_probs'
            
        Returns:
            Array of mean log prob values (higher = more confident)
        """
        log_probs = stats["tfb_sample_log_probs"]
        
        results = []
        for sample_log_probs in log_probs:
            if len(sample_log_probs) == 0:
                results.append(-float('inf'))
            else:
                results.append(np.mean(sample_log_probs))
        
        return np.array(results)


class TFBSampleVariance(Estimator):
    """
    Variance of log probabilities across TFB samples.
    
    High variance indicates the model produces very different likelihood
    estimates under different weight samples, suggesting uncertainty.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_probs"], "sequence")

    def __str__(self):
        return "TFBSampleVariance"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute variance of log probs across TFB samples.
        
        Args:
            stats: Dictionary containing 'tfb_sample_log_probs'
            
        Returns:
            Array of variance values, one per input text
        """
        log_probs = stats["tfb_sample_log_probs"]
        
        results = []
        for sample_log_probs in log_probs:
            if len(sample_log_probs) < 2:
                results.append(0.0)
            else:
                results.append(np.var(sample_log_probs))
        
        return np.array(results)


class TFBSampleStd(Estimator):
    """
    Standard deviation of log probabilities across TFB samples.
    
    Similar to TFBSampleVariance but in the same units as log probability.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_probs"], "sequence")

    def __str__(self):
        return "TFBSampleStd"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute std of log probs across TFB samples.
        
        Args:
            stats: Dictionary containing 'tfb_sample_log_probs'
            
        Returns:
            Array of std values, one per input text
        """
        log_probs = stats["tfb_sample_log_probs"]
        
        results = []
        for sample_log_probs in log_probs:
            if len(sample_log_probs) < 2:
                results.append(0.0)
            else:
                results.append(np.std(sample_log_probs))
        
        return np.array(results)


class TFBLexicalSimilarity(Estimator):
    """
    Lexical diversity across TFB samples (1 - average pairwise ROUGE).
    
    If TFB samples produce very different texts, this indicates
    the model is uncertain about the correct response.
    
    Higher values = more diverse samples = higher uncertainty.
    """

    def __init__(self, metric: str = "rouge1"):
        """
        Args:
            metric: ROUGE metric to use. Options: 'rouge1', 'rouge2', 'rougeL'.
                    Default: 'rouge1'
        """
        super().__init__(["tfb_sample_texts"], "sequence")
        self.metric = metric

    def __str__(self):
        return f"TFBLexicalSim_{self.metric}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute lexical diversity from TFB sample texts.
        
        Args:
            stats: Dictionary containing 'tfb_sample_texts'
            
        Returns:
            Array of diversity scores (1 - avg similarity), one per input
        """
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer([self.metric], use_stemmer=True)
        sample_texts_batch = stats["tfb_sample_texts"]
        
        results = []
        for sample_texts in sample_texts_batch:
            if len(sample_texts) < 2:
                results.append(0.0)
                continue
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(sample_texts)):
                for j in range(i + 1, len(sample_texts)):
                    if sample_texts[i] and sample_texts[j]:
                        score = scorer.score(sample_texts[i], sample_texts[j])
                        similarities.append(score[self.metric].fmeasure)
            
            if len(similarities) == 0:
                results.append(0.0)
            else:
                # Diversity = 1 - similarity
                results.append(1.0 - np.mean(similarities))
        
        return np.array(results)


class TFBNumberUniqueSamples(Estimator):
    """
    Number of unique text samples generated by TFB.
    
    More unique samples suggests the model is uncertain and produces
    different outputs under weight perturbations.
    """

    def __init__(self):
        super().__init__(["tfb_sample_texts"], "sequence")

    def __str__(self):
        return "TFBNumUnique"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Count unique samples for each input.
        
        Args:
            stats: Dictionary containing 'tfb_sample_texts'
            
        Returns:
            Array of unique sample counts
        """
        sample_texts_batch = stats["tfb_sample_texts"]
        
        results = []
        for sample_texts in sample_texts_batch:
            unique_texts = set(sample_texts)
            results.append(len(unique_texts))
        
        return np.array(results)


class TFBTokenLevelEntropy(Estimator):
    """
    Average token-level entropy across TFB samples.
    
    Computes the variance of log likelihoods at each token position,
    then averages across positions. High values indicate the model
    is uncertain about individual tokens.
    """

    def __init__(self):
        super().__init__(["tfb_sample_log_likelihoods"], "sequence")

    def __str__(self):
        return "TFBTokenEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute average token-level variance from TFB samples.
        
        Args:
            stats: Dictionary containing 'tfb_sample_log_likelihoods'
            
        Returns:
            Array of average token entropy values
        """
        log_likelihoods = stats["tfb_sample_log_likelihoods"]
        
        results = []
        for sample_lls in log_likelihoods:
            if len(sample_lls) < 2:
                results.append(0.0)
                continue
            
            # Find minimum length across samples
            min_len = min(len(ll) for ll in sample_lls if len(ll) > 0)
            if min_len == 0:
                results.append(0.0)
                continue
            
            # Compute variance at each position
            token_variances = []
            for pos in range(min_len):
                pos_log_probs = [ll[pos] for ll in sample_lls if len(ll) > pos]
                if len(pos_log_probs) >= 2:
                    token_variances.append(np.var(pos_log_probs))
            
            if len(token_variances) == 0:
                results.append(0.0)
            else:
                # Average variance = estimate of token-level uncertainty
                results.append(np.mean(token_variances))
        
        return np.array(results)
