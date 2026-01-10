"""
TFB (Training-Free Bayesianization) sampling calculator.

Generates multiple stochastic samples from a LoRA-finetuned model
using TFB weight perturbations.
"""

import torch
import numpy as np
import warnings

from typing import Dict, List, Tuple, Union

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.tfb import (
    apply_tfb,
    enable_tfb_sampling,
    disable_tfb_sampling,
    fit_tfb_beta,
)


class TFBSamplingCalculator(StatCalculator):
    """
    Generates multiple stochastic samples using TFB (Training-Free Bayesianization).
    
    TFB adds calibrated noise to LoRA weights during inference, producing
    diverse outputs that reflect model uncertainty. Unlike MC Dropout or
    temperature sampling, TFB noise is derived from the learned LoRA structure
    via SVD, providing principled Bayesian uncertainty.
    
    Requirements:
        - Model must be a PEFT model with LoRA adapters
        - PEFT library must be installed
    
    Example:
        >>> from lm_polygraph.stat_calculators import TFBSamplingCalculator
        >>> calculator = TFBSamplingCalculator(n_samples=10, beta=0.2)
        >>> stats = calculator({}, texts, model, max_new_tokens=100)
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        
        Statistics produced:
            - tfb_sample_log_probs: Sum of log probs for each sample
            - tfb_sample_tokens: Token IDs for each sample
            - tfb_sample_texts: Decoded text for each sample
            - tfb_sample_log_likelihoods: Per-token log probs for each sample
        """
        return [
            "tfb_sample_log_probs",
            "tfb_sample_tokens",
            "tfb_sample_texts",
            "tfb_sample_log_likelihoods",
        ], []

    def __init__(
        self,
        n_samples: int = 10,
        beta: float = 0.2,
        use_softplus: bool = False,
        auto_apply_tfb: bool = True,
        parallel: bool = False,
    ):
        """
        Initialize TFB sampling calculator.
        
        Args:
            n_samples: Number of stochastic samples to generate per input.
                       More samples = better uncertainty estimates but slower.
                       Default: 10
            beta: Variance scaling parameter for TFB noise. Higher values
                  produce more diverse samples. Default: 0.2
            use_softplus: If True, use softplus variance parameterization.
                          If False, use squared parameterization (default).
            auto_apply_tfb: If True, automatically apply TFB to model on first
                            call if not already applied. Default: True
        """
        super().__init__()
        self.n_samples = n_samples
        self.beta = beta
        self.use_softplus = use_softplus
        self.auto_apply_tfb = auto_apply_tfb
        self._tfb_applied = False
        self.parallel = parallel

    def _ensure_tfb_applied(self, model: WhiteboxModel) -> None:
        """
        Ensure TFB modifications are applied to the model.
        
        Args:
            model: WhiteboxModel wrapping a PEFT model
            
        Raises:
            ValueError: If model is not compatible with TFB
        """
        if self._tfb_applied:
            return
        
        
        # Check if TFB is already applied
        tfb_already_applied = any(
            hasattr(m, 'tfb_sampling_enabled') 
            for m in model.model.modules()
        )
        
        if tfb_already_applied:
            self._tfb_applied = True
            return
        
        if not self.auto_apply_tfb:
            raise ValueError(
                "TFB has not been applied to this model. Either call "
                "apply_tfb(model.model) first, or set auto_apply_tfb=True."
            )
        
        # Auto-apply TFB
        try:
            apply_tfb(model.model, beta=self.beta, use_softplus=self.use_softplus)
            self._tfb_applied = True
        except ValueError as e:
            raise ValueError(
                f"Failed to apply TFB: {e}. "
                "Ensure your model is a PEFT model with LoRA adapters."
            ) from e

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate TFB samples for input texts.
        
        Args:
            dependencies: Input statistics (not used, can be empty)
            texts: Input texts batch for generation
            model: WhiteboxModel with PEFT/LoRA model
            max_new_tokens: Maximum tokens to generate. Default: 100
            
        Returns:
            Dictionary with:
                - tfb_sample_log_probs: List[List[float]] - sum log prob per sample
                - tfb_sample_tokens: List[List[List[int]]] - tokens per sample
                - tfb_sample_texts: List[List[str]] - decoded texts per sample
                - tfb_sample_log_likelihoods: List[List[List[float]]] - per-token log probs
        """
        self._ensure_tfb_applied(model)
        
        # Tokenize inputs
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        
        batch_size = len(texts)
        
        # Storage for all samples
        all_sequences = []
        all_logits = []
        
        # Enable TFB sampling
        enable_tfb_sampling(model.model)
        
        # Generate samples
        with torch.no_grad():
            if self.parallel and self.n_samples > 1:
                # Parallel generation: batch repetition
                batch_repeated = {}
                for k, v in batch.items():
                    # k is 'input_ids' or 'attention_mask' of shape [batch, len]
                    # we need [batch*n, len]
                    # repeat_interleave keeps (b1, b1, ..., b2, b2) grouping
                    batch_repeated[k] = v.repeat_interleave(self.n_samples, dim=0)
                
                out = model.generate(
                    **batch_repeated,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                )
                
                # Unpack results: [batch*n, len] -> n x [batch, len]
                # Since we used repeat_interleave, results are grouped by batch elem
                # b1_s1, b1_s2, ..., b2_s1, b2_s2
                
                total_sequences = out.sequences
                total_scores = torch.stack(out.scores, dim=1) # [batch*n, gen_len, vocab]
                
                for sample_idx in range(self.n_samples):

                    indices = torch.arange(sample_idx, len(total_sequences), self.n_samples, device=model.device())

                    all_sequences.append(total_sequences[indices])
                    all_logits.append(total_scores[indices])

            else:
                # Sequential generation
                for _ in range(self.n_samples):
                    out = model.generate(
                        **batch,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=2,
                        do_sample=False,  # Greedy - stochasticity comes from TFB
                        num_beams=1,
                        num_return_sequences=1,
                    )
                    all_sequences.append(out.sequences)
                    # Stack scores: [seq_len, batch, vocab] -> [batch, seq_len, vocab]
                    stacked_scores = torch.stack(out.scores, dim=1)
                    all_logits.append(stacked_scores)
        
        # Disable sampling after generation
        disable_tfb_sampling(model.model)
        
        # Process outputs into statistics
        log_probs = [[] for _ in range(batch_size)]
        tokens = [[] for _ in range(batch_size)]
        decoded_texts = [[] for _ in range(batch_size)]
        log_likelihoods = [[] for _ in range(batch_size)]
        
        for sample_idx in range(self.n_samples):
            sequences = all_sequences[sample_idx]
            logits = all_logits[sample_idx]
            
            for batch_idx in range(batch_size):
                log_prob = 0.0
                ll = []
                toks = []
                
                # Input size to skip prompt tokens
                inp_size = len(batch["input_ids"][batch_idx])
                seq = sequences[batch_idx]
                seq_logits = logits[batch_idx]  # [gen_len, vocab]
                
                # Extract token-level statistics
                gen_len = len(seq) - inp_size
                for j in range(min(gen_len, len(seq_logits))):
                    cur_token = seq[j + inp_size].item()
                    
                    # Get log probability of generated token
                    token_log_prob = seq_logits[j][cur_token].item()
                    log_prob += token_log_prob
                    
                    # Stop at EOS
                    if cur_token == model.tokenizer.eos_token_id:
                        break
                    
                    ll.append(token_log_prob)
                    toks.append(cur_token)
                
                # Only add if we got tokens
                if len(toks) > 0:
                    log_probs[batch_idx].append(log_prob)
                    tokens[batch_idx].append(toks)
                    decoded_texts[batch_idx].append(model.tokenizer.decode(toks))
                    log_likelihoods[batch_idx].append(ll)
                else:
                    warnings.warn(
                        f"No tokens generated for batch {batch_idx}, sample {sample_idx}. "
                        "This sample will be skipped."
                    )
        
        return {
            "tfb_sample_log_probs": log_probs,
            "tfb_sample_tokens": tokens,
            "tfb_sample_texts": decoded_texts,
            "tfb_sample_log_likelihoods": log_likelihoods,
        }
