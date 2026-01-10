"""
TFB (Training-Free Bayesianization) sampling calculator.

Generates stochastic samples from LoRA-finetuned models using TFB weight
perturbations and computes BMA-based uncertainty statistics.
"""

import torch
import numpy as np
import warnings

from typing import Dict, List, Tuple, Optional

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.tfb import (
    apply_tfb,
    enable_tfb_sampling,
    disable_tfb_sampling,
)


class TFBSamplingCalculator(StatCalculator):
    """
    Computes TFB-based uncertainty using Bayesian Model Averaging.
    
    For each input, runs n_samples stochastic forward passes with TFB weight
    perturbations, then averages probabilities (BMA) for uncertainty estimation.
    
    Optionally generates text samples when max_new_tokens > 0.
    
    Requirements:
        - Model must be a PEFT model with LoRA adapters
    
    Example:
        >>> calculator = TFBSamplingCalculator(n_samples=10, beta=0.01)
        >>> stats = calculator({}, texts, model)
        >>> uncertainty = stats["tfb_mean_std"]
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "tfb_bma_probs",
            "tfb_mean_std",
            "tfb_sample_log_probs",
            "tfb_sample_texts",
        ], []

    def __init__(
        self,
        n_samples: int = 10,
        beta: float = 0.01,
        target_ids: Optional[List[int]] = None,
        use_softplus: bool = False,
        auto_apply_tfb: bool = True,
    ):
        """
        Args:
            n_samples: Stochastic samples for BMA. Default: 10
            beta: Variance scale for TFB noise. Default: 0.01
            target_ids: Token IDs to restrict classification. None = full vocab.
            use_softplus: Variance parameterization. Default: False (squared).
            auto_apply_tfb: Auto-apply TFB if not applied. Default: True
        """
        super().__init__()
        self.n_samples = n_samples
        self.beta = beta
        self.target_ids = target_ids
        self.use_softplus = use_softplus
        self.auto_apply_tfb = auto_apply_tfb
        self._tfb_applied = False

    def _ensure_tfb_applied(self, model: WhiteboxModel) -> None:
        if self._tfb_applied:
            return
        
        tfb_already_applied = any(
            hasattr(m, 'tfb_sampling_enabled') 
            for m in model.model.modules()
        )
        
        if tfb_already_applied:
            self._tfb_applied = True
            return
        
        if not self.auto_apply_tfb:
            raise ValueError("TFB not applied. Call apply_tfb() or set auto_apply_tfb=True.")
        
        apply_tfb(model.model, beta=self.beta, use_softplus=self.use_softplus)
        self._tfb_applied = True

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 0,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute TFB statistics with BMA.
        
        Args:
            dependencies: Not used
            texts: Input texts
            model: WhiteboxModel with LoRA
            max_new_tokens: If > 0, also generate text samples
            
        Returns:
            tfb_bma_probs: BMA-averaged next-token probs
            tfb_mean_std: Uncertainty score (mean prob std across samples)
            tfb_sample_log_probs: Per-sample sequence log probs
            tfb_sample_texts: Generated texts (only if max_new_tokens > 0)
        """
        self._ensure_tfb_applied(model)
        
        batch = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        batch_size = len(texts)
        device = model.device()
        
        target_tensor = None
        if self.target_ids is not None:
            target_tensor = torch.tensor(self.target_ids, device=device)
        
        # Collect stochastic samples
        enable_tfb_sampling(model.model)
        all_probs = []
        all_log_probs = []
        all_texts = [[] for _ in range(batch_size)]
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                if max_new_tokens > 0:
                    out = model.generate(
                        **batch,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                    sequences = out.sequences
                    scores = torch.stack(out.scores, dim=1)
                    
                    for i in range(batch_size):
                        inp_len = batch["input_ids"][i].shape[0]
                        gen_tokens = sequences[i][inp_len:]
                        text = model.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        all_texts[i].append(text)
                        
                        # Compute log prob of generated sequence
                        log_prob = 0.0
                        for j, tok in enumerate(gen_tokens):
                            if j < scores.shape[1]:
                                log_prob += torch.log_softmax(scores[i, j], dim=-1)[tok].item()
                        all_log_probs.append(log_prob) if i == 0 else None
                    
                    # Next-token probs from first position
                    logits = scores[:, 0, :]
                else:
                    output = model.model(**batch)
                    logits = output.logits[:, -1, :]
                    all_log_probs.append(None)
                
                if target_tensor is not None:
                    logits = logits[:, target_tensor]
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)
        
        disable_tfb_sampling(model.model)
        
        # BMA
        sample_probs = torch.stack(all_probs, dim=0)
        bma_probs = sample_probs.mean(dim=0)
        prob_std = sample_probs.std(dim=0)
        mean_std = prob_std.mean(dim=-1)
        
        # Format output
        result = {
            "tfb_bma_probs": [bma_probs[i].cpu().numpy().tolist() for i in range(batch_size)],
            "tfb_mean_std": mean_std.cpu().numpy().tolist(),
            "tfb_sample_log_probs": [[lp for lp in all_log_probs if lp is not None] for _ in range(batch_size)],
            "tfb_sample_texts": all_texts if max_new_tokens > 0 else [[] for _ in range(batch_size)],
        }
        
        return result
