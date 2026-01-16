import hashlib
import json
import logging
import os
from typing import Literal

import numpy as np
import torch
from pyarrow import set_cpu_count
from torch import nn

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel

log = logging.getLogger("lm_polygraph")


def patch_model_for_tfb(
    model: nn.Module, initial_beta: float = 0.0, use_softplus: bool = False
) -> bool:
    # Check if already patched
    if hasattr(model, "_tfb_patched"):
        log.info("Model already patched for TFB. Skipping SVD re-computation.")
        return False

    patched_any = False

    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        # Skip if already patched
        if hasattr(module, "_tfb_patched"):
            continue

        # Store TFB state per adapter: {adapter_name: (rho, singular_values)}
        module.tfb_rho = nn.ParameterDict()
        module.tfb_singular_values = {}  # Not parameters, just cached
        module.tfb_beta = initial_beta
        module.tfb_enabled = False
        module.tfb_use_softplus = use_softplus

        # Initialize for each adapter
        for adapter_name in module.lora_A.keys():
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            # SVD of B: B = U @ diag(D) @ Vh
            U, D, Vh = torch.linalg.svd(lora_B.weight.float(), full_matrices=False)

            in_features = module.in_features

            # Prevent division by zero with small epsilon
            D_safe = D.reshape(-1, 1).expand(-1, in_features) + 1e-6
            target_sigma = torch.sqrt(initial_beta / D_safe)

            if use_softplus:
                rho = torch.log(torch.exp(target_sigma) - 1 + 1e-6)
            else:
                rho = torch.sqrt(target_sigma)

            rho = rho.to(lora_A.weight.dtype)

            # Store state
            module.tfb_rho[adapter_name] = nn.Parameter(rho)
            module.tfb_singular_values[adapter_name] = D  # Cache for beta updates

            # Transform weights to SVD basis (one-time)
            lora_B.weight = nn.Parameter((U @ torch.diag(D)).to(lora_B.weight.dtype))
            lora_A.weight = nn.Parameter(
                (Vh @ lora_A.weight.float()).to(lora_A.weight.dtype)
            )

        # Replace forward method
        original_forward = module.forward
        module.forward = _create_tfb_forward(original_forward, module)

        module._tfb_patched = True
        patched_any = True

    # Mark model as globally patched
    model._tfb_patched = True
    return patched_any


def _create_tfb_forward(original_forward, module):
    """Creates the TFB-aware forward function (called once during patching)"""

    def forward(x: torch.Tensor, *args, **kwargs):
        dtype = x.dtype

        # 1. Deterministic LoRA pass
        result = module.base_layer(x, *args, **kwargs)
        for adapter in module.active_adapters:
            if adapter not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter]
            lora_B = module.lora_B[adapter]
            dropout = module.lora_dropout[adapter]
            scaling = module.scaling[adapter]

            x_casted = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x_casted))) * scaling

        # 2. Stochastic noise (if enabled)
        if module.tfb_enabled:
            for adapter in module.active_adapters:
                if adapter not in module.tfb_rho:
                    continue

                lora_A = module.lora_A[adapter]
                lora_B = module.lora_B[adapter]
                dropout = module.lora_dropout[adapter]
                scaling = module.scaling[adapter]

                rho = module.tfb_rho[adapter]

                if module.tfb_use_softplus:
                    sigma = torch.nn.functional.softplus(rho)
                else:
                    sigma = rho**2

                x_casted = x.to(lora_A.weight.dtype)
                x_dropped = dropout(x_casted)

                # Flipout: pseudo-independent noise using Rademacher signs
                bs = x_dropped.size(0)
                seq_len = x_dropped.size(1) if x_dropped.dim() == 3 else 1
                r_dim = module.in_features

                if x_dropped.dim() == 2:
                    r = (
                        torch.randint(0, 2, (bs, r_dim), device=x.device).float() * 2
                        - 1
                    )
                    s = (
                        torch.randint(
                            0, 2, (bs, module.r[adapter]), device=x.device
                        ).float()
                        * 2
                        - 1
                    )
                else:
                    r = (
                        torch.randint(
                            0, 2, (bs, seq_len, r_dim), device=x.device
                        ).float()
                        * 2
                        - 1
                    )
                    s = (
                        torch.randint(
                            0, 2, (bs, seq_len, module.r[adapter]), device=x.device
                        ).float()
                        * 2
                        - 1
                    )

                noise_base = torch.randn_like(lora_A.weight)
                noise_A = noise_base * sigma

                # Apply Flipout transformation
                perturb = (((x_dropped * r) @ noise_A.T) * s) @ lora_B.weight.T
                result = result + perturb * scaling

        return result.to(dtype)

    return forward


def update_tfb_beta(model: nn.Module, beta: float):
    """
    Update noise scale across all TFB modules.
    Fast: uses cached singular values, no re-SVD.
    """
    for module in model.modules():
        if not hasattr(module, "tfb_rho"):
            continue

        module.tfb_beta = beta
        use_softplus = module.tfb_use_softplus

        for adapter_name in module.tfb_rho.keys():
            D = module.tfb_singular_values[adapter_name]
            in_features = module.in_features

            D_safe = D.reshape(-1, 1).expand(-1, in_features) + 1e-6

            target_sigma = torch.sqrt(beta / D_safe)

            if use_softplus:
                new_rho = torch.log(torch.exp(target_sigma) - 1 + 1e-6)
            else:
                new_rho = torch.sqrt(target_sigma)

            module.tfb_rho[adapter_name].data.copy_(
                new_rho.to(module.tfb_rho[adapter_name].dtype)
            )


def set_tfb_mode(model: nn.Module, enabled: bool):
    """Toggle stochastic sampling on/off"""
    for module in model.modules():
        if hasattr(module, "tfb_enabled"):
            module.tfb_enabled = enabled


class TFBStatCalculator(StatCalculator):
    def __init__(
        self,
        stats_key: str,
        anchor_inputs: list[str] = None,
        anchor_targets: list[str] = None,
        n_samples: int = 5,
        target_epsilon: float = 0.003,
        beta: float | None = None,
        beta_range: tuple[float, float] = (0.001, 0.2),
        beta_search_steps: int = 10,
        use_softplus: bool = False,
        batch_size: int = 8,
        calibration_mode: Literal["seq_nll", "exact_match"] = "seq_nll",
    ):
        self.stats_key = stats_key
        super().__init__(
            stats=[
                f"{self.stats_key}_texts",
                f"{self.stats_key}_log_probs",
                f"{self.stats_key}_tokens",
            ],
            stats_dependencies=[],
        )

        self.anchor_inputs = anchor_inputs or []
        self.anchor_targets = anchor_targets or []
        self.n_samples = n_samples
        self.epsilon = target_epsilon
        self.beta = beta
        self.beta_min, self.beta_max = beta_range
        self.beta_search_steps = beta_search_steps
        self.use_softplus = use_softplus
        self.batch_size = batch_size
        self.calibration_mode = calibration_mode

        # State
        self._is_calibrated = self.beta is None
        # Store comparison baseline if needed
        self._calibration_baseline = None

    def _metric_seq_nll(self, model, inputs, targets) -> float:
        """Helper to calculate NLL for the anchor set (Self-Consistency)."""
        # TODO: should there be more than 1 sample in calibration?
        nlls = []
        hf_model = model.model
        tokenizer = model.tokenizer

        for inp, trg in zip(inputs, targets):
            try:
                full_text = inp + trg
                enc = tokenizer(full_text, return_tensors="pt").to(model.device())
                labels = enc.input_ids.clone()

                # Mask out input part
                inp_len = tokenizer(inp, return_tensors="pt").input_ids.shape[1]
                labels[:, :inp_len] = -100

                with torch.no_grad():
                    outputs = hf_model(**enc, labels=labels)
                    nlls.append(outputs.loss.item())
            except Exception as e:
                log.warning(f"TFB Eval Error (seq_nll): {e}")
                return 100.0

        return np.mean(nlls)

    def _chunked_generate_texts(self, model, texts, max_new_tokens):
        results = []
        bs = self.batch_size if self.batch_size > 0 else 4
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            results.extend(model.generate_texts(batch, max_new_tokens=max_new_tokens))
        return results

    def _metric_exact_match(self, model, inputs, targets) -> float:
        """Helper to calculate mismatch rate (Flip Rate)."""
        mismatches = 0
        total = 0

        try:
            generated_texts = model.generate_texts(
                inputs,
                max_new_tokens=50,  # TODO: is there a counterexample for this optimization?
            )
            for gen, target in zip(generated_texts, targets):
                if gen.strip() != target.strip():
                    mismatches += 1
                total += 1
        except Exception as e:
            log.warning(f"TFB Eval Error (exact_match): {e}")
            return 1.0 # Max error

        return mismatches / total if total > 0 else 0.0

    def _get_calibration_metric_fn(self):
        if self.calibration_mode == "seq_nll":
            return self._metric_seq_nll
        elif self.calibration_mode == "exact_match":
            return self._metric_exact_match
        else:
            raise ValueError(f"Unknown calibration mode: {self.calibration_mode}")

    def _binary_search(self, model: WhiteboxModel) -> float:
        """Performs the TFB calibration."""
        if not self.anchor_inputs:
            raise ValueError("No beta and no anchor dataset provided.")

        log.info(f"TFB: Starting calibration (Target degradation < {self.epsilon})...")

        hf_model = model.model
        metric_fn = self._get_calibration_metric_fn()

        # 1. Baseline (Deterministic)
        set_tfb_mode(hf_model, False)

        # Determine targets for calibration
        if self.anchor_targets:
            calibration_targets = self.anchor_targets
        else:
            log.info("TFB: Generating baseline targets for calibration...")
            calibration_targets = self._chunked_generate_texts(
                model, self.anchor_inputs, max_new_tokens=20
            )

        if self.calibration_mode == "seq_nll":
             baseline_val = metric_fn(model, self.anchor_inputs, calibration_targets)
        else:
             baseline_val = 0.0 

        log.info(f"TFB: Baseline Metric ({self.calibration_mode}) = {baseline_val:.4f}")

        # 2. Search
        low, high = self.beta_min, self.beta_max
        best_beta = low

        for i in range(self.beta_search_steps):
            mid = (low + high) / 2
            set_tfb_mode(hf_model, True)
            update_tfb_beta(hf_model, mid)

            curr_val = metric_fn(model, self.anchor_inputs, calibration_targets)
            degradation = curr_val - baseline_val

            log.info(
                f"TFB: Iter {i + 1}, Beta={mid:.4f}, Metric={curr_val:.4f}, Delta={degradation:.4f}"
            )

            if degradation < self.epsilon:
                best_beta = mid
                low = mid  # Try more noise
            else:
                high = mid  # Too much noise

        return best_beta

    def __call__(
        self,
        dependencies: dict[str, np.array],
        texts: list[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> dict[str, np.ndarray]:
        patch_model_for_tfb(model.model, use_softplus=self.use_softplus)

        if self.beta is None:
            if not self._is_calibrated:
                self.beta = self._binary_search(model)
                self._is_calibrated = True
            current_beta = self.beta
        else:
            current_beta = self.beta

        try:
            update_tfb_beta(model.model, current_beta)
            set_tfb_mode(model.model, True)

            expanded_texts = []
            for t in texts:
                expanded_texts.extend([t] * self.n_samples)

            batch_tokens = model.tokenize(expanded_texts)
            batch_tokens = {k: v.to(model.device()) for k, v in batch_tokens.items()}

            with torch.no_grad():
                out = model.generate(
                    **batch_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    num_return_sequences=1,
                )

            sequences = out.sequences
            scores = torch.stack(out.scores, dim=1) if out.scores else None

            if model.model_type == "CausalLM":
                input_len = batch_tokens["input_ids"].shape[1]
                gen_sequences = sequences[:, input_len:]
            elif model.model_type == "Seq2SeqLM":
                gen_sequences = sequences[:, 1:]
            else:
                input_len = batch_tokens["input_ids"].shape[1]
                gen_sequences = sequences[:, input_len:]

            res_texts = []
            res_log_probs = []
            res_tokens = []

            cpu_seqs = gen_sequences.cpu().tolist()
            cpu_scores = scores.cpu() if scores is not None else None

            for i in range(len(texts)):
                start = i * self.n_samples
                end = start + self.n_samples

                sample_texts = []
                sample_log_probs = []
                sample_tokens = []

                for j in range(start, end):
                    seq = cpu_seqs[j]
                    if model.tokenizer.eos_token_id in seq:
                        eos_idx = seq.index(model.tokenizer.eos_token_id)
                        seq = seq[:eos_idx]
                    text = model.tokenizer.decode(seq)

                    if cpu_scores is not None:
                        slen = min(len(seq), cpu_scores.shape[1])
                        current_scores = cpu_scores[j, :slen, :]
                        current_log_probs = torch.log_softmax(current_scores, dim=-1)

                        token_ids = torch.tensor(seq[:slen]).unsqueeze(-1)
                        token_log_probs = torch.gather(
                            current_log_probs, -1, token_ids
                        ).squeeze(-1)
                        sample_log_probs.append(token_log_probs.tolist())
                    else:
                        sample_log_probs.append([])

                    sample_texts.append(text)
                    sample_tokens.append(seq)

                res_texts.append(sample_texts)
                res_log_probs.append(sample_log_probs)
                res_tokens.append(sample_tokens)

        finally:
            set_tfb_mode(model.model, False)

        return {
            f"{self.stats_key}_texts": res_texts,
            f"{self.stats_key}_log_probs": res_log_probs,
            f"{self.stats_key}_tokens": res_tokens,
        }
