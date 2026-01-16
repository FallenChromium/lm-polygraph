import hashlib
import json
import logging
import os

import numpy as np
import torch
from pyarrow import set_cpu_count
from torch import nn

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel

log = logging.getLogger("lm_polygraph")


def patch_model_for_tfb(model: nn.Module, initial_beta: float = 0.0) -> bool:
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

        # Initialize for each adapter
        for adapter_name in module.lora_A.keys():
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            # SVD of B: B = U @ diag(D) @ Vh
            U, D, Vh = torch.linalg.svd(lora_B.weight.float(), full_matrices=False)

            # Compute variance: rho = sqrt(beta / D)
            in_features = module.in_features
            lora_std = initial_beta / (D.reshape(-1, 1).expand(-1, in_features) + 1e-6)
            rho = torch.sqrt(lora_std).to(lora_A.weight.dtype)

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
                sigma_sq = rho**2

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

                # Sample noise ~ N(0, sigma^2)
                noise_A = sigma_sq * torch.randn_like(lora_A.weight)

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

        for adapter_name in module.tfb_rho.keys():
            D = module.tfb_singular_values[adapter_name]
            in_features = module.in_features

            # Recompute rho from singular values
            lora_std = beta / (D.reshape(-1, 1).expand(-1, in_features) + 1e-6)
            new_rho = torch.sqrt(lora_std).to(module.tfb_rho[adapter_name].dtype)

            module.tfb_rho[adapter_name].data.copy_(new_rho)


def set_tfb_mode(model: nn.Module, enabled: bool):
    """Toggle stochastic sampling on/off"""
    for module in model.modules():
        if hasattr(module, "tfb_enabled"):
            module.tfb_enabled = enabled


def is_tfb_patched(model: nn.Module) -> bool:
    """Check if model has been patched"""
    for module in model.modules():
        if hasattr(module, "_tfb_patched"):
            return True
    return False


def get_model_fingerprint(model: nn.Module) -> str:
    """
    Generate a fingerprint for the model to detect swaps.
    Uses first LoRA weight's pointer address.
    """
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.keys():
                # Use data pointer as fingerprint
                return str(module.lora_A[adapter].weight.data_ptr())
    return "no_lora_found"


class TFBStatCalculator(StatCalculator):
    def __init__(
        self,
        config_hash: str,
        anchor_inputs: list[str],  # inputs
        anchor_targets: list[str],  # targets
        n_samples: int,
        target_epsilon: float,
        beta: float | None = None,
        beta_range: tuple[float, float] = (0.001, 0.2),
        beta_search_steps: int = 10,
        save_dir: str | None = None,
        batch_size: int = 8,
    ):
        # Unique key to allow multiple TFB configs in one run
        self.stats_key = f"tfb_samples_{config_hash}"
        super().__init__(stats=[self.stats_key], stats_dependencies=[])

        self.anchor_inputs = anchor_inputs
        self.anchor_targets = anchor_targets
        self.n_samples = n_samples
        self.epsilon = target_epsilon
        self.beta = beta
        self.beta_min, self.beta_max = beta_range
        self.beta_search_steps = beta_search_steps
        self.save_dir = save_dir
        self._is_calibrated = False
        self._optimal_beta = 0.0
        self.batch_size = batch_size

    def _calculate_nll(self, model, inputs, targets) -> float:
        """Helper to calculate NLL for the anchor set."""
        # Note: This is a simplified sequential eval for brevity.
        # In production, this should be batched using model.tokenizer.
        nlls = []
        for inp, trg in zip(inputs, targets):
            # We assume WhiteboxModel exposes 'log_probs' or similar scoring
            # For pure PyTorch generation loop:
            try:
                # Use polygraph model's internal tokenizer logic if accessible
                # or rely on standard formatting. Here we try a generic approach:
                # Calculate loss (NLL) of generating 'trg' given 'inp'
                # WhiteboxModel usually has model.model (HF) and model.tokenizer
                hf_model = model.model
                tokenizer = model.tokenizer

                full_text = inp + trg
                enc = tokenizer(full_text, return_tensors="pt").to(model.device())
                labels = enc.input_ids.clone()

                # Mask out input part for loss calculation
                inp_len = tokenizer(inp, return_tensors="pt").input_ids.shape[1]
                labels[:, :inp_len] = -100

                with torch.no_grad():
                    outputs = hf_model(**enc, labels=labels)
                    nlls.append(outputs.loss.item())
            except Exception as e:
                log.warning(f"TFB Eval Error: {e}")
                return 100.0  # High penalty

        return np.mean(nlls)

    def _binary_search(self, model: WhiteboxModel) -> float:
        """Performs the TFB calibration."""
        log.info(f"TFB: Starting calibration (Target degradation < {self.epsilon})...")

        hf_model = model.model

        # 1. Baseline (Deterministic)
        set_tfb_mode(hf_model, False)
        baseline_nll = self._calculate_nll(
            model, self.anchor_inputs, self.anchor_targets
        )
        log.info(f"TFB: Baseline NLL = {baseline_nll:.4f}")

        # 2. Search
        low, high = self.beta_min, self.beta_max
        best_beta = low

        for i in range(self.beta_search_steps):  # Fixed iterations
            mid = (low + high) / 2
            set_tfb_mode(hf_model, True)
            update_tfb_beta(hf_model, mid)

            # Stochastic NLL (average of 1 run implies weak estimate, but fast)
            # Ideally average multiple runs here
            curr_nll = self._calculate_nll(
                model, self.anchor_inputs, self.anchor_targets
            )
            degradation = curr_nll - baseline_nll

            log.info(
                f"TFB: Iter {i + 1}, Beta={mid:.4f}, NLL={curr_nll:.4f}, Delta={degradation:.4f}"
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
        # --- Lazy Initialization ---
        if not self._is_calibrated:
            # 1. Apply TFB Architecture Changes
            patch_model_for_tfb(model.model)

            self.beta = self.beta or self._binary_search(model)

            # 3. Finalize
            update_tfb_beta(model.model, self.beta)
            set_tfb_mode(model.model, True)
            self._is_calibrated = True

        results = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            batch_samples = [[] for _ in batch_texts]
            for _ in range(self.n_samples):
                # Ensure TFB mode is on
                set_tfb_mode(model.model, True)
                out = model.generate(batch_texts, max_new_tokens=max_new_tokens)
                for i, gen in enumerate(out):
                    batch_samples[i].append(gen)
            results.extend(batch_samples)

        return {self.stats_key: results}
