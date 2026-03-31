"""
Training-Free Bayesianization (TFB) stat calculator.

Adds calibrated Gaussian noise to LoRA adapter weights via Flipout to obtain
a posterior approximation *without* retraining.  Multiple stochastic forward
passes yield a distribution over predictions whose spread is used as an
uncertainty signal.

Reference
---------
Wang et al., "Training-Free Bayesianization for Low-Rank Adaptation of
Large Language Models", arXiv 2412.05723, 2024.
"""

import logging
from collections.abc import Sequence as SequenceABC
from typing import List, Literal, Optional, Sequence

import numpy as np
import torch
from torch import nn

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel

log = logging.getLogger("lm_polygraph")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TFB_SIGMA_EPS = 1e-6


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _compute_target_sigma(
    beta: float, singular_values: torch.Tensor, in_features: int
) -> torch.Tensor:
    """Reference-compatible sigma target: ``beta / (D + eps)``."""
    beta_value = max(float(beta), 0.0)
    d_safe = singular_values.reshape(-1, 1).expand(-1, in_features) + TFB_SIGMA_EPS
    return beta_value / d_safe


def _sigma_to_rho(target_sigma: torch.Tensor, use_softplus: bool) -> torch.Tensor:
    """Convert target sigma to the rho parameterisation used during sampling."""
    if use_softplus:
        return torch.log(torch.expm1(target_sigma) + TFB_SIGMA_EPS)
    return torch.sqrt(torch.clamp(target_sigma, min=0.0))


# ---------------------------------------------------------------------------
# Model patching – SVD rotation + Flipout forward
# ---------------------------------------------------------------------------
def patch_model_for_tfb(
    model: nn.Module, initial_beta: float = 0.0, use_softplus: bool = False
) -> bool:
    """Decompose LoRA-B via SVD and install the Flipout forward hook.

    Raises
    ------
    ValueError
        If the model contains no LoRA adapter layers (``lora_A`` / ``lora_B``).
    """
    if hasattr(model, "_tfb_patched"):
        log.info("Model already patched for TFB – skipping SVD re-computation.")
        return False

    patched_any = False

    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if hasattr(module, "_tfb_patched"):
            continue

        module.tfb_rho = nn.ParameterDict()
        module.tfb_singular_values = {}
        module.tfb_beta = initial_beta
        module.tfb_enabled = False
        module.tfb_use_softplus = use_softplus

        for adapter_name in module.lora_A.keys():
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            # SVD of B: B = U @ diag(D) @ Vh
            # Perform SVD on CPU (small matrices, avoids CUDA compat issues)
            orig_device = lora_B.weight.device
            B_cpu = lora_B.weight.detach().float().cpu()
            A_cpu = lora_A.weight.detach().float().cpu()
            U, D, Vh = torch.linalg.svd(B_cpu, full_matrices=False)
            in_features = module.in_features

            target_sigma = _compute_target_sigma(initial_beta, D, in_features)
            rho = _sigma_to_rho(target_sigma, use_softplus).to(lora_A.weight.dtype)

            module.tfb_rho[adapter_name] = nn.Parameter(rho.to(orig_device))
            module.tfb_singular_values[adapter_name] = D.to(orig_device)

            # Rotate weights into SVD basis (one-time transformation)
            new_B = (U @ torch.diag(D)).to(
                dtype=lora_B.weight.dtype, device=orig_device
            )
            new_A = (Vh @ A_cpu).to(dtype=lora_A.weight.dtype, device=orig_device)
            lora_B.weight = nn.Parameter(new_B)
            lora_A.weight = nn.Parameter(new_A)

        module.forward = _create_tfb_forward(module.forward, module)
        module._tfb_patched = True
        patched_any = True

    if not patched_any:
        raise ValueError(
            "TFB requires a model with LoRA adapters (lora_A / lora_B modules). "
            "No LoRA layers were found.  Make sure you have applied a LoRA "
            "config (e.g. via peft.get_peft_model) before calling "
            "patch_model_for_tfb."
        )

    model._tfb_patched = True
    return patched_any


def _create_tfb_forward(original_forward, module):
    """Return a Flipout-aware forward replacing the original LoRA forward."""

    def forward(x: torch.Tensor, *args, **kwargs):
        dtype = x.dtype

        # --- deterministic LoRA pass ---
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

        # --- stochastic noise via Flipout ---
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

                bs = x_dropped.size(0)
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
                    seq_len = x_dropped.size(1)
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

                noise_A = torch.randn_like(lora_A.weight) * sigma
                perturb = (((x_dropped * r) @ noise_A.T) * s) @ lora_B.weight.T
                result = result + perturb * scaling

        return result.to(dtype)

    return forward


# ---------------------------------------------------------------------------
# Runtime helpers – beta update & mode toggle
# ---------------------------------------------------------------------------
def update_tfb_beta(model: nn.Module, beta: float):
    """Update the noise scale across all patched LoRA layers (no re-SVD)."""
    for module in model.modules():
        if not hasattr(module, "tfb_rho"):
            continue
        module.tfb_beta = beta
        use_softplus = module.tfb_use_softplus
        for adapter_name in module.tfb_rho.keys():
            D = module.tfb_singular_values[adapter_name]
            target_sigma = _compute_target_sigma(beta, D, module.in_features)
            new_rho = _sigma_to_rho(target_sigma, use_softplus)
            module.tfb_rho[adapter_name].data.copy_(
                new_rho.to(module.tfb_rho[adapter_name].dtype)
            )


def set_tfb_mode(model: nn.Module, enabled: bool):
    """Toggle stochastic sampling on / off."""
    for module in model.modules():
        if hasattr(module, "tfb_enabled"):
            module.tfb_enabled = enabled


# ---------------------------------------------------------------------------
# Stat-key helper
# ---------------------------------------------------------------------------
def tfb_stats_for_key(stats_key: str = "tfb") -> List[str]:
    """Return the list of stat names produced by a TFBStatCalculator."""
    return [
        f"{stats_key}_texts",
        f"{stats_key}_log_probs",
        f"{stats_key}_tokens",
        f"{stats_key}_target_probs",
        f"{stats_key}_metadata",
    ]


# ---------------------------------------------------------------------------
# TFBStatCalculator
# ---------------------------------------------------------------------------
class TFBStatCalculator(StatCalculator):
    """Compute stochastic forward-pass statistics for classification tasks.

    In **classification mode** (``target_ids`` or ``target_labels`` provided),
    the calculator extracts next-token class probabilities from the last input
    position across ``n_samples`` stochastic forward passes.

    Free-form generation mode is *not* supported – the semantic interpretation
    of TFB uncertainty for open-ended text has not been established.

    Parameters
    ----------
    stats_key : str
        Unique prefix for the stat names this instance produces.
    target_ids : sequence, optional
        Per-class token-id groups, e.g. ``[[32, 319], [33, 347], ...]``.
        Mutually complementary with *target_labels* (at least one required).
    target_labels : list[str], optional
        Human-readable class labels (e.g. ``["A", "B", "C", "D"]``).
        Resolved to token ids at runtime using the model tokenizer.
    anchor_inputs / anchor_targets : list[str], optional
        Held-out prompts (and optionally their gold completions) used by the
        binary-search beta calibration.
    beta : float, optional
        Fixed noise scale.  If ``None``, beta is calibrated automatically
        (requires *anchor_inputs*).
    n_samples : int
        Number of stochastic forward passes per input.
    max_seq_len : int, optional
        Truncation limit passed to the tokenizer.  ``None`` = no truncation.

    Notes
    -----
    This method requires a **LoRA-adapted** model.  ``patch_model_for_tfb``
    will raise ``ValueError`` if no LoRA layers are found.

    The simple ``estimate_uncertainty(model, estimator, text)`` API cannot be
    used with TFB because TFB needs dataset-level configuration (target ids,
    anchor set, LoRA model).  Use ``polygraph_eval`` with a YAML config or
    instantiate ``UEManager`` directly.
    """

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        # Static fallback (stats_key="tfb").  Instances override ``_stats``
        # in __init__ to match their actual stats_key.
        return (tfb_stats_for_key("tfb"), [])

    def __init__(
        self,
        stats_key: str = "tfb",
        anchor_inputs: Optional[List[str]] = None,
        anchor_targets: Optional[List[str]] = None,
        target_ids: Optional[Sequence] = None,
        target_labels: Optional[List[str]] = None,
        n_samples: int = 5,
        target_epsilon: float = 0.003,
        beta: Optional[float] = None,
        beta_range: tuple = (0.001, 0.2),
        beta_search_steps: int = 10,
        use_softplus: bool = False,
        batch_size: int = 8,
        max_seq_len: Optional[int] = None,
        calibration_mode: Literal["seq_nll", "exact_match"] = "seq_nll",
    ):
        self.stats_key = stats_key
        super().__init__()
        self._stats = tfb_stats_for_key(stats_key)

        self.anchor_inputs = anchor_inputs or []
        self.anchor_targets = anchor_targets or []
        self.target_ids = target_ids
        self.target_labels = target_labels
        self._target_token_indices: Optional[List[torch.Tensor]] = None
        self._class_primary_tokens: List[int] = []
        self.n_samples = n_samples
        self.epsilon = target_epsilon
        self.beta = beta
        self.beta_min, self.beta_max = beta_range
        self.beta_search_steps = beta_search_steps
        self.use_softplus = use_softplus
        self.batch_size = max(batch_size, 1)
        self.max_seq_len = (
            max_seq_len if max_seq_len is None or max_seq_len > 0 else None
        )
        self.calibration_mode = calibration_mode

        self._is_calibrated = self.beta is not None
        self._calibration_summary: dict = {}

        if self.target_ids is not None:
            self._configure_target_ids(self.target_ids)

    # ------------------------------------------------------------------
    # Target-ID configuration
    # ------------------------------------------------------------------
    def _resolve_target_labels(self, tokenizer) -> None:
        """Convert ``self.target_labels`` to token-id groups using *tokenizer*.

        Each label string is tokenized with and without a leading space;
        the last token of each variant is used as the class representative.
        Results are cached so this runs only once.
        """
        if self._target_token_indices is not None:
            return  # already resolved
        if not self.target_labels:
            return

        target_ids: list[list[list[int]]] = []
        debug_map = {}
        for label in self.target_labels:
            variants = [f" {label}", label]
            class_sequences: list[list[int]] = []
            decoded: list[str] = []
            for variant in variants:
                ids = tokenizer(variant, add_special_tokens=False).input_ids
                if ids:
                    seq = [ids[-1]]
                    if seq not in class_sequences:
                        class_sequences.append(seq)
                        decoded.append(tokenizer.decode(seq))
            if not class_sequences:
                # Absolute fallback – encode the raw label
                ids = tokenizer.encode(label, add_special_tokens=False)
                class_sequences = [[ids[-1]]] if ids else [[0]]
                decoded = [tokenizer.decode(class_sequences[0])]
            target_ids.append(class_sequences)
            debug_map[label] = decoded

        log.info(f"TFB: resolved target_labels → token map: {debug_map}")
        self._configure_target_ids(target_ids)

    def _configure_target_ids(self, target_ids: Sequence) -> None:
        """Normalise *target_ids* (various formats) into ``_target_token_indices``."""
        if not target_ids:
            raise ValueError("target_ids must not be empty when provided.")

        first = target_ids[0]
        normalized: list[list[int]] = []

        if isinstance(first, int):
            seen: set = set()
            for tid in target_ids:
                tid_int = int(tid)
                if tid_int not in seen:
                    seen.add(tid_int)
                    normalized.append([tid_int])
        else:

            def _collect(node, buf: list):
                if isinstance(node, torch.Tensor):
                    buf.extend(node.reshape(-1).tolist()) if node.numel() else None
                elif isinstance(node, np.ndarray):
                    buf.extend(node.reshape(-1).tolist()) if node.size else None
                elif isinstance(node, SequenceABC) and not isinstance(
                    node, (str, bytes, bytearray)
                ):
                    for child in node:
                        _collect(child, buf)
                else:
                    buf.append(int(node))

            for group in target_ids:
                if not isinstance(group, SequenceABC) or len(group) == 0:
                    raise ValueError(
                        "Each target_id group must be a non-empty sequence."
                    )
                flat: list[int] = []
                _collect(group, flat)
                seen_g: set = set()
                deduped = [t for t in flat if t not in seen_g and not seen_g.add(t)]
                if not deduped:
                    raise ValueError(
                        "Each target_id group must contain at least one unique token id."
                    )
                normalized.append(deduped)

        self._target_token_indices = [
            torch.tensor(group, dtype=torch.long) for group in normalized
        ]
        self._class_primary_tokens = [group[0] for group in normalized]

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------
    def _get_last_token_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return (
            attention_mask.size(1) - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
        ).long()

    def _compute_class_probs(
        self, logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract and normalise class probabilities from the last-token logits."""
        if not self._target_token_indices:
            raise ValueError("target_ids must be configured for classification mode.")
        last_indices = self._get_last_token_indices(attention_mask)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_idx, last_indices]
        token_log_probs = torch.log_softmax(last_logits, dim=-1)

        class_probs = []
        for idx_tensor in self._target_token_indices:
            idx = idx_tensor.to(token_log_probs.device)
            class_probs.append(
                torch.index_select(token_log_probs, dim=-1, index=idx).exp().sum(dim=-1)
            )
        probs = torch.stack(class_probs, dim=-1)
        return probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)

    def _tokenize_batch(self, model: WhiteboxModel, texts: List[str]) -> dict:
        """Tokenise a batch of texts with optional truncation."""
        if self.max_seq_len is None:
            return model.tokenize(texts)

        if model.instruct:
            formatted = [
                model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for t in texts
            ]
            return model.tokenizer(
                formatted,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,
            )

        return model.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

    # ------------------------------------------------------------------
    # Beta calibration (binary search)
    # ------------------------------------------------------------------
    def _metric_seq_nll(self, model, inputs, targets) -> float:
        nlls = []
        hf_model = model.model
        tokenizer = model.tokenizer
        for inp, trg in zip(inputs, targets):
            try:
                full_text = inp + trg
                enc = tokenizer(full_text, return_tensors="pt").to(model.device())
                labels = enc.input_ids.clone()
                inp_len = tokenizer(inp, return_tensors="pt").input_ids.shape[1]
                labels[:, :inp_len] = -100
                with torch.no_grad():
                    outputs = hf_model(**enc, labels=labels)
                    nlls.append(outputs.loss.item())
            except Exception as e:
                log.warning(f"TFB calibration error (seq_nll): {e}")
                return 100.0
        return float(np.mean(nlls))

    def _chunked_generate_texts(self, model, texts, max_new_tokens):
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results.extend(model.generate_texts(batch, max_new_tokens=max_new_tokens))
        return results

    def _metric_exact_match(self, model, inputs, targets) -> float:
        try:
            generated = self._chunked_generate_texts(model, inputs, max_new_tokens=10)
            mismatches = sum(
                1 for g, t in zip(generated, targets) if g.strip() != t.strip()
            )
            return mismatches / max(len(targets), 1)
        except Exception as e:
            log.warning(f"TFB calibration error (exact_match): {e}")
            return 1.0

    def _binary_search(self, model: WhiteboxModel) -> float:
        """Find the largest beta whose calibration degradation stays below epsilon."""
        if not self.anchor_inputs:
            raise ValueError(
                "Beta calibration requires anchor_inputs.  Either provide a "
                "fixed beta or supply an anchor dataset."
            )
        log.info(f"TFB: starting calibration (target degradation < {self.epsilon})")

        metric_fn = {
            "seq_nll": self._metric_seq_nll,
            "exact_match": self._metric_exact_match,
        }[self.calibration_mode]

        hf_model = model.model
        set_tfb_mode(hf_model, False)

        if self.anchor_targets:
            calibration_targets = self.anchor_targets
        else:
            log.info("TFB: generating baseline targets for calibration …")
            calibration_targets = self._chunked_generate_texts(
                model, self.anchor_inputs, max_new_tokens=20
            )

        baseline_val = metric_fn(model, self.anchor_inputs, calibration_targets)
        log.info(f"TFB: baseline {self.calibration_mode} = {baseline_val:.4f}")

        low, high = self.beta_min, self.beta_max
        best_beta = low
        history = []

        for i in range(self.beta_search_steps):
            mid = (low + high) / 2
            set_tfb_mode(hf_model, True)
            update_tfb_beta(hf_model, mid)
            curr_val = metric_fn(model, self.anchor_inputs, calibration_targets)
            degradation = curr_val - baseline_val
            history.append(
                {"beta": mid, "metric": curr_val, "degradation": degradation}
            )
            log.info(
                f"TFB: iter {i + 1}, beta={mid:.4f}, "
                f"metric={curr_val:.4f}, delta={degradation:.4f}"
            )
            if degradation < self.epsilon:
                best_beta = mid
                low = mid
            else:
                high = mid

        self._calibration_summary = {
            "mode": self.calibration_mode,
            "baseline_metric": baseline_val,
            "epsilon": self.epsilon,
            "history": history,
            "best_beta": best_beta,
            "anchor_size": len(calibration_targets),
        }
        return best_beta

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _build_metadata(self) -> dict:
        return {
            "beta": self.beta,
            "calibration": self._calibration_summary,
            "n_samples": self.n_samples,
            "max_seq_len": self.max_seq_len,
            "mode": "classification",
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        dependencies: dict,
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> dict:
        patch_model_for_tfb(model.model, use_softplus=self.use_softplus)

        # Resolve target_labels → target_ids on first call (needs tokenizer)
        if self._target_token_indices is None and self.target_labels:
            self._resolve_target_labels(model.tokenizer)

        if self._target_token_indices is None:
            raise NotImplementedError(
                "TFB is currently supported only in classification mode.  "
                "Set target_ids or target_labels to define class tokens.  "
                "Free-form generation uncertainty via TFB has undefined "
                "semantics and may be supported in a future release."
            )

        # Calibrate beta if needed
        if self.beta is None:
            if not self._is_calibrated:
                self.beta = self._binary_search(model)
                self._is_calibrated = True

        # Result accumulators
        all_texts: list = []
        all_log_probs: list = []
        all_tokens: list = []
        all_target_probs: list = []

        try:
            update_tfb_beta(model.model, self.beta)
            set_tfb_mode(model.model, True)

            for i in range(0, len(texts), self.batch_size):
                chunk_texts = texts[i : i + self.batch_size]
                batch_enc = self._tokenize_batch(model, chunk_texts)
                batch_enc = {k: v.to(model.device()) for k, v in batch_enc.items()}
                chunk_size = len(chunk_texts)

                # [n_samples][chunk_size, n_classes]
                sample_probs_list: list[torch.Tensor] = []

                with torch.no_grad():
                    for _ in range(self.n_samples):
                        logits = model.model(**batch_enc).logits
                        class_probs = self._compute_class_probs(
                            logits, batch_enc["attention_mask"]
                        )
                        sample_probs_list.append(class_probs.detach().cpu())

                # Reshape into per-item results
                for k in range(chunk_size):
                    item_probs = []  # [n_samples, n_classes]
                    item_log_probs = []  # [n_samples, n_classes]
                    item_texts = []  # [n_samples]

                    for s in range(self.n_samples):
                        probs_k = sample_probs_list[s][k]
                        item_probs.append(probs_k.tolist())
                        item_log_probs.append(
                            torch.log(torch.clamp(probs_k, min=1e-12)).tolist()
                        )
                        pred_idx = int(probs_k.argmax().item())
                        pred_token_id = self._class_primary_tokens[pred_idx]
                        item_texts.append(model.tokenizer.decode([pred_token_id]))

                    all_target_probs.append(item_probs)
                    all_log_probs.append(item_log_probs)
                    all_texts.append(item_texts)
                    all_tokens.append([[] for _ in range(self.n_samples)])

                del batch_enc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            set_tfb_mode(model.model, False)

        return {
            f"{self.stats_key}_texts": all_texts,
            f"{self.stats_key}_log_probs": all_log_probs,
            f"{self.stats_key}_tokens": all_tokens,
            f"{self.stats_key}_target_probs": all_target_probs,
            f"{self.stats_key}_metadata": self._build_metadata(),
        }
