"""
Training-Free Bayesianization (TFB) for LoRA-finetuned models.

TFB converts pre-trained LoRA weights into Bayesian posteriors via SVD-based
variance inference, enabling stochastic sampling without additional training.

Reference:
    Shi et al. "Training-Free Bayesianization for Low-Rank Adapters of Large
    Language Models" (arXiv:2412.05723)
"""

import torch
import torch.nn as nn
from typing import List, Callable, Optional, Tuple


def _extract_lora_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Extract all LoRA layers from a PEFT model.
    
    Args:
        model: A PEFT model with LoRA adapters
        
    Returns:
        List of (name, module) tuples for modules with lora_A and lora_B
    """
    return [
        (name, module) for name, module in model.named_modules()
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')
    ]


def _compute_rho_from_d(
    D: torch.Tensor,
    in_features: int,
    beta: float,
    use_softplus: bool = False,
) -> torch.Tensor:
    """
    Compute variance parameter (rho) from singular values D.
    """
    # Infer std: σ = β / (D + ε), broadcast to [r, in_features]
    lora_std = beta / (D.reshape(-1, 1).expand(-1, in_features) + 1e-6)
    
    # Convert std to rho (inverse transform)
    if use_softplus:
        # softplus inverse: rho = log(exp(σ) - 1)
        rho = torch.log(torch.exp(lora_std) - 1)
    else:
        # squared inverse: rho = sqrt(σ)
        rho = torch.sqrt(lora_std)
    
    return rho


def _compute_variance_from_svd(
    lora_B_weight: torch.Tensor,
    in_features: int,
    beta: float,
    use_softplus: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute variance parameter (rho) and SVD decomposition of LoRA-B.
    """
    # SVD decomposition: B = U @ diag(D) @ V^T
    U, D, V = torch.linalg.svd(lora_B_weight, full_matrices=False)
    
    rho = _compute_rho_from_d(D, in_features, beta, use_softplus)
    
    return rho, U, D, V


def _tfb_forward_factory(use_softplus: bool = False):
    """
    Factory that creates the TFB forward function with noise injection.
    
    This implements the Flipout-style efficient sampling from the paper:
    noise = ((x * r_A) @ noise_A^T) * s_A) @ B^T
    
    where r_A, s_A are Rademacher random signs.
    """
    
    def tfb_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Store original dtype for output
        previous_dtype = x.dtype
        
        # Handle disabled/merged states
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Standard LoRA forward pass
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x_casted = x.to(lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x_casted))) * scaling
        
        # Add TFB stochastic noise if sampling is enabled
        if getattr(self, 'tfb_sampling_enabled', False):
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                if active_adapter not in getattr(self, 'lora_A_rho', {}):
                    continue
                
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                scaling = self.scaling[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                
                # Convert rho → sigma
                rho = self.lora_A_rho[active_adapter]
                if use_softplus:
                    A_sigma = torch.log1p(torch.exp(rho))
                else:
                    A_sigma = rho ** 2
                
                x_casted = x.to(lora_A.weight.dtype)
                x_dropped = dropout(x_casted)
                
                # Rademacher random signs for efficient Flipout sampling
                if x_dropped.dim() == 2:  # [batch, features]
                    r_A = torch.empty(
                        (x_dropped.size(0), self.in_features),
                        device=x.device, dtype=x_casted.dtype
                    ).uniform_(-1, 1).sign()
                    s_A = torch.empty(
                        (x_dropped.size(0), self.r[active_adapter]),
                        device=x.device, dtype=x_casted.dtype
                    ).uniform_(-1, 1).sign()
                else:  # [batch, seq_len, features]
                    r_A = torch.empty(
                        (x_dropped.size(0), x_dropped.size(1), self.in_features),
                        device=x.device, dtype=x_casted.dtype
                    ).uniform_(-1, 1).sign()
                    s_A = torch.empty(
                        (x_dropped.size(0), x_dropped.size(1), self.r[active_adapter]),
                        device=x.device, dtype=x_casted.dtype
                    ).uniform_(-1, 1).sign()
                
                # Sample noise for LoRA-A weights: noise_A ~ N(0, σ²)
                lora_noise_a = A_sigma * torch.randn_like(lora_A.weight)
                
                # Compute noise contribution using Flipout:
                # noise = ((x * r_A) @ noise_A^T) * s_A) @ B^T
                noise = (
                    ((x_dropped * r_A) @ lora_noise_a.T) * s_A
                ) @ lora_B.weight.T
                
                result = result + noise * scaling
        
        return result.to(previous_dtype)
    
    return tfb_forward


def apply_tfb(
    model: nn.Module,
    beta: float = 0.2,
    use_softplus: bool = False,
) -> List[nn.Module]:
    """
    Apply Training-Free Bayesianization to a LoRA-finetuned model.
    
    This modifies the model in-place by:
    1. Performing SVD on each LoRA-B weight matrix
    2. Inferring variance parameters for LoRA-A
    3. Replacing forward methods with stochastic versions
    
    Args:
        model: A PEFT model with LoRA adapters
        beta: Variance scaling parameter. Higher = more noise. Default: 0.2
        use_softplus: If True, use softplus for variance parameterization.
                      If False (default), use squared parameterization.
                      
    Returns:
        List of modified LoRA layers
        
    Raises:
        ValueError: If model is not a PEFT model with LoRA adapters
        
    Example:
        >>> from peft import PeftModel
        >>> model = PeftModel.from_pretrained(base_model, lora_path)
        >>> lora_layers = apply_tfb(model, beta=0.2)
        >>> enable_tfb_sampling(model)  # Enable stochastic forward
    """
    lora_layer_items = _extract_lora_layers(model)
    
    if len(lora_layer_items) == 0:
        raise ValueError(
            "No LoRA layers found in model. Ensure you're using a PEFT model "
            "with LoRA adapters loaded via PeftModel.from_pretrained()."
        )
    
    tfb_forward = _tfb_forward_factory(use_softplus)
    modified_layers = []
    
    for name, layer in lora_layer_items:
        layer.lora_A_rho = nn.ParameterDict({})
        layer.tfb_sampling_enabled = False
        layer.tfb_use_softplus = use_softplus
        layer.tfb_beta = beta
        
        layer.tfb_singular_values = nn.ParameterDict({})
        
        for adapter_name in layer.lora_A.keys():
            lora_A = layer.lora_A[adapter_name]
            lora_B = layer.lora_B[adapter_name]
            
            dtype_A = lora_A.weight.dtype
            dtype_B = lora_B.weight.dtype
            
            A_weight = lora_A.weight.float()
            B_weight = lora_B.weight.float()
            
            rho, U, D, V = _compute_variance_from_svd(
                B_weight, layer.in_features, beta, use_softplus
            )
            
            layer.lora_A_rho[adapter_name] = nn.Parameter(rho.to(dtype_A))
            
            # Cache singular values for fast re-calibration
            layer.tfb_singular_values[adapter_name] = nn.Parameter(D.to(dtype_B), requires_grad=False)
            
            # Transform: B' = U @ diag(D), A' = V^T @ A
            # Note: _compute_variance_from_svd uses linalg.svd which returns Vh (V^T).
            # So V variable holds V^T. We want V^T @ A.
            lora_B.weight = nn.Parameter((U @ torch.diag(D)).to(dtype_B))
            lora_A.weight = nn.Parameter((V @ A_weight).to(dtype_A))
        
        layer._tfb_original_forward = layer.forward
        layer.forward = tfb_forward.__get__(layer, type(layer))
        modified_layers.append(layer)
    
    return modified_layers


def enable_tfb_sampling(model: nn.Module) -> None:
    """
    Enable TFB stochastic sampling for all LoRA layers.
    
    When enabled, each forward pass will sample different weights,
    producing stochastic outputs useful for uncertainty estimation.
    
    Args:
        model: Model with TFB applied via apply_tfb()
    """
    for module in model.modules():
        if hasattr(module, 'tfb_sampling_enabled'):
            module.tfb_sampling_enabled = True


def disable_tfb_sampling(model: nn.Module) -> None:
    """
    Disable TFB stochastic sampling, returning to deterministic forward.
    
    Args:
        model: Model with TFB applied via apply_tfb()
    """
    for module in model.modules():
        if hasattr(module, 'tfb_sampling_enabled'):
            module.tfb_sampling_enabled = False


def update_tfb_beta(
    model: nn.Module,
    beta: float,
) -> None:
    """
    Update the beta (variance scale) parameter for all TFB layers.
    
    This recomputes the variance parameters using SVD with the new beta.
    Useful during calibration to find optimal beta.
    
    Args:
        model: Model with TFB applied
        beta: New variance scaling parameter
    """
    for module in model.modules():
        if not hasattr(module, 'lora_A_rho'):
            continue
        
        use_softplus = getattr(module, 'tfb_use_softplus', False)
        module.tfb_beta = beta
        
        for adapter_name in module._active_adapter:
            # Use cached singular values if available
            if hasattr(module, 'tfb_singular_values') and adapter_name in module.tfb_singular_values:
                D = module.tfb_singular_values[adapter_name].float()
                rho = _compute_rho_from_d(D, module.in_features, beta, use_softplus)
            else:
                # Fallback to SVD (slower)
                lora_B = module.lora_B[adapter_name].weight.float()
                rho, _, _, _ = _compute_variance_from_svd(
                    lora_B, module.in_features, beta, use_softplus
                )
            
            dtype = module.lora_A_rho[adapter_name].dtype
            module.lora_A_rho[adapter_name] = nn.Parameter(rho.to(dtype))


# Type alias for calibration metric function
# Takes (model, inputs, n_samples) -> scalar loss
CalibrationMetricFn = Callable[[nn.Module, dict, int], torch.Tensor]


def _default_nll_metric(
    model: nn.Module,
    inputs: dict,
    n_samples: int,
    parallel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Default NLL-based calibration metric.
    
    Args:
        model: Model with TFB applied
        inputs: Tokenized inputs dict
        n_samples: Number of stochastic samples
        parallel: If True, process samples in a single batch (VRAM heavy)
        
    Returns:
        Tuple of (nll_loss, deterministic_probs)
    """
    # Get deterministic prediction
    disable_tfb_sampling(model)
    with torch.no_grad():
        det_output = model(**inputs)
        det_logits = det_output.logits[:, -1, :]  # Last token logits
        det_probs = torch.softmax(det_logits, dim=-1)
        det_preds = det_logits.argmax(dim=-1)
    
    # Get stochastic predictions
    enable_tfb_sampling(model)
    all_probs = []
    
    with torch.no_grad():
        if parallel and n_samples > 1:
            # Parallel: repeat inputs and run once
            batch_inputs = {}
            for k, v in inputs.items():
                # [batch, seq] -> [batch * n, seq]
                v_repeated = v.repeat_interleave(n_samples, dim=0)
                batch_inputs[k] = v_repeated
                
            output = model(**batch_inputs)
            logits = output.logits[:, -1, :]  # [batch * n, vocab]
            probs = torch.softmax(logits, dim=-1)
            
            # Reshape stats: [batch * n, vocab] -> [n, batch, vocab]
            # Since repeat_interleave groups samples: b1s1, b1s2... b2s1, b2s2
            # We need to be careful with reshaping
            batch_size = inputs['input_ids'].size(0)
            probs = probs.view(batch_size, n_samples, -1).transpose(0, 1) # [n, batch, vocab]
            
            for i in range(n_samples):
                all_probs.append(probs[i])
        else:
            # Sequential: run n times
            for _ in range(n_samples):
                output = model(**inputs)
                logits = output.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs)
    
    # Average probabilities across samples
    mean_probs = torch.stack(all_probs).mean(dim=0)
    
    # NLL of deterministic prediction under stochastic model
    nll = -torch.log(mean_probs.gather(1, det_preds.unsqueeze(1)) + 1e-12).mean()
    
    return nll, det_probs


def fit_tfb_beta(
    model: nn.Module,
    calibration_inputs: List[dict],
    target_metric_ratio: float = 0.01,
    max_iters: int = 10,
    n_samples: int = 5,
    initial_beta: float = 0.2,
    metric_fn: Optional[CalibrationMetricFn] = None,
    verbose: bool = False,
    parallel: bool = True,
) -> float:
    """
    Find optimal beta via binary search on calibration data.
    
    Searches for the largest beta such that the metric degradation
    (compared to deterministic) stays below target_metric_ratio.
    
    Args:
        model: Model with TFB applied
        calibration_inputs: List of tokenized input dicts for calibration.
                           Each dict should have 'input_ids' and 'attention_mask'.
        target_metric_ratio: Target ratio of metric change. Default 0.01 (1%).
        max_iters: Maximum binary search iterations. Default 10.
        n_samples: Samples per forward pass during calibration. Default 5.
        initial_beta: Starting beta value (upper bound). Default 0.2.
        metric_fn: Optional custom metric function. Should take 
                   (model, inputs, n_samples, parallel) and return (metric_val, baseline_info).
                   Default uses NLL-based metric.
        verbose: If True, print progress during search.
        parallel: If True, use batching for calibration forward passes (faster).
        
    Returns:
        Optimal beta value
        
    Example:
        >>> # Prepare calibration data
        >>> cal_texts = ["What is 2+2?", "Who wrote Hamlet?"]
        >>> cal_inputs = [tokenizer(t, return_tensors='pt').to(device) for t in cal_texts]
        >>> 
        >>> # Find optimal beta
        >>> optimal_beta = fit_tfb_beta(model, cal_inputs, target_metric_ratio=0.01)
    """
    if metric_fn is None:
        metric_fn = _default_nll_metric
    
    low, high = 0.0, initial_beta
    best_beta = low
    
    # Compute baseline metric with zero noise
    update_tfb_beta(model, 0.0)
    disable_tfb_sampling(model)
    
    baseline_metrics = []
    
    with torch.no_grad():
        for inputs in calibration_inputs:
            metric_val, det_probs = metric_fn(model, inputs, n_samples, parallel)
            baseline_metrics.append(metric_val)
    
    baseline_metric = torch.stack(baseline_metrics).mean().item()
    
    if verbose:
        print(f"Baseline metric: {baseline_metric:.6f}")
    
    # Binary search
    # We want to find max beta where the metric does not deviate significantly from baseline.
    for iteration in range(max_iters):
        mid = (low + high) / 2
        update_tfb_beta(model, mid)
        enable_tfb_sampling(model)
        
        current_metrics = []
        with torch.no_grad():
            for inputs in calibration_inputs:
                metric_val, _ = metric_fn(model, inputs, n_samples, parallel)
                current_metrics.append(metric_val)
        
        current_metric = torch.stack(current_metrics).mean().item()
        
        # Determine if we exceeded the threshold
        if baseline_metric == 0:
             metric_ratio = current_metric
        else:
             metric_ratio = abs(current_metric - baseline_metric) / (baseline_metric + 1e-12)

        if verbose:
            print(f"Iter {iteration}: beta={mid:.6f}, metric={current_metric:.6f}, "
                  f"ratio={metric_ratio:.6f}")
        
        # Adjust search range
        if metric_ratio > target_metric_ratio:
            # Metric degradation is too high; need lower beta
            high = mid
        else:
            # Metric degradation is acceptable; try higher beta
            best_beta = mid
            low = mid
    
    # Set final beta
    update_tfb_beta(model, best_beta)
    
    if verbose:
        print(f"Optimal beta: {best_beta:.6f}")
    
    return best_beta


def compute_flip_ratio(
    model: nn.Module,
    inputs: dict,
    n_samples: int = 10,
) -> float:
    """
    Compute the flip ratio: fraction of samples where prediction differs from deterministic.
    
    Useful for monitoring calibration quality.
    
    Args:
        model: Model with TFB applied
        inputs: Tokenized inputs
        n_samples: Number of stochastic samples
        
    Returns:
        Flip ratio in [0, 1]
    """
    # Get deterministic prediction
    disable_tfb_sampling(model)
    with torch.no_grad():
        det_output = model(**inputs)
        det_pred = det_output.logits[:, -1, :].argmax(dim=-1)
    
    # Get stochastic predictions
    enable_tfb_sampling(model)
    flip_count = 0
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(**inputs)
            stoch_pred = output.logits[:, -1, :].argmax(dim=-1)
            flip_count += (stoch_pred != det_pred).sum().item()
    
    return flip_count / (n_samples * det_pred.numel())
