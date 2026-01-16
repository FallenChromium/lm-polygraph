import hashlib
import json
from typing import Literal

import numpy as np

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.stat_calculators.tfb import TFBStatCalculator


class TFBEstimator(Estimator):
    @property
    def epsilon(self) -> float:
        return self._config["target_nll_epsilon"]

    @property
    def n_samples(self) -> int:
        return self._config["n_samples"]

    @property
    def metric(self) -> str:
        return self._config["metric"]

    """
    Training-Free Bayesianization (TFB) for LoRA-finetuned models.

    TFB converts pre-trained LoRA weights into Bayesian posteriors via SVD-based
    variance inference, enabling stochastic sampling without additional training.

    Reference:
        Shi et al. "Training-Free Bayesianization for Low-Rank Adapters of Large
        Language Models" (arXiv:2412.05723)
    """

    def __init__(
        self,
        anchor_texts: list[str],
        anchor_targets: list[str],
        n_samples: int = 10,
        target_epsilon: float = 0.003,
        beta: float | None = None,
        beta_search_range: tuple[float, float] = (0.001, 0.1),
        beta_search_steps: int = 10,
        save_dir: str | None = None,
    ):
        self._config = {
            "anchor_inputs": anchor_texts,
            "anchor_targets": anchor_targets,
            "n_samples": n_samples,
            "target_epsilon": target_epsilon,
            "save_dir": save_dir,
        }

        # 2. Compute Hash based on values that affect calibration/sampling
        hash_payload = {
            "n": n_samples,
            "eps": target_epsilon,
            "beta": beta,
            "beta_range": beta_search_range,
            "beta_steps": beta_search_steps,
            "anch_len": len(anchor_texts),
            "anch_sample": anchor_texts[0][:20] if anchor_texts else "",
        }
        config_str = json.dumps(hash_payload, sort_keys=True)
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # 3. Define the dependency key
        self._target_stat = f"tfb_samples_{self._config_hash}"

        # Initialize parent
        super().__init__([self._target_stat], "sequence")

        # Cache for the calculator factory
        self._cached_calc: TFBStatCalculator | None = None

        # --- Read-Only Properties to Prevent accidental mutation ---

    def create_stat_calculator(self) -> TFBStatCalculator:
        if (
            self._cached_calc is None
            or self._cached_calc.config_hash != self._config_hash
        ):
            # Pass the private config + the hash
            self._cached_calc = TFBStatCalculator(
                config_hash=self._config_hash, **self._config
            )
        return self._cached_calc

    def __str__(self):
        return f"TFB_Uncertainty({', '.join([k + '=' + v for k, v in self._config.items()])})"

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
