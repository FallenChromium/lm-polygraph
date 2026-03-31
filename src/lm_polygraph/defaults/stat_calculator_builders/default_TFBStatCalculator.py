from lm_polygraph.stat_calculators.tfb import TFBStatCalculator


def load_stat_calculator(config, builder):
    """
    Instantiate a TFBStatCalculator from a YAML config node.

    The builder environment is accepted for API parity with other factory
    functions but is not used by TFB.
    """

    def _get(attr, default=None):
        return getattr(config, attr, default)

    beta_range = tuple(_get("beta_range", (0.001, 0.2)))

    # target_ids (raw token id groups) and target_labels (string labels
    # resolved at runtime) are both accepted.  At least one must be set.
    target_ids = _get("target_ids")
    target_labels = _get("target_labels")

    return TFBStatCalculator(
        stats_key=_get("stats_key", "tfb"),
        anchor_inputs=_get("anchor_inputs"),
        anchor_targets=_get("anchor_targets"),
        target_ids=target_ids,
        target_labels=target_labels,
        n_samples=_get("n_samples", 10),
        target_epsilon=_get("target_epsilon", 0.003),
        beta=_get("beta"),
        beta_range=beta_range,
        beta_search_steps=_get("beta_search_steps", 10),
        use_softplus=_get("use_softplus", False),
        batch_size=_get("batch_size", 8),
        max_seq_len=_get("max_seq_len"),
        calibration_mode=_get("calibration_mode", "seq_nll"),
    )
