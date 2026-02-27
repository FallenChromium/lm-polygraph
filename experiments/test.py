import argparse
import json
import pickle
import platform
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.estimators.tfb import TFBSequenceEstimator
from lm_polygraph.stat_calculators.tfb import TFBStatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.tfb_metrics import tfb_classification_metrics

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
ADAPTER_REPO = "FlyLee/bayesian-peft"
ADAPTER_SUBFOLDER = (
    "blob/meta-llama/Meta-Llama-3.1-8B/obqa/"
    "blob-obqa-sample10-eps0.05-kllr0.0075-beta0.15-seed1"
)
DATASET_NAME = "ai2_arc"
DATASET_SUBSET = "ARC-Easy"


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class TraceWriter:
    def __init__(self, trace_dir: str | None, run_id: str, seed: int):
        self.enabled = bool(trace_dir)
        self.run_id = run_id
        self.seed = seed
        self.trace_dir = (
            (Path(trace_dir).resolve() / run_id) if trace_dir else None
        )
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: dict[str, Any]):
        if not self.enabled:
            return
        path = self.trace_dir / filename
        with path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2, default=_json_default)

    def append_jsonl(self, filename: str, payload: dict[str, Any]):
        if not self.enabled:
            return
        path = self.trace_dir / filename
        with path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, default=_json_default))
            fout.write("\n")

    def event(self, stage: str, event: str, **kwargs):
        record = {
            "ts": time.time(),
            "stage": stage,
            "event": event,
            "run_id": self.run_id,
            "seed": self.seed,
        }
        record.update(kwargs)
        self.append_jsonl("events.jsonl", record)


def setup_oom_snapshot(filename: str = "oom_snapshot_light.pickle"):
    if not torch.cuda.is_available():
        return

    torch.cuda.memory._record_memory_history(max_entries=100000, context="state")

    def _oom_observer(device, alloc, device_alloc, device_free):
        print("!! OOM Detected !! Dumping memory summary stats:")
        print(torch.cuda.memory_summary())
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
            snapshot = torch.cuda.memory._snapshot()
            with open(filename, "wb") as fout:
                pickle.dump(snapshot, fout)
            print(f"Saved {filename}")
        except Exception as exc:
            print(f"Could not save OOM snapshot: {exc}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)


def _normalize_active_adapters(adapter_field) -> list[str]:
    if adapter_field is None:
        return []
    if isinstance(adapter_field, str):
        return [adapter_field]
    if isinstance(adapter_field, (list, tuple, set)):
        return [str(item) for item in adapter_field]
    return [str(adapter_field)]


def _load_blob_checkpoint_manually(model, repo_id: str, subfolder: str) -> dict[str, Any]:
    ckpt_path = hf_hub_download(
        repo_id,
        "adapter_model.safetensors",
        subfolder=subfolder,
    )
    safe_device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_weights = {}
    with safe_open(ckpt_path, framework="pt", device=safe_device) as handle:
        for key in handle.keys():
            ckpt_weights[key] = handle.get_tensor(key)

    print(f"[DEBUG] Checkpoint has {len(ckpt_weights)} keys")

    model_state = dict(model.named_parameters())
    loaded_count = 0
    skipped_keys = []
    shape_mismatches = []

    for ckpt_key, ckpt_tensor in ckpt_weights.items():
        if "lora_A_rho" in ckpt_key or "base_layer" in ckpt_key:
            skipped_keys.append(ckpt_key)
            continue

        remapped = ckpt_key
        if ckpt_key.startswith("base_model.model.base_model.model."):
            remapped = ckpt_key.replace(
                "base_model.model.base_model.model.",
                "base_model.model.",
                1,
            )

        if ".lora_A.weight" in remapped:
            remapped = remapped.replace(".lora_A.weight", ".lora_A.default.weight")
        elif ".lora_B.weight" in remapped:
            remapped = remapped.replace(".lora_B.weight", ".lora_B.default.weight")

        if remapped not in model_state:
            skipped_keys.append(ckpt_key)
            continue

        if model_state[remapped].shape != ckpt_tensor.shape:
            shape_mismatches.append(
                {
                    "checkpoint_key": ckpt_key,
                    "model_key": remapped,
                    "checkpoint_shape": list(ckpt_tensor.shape),
                    "model_shape": list(model_state[remapped].shape),
                }
            )
            continue

        model_state[remapped].data.copy_(ckpt_tensor)
        loaded_count += 1

    print(
        f"[DEBUG] Loaded {loaded_count} weights, "
        f"skipped {len(skipped_keys)} keys, "
        f"shape mismatches: {len(shape_mismatches)}"
    )

    return {
        "checkpoint_path": ckpt_path,
        "checkpoint_key_count": len(ckpt_weights),
        "loaded_count": loaded_count,
        "skipped_keys_count": len(skipped_keys),
        "skipped_keys_head": skipped_keys[:30],
        "shape_mismatches_count": len(shape_mismatches),
        "shape_mismatches_head": shape_mismatches[:30],
    }


def _check_lora_weights_loaded(model) -> dict[str, Any]:
    checks = []
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        adapter_names = _normalize_active_adapters(
            getattr(module, "_active_adapter", None)
        )
        if not adapter_names:
            adapter_names = _normalize_active_adapters(
                getattr(module, "active_adapters", None)
            )

        for adapter_name in adapter_names:
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue
            a_norm = module.lora_A[adapter_name].weight.float().norm().item()
            b_norm = module.lora_B[adapter_name].weight.float().norm().item()
            checks.append(
                {
                    "module": module_name,
                    "adapter": adapter_name,
                    "lora_A_norm": a_norm,
                    "lora_B_norm": b_norm,
                }
            )

    if not checks:
        raise RuntimeError("No active LoRA adapters were validated.")

    zero_layers = [
        c for c in checks if c["lora_A_norm"] < 1e-6 and c["lora_B_norm"] < 1e-6
    ]
    if zero_layers:
        first = zero_layers[0]
        raise RuntimeError(
            "LoRA weights appear zero for module="
            f"{first['module']} adapter={first['adapter']}"
        )

    print(f"  ... checked {len(checks)} LoRA layer-adapter pairs")

    return {
        "checked_pairs": len(checks),
        "adapters": sorted({c["adapter"] for c in checks}),
        "max_lora_A_norm": max(c["lora_A_norm"] for c in checks),
        "max_lora_B_norm": max(c["lora_B_norm"] for c in checks),
        "min_lora_A_norm": min(c["lora_A_norm"] for c in checks),
        "min_lora_B_norm": min(c["lora_B_norm"] for c in checks),
    }


def format_options(example) -> str:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    parts = []
    for txt, lbl in zip(texts, labels):
        # Kept for parity with reference ARC path.
        parts.append(f"{txt}) {lbl}")
    return "\n".join(parts)


def _encode_variant(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if encoded.numel() == 0:
        raise ValueError(f"Variant '{text}' produced no tokens.")
    return [int(encoded[-1].item())]


def build_target_ids(tokenizer, labels: list[str], include_space_variants: bool = True):
    target_ids: list[list[list[int]]] = []
    token_debug = []
    for label in labels:
        variants = [f" {label}"] if include_space_variants else [label]
        class_sequences: list[list[int]] = []
        for variant in variants:
            seq = _encode_variant(tokenizer, variant)
            if seq not in class_sequences:
                class_sequences.append(seq)
        target_ids.append(class_sequences)
        token_debug.append([tokenizer.decode(seq) for seq in class_sequences])
    return target_ids, token_debug


def map_arc_label(answer_key: Any) -> int:
    key = str(answer_key).strip()
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }
    if key not in mapping:
        raise ValueError(
            "Unsupported answerKey encountered: "
            f"'{answer_key}'. Expected one of A-E or 1-5."
        )
    return mapping[key]


def _ece_histogram_with_right_edge(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int,
) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = max(len(confidences), 1)

    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == num_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / total) * abs(acc - conf)

    return float(ece)


def _compute_batch_summary(
    probs_slice: np.ndarray,
    labels_slice: np.ndarray,
    num_bins: int,
) -> dict[str, Any]:
    probs_slice = np.asarray(probs_slice, dtype=np.float64)
    probs_slice = np.nan_to_num(probs_slice, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs_slice.sum(axis=-1, keepdims=True)
    valid = denom > 0
    probs_slice = probs_slice / np.clip(denom, a_min=1e-12, a_max=None)
    if not bool(valid.all()):
        probs_slice[~valid[..., 0]] = 1.0 / probs_slice.shape[-1]

    mean_probs = probs_slice.mean(axis=1)
    preds = mean_probs.argmax(axis=1)
    conf = mean_probs.max(axis=1)
    correctness = (preds == labels_slice).astype(np.float32)
    label_probs = np.clip(mean_probs[np.arange(len(labels_slice)), labels_slice], 1e-12, 1.0)

    per_item_flip = []
    for item_probs in probs_slice:
        sampled_pred = item_probs.argmax(axis=1)
        ref = sampled_pred[0]
        per_item_flip.append(float((sampled_pred != ref).mean()))

    num_classes = mean_probs.shape[1]
    class_dist = np.bincount(preds, minlength=num_classes).tolist()

    return {
        "n_items": int(len(labels_slice)),
        "acc": float(correctness.mean()) if len(correctness) else 0.0,
        "nll": float(-np.log(label_probs).mean()) if len(label_probs) else 0.0,
        "ece": _ece_histogram_with_right_edge(conf, correctness, num_bins=num_bins),
        "brier": float(
            (
                (mean_probs - np.eye(num_classes, dtype=np.float64)[labels_slice]) ** 2
            ).sum(axis=1).mean()
        )
        if len(labels_slice)
        else 0.0,
        "flip_ratio": float(np.mean(per_item_flip)) if per_item_flip else 0.0,
        "class_dist": class_dist,
        "confidence_stats": {
            "min": float(conf.min()) if len(conf) else 0.0,
            "mean": float(conf.mean()) if len(conf) else 0.0,
            "max": float(conf.max()) if len(conf) else 0.0,
        },
    }


def _git_head() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TFB ARC-Easy parity harness with structured traces."
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--adapter-subfolder", default=ADAPTER_SUBFOLDER)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-subset", default=DATASET_SUBSET)

    parser.add_argument("--anchor-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--th", type=float, default=0.003)
    parser.add_argument("--max-seq-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-bins", type=int, default=15)

    parser.add_argument("--beta-min", type=float, default=0.001)
    parser.add_argument("--beta-max", type=float, default=0.2)
    parser.add_argument("--calibration-mode", choices=["seq_nll", "exact_match"], default="seq_nll")

    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj", "lm_head"],
        help="LoRA target modules for parity with --apply-classhead-lora.",
    )
    parser.add_argument(
        "--use-softplus",
        action="store_true",
        help="Enable softplus parameterization. Default is disabled for parity.",
    )
    parser.add_argument(
        "--no-space-target-variants",
        action="store_true",
        help="Disable leading-space class-token variants.",
    )
    parser.add_argument("--tfb-key", default="tfb_arc_repro")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument(
        "--trace-save-probs",
        action="store_true",
        help="Persist full mean/per-sample probability tensors for deep debugging.",
    )
    parser.add_argument("--enable-oom-snapshot", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = f"tfb-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    trace = TraceWriter(args.trace_dir, run_id=run_id, seed=args.seed)

    if args.enable_oom_snapshot:
        setup_oom_snapshot()

    print(f"Loading Base Model: {args.model_name}")
    trace.event("setup", "start", params=vars(args))

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(base_model, peft_config)

    trace.event(
        "model",
        "peft_initialized",
        checks={
            "target_modules": list(args.lora_target_modules),
            "include_lm_head": "lm_head" in args.lora_target_modules,
        },
    )

    print("Loading BLoB checkpoint manually...")
    load_report = _load_blob_checkpoint_manually(
        model,
        args.adapter_repo,
        args.adapter_subfolder,
    )
    adapter_report = _check_lora_weights_loaded(model)
    trace.event(
        "model",
        "adapter_loaded",
        checks={
            "checkpoint": load_report,
            "adapter_coverage": adapter_report,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.add_eos_token = False
    tokenizer.pad_token = tokenizer.bos_token

    wb_model = WhiteboxModel(model=model, tokenizer=tokenizer, model_type="CausalLM")

    print(f"Loading Dataset: {args.dataset_name}/{args.dataset_subset}")
    dataset_full = load_dataset(args.dataset_name, args.dataset_subset)
    anchor_ds = (
        dataset_full["train"]
        .shuffle(seed=args.seed)
        .select(range(min(args.anchor_size, len(dataset_full["train"]))))
    )
    eval_ds = dataset_full["validation"]

    preamble = """Answer the science question by choosing the correct option letter.

Question: {question}
Options:
{options}
Answer:"""

    anchor_inputs = [
        preamble.format(question=example["question"], options=format_options(example))
        for example in anchor_ds
    ]
    eval_inputs = [
        preamble.format(question=example["question"], options=format_options(example))
        for example in eval_ds
    ]

    labels = ["A", "B", "C", "D", "E"]
    target_ids, debug_tokens = build_target_ids(
        tokenizer,
        labels,
        include_space_variants=not args.no_space_target_variants,
    )
    print(f"[DEBUG] target_ids (per class): {debug_tokens}")

    eval_labels = [map_arc_label(example["answerKey"]) for example in eval_ds]
    print(f"[DEBUG] First 10 eval labels: {eval_labels[:10]}")

    trace.write_json(
        "run_meta.json",
        {
            "run_id": run_id,
            "params": vars(args),
            "trace_output_dir": str(trace.trace_dir) if trace.enabled else None,
            "model": {
                "name": args.model_name,
                "dtype": str(next(model.parameters()).dtype),
                "device_count": torch.cuda.device_count(),
            },
            "tokenizer": {
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
            },
            "dataset": {
                "name": args.dataset_name,
                "subset": args.dataset_subset,
                "anchor_size": len(anchor_inputs),
                "eval_size": len(eval_inputs),
                "drop_last_note": (
                    "Reference loaders may use drop_last=True; this harness evaluates "
                    "the full validation split unless explicitly changed."
                ),
            },
            "checks": {
                "target_token_map": debug_tokens,
                "checkpoint": load_report,
                "adapter_coverage": adapter_report,
            },
            "env": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "git_head": _git_head(),
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
            },
        },
    )

    if args.iter > 0:
        beta_for_run = None
        beta_search_steps = args.iter
    else:
        beta_for_run = args.beta
        beta_search_steps = 1

    calc = TFBStatCalculator(
        stats_key=args.tfb_key,
        anchor_inputs=anchor_inputs,
        target_ids=target_ids,
        calibration_mode=args.calibration_mode,
        batch_size=args.batch_size,
        target_epsilon=args.th,
        n_samples=args.n_samples,
        use_softplus=args.use_softplus,
        beta=beta_for_run,
        beta_range=(args.beta_min, args.beta_max),
        beta_search_steps=beta_search_steps,
        max_seq_len=args.max_seq_len,
    )

    trace.event(
        "tfb",
        "calculator_initialized",
        params={
            "beta": beta_for_run,
            "iter": args.iter,
            "use_softplus": args.use_softplus,
            "max_seq_len": args.max_seq_len,
            "n_samples": args.n_samples,
        },
    )

    print("--- Running Inference ---")
    stats = calc(dependencies={}, texts=eval_inputs, model=wb_model, max_new_tokens=1)

    metadata = stats.get(f"{args.tfb_key}_metadata", {})
    print(f"[DEBUG] TFB metadata: {metadata}")
    trace.event("tfb", "inference_complete", beta=calc.beta, metadata=metadata)

    if isinstance(metadata, dict):
        calibration = metadata.get("calibration", {})
        for idx, item in enumerate(calibration.get("history", []) or []):
            record = {
                "ts": time.time(),
                "run_id": run_id,
                "seed": args.seed,
                "step": idx,
            }
            record.update(item)
            trace.append_jsonl("calibration_history.jsonl", record)

    metrics = tfb_classification_metrics(
        stats,
        args.tfb_key,
        eval_labels,
        num_bins=args.num_bins,
        expected_classes=len(labels),
    )

    print("\n========================================")
    print(f"  Classification NLL: {metrics['nll']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print("========================================")

    probs_key = f"{args.tfb_key}_target_probs"
    probs_array = np.asarray(stats[probs_key], dtype=np.float64)
    labels_array = np.asarray(eval_labels, dtype=np.int64)

    for start in range(0, len(labels_array), args.batch_size):
        end = min(start + args.batch_size, len(labels_array))
        batch_summary = _compute_batch_summary(
            probs_array[start:end],
            labels_array[start:end],
            num_bins=args.num_bins,
        )
        batch_summary.update(
            {
                "run_id": run_id,
                "seed": args.seed,
                "batch_idx": start // args.batch_size,
                "stage": "inference",
            }
        )
        trace.append_jsonl("batch_summary.jsonl", batch_summary)

    final_metrics = {
        "run_id": run_id,
        "seed": args.seed,
        "metrics": metrics,
        "evaluated_count": int(len(eval_labels)),
        "beta": calc.beta,
        "tfb_metadata": metadata,
    }
    trace.write_json("final_metrics.json", final_metrics)
    trace.event(
        "metrics",
        "final",
        metrics=final_metrics["metrics"],
        evaluated_count=final_metrics["evaluated_count"],
        beta=final_metrics["beta"],
    )
    if trace.enabled and args.trace_save_probs:
        np.savez_compressed(
            trace.trace_dir / "prob_tensors.npz",
            mean_probs=np.asarray(metrics["mean_probs"], dtype=np.float32),
            per_sample_probs=np.asarray(metrics["per_sample_probs"], dtype=np.float32),
        )

    seq_est = TFBSequenceEstimator(stats_key=args.tfb_key)
    uncertainties = seq_est(stats)

    print("\n--- Sample Outputs ---")
    for i in range(min(5, len(eval_inputs))):
        print(f"\nSample {i}:")
        print(f"  Question: {eval_ds[i]['question'][:60]}...")
        print(f"  True Answer: {eval_ds[i]['answerKey']}")
        print(f"  Generations: {stats[f'{args.tfb_key}_texts'][i][:3]}")
        print(f"  Semantic Entropy: {uncertainties[i]:.4f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
