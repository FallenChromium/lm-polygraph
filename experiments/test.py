import pickle
import warnings

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lm_polygraph.estimators.tfb import TFBSequenceEstimator, TFBTokenEstimator
from lm_polygraph.stat_calculators.tfb import TFBStatCalculator

# LM-Polygraph Imports
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.tfb_metrics import tfb_classification_metrics

# ==========================================
# 1. Precise Loading Helpers (from benchmark_tfb.py)
# ==========================================

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
ADAPTER_REPO = "FlyLee/bayesian-peft"
ADAPTER_SUBFOLDER = "blob/meta-llama/Meta-Llama-3.1-8B/obqa/blob-obqa-sample10-eps0.05-kllr0.0075-beta0.15-seed1"
DATASET_NAME = "ai2_arc"
DATASET_SUBSET = "ARC-Easy"
ANCHOR_SIZE = 150


def setup_oom_snapshot(filename="oom_snapshot.pickle"):
    # 1. Enable memory history (required for the snapshot to be useful)
    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        context="state",  # Change this from 'all' (default) to 'state'
    )

    def _oom_observer(device, alloc, device_alloc, device_free):
        print("!! OOM Detected !! Dumping memory summary stats:")
        # This will NOT crash because it doesn't use the stack unwinder
        print(torch.cuda.memory_summary())

        # Optional: Try to dump a snapshot WITHOUT history (no stack traces)
        # This might still work for seeing fragmentation segments
        try:
            torch.cuda.memory._record_memory_history(
                enabled=None
            )  # Stop recording first
            print("Attempting 'light' snapshot...")
            snapshot = torch.cuda.memory._snapshot()
            with open("oom_snapshot_light.pickle", "wb") as f:
                pickle.dump(snapshot, f)
            print("Saved oom_snapshot_light.pickle (no stack traces)")
        except Exception as e:
            print(f"Could not save light snapshot: {e}")

    # 3. Register the observer
    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)


def _load_blob_checkpoint_manually(model, repo_id, subfolder):
    ckpt_path = hf_hub_download(
        repo_id, "adapter_model.safetensors", subfolder=subfolder
    )

    ckpt_weights = {}
    with safe_open(ckpt_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            ckpt_weights[key] = f.get_tensor(key)

    print(f"[DEBUG] Checkpoint has {len(ckpt_weights)} keys")

    model_state = dict(model.named_parameters())
    loaded_count = 0
    skipped_keys = []

    for ckpt_key, ckpt_tensor in ckpt_weights.items():
        if "lora_A_rho" in ckpt_key or "base_layer" in ckpt_key:
            skipped_keys.append(ckpt_key)
            continue

        if ckpt_key.startswith("base_model.model.base_model.model."):
            remapped = ckpt_key.replace(
                "base_model.model.base_model.model.", "base_model.model.", 1
            )
        else:
            remapped = ckpt_key

        if ".lora_A.weight" in remapped:
            remapped = remapped.replace(".lora_A.weight", ".lora_A.default.weight")
        elif ".lora_B.weight" in remapped:
            remapped = remapped.replace(".lora_B.weight", ".lora_B.default.weight")

        if remapped in model_state:
            if model_state[remapped].shape == ckpt_tensor.shape:
                model_state[remapped].data.copy_(ckpt_tensor)
                loaded_count += 1
            else:
                print(f"[WARN] Shape mismatch: {remapped}")
        else:
            skipped_keys.append(ckpt_key)

    print(f"[DEBUG] Loaded {loaded_count} weights, skipped {len(skipped_keys)} keys")
    return loaded_count


def _check_lora_weights_loaded(model):
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name in getattr(module, "_active_adapter", []):
                if adapter_name in module.lora_A:
                    A_norm = module.lora_A[adapter_name].weight.float().norm().item()
                    B_norm = module.lora_B[adapter_name].weight.float().norm().item()
                    if A_norm < 1e-6 and B_norm < 1e-6:
                        raise RuntimeError(f"LoRA weights at {name} are zero!")
                    count += 1
    print(f"  ... checked {count} LoRA layers, all have non-zero weights")


# ==========================================
# 2. Model Loading
# ==========================================
setup_oom_snapshot()
print(f"Loading Base Model: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(base_model, peft_config)

print(f"Loading BLoB Checkpoint manually...")
_load_blob_checkpoint_manually(model, ADAPTER_REPO, ADAPTER_SUBFOLDER)
_check_lora_weights_loaded(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.add_eos_token = False
tokenizer.pad_token = tokenizer.bos_token

# FIXED: Removed 'tokenizer_args' which caused the TypeError
wb_model = WhiteboxModel(model=model, tokenizer=tokenizer, model_type="CausalLM")

# ==========================================
# 3. Data Setup
# ==========================================
print(f"Loading Dataset: {DATASET_NAME}/{DATASET_SUBSET}")
dataset_full = load_dataset(DATASET_NAME, DATASET_SUBSET)
anchor_ds = (
    dataset_full["train"]
    .shuffle(seed=42)
    .select(range(min(ANCHOR_SIZE, len(dataset_full["train"]))))
)
eval_ds = dataset_full["validation"].select(range(50))

preamble = """Answer the science question by choosing the correct option letter.

Question: {question}
Options:
{options}
Answer:"""


def format_options(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    parts = []
    for txt, lbl in zip(texts, labels):
        parts.append(f"{txt}) {lbl}")
    return "\n".join(parts)


anchor_inputs = [
    preamble.format(question=e["question"], options=format_options(e))
    for e in anchor_ds
]
eval_inputs = [
    preamble.format(question=e["question"], options=format_options(e)) for e in eval_ds
]

# ==========================================
# 4. Prepare target_ids for classification
# ==========================================
labels = ["A", "B", "C", "D", "E"]
label_variants = [[lbl, f" {lbl}"] for lbl in labels]


def _encode_variant(text: str) -> list[int]:
    encoded = (
        tokenizer(text, add_special_tokens=False, return_tensors="pt")
        .input_ids[0]
        .tolist()
    )
    if not encoded:
        raise ValueError(f"Variant '{text}' produced no tokens.")
    return [encoded[-1]]


target_ids = []
for variants in label_variants:
    class_sequences: list[list[int]] = []
    for variant in variants:
        seq = _encode_variant(variant)
        if seq not in class_sequences:
            class_sequences.append(seq)
    target_ids.append(class_sequences)

debug_tokens = [
    [tokenizer.decode(seq) for seq in class_sequences] for class_sequences in target_ids
]
print(f"[DEBUG] target_ids (per class): {debug_tokens}")

# Ground truth labels for evaluation
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
eval_labels = [label_map.get(e["answerKey"], 0) for e in eval_ds]
print(f"[DEBUG] First 10 eval labels: {eval_labels[:10]}")

# ==========================================
# 5. Run TFB Pipeline (Classification Mode)
# ==========================================
tfb_key = "tfb_arc_repro"

print("\n--- Initializing TFB Calculator ---")
calc = TFBStatCalculator(
    stats_key=tfb_key,
    anchor_inputs=anchor_inputs,
    target_ids=target_ids,  # Enable classification mode
    calibration_mode="seq_nll",
    batch_size=1,
    target_epsilon=0.003,
    n_samples=10,
    use_softplus=True,
    beta=None,
)

print("--- Running Inference ---")
# For classification, we don't generate - we just get logits at last position
# Set max_new_tokens=1 (minimal generation, we only care about first token logits)
stats = calc(dependencies={}, texts=eval_inputs, model=wb_model, max_new_tokens=1)

print("\n--- Results ---")
print(f"Optimal Beta Found: {calc.beta}")

# ==========================================
# 6. Compute Classification NLL (matching benchmark)
# ==========================================
import numpy as np

metadata = stats.get(f"{tfb_key}_metadata", {})
print(f"[DEBUG] TFB metadata: {metadata}")

metrics = tfb_classification_metrics(
    stats,
    tfb_key,
    eval_labels,
    num_bins=15,
    expected_classes=len(labels),
)
classification_nll = metrics["nll"]
print(f"\n========================================")
print(f"  Classification NLL: {classification_nll:.4f}")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  ECE: {metrics['ece']:.4f}")
print(f"========================================")

# ==========================================
# 7. Optional: Uncertainty Analysis
# ==========================================
seq_est = TFBSequenceEstimator(stats_key=tfb_key)
uncertainties = seq_est(stats)

print(f"\n--- Sample Outputs ---")
for i in range(min(5, len(eval_inputs))):
    print(f"\nSample {i}:")
    print(f"  Question: {eval_ds[i]['question'][:60]}...")
    print(f"  True Answer: {eval_ds[i]['answerKey']}")
    print(f"  Generations: {stats[f'{tfb_key}_texts'][i][:3]}")  # First 3 samples
    print(f"  Semantic Entropy: {uncertainties[i]:.4f}")
