import subprocess
import re
import sys
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from lm_polygraph.utils.tfb import apply_tfb, enable_tfb_sampling, disable_tfb_sampling, update_tfb_beta, fit_tfb_beta

def run_reference_bayesian_peft(model_path, adapter_path, beta=0.01, n_samples=10):
    """
    Runs bayesian-peft/run/main.py as a subprocess to get reference metrics.
    """
    print(f"--- [Reference] Running Bayesian-PEFT (beta={beta}, n_samples={n_samples}) ---")
    
    # Construct command
    # modelwrapper: tfblora_acc is best for classification metrics in their codebase
    cmd = [
        "python", "bayesian-peft/run/main.py",
        "--dataset-type", "mcdataset",
        "--dataset", "pqa_labeled",
        "--model-type", "causallm",
        "--model", model_path, 
        "--modelwrapper", "tfblora_acc",
        "--load-lora-path", adapter_path,
        "--bayes-beta", str(beta),
        "--bayes-eval-n-samples-final", str(n_samples),
        "--evaluate",
        # Minimal training args to satisfy parser
        "--lr", "1e-4", "--batch-size", "8", "--max-seq-len", "512",
        "--nowand", # Disable wandb
        "--iter", "10",
        "--testing-set", "train_train_val",
        "--anchor-size", "50", # calibration set size
        "--th", "0.01", # Target change ratio
        "--bayes-train-n-samples", "5", # Ensure sufficient resolution (1/250 = 0.4%)
    ]
    
    print("Executing:", " ".join(cmd))
    
    # Run inside bayesian-peft directory to fix "dataset" path issue
    cwd_path = "bayesian-peft"
    
    # Use relative path from bayesian-peft directory
    cmd[1] = "run/main.py"
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd_path)
    
    if result.returncode != 0:
        print("Error running reference script:")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return None
    
    # Print the subprocess output for debugging
    print("=== REFERENCE SUBPROCESS OUTPUT ===")
    print(result.stdout)
    print("=== END SUBPROCESS OUTPUT ===")
    
    # Instead of stdout, we MUST read the log file because we reverted the print changes in the lib.
    log_file_path = os.path.join(cwd_path, "checkpoints", "tfblora_acc", model_path, "pqa_labeled", "default", "log.txt")
    
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}")
        print("Subprocess Output:", result.stdout[-500:])
        return None
        
    with open(log_file_path, 'r') as f:
        output = f.read()
    
    print("=== LOG FILE CONTENT ===")
    print(output)
    print("=== END LOG FILE ===")
    
    # Parse NLL, ACC from logs
    # Log format: val_acc: 0.5, val_ece: 0.1, val_nll: 2.3, val_brier: 0.2
    
    nll_match = re.findall(r"val_nll:\s*([0-9.]+)", output)
    acc_match = re.findall(r"val_acc:\s*([0-9.]+)", output)
    
    if not nll_match:
        print("Could not parse NLL from log file.")
        return None
        
    # Take the last evaluation result
    nll = float(nll_match[-1])
    acc = float(acc_match[-1]) if acc_match else 0.0
    
    return {"nll": nll, "acc": acc}

def run_candidate_lm_polygraph(model_path, adapter_path, beta=0.01, n_samples=10, anchor_size=50):
    """Runs lm-polygraph implementation in-process with tokenizer-agnostic last-token selection."""
    print(f"\n--- [Candidate] Running LM-Polygraph (beta={beta}, n_samples={n_samples}) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)

    # Tokenizer aligned with reference defaults, but selection is last-nonpad (Qwen might have issues with left-padding inference though)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset split consistent with reference (seed=42, 90/10)
    dataset_full = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds_split = dataset_full.train_test_split(test_size=0.1, seed=42)
    anchor_ds = ds_split["train"].select(range(min(anchor_size, len(ds_split["train"]))))
    eval_ds = ds_split["test"]

    # Targets from tokenizer (leading space labels like reference)
    labels = [" yes", " no", " maybe"]
    label_map = {"yes": 0, "no": 1, "maybe": 2}
    target_ids = tokenizer(labels, add_special_tokens=False).input_ids
    target_ids = [ids[0] for ids in target_ids]
    target_ids_tensor = torch.tensor(target_ids, device=device)

    preamble = """Answer the question with yes, no, or maybe based on the context.

Context: {context}
Question: {question}
Answer:"""

    def last_nonpad(mask: torch.Tensor) -> torch.Tensor:
        return mask.sum(dim=1) - 1

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Apply TFB
    apply_tfb(model, beta=beta)

    # Prepare calibration batches - FIXED to match original logic exactly
    cal_batches = []
    baselines = []
    
    print("Pre-computing Calibration Baselines (beta=0)...")
    update_tfb_beta(model, 0.0)  # Set beta=0 for deterministic baseline
    disable_tfb_sampling(model)
    
    with torch.no_grad():
        for i in range(0, len(anchor_ds), 8):
            chunk = anchor_ds.select(range(i, min(i + 8, len(anchor_ds))))
            prompts = [
                preamble.format(
                    context=" ".join(e["context"]["contexts"]),
                    question=e["question"],
                )
                for e in chunk
            ]
            tokenized = [tokenizer(p, truncation=True, max_length=512) for p in prompts]
            batch = collator(tokenized)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits
            idx = last_nonpad(batch["attention_mask"])
            b_idx = torch.arange(logits.size(0), device=logits.device)
            logits = logits[b_idx, idx][:, target_ids_tensor]
            det_probs = torch.softmax(logits, dim=-1)
            det_preds = det_probs.argmax(dim=-1)

            cal_batches.append(batch)
            baselines.append(det_preds)

    # Flatten baselines to match original format
    all_baseline_preds = torch.cat(baselines, dim=0)

    def flip_metric_debug(m, inputs_list, n_s, parallel=False):
        """Flip ratio metric that matches the original bayesian-peft exactly"""
        all_stoch_preds = []
        
        enable_tfb_sampling(m)
        with torch.no_grad():
            for inputs in inputs_list:
                batch_preds = []
                for _ in range(n_s):
                    logits = m(**inputs).logits
                    idx = last_nonpad(inputs["attention_mask"])
                    b_idx = torch.arange(logits.size(0), device=logits.device)
                    logits = logits[b_idx, idx][:, target_ids_tensor]
                    probs = torch.softmax(logits, dim=-1)
                    batch_preds.append(probs)
                
                # Average across samples, then argmax (like original)
                mean_probs = torch.stack(batch_preds).mean(dim=0)
                stoch_pred = mean_probs.argmax(dim=-1)
                all_stoch_preds.append(stoch_pred)
        
        all_stoch_preds = torch.cat(all_stoch_preds, dim=0)
        flip_ratio = (all_stoch_preds != all_baseline_preds).float().mean().item()
        
        print(f"    DEBUG: flip_ratio = {flip_ratio:.4f} (target: 0.01)")
        return flip_ratio, None

    print("Running Calibration via fit_tfb_beta...")
    best_beta = fit_tfb_beta(
        model,
        cal_batches,
        target_metric_ratio=0.01,
        max_iters=10,
        n_samples=5,
        initial_beta=beta,
        metric_fn=flip_metric_debug,
        verbose=True,
    )

    print(f"Optimal Beta found: {best_beta:.6f}")
    update_tfb_beta(model, best_beta)
    enable_tfb_sampling(model)

    nll_vals = []
    with torch.no_grad():
        for item in eval_ds:
            prompt = preamble.format(
                context=" ".join(item["context"]["contexts"]),
                question=item["question"],
            )
            batch = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            probs = []
            for _ in range(n_samples):
                logits = model(**batch).logits
                idx = last_nonpad(batch["attention_mask"])
                b_idx = torch.arange(logits.size(0), device=logits.device)
                logits = logits[b_idx, idx][:, target_ids_tensor]
                probs.append(torch.softmax(logits, dim=-1))

            mean_probs = torch.stack(probs).mean(dim=0)
            true_idx = label_map[item["final_decision"]]
            nll_vals.append(-np.log(mean_probs[0, true_idx].item() + 1e-12))

    return {"nll": float(np.mean(nll_vals))}

def main():
    # Use a small model for speed
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_name = "ShahzebKhoso/qwen2.5-instruct-0.5B-pubmedqa-lora"
    model_cache = "model_cache"
    
    # 1. Run Reference
    ref_metrics = run_reference_bayesian_peft(model_name, adapter_name)
    
    if ref_metrics is None:
        print("Reference run failed. Aborting.")
        return

    # 2. Run Candidate
    cand_metrics = run_candidate_lm_polygraph(model_name, adapter_name)
    
    # 3. Compare
    print("\n========================================")
    print("       BENCHMARK RESULTS")
    print("========================================")
    print(f"Ref NLL: {ref_metrics['nll']:.4f}")
    print(f"Can NLL: {cand_metrics['nll']:.4f}")
    
    diff = abs(ref_metrics['nll'] - cand_metrics['nll'])
    print(f"Diff:    {diff:.4f}")
    
    if diff < 0.1: # Allow some slush for float/implementation variance
        print("✅ SUCCESS: Matches Reference")
    else:
        print("❌ FAILURE: Significant Mismatch")

if __name__ == "__main__":
    main()
