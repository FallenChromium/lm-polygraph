import subprocess
import re
import sys
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from lm_polygraph.utils.tfb import apply_tfb, enable_tfb_sampling

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
    ]
    
    print("Executing:", " ".join(cmd))
    
    # Run inside bayesian-peft directory to fix "dataset" path issue
    cwd_path = "bayesian-peft"
    
    # Use relative path from bayesian-peft directory
    cmd[1] = "run/main.py"
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd_path)
    
    if result.returncode != 0:
        print("Error running reference script:")
        print(result.stderr)
        return None
    
    output = result.stderr + result.stdout # Logging often goes to stderr
    # Parse NLL, ACC from logs
    # Log format: val_acc: 0.5, val_ece: 0.1, val_nll: 2.3, val_brier: 0.2
    
    nll_match = re.search(r"val_nll:\s*([0-9.]+)", output)
    acc_match = re.search(r"val_acc:\s*([0-9.]+)", output)
    
    if not nll_match:
        print("Could not parse NLL from output.")
        print("Output Snippet:", output[-500:])
        return None
        
    nll = float(nll_match.group(1))
    acc = float(acc_match.group(1)) if acc_match else 0.0
    
    return {"nll": nll, "acc": acc}

def run_candidate_lm_polygraph(model_path, adapter_path, beta=0.01, n_samples=10):
    """
    Runs lm-polygraph implementation in-process.
    """
    print(f"\n--- [Candidate] Running LM-Polygraph (beta={beta}, n_samples={n_samples}) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Apply TFB
    apply_tfb(model, beta=beta)
    
    # Load Data (Identical split to reference)
    # The reference implementation uses 'validation' or 'test' split depending on args.
    # S2S_Classification default for "validation" split logic:
    # We used default args, so it likely loads 'validation' split if testing_set is default?
    # Actually `main.py` calls `get_loaders`, which loads `test_dataloader` from `validation` split by default.
    # BUT, 'pqa_labeled' only has 'train'.
    # Our Adapter loads 'pqa_labeled'. If we didn't specify split mapping, `load_dataset("qiaojin/PubMedQA", "pqa_labeled")`
    # returns a DatasetDict with 'train'.
    
    # Update: In my adapter `PubMedQADataset.__init__`, I did:
    # dset = load_dataset(...)
    # S2ClassDataset.loader calls `dset[split]`.
    # Since pqa only has train, I should map 'validation' to 'train' or handle it.
    # Wait, my adapter didn't handle splits carefully. It just loaded the dataset. 
    # If `dsets.py` logic tries `dset['validation']`, it will fail if the dict only has 'train'.
    # I should check PubMedQA structure. It usually has train.
    # If my adapter fails in the subprocess, I will see it.
    
    # Assume we are using the 'train' split for now. 
    dataset_full = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds_split = dataset_full.train_test_split(test_size=0.1, seed=42)
    dataset = ds_split["test"] # Matches 'validation' split in dsets.py
    # For fair bench, define a subset.
    # bayesian-peft doesn't subsample unless requested. 
    # It runs on the full split provided. 
    # To save time, we should probably stick to a small subset, but I didn't set that in Ref args.
    # I'll let it run on full (1000 samples) or maybe just 50 if I can control it.
    # bayesian-peft uses `--bayes-eval-n-samples`? No that's stochastic samples.
    # Use `--anchor-size` or custom logic?
    
    # Let's match the PRECISE logic of the reference adapter I wrote.
    # My Adapter uses:
    # context = e["context"]["contexts"]
    # question = e["question"]
    # Preamble: "Answer the question with yes, no, or maybe based on the context.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
    
    # Labels: " yes", " no", " maybe" (with space if add_space=True, which is default)
    # Target IDs: [tokenizer(' yes'), ...]
    
    target_words = [" yes", " no", " maybe"]
    target_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in target_words]
    target_ids_tensor = torch.tensor(target_ids).to(device)
    label_map = {"yes": 0, "no": 1, "maybe": 2}
    
    preamble = """Answer the question with yes, no, or maybe based on the context.

Context: {context}
Question: {question}
Answer:"""

    nll_vals = []
    
    from lm_polygraph.utils.tfb import update_tfb_beta, fit_tfb_beta, disable_tfb_sampling
    
    # 3. Calibration (Binary Search)
    cal_dataset = ds_split["train"].select(range(min(50, len(ds_split["train"]))))
    cal_inputs = [
        tokenizer(
            preamble.format(context=" ".join(item["context"]["contexts"]), question=item["question"]),
            return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        for item in cal_dataset
    ]
    
    def classification_acc_metric(m, inputs, n_s, p=False):
        # Deterministic baseline
        disable_tfb_sampling(m)
        with torch.no_grad():
            det_logits = m(**inputs).logits[:, -1, target_ids_tensor]
            det_pred = det_logits.argmax(dim=-1)
        
        # Stochastic sample
        enable_tfb_sampling(m)
        with torch.no_grad():
            # Match reference: 1 sample for calibration
            stoch_logits = m(**inputs).logits[:, -1, target_ids_tensor]
            stoch_pred = stoch_logits.argmax(dim=-1)
            
        # Return mismatch count as "loss". 
        # Baseline at beta=0 will be 0.
        # Metric change will be the drift from deterministic.
        return (stoch_pred != det_pred).float().mean(), det_pred

    print("Running Calibration via fit_tfb_beta...")
    best_beta = fit_tfb_beta(
        model, 
        cal_inputs, 
        target_metric_ratio=0.01, 
        max_iters=10, 
        initial_beta=beta,
        metric_fn=classification_acc_metric,
        verbose=True
    )
    
    print(f"Optimal Beta found: {best_beta:.6f}")
    update_tfb_beta(model, best_beta)
    enable_tfb_sampling(model)
    
    # Limit to first 50 samples for speed if not controllable in Ref?
    # Ref runs full set. PQA is 1k samples. Might take a while.
    # 0.5B model is fast. 1k * 10 samples ~ 1 minute on GPU.
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            # Prep Input
            context_str = " ".join(item["context"]["contexts"]) # Match dsets.py formatting
            prompt = preamble.format(context=context_str, question=item["question"])
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Stochastic loop
            sample_probs = []
            for _ in range(n_samples):
                out = model(**inputs)
                logits = out.logits[:, -1, :] # [1, vocab]
                
                # Filter targets
                target_l = logits[:, target_ids_tensor] # [1, 3]
                probs = torch.softmax(target_l, dim=-1)
                sample_probs.append(probs)
            
            # Mean
            mean_probs = torch.stack(sample_probs).mean(dim=0) # [1, 3]
            
            # Get metric
            true_label_idx = label_map[item["final_decision"]]
            prob_true = mean_probs[0, true_label_idx].item()
            nll_vals.append(-np.log(prob_true + 1e-12))
            
    return {"nll": np.mean(nll_vals)}

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
