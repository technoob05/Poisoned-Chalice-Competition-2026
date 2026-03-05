
"""
EXPERIMENT 12: Hessian Trace Estimation (White-Box)
Method: Hutchinson's Method to estimate the Trace of the Hessian w.r.t Input Embeddings.
Goal: Measure "Loss Curvature" (Sharpness).
      - Member data -> Sharp Minima (Overfitted) OR Flat Minima?
      - Non-member -> Distinctive curvature profile.
      - Typically, Sharp Minima = Poor Generalization (Memorization?).
Reference: "Hutchinson's Method", "PyTorch Hessian Power Iteration".
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Kaggle & Environment Setup
# ============================================================================

def setup_environment():
    print("--- Environment Setup Starting ---")
    try:
        import transformers
        import datasets
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub")
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Login Note: {e}")

    kaggle_path = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
    if os.path.exists(kaggle_path):
        print(f"Dataset found: {kaggle_path}")
    else:
        print("Warning: Standard dataset path not found.")
    print("--- Environment Setup Complete ---")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        dtype=dtype, 
        device_map="auto"
    )
    # We need to compute second derivatives (double backprop)
    # So we must NOT use eval? Actually eval doesn't disable grads, just dropout.
    # But we need input tensors to require grad.
    model.eval()
    return model, tokenizer

# ============================================================================
# Hessian Trace Attack (Hutchinson)
# ============================================================================

class HessianTraceAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.num_iters = 1 # Keep it fast for competition context. Accuracy grows with N.

    @property
    def name(self):
        return "hessian_trace"

    def compute_hessian_trace(self, text: str) -> float:
        """
        Estimates Trace(H) w.r.t Input Embeddings using Hutchinson's Method.
        Trace(H) = E[v^T H v] where v is Rademacher vector.
        Hv is computed via gradient of (grad * v).
        """
        if not text: return np.nan
        
        try:
            # 1. Prepare Input
            # We need to intercept the embeddings to set requires_grad=True
            # because standard forward() takes integer IDs.
            # Only model.embed_tokens(input_ids) gives embeddings.
            
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            input_ids = inputs["input_ids"]
            
            # Identify Embedding Layer
            # Starcoder2: model.model.embed_tokens
            embed_layer = self.model.get_input_embeddings()
            
            # 2. Get Embeddings Manually
            with torch.no_grad():
                input_embeds = embed_layer(input_ids)
            
            # Enable Gradient Tracking for Inputs
            input_embeds.requires_grad_(True)
            input_embeds.retain_grad()
            
            # 3. Forward Pass with Embeddings
            # Pass inputs_embeds to model
            outputs = self.model(inputs_embeds=input_embeds, labels=input_ids)
            loss = outputs.loss
            
            # 4. First Gradient (Grad)
            # Create Graph is needed for higher order derivatives
            grads = torch.autograd.grad(loss, input_embeds, create_graph=True)[0]
            
            # 5. Hutchinson's Trace Estimation
            trace_accum = 0.0
            for _ in range(self.num_iters):
                # Sample Rademacher vector v (same shape as inputs, values {-1, 1})
                v = torch.randint_like(input_embeds, high=2) * 2 - 1
                v = v.float() # Ensure float matching dtype
                
                # Compute (Grad * v)
                grad_v_prod = torch.sum(grads * v)
                
                # Compute Gradient of (Grad * v) w.r.t Inputs -> Hv
                # This equals H * v
                h_v = torch.autograd.grad(grad_v_prod, input_embeds, create_graph=False)[0]
                
                # Trace estimate = v^T * (H v)
                # Since v is {-1, 1}, v*v = 1.
                # So v^T * H * v is projection.
                curr_trace = torch.sum(v * h_v).item()
                trace_accum += curr_trace
                
            est_trace = trace_accum / self.num_iters
            
            # Clear graph
            self.model.zero_grad()
            del grads, h_v, loss
            
            return est_trace
            
        except Exception as e:
            # print(f"Hessian Error: {e}")
            return np.nan

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores (Hutchinson Trace)...")
        scores = []
        
        for text in tqdm(texts, desc="Hessian Analysis"):
            trace = self.compute_hessian_trace(text)
            
            # Metric interpretation:
            # High Trace -> Sharp Minima -> Unstable / Overfitted?
            # Low Trace -> Flat Minima -> Generalized / Robust.
            # In MIA, Memorized samples often have sharper minima?
            # Or very flat if heavily overtrained SGD?
            # "Sharpness Correlates with Generalization" (Jiang et al.) -> Sharp = Bad Gen.
            # So Member (Memorized) might be Sharp? Or Flat?
            # Let's save the raw Trace value.
            scores.append(trace)
            
        return scores

# ============================================================================
# Experiment Runners
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self):
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading data from {self.args.dataset}...")
        for subset in subsets:
            if is_local:
                path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(path): continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            dfs.append(ds.to_pandas())
        if not dfs: raise ValueError("No data found!")
        df = pd.concat(dfs, ignore_index=True)
        df['is_member'] = df['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        attacker = HessianTraceAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP12_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {fname}")
        
        try:
            auc = roc_auc_score(df['is_member'], df[f"{attacker.name}_score"].fillna(0))
            print(f"AUC ({attacker.name}): {auc:.4f}")
        except Exception as e:
            print(f"AUC Error: {e}")

if __name__ == "__main__":
    setup_environment()
    
    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.05 # Double backprop is heavy
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP12] Model: {Args.model_name}")
    Experiment(Args).run()
