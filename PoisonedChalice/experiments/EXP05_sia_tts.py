
"""
EXPERIMENT 05: SIA-TTS (Structural Integrity Auditing via Test-Time Scaling)
Method: Gradient-based Self-Influence Analysis on Semantic Neighborhoods.
Goal: Detect brittle memorization by measuring stability of influence scores across
      semantically equivalent variants (SECT).
Reference: "Structural Integrity Auditing via Test-Time Scaling" (Proposed 2026).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import ast
import random
import os
import sys
import copy
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
# Core: SECT Generator (Variant Factory)
# ============================================================================

class RandomizedRenamer(ast.NodeTransformer):
    """Renames variables to random 'v_{random}' names to create diverse variants."""
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.mapping = {}
        self.used_names = set()
        self.preserve = {
            'self', 'print', 'len', 'range', 'enumerate', 'int', 'float', 'str', 'list', 'dict', 'set',
            'min', 'max', 'sum', 'zip', 'map', 'filter', 'sorted', 'open', 'super', 'isinstance'
        }

    def get_new_name(self, old_name):
        if old_name in self.preserve: return old_name
        if old_name.startswith("__") and old_name.endswith("__"): return old_name
        
        if old_name not in self.mapping:
            while True:
                # Generate a random variable name like v_123, var_456
                suffix = self.rng.randint(100, 9999)
                new_name = f"var_{suffix}"
                if new_name not in self.used_names:
                    self.used_names.add(new_name)
                    self.mapping[old_name] = new_name
                    break
        return self.mapping[old_name]

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
             node.id = self.get_new_name(node.id)
        return node
    
    def visit_arg(self, node):
        node.arg = self.get_new_name(node.arg)
        return node

def generate_variants(code: str, n_variants: int) -> List[str]:
    """Generates N semantically equivalent variants of the code."""
    variants = []
    try:
        tree = ast.parse(code)
    except:
        return []

    for i in range(n_variants):
        try:
            # Re-parse fresh tree for each variant to avoid accumulating changes
            # (Though NodeTransformer modifies in-place, better to be safe)
            current_tree = copy.deepcopy(tree)
            renamer = RandomizedRenamer(seed=i + random.randint(0, 10000))
            transformed = renamer.visit(current_tree)
            
            if sys.version_info >= (3, 9):
                variants.append(ast.unparse(transformed))
        except:
            continue
    return variants

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
        torch_dtype=dtype, 
        device_map="auto"
    )
    # Important: We need gradients, so don't just eval() without thinking.
    # But usually we eval() for inference. For gradient calc:
    # We will enable gradients temporarily.
    model.eval() 
    return model, tokenizer

# ============================================================================
# SIA-TTS Attack
# ============================================================================

class SIATTSAttack:
    """
    SIA-TTS: Structural Integrity Auditing via Test-Time Scaling.
    Measures Gradient Norm divergence across Semantic Neighborhoods.
    """
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.n_variants = 5 # Budget-constrained

    @property
    def name(self):
        return "sia_tts"

    def compute_influence(self, text: str) -> float:
        if not text or len(text.strip()) == 0: return np.nan
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            
            # Enable grad for input embeddings or weights
            # To get gradients w.r.t parameters efficiently w/o optimizer:
            # We can use the embedding layer weights.
            
            # Ensure model params require grad (might be disabled if configured for pure inference elsewhere)
            # But usually load_pretrained expects full model.
            
            # Simple Proxy: Gradient Norm of the Embedding Layer
            embeddings = self.model.get_input_embeddings()
            
            # Clear prev grads
            self.model.zero_grad()
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            # Calculate Norm of Embedding Gradients
            # Note: This is huge [Vocab, Dim]. 
            # We can use the gradients of the specific tokens used in input?
            # Or just the full norm. Full norm is safer "global" influence proxy.
            
            if embeddings.weight.grad is not None:
                grad_norm = embeddings.weight.grad.norm().item()
            else:
                grad_norm = 0.0
                
            # Cleanup to save memory
            self.model.zero_grad()
            
            return grad_norm
        except Exception as e:
            # print(f"Grad error: {e}")
            return np.nan

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores (SIA-TTS)...")
        scores = []
        
        for text in tqdm(texts, desc="SIA-TTS Auditing"):
            # 1. Base Influence
            base_inf = self.compute_influence(text)
            
            if np.isnan(base_inf):
                scores.append(np.nan)
                continue
                
            # 2. Generate Variants
            variants = generate_variants(text, self.n_variants)
            if not variants:
                scores.append(np.nan)
                continue
            
            # 3. Compute Variant Influences
            variant_infs = []
            for var_code in variants:
                inf = self.compute_influence(var_code)
                if not np.isnan(inf):
                    variant_infs.append(inf)
            
            if not variant_infs:
                scores.append(np.nan)
                continue

            # 4. Compute SII (Structural Integrity Index)
            # SII = Mean(|(Base - Var) / Base|)
            # High divergence -> Base was "Special" -> Memorized
            base_inf_safe = base_inf + 1e-9 # Avoid div by zero
            divergences = [abs((base_inf - v_inf) / base_inf_safe) for v_inf in variant_infs]
            
            sii = np.mean(divergences)
            scores.append(sii)
            
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
        attacker = SIATTSAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP05_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {fname}")
        
        # AUC
        try:
            auc = roc_auc_score(df['is_member'], df[f"{attacker.name}_score"].fillna(-999))
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
        sample_fraction = 0.05 # Lower fraction because gradients are slow
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP05] Model: {Args.model_name}")
    Experiment(Args).run()
