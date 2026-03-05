
"""
EXPERIMENT 06: SIA-TTS Funnel (Cascading Architecture)
Method: Multi-stage filtering pipeline combining Min-K%++, SECT-Loss, and Gradient-SIA.
Goal: Efficiently allocate compute (Test-Time Scaling) to suspicious samples.
      Stage 1 (Fast): Min-K%++
      Stage 2 (Medium): SECT-Loss Divergence
      Stage 3 (Deep): Gradient-SIA (Influence)
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
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
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
# Shared Utilities (SECT, etc)
# ============================================================================

class VariableRenamer(ast.NodeTransformer):
    def __init__(self, seed=None):
        self.rng = random.Random(seed) if seed else random
        self.mapping = {}
        self.count = 0
        self.preserve = {
            'self', 'print', 'len', 'range', 'enumerate', 'int', 'float', 'str', 'list', 'dict', 'set',
            'min', 'max', 'sum', 'zip', 'map', 'filter', 'sorted', 'open', 'super', 'isinstance'
        }

    def get_new_name(self, old_name):
        if old_name in self.preserve: return old_name
        if old_name.startswith("__") and old_name.endswith("__"): return old_name
        if old_name not in self.mapping:
            self.count += 1
            # Simple v1, v2 style for Stage 2 (Filter 2)
            self.mapping[old_name] = f"v{self.count}"
        return self.mapping[old_name]

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
             node.id = self.get_new_name(node.id)
        return node
    
    def visit_arg(self, node):
        node.arg = self.get_new_name(node.arg)
        return node

def apply_sect_rename(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        renamer = VariableRenamer()
        transformed = renamer.visit(tree)
        if sys.version_info >= (3, 9):
            return ast.unparse(transformed)
    except:
        return None
    return None

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # We need grads for Stage 3
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype, 
        device_map="auto"
    )
    # Important logic: 
    # Stage 1 & 2 are inference (eval mode).
    # Stage 3 needs gradients. Standard flow: eval() then manually enable grad context?
    # Actually, backward() works in eval mode if inputs have requires_grad=True or we force it.
    model.eval()
    return model, tokenizer

# ============================================================================
# The Funnel (Cascading Architecture)
# ============================================================================

class FunnelAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        # Funnel Thresholds (heuristic based on typical distributions)
        # In a real scenario, these would be calibrated on a validation set.
        # Here we use percentiles or absolute values.
        # For simplicity, we use "pass all" to "compute all" but weigh scores.
        # But to demonstrate "Scaling", we should skip checks.
        
        # Let's say: 
        # Low MinK score -> Likely Non-Member -> Skip detailed check -> Final Score = MinK
        # High MinK score -> Potential Member -> Run Next Stage

    @property
    def name(self):
        return "sia_funnel"

    # --- Stage 1: Min-K%++ ---
    def stage1_mink_pp(self, text: str) -> float:
        if not text: return np.nan
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_probs = log_softmax(outputs.logits, dim=-1)
            vocab_mean = log_probs.mean(dim=-1)
            vocab_std = log_probs.std(dim=-1)
            z_scores_dist = (log_probs - vocab_mean.unsqueeze(-1)) / (vocab_std.unsqueeze(-1) + 1e-8)
            
            target_z_scores = []
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1]
                target_z_scores.append(z_scores_dist[0, i, token_id].item())
        
        if not target_z_scores: return np.nan
        sorted_z = np.sort(target_z_scores)
        k_len = max(1, int(len(sorted_z) * 0.2)) # k=0.2
        return np.mean(sorted_z[:k_len])

    # --- Stage 2: SECT-Loss Divergence ---
    def stage2_sect_loss(self, text: str, base_loss: float = None) -> float:
        # 1. Transform
        trans_code = apply_sect_rename(text)
        if not trans_code: return 0.0 # Failed transform -> Assume 0 divergence (Conservative)
        
        # 2. Compute Loss
        # Need base loss if not provided
        if base_loss is None:
             # Recompute
             with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
                base_loss = self.model(**inputs, labels=inputs["input_ids"]).loss.item()

        with torch.no_grad():
             inputs = self.tokenizer(trans_code, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
             trans_loss = self.model(**inputs, labels=inputs["input_ids"]).loss.item()
        
        # 3. Divergence
        # If Member: Structure is fragile -> Transformed Loss >> Base Loss -> High Diff
        # If Non-Member: Generalized -> Transformed Loss ~= Base Loss -> Low Diff
        return trans_loss - base_loss

    # --- Stage 3: Gradient Influence (SIA-TTS) ---
    def stage3_gradient_sia(self, text: str) -> float:
        # Simplified for Funnel: Just gradient norm of original (Base Influence)
        # Full SIA-TTS compares this against variants, but let's use 
        # "Grad Norm" as a feature itself. 
        # High Influence (Gradient Norm) often correlates with "Outliers" or memorized samples in Hessian analysis.
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            embeddings = self.model.get_input_embeddings()
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            if embeddings.weight.grad is not None:
                grad_norm = embeddings.weight.grad.norm().item()
            else:
                grad_norm = 0.0
            self.model.zero_grad()
            return grad_norm
        except:
            return 0.0

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores (Cascading)...")
        scores = []
        
        # Stats for report
        s1_count = 0
        s2_count = 0
        s3_count = 0
        
        # Heuristic Thresholds (Assumed based on distribution)
        # MinK scores are usually negative (log prob Z-scores). High = Closer to 0 (or positive).
        # Wait, Z-scores centered at 0.
        # "Unexpected" tokens have Low Z-scores (negative).
        # Memorized samples have "Higher" Min-K scores (less unexpected).
        # So Threshold: if score > X, proceed.
        # Let's set a relative threshold after a warmup or just use fixed.
        # For this script, we'll run ALL to gather data, but calculate final score as weighted sum.
        # Real "Scaling" would break early.
        
        # Implementation of "Soft Funnel":
        # Final Score = w1*S1 + w2*S2 + w3*S3
        # But we compute S2 only if S1 is suspicious, etc.
        
        for text in tqdm(texts, desc="Funnel Analysis"):
            if not text or len(text.strip()) == 0:
                scores.append(np.nan)
                continue
                
            # Stage 1
            s1 = self.stage1_mink_pp(text)
            if np.isnan(s1):
                scores.append(np.nan)
                continue
            s1_count += 1
            
            # Decision Gate 1
            # Assuming Z-score distribution centered ~0.
            # If s1 < -2.0 (Very likely non-member/anomalous in bad way?), skip?
            # Actually, standard MIA: High Score = Member.
            # Let's compute S2 if S1 > -0.5 (Top 50% likelihood)
            
            final_score = s1 # Base score
            
            if s1 > -1.0: # Suspiciously high likelihood
                s2 = self.stage2_sect_loss(text)
                s2_count += 1
                
                # Combine: S1 + S2
                # S2 is Divergence. High Divergence = Member.
                # S1 is Likelihood. High Likelihood = Member.
                final_score += s2
                
                if s2 > 0.5: # Structure is also brittle!
                    s3 = self.stage3_gradient_sia(text)
                    s3_count += 1
                    
                    # S3 is Grad Norm. 
                    # Relation to membership is complex, typically higher for members (influence).
                    # Normalize it?
                    final_score += (s3 * 0.01) # Scale down grad norm
            
            scores.append(final_score)
            
        print(f"\n[Funnel Stats] Processed: {len(texts)}")
        print(f"Stage 1 (MinK): {s1_count}")
        print(f"Stage 2 (SECT): {s2_count} ({s2_count/len(texts):.1%})")
        print(f"Stage 3 (Grad): {s3_count} ({s3_count/len(texts):.1%})")
        
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
        attacker = FunnelAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP06_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {fname}")
        
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
        sample_fraction = 0.05 # Lower fraction for cascading
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP06] Model: {Args.model_name}")
    Experiment(Args).run()
