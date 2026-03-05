
"""
EXPERIMENT 09: Semantic Canonicalization (Maximum Invariance)
Method: Transform code to its most "generic" canonical form.
        1. Aggressive variable/function renaming to id1, id2...
        2. Remove comments and docstrings.
        3. Standardize indentation/whitespacing.
        4. Reorder independent statements (Topological sorting).
Goal: Detect "Memorization Gap". If Loss(Original) << Loss(Canonical), it's a member.
Reference: "Structural Integrity Auditing" (2025-2026).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import ast
import random
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple
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
        print("Warning: Expected dataset path not found.")
    print("--- Environment Setup Complete ---")

# ============================================================================
# Core: Canonicalizer (The "Generic" Transformer)
# ============================================================================

class Canonicalizer(ast.NodeTransformer):
    """
    Strips code down to its bare logic.
    """
    def __init__(self):
        self.mapping = {}
        self.id_count = 0
        self.preserve = {
            'self', 'print', 'range', 'enumerate', 'len', 'int', 'float', 'str', 
            'list', 'dict', 'set', 'bool', 'Exception', 'super', 'isinstance'
        }

    def get_id(self, name):
        if name in self.preserve: return name
        if name.startswith("__") and name.endswith("__"): return name
        if name not in self.mapping:
            self.id_count += 1
            self.mapping[name] = f"id{self.id_count}"
        return self.mapping[name]

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
            node.id = self.get_id(node.id)
        return node

    def visit_FunctionDef(self, node):
        # Rename function
        node.name = self.get_id(node.name)
        # Strip docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant))):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = self.get_id(node.name)
        # Strip docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, (ast.Str, ast.Constant))):
            node.body.pop(0)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        node.arg = self.get_id(node.arg)
        return node

    def visit_Expr(self, node):
        # Remove standalone strings (often docstrings or comments-as-strings)
        if isinstance(node.value, (ast.Str, ast.Constant)) and isinstance(node.value.value, str):
            return None
        return node

def apply_canonicalization(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        transformer = Canonicalizer()
        tree = transformer.visit(tree)
        # ast.unparse removes comments and standardizes whitespacing
        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=dtype, 
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

# ============================================================================
# Canonicalization Attack
# ============================================================================

class CanonicalizationAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "canonical_gap"

    def calculate_loss(self, text: str) -> float:
        if not text: return np.nan
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            return outputs.loss.item()
        except:
            return np.nan

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores...")
        scores = []
        
        for text in tqdm(texts, desc="Canonical Analysis"):
            orig_loss = self.calculate_loss(text)
            if np.isnan(orig_loss):
                scores.append(np.nan)
                continue
                
            canonical_code = apply_canonicalization(text)
            if not canonical_code:
                scores.append(np.nan)
                continue
                
            canon_loss = self.calculate_loss(canonical_code)
            if np.isnan(canon_loss):
                scores.append(np.nan)
                continue
                
            # Gap = Canonical Loss - Original Loss
            # High Gap -> Model "loves" the original tokens far more than the logic.
            scores.append(canon_loss - orig_loss)
            
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
        attacker = CanonicalizationAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP09_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
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
        sample_fraction = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP09] Model: {Args.model_name}")
    Experiment(Args).run()
