
"""
EXPERIMENT 04: Structural Divergence (SIA-TTS Filter 2)
Method: Loss Difference between Original and Variable-Renamed Code.
Goal: Detect brittle memorization where model relies on exact variable names.
Reference: "Structural Integrity Auditing via Test-Time Scaling" (Proposed 2026).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import ast
import random
import os
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
# Core: Semantic Equivalent Code Transformation (SECT) - Renamer
# ============================================================================

class VariableRenamer(ast.NodeTransformer):
    """
    Renames all local variables and function arguments to standardized names (v1, v2, ...).
    Keeps structure but destroys surface-level token patterns.
    """
    def __init__(self):
        self.mapping = {}
        self.count = 0
        # Preserve built-ins and common keywords to keep code vaguely executable/valid looking
        self.preserve = {
            'self', 'print', 'len', 'range', 'enumerate', 'int', 'float', 'str', 'list', 'dict', 'set',
            'min', 'max', 'sum', 'zip', 'map', 'filter', 'sorted', 'open', 'super', 'isinstance'
        }

    def get_new_name(self, old_name):
        if old_name in self.preserve:
            return old_name
        if old_name.startswith("__") and old_name.endswith("__"):
             return old_name
        if old_name not in self.mapping:
            self.count += 1
            self.mapping[old_name] = f"v{self.count}"
        return self.mapping[old_name]

    def visit_Name(self, node):
        # Only rename variables being loaded or stored, not globals/attributes usually
        # But for 'loss divergence', agressively renaming everything (except keywords) is fine
        # as long as consistency is maintained in the scope.
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
             node.id = self.get_new_name(node.id)
        return node
    
    def visit_arg(self, node):
        node.arg = self.get_new_name(node.arg)
        return node
    
    # We might want to skip renaming import aliases or function names 
    # if we want to be "strictly" semantic preserving, but for MIA/Hallucination check
    # modifying function names (if defined locally) is also good.
    # For simplicity, we stick to Name nodes which covers most usages.

def apply_sect_rename(code: str) -> Optional[str]:
    """Applies variable renaming to the code. Returns None if parsing fails."""
    try:
        tree = ast.parse(code)
        renamer = VariableRenamer()
        transformed = renamer.visit(tree)
        # unparse requires Python 3.9+
        if sys.version_info >= (3, 9):
            return ast.unparse(transformed)
        else:
            # Fallback for older python (unlikely on Kaggle/Colab now, but safety)
            return None
    except Exception as e:
        # If syntax error (snippet might not be valid full code), return None
        return None

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Use bfloat16 or float16
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
# Structural Divergence Attack
# ============================================================================

class StructuralDivergenceAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "sect_divergence"

    def calculate_loss(self, text: str) -> float:
        if not text or len(text.strip()) == 0: return np.nan
        
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
        
        for text in tqdm(texts, desc="SECT Analysis"):
            # 1. Original Loss
            orig_loss = self.calculate_loss(text)
            
            # 2. Transformed (SECT) Loss
            transformed_code = apply_sect_rename(text)
            
            if transformed_code is None or np.isnan(orig_loss):
                scores.append(np.nan)
                continue
                
            trans_loss = self.calculate_loss(transformed_code)
            
            if np.isnan(trans_loss):
                scores.append(np.nan)
                continue
                
            # 3. Score = Loss(Transformed) - Loss(Original)
            # Logic: 
            # - Memorized code: Relies on specific tokens -> Changing them destroys 'low perplexity' -> High Diff
            # - Generalized code: Relies on structure -> Changing names keeps structure -> Low Diff
            score = trans_loss - orig_loss
            scores.append(score)
            
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
        attacker = StructuralDivergenceAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP04_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
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
        sample_fraction = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP04] Model: {Args.model_name}")
    Experiment(Args).run()
