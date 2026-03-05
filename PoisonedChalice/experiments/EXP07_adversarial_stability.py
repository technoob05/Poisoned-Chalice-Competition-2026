
"""
EXPERIMENT 07: Adversarial Prompt Stability (Context Perturbation)
Method: Perturb the first 50% of the code (Context) and measure the stability 
        of the target probability (Completion).
Goal: Detect membership via "Stability under Noise". 
      Members are robust to context noise (because they memorized the whole file).
      Non-members are brittle (changing context changes prediction).
Reference: Jiang et al. (ASE 2025) - "Adversarial Prompts for MIA".
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
# Perturbation Logic
# ============================================================================

def perturb_context(context_code: str) -> str:
    """
    Apply simple adversarial perturbations to context:
    1. Add random comments.
    2. Add extra newlines.
    3. Rename variables (lightly).
    """
    pg = random.choice(['comments', 'newlines', 'rename'])
    
    if pg == 'comments':
        return f"# CONTEXT PERTURBATION\n{context_code}\n# END CONTEXT"
    elif pg == 'newlines':
        return context_code.replace("\n", "\n\n")
    elif pg == 'rename':
        # Simple string replace for common vars (risky but okay for adversarial noise)
        for var in ['i', 'x', 'data', 'result']:
            context_code = context_code.replace(f" {var} ", f" {var}_adv ")
        return context_code
    return context_code

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
# Stability Attack
# ============================================================================

class AdversarialStabilityAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "adv_stability"

    def get_target_log_prob(self, context: str, target: str) -> float:
        # Full text = Context + Target
        # We want P(Target | Context)
        full_text = context + target
        
        inputs = self.tokenizer(full_text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        
        # Find where target starts? 
        # Approximate: Tokenize context separately to find length.
        ctx_len = len(self.tokenizer.encode(context, add_special_tokens=False))
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
            log_probs = log_softmax(outputs.logits, dim=-1)
            
            # Sum log probs of target tokens
            # Target tokens start at ctx_len
            total_log_prob = 0.0
            count = 0
            
            # Shift by 1 for next token prediction
            # input_ids[0, i+1] is predicted by logits[0, i]
            
            start_idx = max(0, ctx_len - 1) 
            end_idx = input_ids.shape[1] - 1
            
            for i in range(start_idx, end_idx):
                token_id = input_ids[0, i + 1]
                prob = log_probs[0, i, token_id].item()
                total_log_prob += prob
                count += 1
                
            return total_log_prob / max(1, count)

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores...")
        scores = []
        
        for text in tqdm(texts, desc="Adversarial Stability"):
            if not text or len(text) < 50:
                scores.append(np.nan)
                continue
                
            # Split Context/Target (50/50 split by characters)
            split_idx = len(text) // 2
            context = text[:split_idx]
            target = text[split_idx:]
            
            # 1. Original Score
            orig_score = self.get_target_log_prob(context, target)
            
            # 2. Perturbed Score
            adv_context = perturb_context(context)
            adv_score = self.get_target_log_prob(adv_context, target)
            
            # 3. Stability Score
            # If Member: Robust -> Orig ~= Adv
            # If Non-Member: Brittle -> Orig >> Adv (or just different)
            # Stability = -|Diff| (Lower diff = More stable = Higher score?)
            # Wait, usually Membership score implies Higher = Member.
            # If Member is Stable -> Diff is Low.
            # If Non-Member is Unstable -> Diff is High.
            # So Score should be -Diff.
            
            diff = abs(orig_score - adv_score)
            scores.append(-diff) 
            
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
        attacker = AdversarialStabilityAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP07_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
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

    print(f"[EXP07] Model: {Args.model_name}")
    Experiment(Args).run()
