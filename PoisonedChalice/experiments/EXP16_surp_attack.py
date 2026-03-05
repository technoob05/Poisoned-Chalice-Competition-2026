
"""
EXPERIMENT 16: SURP Attack (Statistical Unsupervised Robustness Prediction)
Method: Calibrates the average log-likelihood using the standard deviation of token probabilities.
Goal: Differentiate between low loss due to general proficiency vs. low loss due to memorization.
      - Memorized: High Mean, Low Std (Very stable)
      - General: High Mean, Higher Std (Less stable across tokens)
Formula: Score = Mean(log_probs) - Std(log_probs)
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
    except: pass
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
    model.eval()
    return model, tokenizer

# ============================================================================
# SURP Attack
# ============================================================================

class SURPAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "surp"

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"Computing {self.name} scores...")
        scores = []
        
        for text in tqdm(texts, desc="SURP Analysis"):
            if not text:
                scores.append(np.nan)
                continue
                
            try:
                inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits
                    log_probs = log_softmax(logits, dim=-1)
                    
                    target_probs = []
                    for i in range(inputs["input_ids"].shape[1] - 1):
                        token_id = inputs["input_ids"][0, i + 1]
                        target_probs.append(log_probs[0, i, token_id].item())
                    
                    if len(target_probs) == 0:
                        scores.append(np.nan)
                        continue
                        
                    # SURP core logic: Mean - Std
                    # Members: High Mean, Low Std -> High Score
                    # Non-members: Lower Mean, High Std -> Lower Score
                    mean_lp = np.mean(target_probs)
                    std_lp = np.std(target_probs)
                    
                    score = mean_lp - std_lp
                    scores.append(score)
                    
            except Exception as e:
                scores.append(np.nan)
                
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
        attacker = SURPAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP16_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
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

    print(f"[EXP16] Model: {Args.model_name}")
    Experiment(Args).run()
