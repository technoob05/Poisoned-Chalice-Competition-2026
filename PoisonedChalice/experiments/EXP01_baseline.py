
"""
EXPERIMENT 01: Official Baseline Suite
Methods: Loss (Perplexity), Min-K% Prob, and PAC (Polarized-Augment Calibration).
Goal: 100% Alignment with official competition baseline scripts.
Reference: PoisonedChalice/Pac.py, PoisonedChalice/MinKProbAttack.py, PoisonedChalice/Loss.py
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import argparse
import json
import random
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type, Any, Tuple
from abc import ABC, abstractmethod

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
# Model Loading (Standardized)
# ============================================================================

def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=True, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False, trust_remote_code=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype, device_map="auto"
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)
        if torch.cuda.is_available(): model = model.to("cuda")
    model.eval()
    return model, tokenizer

# ============================================================================
# Base Attack Class
# ============================================================================

class MIAttack(ABC):
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def name(self) -> str: pass

    @abstractmethod
    def compute_scores(self, texts: List[str]) -> List[float]: pass

# ============================================================================
# 1. Official Loss Attack
# ============================================================================

class LossAttack(MIAttack):
    @property
    def name(self) -> str: return "loss"
    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="Official Loss Attack"):
            try:
                inputs = self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                scores.append(-outputs.loss.item())
            except: scores.append(np.nan)
        return scores

# ============================================================================
# 2. Official Min-K% Prob Attack
# ============================================================================

class MinKProbAttack(MIAttack):
    @property
    def name(self) -> str: return "mkp"
    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        k = 0.2
        for text in tqdm(texts, desc="Official Min-K Attack"):
            try:
                inputs = self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    log_probs = log_softmax(outputs.logits, dim=-1)
                    
                    token_log_probs = []
                    for i in range(inputs["input_ids"].shape[1] - 1):
                        token_id = inputs["input_ids"][0, i + 1]
                        token_log_probs.append(log_probs[0, i, token_id].item())
                    
                    sorted_probs = np.sort(token_log_probs)
                    k_len = max(1, int(len(sorted_probs) * k))
                    scores.append(np.mean(sorted_probs[:k_len]))
            except: scores.append(np.nan)
        return scores

# ============================================================================
# 3. Official PAC Attack (Polarized-Augment Calibration)
# ============================================================================

class PACAttack(MIAttack):
    @property
    def name(self) -> str: return "pac"

    def compute_polarized_distance(self, log_probs: np.ndarray) -> float:
        if len(log_probs) == 0: return 0.0
        sorted_p = np.sort(log_probs)
        far_count, near_count = 5, 30
        
        # Scaling logic from Pac.py
        list_length = len(sorted_p)
        if near_count + far_count > list_length:
            scale = list_length / (near_count + far_count)
            near_count = max(1, int(near_count * scale))
            far_count = max(1, int(far_count * scale))
            
        return np.mean(sorted_p[::-1][:far_count]) - np.mean(sorted_p[:near_count])

    def get_token_probs(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, max_length=self.args.max_length, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_probs = log_softmax(outputs.logits, dim=-1)
            token_probs = []
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_probs.append(log_probs[0, i, inputs["input_ids"][0, i+1]].item())
            return np.array(token_probs)

    def mutate(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        m_ratio, n_samples = 0.3, 5
        adjacent = []
        for _ in range(n_samples):
            swapped = list(tokens)
            for _ in range(int(m_ratio * len(swapped))):
                if len(swapped) >= 2:
                    i, j = random.sample(range(len(swapped)), 2)
                    swapped[i], swapped[j] = swapped[j], swapped[i]
            adjacent.append(self.tokenizer.decode(swapped, skip_special_tokens=True))
        return adjacent

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="Official PAC Attack"):
            try:
                orig_probs = self.get_token_probs(text)
                orig_pd = self.compute_polarized_distance(orig_probs)
                
                mut_pds = [self.compute_polarized_distance(self.get_token_probs(m)) for m in self.mutate(text)]
                scores.append(orig_pd - np.mean(mut_pds))
            except: scores.append(np.nan)
        return scores

# ============================================================================
# Main Execution
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = load_model(args.model_name)
        self.attacks = [LossAttack, MinKProbAttack, PACAttack]

    def run(self):
        # Load Data
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        for s in subsets:
            path = os.path.join(self.args.dataset, s)
            if os.path.exists(path):
                ds = load_from_disk(path)
                dfs.append(ds["test"].to_pandas() if "test" in ds else ds.to_pandas())
        if not dfs: raise ValueError("Dataset not found!")
        df = pd.concat(dfs).sample(frac=self.args.sample_fraction, random_state=42)
        df['is_member'] = df['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        # Run Attacks
        for atk_cls in self.attacks:
            atk = atk_cls(self.args, self.model, self.tokenizer)
            df[f"{atk.name}_score"] = atk.compute_scores(df['content'].tolist())
            auc = roc_auc_score(df['is_member'], df[f"{atk.name}_score"].fillna(0))
            print(f"AUC ({atk.name}): {auc:.4f}")
        
        os.makedirs("results", exist_ok=True)
        df.to_parquet(f"results/EXP01_Official_Baseline.parquet", index=False)

if __name__ == "__main__":
    setup_environment()
    class Args:
        model_name = "bigcode/starcoder2-3b"
        dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        max_length = 2048
        sample_fraction = 0.1
    Experiment(Args).run()
