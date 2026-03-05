
"""
EXPERIMENT 02: Min-K%++ (SOTA 2025)
Method: Min-K% with Z-score normalization (Vocabulary statistics).
Goal: Outperform standard Min-K% by accounting for token prediction difficulty.
Reference: "Min-K%++: Improved Membership Inference via Vocabulary-wide Statistics" (ICLR 2025).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import argparse
import json
import random
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type, Any

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Kaggle & Environment Setup (Zero-Setup System)
# ============================================================================

def setup_environment():
    """Handles auto-install and Hugging Face login."""
    print("--- Environment Setup Starting ---")
    
    # 1. Auto-install dependencies if running in Kaggle/Colab
    try:
        import transformers
        import datasets
    except ImportError:
        print("Required libraries not found. Installing now...")
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub")
        print("Libraries installed successfully.")

    # 2. Login to Hugging Face
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Login Note: {e} (Standard if running locally or secret missing)")

    # 3. Check Dataset Path
    kaggle_path = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
    if os.path.exists(kaggle_path):
        print(f"Dataset found at standard Kaggle path: {kaggle_path}")
    else:
        print("Warning: Expected dataset path not found. Please ensure the dataset is added to your Kaggle notebook.")
    
    print("--- Environment Setup Complete ---")

# ============================================================================
# Model Loading
# ============================================================================

def load_model_from_directory(model_path: str):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=True, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, use_fast=False, trust_remote_code=True)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype, 
            device_map="auto"
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.eval()
    return model, tokenizer

# ============================================================================
# Min-K%++ Attack (Z-Score Normalization)
# ============================================================================

class MinKPlusPlusAttack:
    """
    Min-K%++ Attack:
    Normalizes token probabilities using the mean and variance of the model's 
    vocabulary distribution at each step.
    
    Score = Mean(Top-K% lowest Z-Scores)
    """
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.k = 0.2
        self.window_size = args.max_length if args.max_length != -1 else 256
        self.use_sliding_window = args.mink_pp_sliding_window
        self.sliding_window_size = args.mink_pp_window_size
        print(f"[Min-K++] Config: k={self.k}, window_size={self.window_size}")

    @property
    def name(self) -> str:
        return "mink_plus_plus"

    def calculate_z_scores_truncation(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return np.array([])
            
        inputs = self.tokenizer(
            text, 
            max_length=self.window_size, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            log_probs = log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # Probability-weighted mean and variance over the vocabulary (paper definition)
            mu = (probs * log_probs).sum(dim=-1)
            var = (probs * (log_probs ** 2)).sum(dim=-1) - (mu ** 2)
            sigma = torch.sqrt(torch.clamp(var, min=1e-12))

            target_z_scores = []
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1]
                token_lp = log_probs[0, i, token_id]
                z_score = (token_lp - mu[0, i]) / sigma[0, i]
                target_z_scores.append(z_score.item())

        return np.array(target_z_scores)

    def calculate_z_scores_sliding(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return np.array([])

        encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encodings.input_ids[0]
        all_z_scores = []

        for i in range(0, len(input_ids), self.sliding_window_size):
            chunk_ids = input_ids[i: i + self.sliding_window_size]
            if len(chunk_ids) < 2:
                continue

            chunk_tensor = chunk_ids.unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(chunk_tensor)
                logits = outputs.logits
                log_probs = log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                mu = (probs * log_probs).sum(dim=-1)
                var = (probs * (log_probs ** 2)).sum(dim=-1) - (mu ** 2)
                sigma = torch.sqrt(torch.clamp(var, min=1e-12))

                for j in range(chunk_ids.shape[0] - 1):
                    token_id = chunk_ids[j + 1]
                    token_lp = log_probs[0, j, token_id]
                    z_score = (token_lp - mu[0, j]) / sigma[0, j]
                    all_z_scores.append(z_score.item())

        return np.array(all_z_scores)

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"\nComputing {self.name} scores...")
        scores = []
        for text in tqdm(texts, desc="Z-Score Calculation"):
            try:
                if self.use_sliding_window:
                    z_scores = self.calculate_z_scores_sliding(text)
                else:
                    z_scores = self.calculate_z_scores_truncation(text)
                
                if len(z_scores) == 0:
                    scores.append(np.nan)
                    continue
                
                # 2. Sort Z-Scores (descending or ascending?)
                # We want the "least likely" tokens (lowest probability/z-score)
                # Lower Z-score = More unexpected = Non-member behavior?
                # Wait. If member, the model should assign HIGH probability (High Z-score).
                # If non-member, low prob (Low Z-score).
                # Min-K checks the *minimum* probabilities. If the minimums are high, it's a member.
                
                sorted_z = np.sort(z_scores) # Ascending: [Low Z, ..., High Z]
                
                # Take top-k% lowest values
                k_len = max(1, int(len(sorted_z) * self.k))
                min_k_z = sorted_z[:k_len]
                
                # Average them
                score = np.mean(min_k_z)
                scores.append(score)
            except Exception as e:
                print(f"Error: {e}")
                scores.append(np.nan)
        return scores

# ============================================================================
# Experiment Orchestrator
# ============================================================================

class MIAExperiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        self.model, self.tokenizer = load_model_from_directory(args.model_name)

    def load_datasets(self) -> pd.DataFrame:
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading dataset from: {self.args.dataset} (Local: {is_local})")
        for subset in subsets:
            if is_local:
                subset_path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(subset_path): continue
                ds = load_from_disk(subset_path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            dfs.append(ds.to_pandas())
        ds = pd.concat(dfs, ignore_index=True)
        ds['is_member'] = ds['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        if self.args.sample_fraction < 1.0:
            ds = ds.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(ds)} examples")
        return ds

    def run(self):
        df = self.load_datasets()
        attacker = MinKPlusPlusAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        
        df[f"{attacker.name}_score"] = scores
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP02_{self.args.model_name.replace('/', '_')}_{timestamp}"
        output_file = self.output_dir / f"{exp_id}.parquet"
        df.to_parquet(output_file, index=False)
        print(f"Saved to {output_file}")
        
        # AUC
        y_true = df['is_member']
        y_scores = df[f"{attacker.name}_score"].fillna(-999)
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"AUC: {auc:.4f}")
        except:
            print("AUC computation failed (maybe all one class?)")

if __name__ == "__main__":
    setup_environment()

    # ==========================================
    # Configuration (Edit these directly)
    # ==========================================
    class Args:
        model_name = "bigcode/starcoder2-3b"
        # Try Kaggle path first, else fall back to HF hub
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
            
        sample_fraction = 0.1  # 10% of data
        output_dir = "results"
        max_length = 2048
        seed = 42
        mink_pp_sliding_window = True
        mink_pp_window_size = 256

    print(f"\n[Config] Model: {Args.model_name}")
    print(f"[Config] Dataset: {Args.dataset}")
    print(f"[Config] Sample Fraction: {Args.sample_fraction}")
    
    MIAExperiment(Args).run()
