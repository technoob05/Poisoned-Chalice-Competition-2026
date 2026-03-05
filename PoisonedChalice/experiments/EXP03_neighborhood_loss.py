
"""
EXPERIMENT 03: Neighborhood Attack (LiRA Approximation)
Method: Self-perturbation calibration.
Goal: Predict membership by comparing target loss to "neighborhood" (mutated) losses.
Concept: If a sample's loss is much lower than its neighbors, it's likely memorized.
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
        if torch.cuda.is_available(): model = model.to("cuda")
    model.eval()
    return model, tokenizer

# ============================================================================
# Neighborhood Attack (LiRA-Approx)
# ============================================================================

class NeighborhoodAttack:
    """
    Neighborhood Attack (LiRA Approximation):
    Estimates the difficulty of a sample by comparing its loss to the loss of 
    neighboring samples (perturbations).
    
    Score = Mean(Loss(Neighbors)) - Loss(Target)
    Higher Score => Member (Target loss is significantly lower than neighbors)
    """
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        # Mutation parameters
        self.m_ratio = 0.3
        self.n_samples = 15 # More samples = Better estimation of local difficulty
        self.max_length = args.max_length if args.max_length != -1 else 256
        print(f"[Neighborhood] Config: ratio={self.m_ratio}, samples={self.n_samples}")

    @property
    def name(self) -> str:
        return "neighborhood"

    def generate_neighbors(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        neighbors = []
        for _ in range(self.n_samples):
            swapped = tokens.copy()
            num_swaps = int(self.m_ratio * len(swapped))
            for _ in range(num_swaps):
                if len(swapped) >= 2:
                    idx1, idx2 = random.sample(range(len(swapped)), 2)
                    swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
            neighbors.append(self.tokenizer.decode(swapped, skip_special_tokens=True))
        return neighbors

    def calculate_loss(self, texts: List[str]) -> np.ndarray:
        losses = []
        for text in texts:
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                losses.append(np.nan)
                continue
                
            try:
                inputs = self.tokenizer(
                    text, 
                    max_length=self.max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                losses.append(outputs.loss.item())
            except:
                losses.append(np.nan)
        return np.array(losses)

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"\nComputing {self.name} scores...")
        scores = []
        for text in tqdm(texts, desc="Neighborhood Analysis"):
            try:
                # 1. Calculate Target Loss
                # We calculate it here to ensure same context/batching as neighbors if needed
                # But separate is fine
                target_loss = self.calculate_loss([text])[0]
                
                # 2. Generate Neighbors
                neighbors = self.generate_neighbors(text)
                
                # 3. Calculate Neighbor Losses
                neighbor_losses = self.calculate_loss(neighbors)
                
                # 4. Compute Score
                # Score = Mean(Neighbor Losses) - Target Loss
                # Positive score -> Target is easier than neighbors -> Member
                if np.isnan(target_loss) or np.all(np.isnan(neighbor_losses)):
                    scores.append(np.nan)
                else:
                    mean_neighbor_loss = np.nanmean(neighbor_losses)
                    score = mean_neighbor_loss - target_loss
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
        attacker = NeighborhoodAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        
        df[f"{attacker.name}_score"] = scores
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"EXP03_{self.args.model_name.replace('/', '_')}_{timestamp}"
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
            print("AUC computation failed")

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

    print(f"\n[Config] Model: {Args.model_name}")
    print(f"[Config] Dataset: {Args.dataset}")
    print(f"[Config] Sample Fraction: {Args.sample_fraction}")
    
    MIAExperiment(Args).run()
