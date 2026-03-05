"""
EXPERIMENT 23: Gradient Direction Variance (White-Box)
Method: Measure gradient direction consistency across multiple forward passes with dropout.
Goal: Members should have stable gradient directions (converged), non-members have varying gradients.
Innovation:
    - Multiple forward passes with dropout enabled
    - Compute cosine similarity between gradient vectors
    - Low variance = member (stable), high variance = non-member (uncertain)
Reference: Gradient stability as memorization signal
Usage: python EXP23_gradient_direction_variance.py or copy to Kaggle
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def setup_environment():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
    except Exception as e:
        print(f"[HF] Note: {e}")

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
    
    # Enable dropout for variance estimation
    model.train()  # Important: enables dropout
    for param in model.parameters():
        param.requires_grad = True
        
    return model, tokenizer

class GradientDirectionVarianceAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, 'max_length', 2048)
        self.num_passes = 5  # Number of forward passes

    @property
    def name(self):
        return "grad_direction_variance"

    def extract_gradient_vector(self) -> torch.Tensor:
        """Extract flattened gradient vector from embedding layer"""
        embed_layer = self.model.get_input_embeddings()
        if embed_layer.weight.grad is None:
            return None
        
        # Flatten gradient to 1D vector
        grad_vector = embed_layer.weight.grad.flatten()
        return grad_vector

    def compute_gradient_variance(self, text: str) -> Dict[str, float]:
        """Compute gradient direction variance over multiple passes"""
        if not text:
            return {'cosine_std': np.nan, 'norm_std': np.nan, 'stability': np.nan}
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            ).to(self.model.device)
            
            gradient_vectors = []
            gradient_norms = []
            
            # Multiple forward passes with dropout
            for _ in range(self.num_passes):
                self.model.zero_grad()
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                
                grad_vec = self.extract_gradient_vector()
                if grad_vec is not None:
                    # Store normalized gradient direction
                    norm = grad_vec.norm(2).item()
                    if norm > 0:
                        gradient_vectors.append((grad_vec / norm).cpu())
                        gradient_norms.append(norm)
            
            self.model.zero_grad()
            
            if len(gradient_vectors) < 2:
                return {'cosine_std': np.nan, 'norm_std': np.nan, 'stability': np.nan}
            
            # Compute pairwise cosine similarities
            cosine_sims = []
            for i in range(len(gradient_vectors)):
                for j in range(i + 1, len(gradient_vectors)):
                    cos_sim = F.cosine_similarity(
                        gradient_vectors[i].unsqueeze(0),
                        gradient_vectors[j].unsqueeze(0)
                    ).item()
                    cosine_sims.append(cos_sim)
            
            # Statistics
            cosine_mean = np.mean(cosine_sims)
            cosine_std = np.std(cosine_sims)
            norm_std = np.std(gradient_norms)
            
            # Stability score: high cosine similarity + low std = member
            # Invert so higher = more stable = member
            stability = cosine_mean - cosine_std
            
            return {
                'cosine_std': cosine_std,
                'norm_std': norm_std,
                'stability': stability
            }
            
        except Exception as e:
            return {'cosine_std': np.nan, 'norm_std': np.nan, 'stability': np.nan}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} scores...")
        all_stats = []
        
        for text in tqdm(texts, desc="Gradient Direction Variance"):
            stats = self.compute_gradient_variance(text)
            all_stats.append(stats)
        
        df = pd.DataFrame(all_stats)
        
        # Higher stability = member
        df['score'] = df['stability']
        
        return df

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
                if not os.path.exists(path):
                    continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys():
                    ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            
            dfs.append(ds.to_pandas())
        
        if not dfs:
            raise ValueError("No data found!")
        
        df = pd.concat(dfs, ignore_index=True)
        df['is_member'] = df['membership'].apply(lambda x: 1 if x == 'member' else 0)
        
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(df)} rows.")
        
        return df

    def run(self):
        df = self.load_data()
        attacker = GradientDirectionVarianceAttack(self.args, self.model, self.tokenizer)
        stats_df = attacker.compute_scores(df['content'].tolist())
        
        # Merge
        df = pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.args.model_name.replace('/', '_')
        fname = f"EXP23_{model_name_safe}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Saved to {fname}")
        
        # Evaluate
        try:
            auc = roc_auc_score(df['is_member'], df['score'].fillna(-999))
            print(f"\n{'='*50}")
            print(f"EXP23 - Gradient Direction Variance")
            print(f"{'='*50}")
            print(f"AUC Score: {auc:.4f}")
            print(f"Mean Stability (Member): {df[df['is_member']==1]['stability'].mean():.4f}")
            print(f"Mean Stability (Non-member): {df[df['is_member']==0]['stability'].mean():.4f}")
            print(f"{'='*50}\n")
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
        sample_fraction = 0.05  # Lower fraction due to multiple passes
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP23] Model: {Args.model_name}")
    Experiment(Args).run()
