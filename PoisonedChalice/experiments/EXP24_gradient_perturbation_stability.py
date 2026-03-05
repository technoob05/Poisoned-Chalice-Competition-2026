"""
EXPERIMENT 24: Multi-pass Gradient Stability with Perturbations (White-Box)
Method: Measure gradient stability across input perturbations (token substitution).
Goal: Members have stable gradients under perturbation, non-members vary significantly.
Innovation:
    - Original text + N perturbed versions (random token substitution)
    - Compare gradient norms across versions
    - Stability metric: coefficient of variation (CV)
    - Low CV = member (stable), high CV = non-member (sensitive)
Usage: python EXP24_gradient_perturbation_stability.py or copy to Kaggle
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
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
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
        
    return model, tokenizer

class PerturbedGradientStabilityAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, 'max_length', 2048)
        self.num_perturbations = 4  # Number of perturbed versions
        self.perturb_ratio = 0.05  # 5% of tokens to substitute

    @property
    def name(self):
        return "perturbed_grad_stability"

    def perturb_tokens(self, token_ids: List[int]) -> List[int]:
        """Randomly substitute tokens"""
        token_ids = token_ids.copy()
        num_to_perturb = max(1, int(len(token_ids) * self.perturb_ratio))
        
        # Avoid special tokens (first and last)
        perturbable_positions = list(range(1, len(token_ids) - 1))
        if not perturbable_positions:
            return token_ids
        
        positions = random.sample(
            perturbable_positions, 
            min(num_to_perturb, len(perturbable_positions))
        )
        
        vocab_size = len(self.tokenizer)
        for pos in positions:
            # Replace with random token
            token_ids[pos] = random.randint(0, vocab_size - 1)
        
        return token_ids

    def compute_grad_norm(self, input_ids: torch.Tensor) -> float:
        """Compute gradient norm for given input"""
        try:
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            
            embed_layer = self.model.get_input_embeddings()
            if embed_layer.weight.grad is None:
                return np.nan
            
            norm = embed_layer.weight.grad.norm(2).item()
            self.model.zero_grad()
            return norm
        except Exception:
            self.model.zero_grad()
            return np.nan

    def compute_stability(self, text: str) -> Dict[str, float]:
        """Compute gradient stability across perturbations"""
        if not text:
            return {'stability_cv': np.nan, 'mean_norm': np.nan, 'std_norm': np.nan}
        
        try:
            # Original tokens
            original_tokens = self.tokenizer.encode(
                text, 
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True
            )
            
            if len(original_tokens) < 5:  # Too short
                return {'stability_cv': np.nan, 'mean_norm': np.nan, 'std_norm': np.nan}
            
            gradient_norms = []
            
            # Original gradient norm
            input_ids = torch.tensor([original_tokens]).to(self.model.device)
            norm = self.compute_grad_norm(input_ids)
            if not np.isnan(norm):
                gradient_norms.append(norm)
            
            # Perturbed versions
            for _ in range(self.num_perturbations):
                perturbed_tokens = self.perturb_tokens(original_tokens)
                input_ids = torch.tensor([perturbed_tokens]).to(self.model.device)
                norm = self.compute_grad_norm(input_ids)
                if not np.isnan(norm):
                    gradient_norms.append(norm)
            
            if len(gradient_norms) < 2:
                return {'stability_cv': np.nan, 'mean_norm': np.nan, 'std_norm': np.nan}
            
            mean_norm = np.mean(gradient_norms)
            std_norm = np.std(gradient_norms)
            
            # Coefficient of Variation (CV): std/mean
            # Lower CV = more stable = member
            cv = std_norm / mean_norm if mean_norm > 0 else np.nan
            
            return {
                'stability_cv': cv,
                'mean_norm': mean_norm,
                'std_norm': std_norm
            }
            
        except Exception:
            return {'stability_cv': np.nan, 'mean_norm': np.nan, 'std_norm': np.nan}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} scores...")
        all_stats = []
        
        for text in tqdm(texts, desc="Perturbed Gradient Stability"):
            stats = self.compute_stability(text)
            all_stats.append(stats)
        
        df = pd.DataFrame(all_stats)
        
        # Lower CV = more stable = member, so negate
        df['score'] = -df['stability_cv']
        
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
        attacker = PerturbedGradientStabilityAttack(self.args, self.model, self.tokenizer)
        stats_df = attacker.compute_scores(df['content'].tolist())
        
        # Merge
        df = pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.args.model_name.replace('/', '_')
        fname = f"EXP24_{model_name_safe}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Saved to {fname}")
        
        # Evaluate
        try:
            auc = roc_auc_score(df['is_member'], df['score'].fillna(-999))
            print(f"\n{'='*50}")
            print(f"EXP24 - Perturbed Gradient Stability")
            print(f"{'='*50}")
            print(f"AUC Score: {auc:.4f}")
            print(f"Mean CV (Member): {df[df['is_member']==1]['stability_cv'].mean():.4f}")
            print(f"Mean CV (Non-member): {df[df['is_member']==0]['stability_cv'].mean():.4f}")
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
        sample_fraction = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP24] Model: {Args.model_name}")
    Experiment(Args).run()
