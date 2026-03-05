"""
EXPERIMENT 21: Gradient Norm with Sliding Window (White-Box Enhanced)
Method: Compute gradient norm over non-overlapping chunks to handle long sequences.
Goal: Improve EXP11 by avoiding truncation artifacts and capturing local memorization patterns.
Innovation:
    - Chunk-based gradient analysis (256 tokens per window)
    - Aggregate statistics: mean, max, std of gradient norms
    - Better signal for long code samples
Usage: python EXP21_gradient_sliding_window.py or copy to Kaggle
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

class SlidingWindowGradientAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = 256  # Tokens per window
        self.max_windows = 8  # Max windows to process

    @property
    def name(self):
        return "grad_sliding_window"

    def compute_window_grad_norm(self, input_ids: torch.Tensor) -> float:
        """Compute gradient norm for a single window"""
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

    def compute_sliding_grad_norms(self, text: str) -> Dict[str, float]:
        """Compute gradient norms over sliding windows"""
        if not text:
            return {'mean': np.nan, 'max': np.nan, 'std': np.nan, 'min': np.nan}
        
        try:
            # Tokenize full text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= self.window_size:
                # Single window
                input_ids = torch.tensor([tokens]).to(self.model.device)
                norm = self.compute_window_grad_norm(input_ids)
                return {'mean': norm, 'max': norm, 'std': 0.0, 'min': norm}
            
            # Multiple non-overlapping windows
            window_norms = []
            num_windows = min(len(tokens) // self.window_size, self.max_windows)
            
            for i in range(num_windows):
                start = i * self.window_size
                end = start + self.window_size
                window_tokens = tokens[start:end]
                
                input_ids = torch.tensor([window_tokens]).to(self.model.device)
                norm = self.compute_window_grad_norm(input_ids)
                
                if not np.isnan(norm):
                    window_norms.append(norm)
            
            if not window_norms:
                return {'mean': np.nan, 'max': np.nan, 'std': np.nan, 'min': np.nan}
            
            return {
                'mean': np.mean(window_norms),
                'max': np.max(window_norms),
                'std': np.std(window_norms),
                'min': np.min(window_norms)
            }
        except Exception:
            return {'mean': np.nan, 'max': np.nan, 'std': np.nan, 'min': np.nan}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} scores...")
        all_stats = []
        
        for text in tqdm(texts, desc="Sliding Window Gradient"):
            stats = self.compute_sliding_grad_norms(text)
            all_stats.append(stats)
        
        df = pd.DataFrame(all_stats)
        # Members have LOW gradient norms, so negate for ranking
        df['score'] = -(df['mean'] + 0.5 * df['max'] - 0.3 * df['std'])
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
        attacker = SlidingWindowGradientAttack(self.args, self.model, self.tokenizer)
        stats_df = attacker.compute_scores(df['content'].tolist())
        
        # Merge statistics
        df = pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.args.model_name.replace('/', '_')
        fname = f"EXP21_{model_name_safe}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Saved to {fname}")
        
        # Evaluate
        try:
            auc = roc_auc_score(df['is_member'], df['score'].fillna(-999))
            print(f"\n{'='*50}")
            print(f"EXP21 - Sliding Window Gradient Norm")
            print(f"{'='*50}")
            print(f"AUC Score: {auc:.4f}")
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
        seed = 42

    print(f"[EXP21] Model: {Args.model_name}")
    Experiment(Args).run()
