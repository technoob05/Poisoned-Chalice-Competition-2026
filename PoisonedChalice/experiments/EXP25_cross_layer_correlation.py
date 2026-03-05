"""
EXPERIMENT 25: Cross-layer Gradient Correlation Analysis (White-Box)
Method: Measure correlation of gradients between different layers.
Goal: Members show coordinated gradient patterns (high correlation), non-members show chaotic patterns.
Innovation:
    - Extract gradients from 3 strategic layers: early, middle, late
    - Compute pairwise correlation between layer gradient distributions
    - High correlation = coordinated optimization = member
    - Low correlation = unoptimized/uncertain = non-member
Usage: python EXP25_cross_layer_correlation.py or copy to Kaggle
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
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

class CrossLayerGradientCorrelation:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, 'max_length', 2048)
        self.target_layers = self._identify_layers()

    def _identify_layers(self) -> Dict[str, torch.nn.Module]:
        """Identify 3 strategic layers"""
        layers = {}
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            
            # Early (10%), Middle (50%), Late (85%)
            early_idx = max(0, num_layers // 10)
            middle_idx = num_layers // 2
            late_idx = int(num_layers * 0.85)
            
            layers['early'] = self.model.model.layers[early_idx]
            layers['middle'] = self.model.model.layers[middle_idx]
            layers['late'] = self.model.model.layers[late_idx]
            
            print(f"[*] Target layers: {early_idx}, {middle_idx}, {late_idx} (Total: {num_layers})")
        
        return layers

    @property
    def name(self):
        return "cross_layer_correlation"

    def extract_layer_gradients(self, layer_module: torch.nn.Module) -> np.ndarray:
        """Extract gradient vector from a layer module"""
        gradients = []
        for param in layer_module.parameters():
            if param.grad is not None:
                gradients.append(param.grad.flatten().cpu().numpy())
        
        if not gradients:
            return np.array([])
        
        return np.concatenate(gradients)

    def compute_correlation_stats(self, text: str) -> Dict[str, float]:
        """Compute cross-layer gradient correlations"""
        if not text:
            return {
                'early_middle_corr': np.nan,
                'middle_late_corr': np.nan,
                'early_late_corr': np.nan,
                'mean_corr': np.nan
            }
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            ).to(self.model.device)
            
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            # Extract gradient vectors from each layer
            layer_grads = {}
            for layer_name, layer_module in self.target_layers.items():
                grad_vec = self.extract_layer_gradients(layer_module)
                if len(grad_vec) > 0:
                    layer_grads[layer_name] = grad_vec
            
            self.model.zero_grad()
            
            if len(layer_grads) < 3:
                return {
                    'early_middle_corr': np.nan,
                    'middle_late_corr': np.nan,
                    'early_late_corr': np.nan,
                    'mean_corr': np.nan
                }
            
            # Sample gradients for correlation (full vectors too large)
            sample_size = 10000
            sampled_grads = {}
            for name, grad_vec in layer_grads.items():
                if len(grad_vec) > sample_size:
                    indices = np.random.choice(len(grad_vec), sample_size, replace=False)
                    sampled_grads[name] = grad_vec[indices]
                else:
                    sampled_grads[name] = grad_vec
            
            # Align to minimum length
            min_len = min(len(v) for v in sampled_grads.values())
            for name in sampled_grads:
                sampled_grads[name] = sampled_grads[name][:min_len]
            
            # Compute pairwise correlations
            early = sampled_grads.get('early', np.array([]))
            middle = sampled_grads.get('middle', np.array([]))
            late = sampled_grads.get('late', np.array([]))
            
            correlations = {}
            
            if len(early) > 0 and len(middle) > 0:
                corr, _ = pearsonr(early, middle)
                correlations['early_middle_corr'] = corr
            else:
                correlations['early_middle_corr'] = np.nan
            
            if len(middle) > 0 and len(late) > 0:
                corr, _ = pearsonr(middle, late)
                correlations['middle_late_corr'] = corr
            else:
                correlations['middle_late_corr'] = np.nan
            
            if len(early) > 0 and len(late) > 0:
                corr, _ = pearsonr(early, late)
                correlations['early_late_corr'] = corr
            else:
                correlations['early_late_corr'] = np.nan
            
            # Mean correlation (higher = more coordinated = member)
            valid_corrs = [v for v in correlations.values() if not np.isnan(v)]
            correlations['mean_corr'] = np.mean(valid_corrs) if valid_corrs else np.nan
            
            return correlations
            
        except Exception as e:
            return {
                'early_middle_corr': np.nan,
                'middle_late_corr': np.nan,
                'early_late_corr': np.nan,
                'mean_corr': np.nan
            }

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} scores...")
        all_stats = []
        
        for text in tqdm(texts, desc="Cross-layer Correlation"):
            stats = self.compute_correlation_stats(text)
            all_stats.append(stats)
        
        df = pd.DataFrame(all_stats)
        
        # Higher correlation = member
        df['score'] = df['mean_corr']
        
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
        attacker = CrossLayerGradientCorrelation(self.args, self.model, self.tokenizer)
        stats_df = attacker.compute_scores(df['content'].tolist())
        
        # Merge
        df = pd.concat([df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.args.model_name.replace('/', '_')
        fname = f"EXP25_{model_name_safe}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Saved to {fname}")
        
        # Evaluate
        try:
            auc = roc_auc_score(df['is_member'], df['score'].fillna(-999))
            print(f"\n{'='*50}")
            print(f"EXP25 - Cross-layer Gradient Correlation")
            print(f"{'='*50}")
            print(f"AUC Score: {auc:.4f}")
            print(f"\nCorrelation Stats:")
            print(f"Mean Early-Middle (Member): {df[df['is_member']==1]['early_middle_corr'].mean():.4f}")
            print(f"Mean Early-Middle (Non-member): {df[df['is_member']==0]['early_middle_corr'].mean():.4f}")
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
        sample_fraction = 0.05  # Lower due to computation intensity
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP25] Model: {Args.model_name}")
    Experiment(Args).run()
