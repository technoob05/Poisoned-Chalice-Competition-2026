"""
EXPERIMENT 22: Selective Layer Gradient Ensemble (White-Box)
Method: Extract gradient norms from strategic layers and ensemble with learned weights.
Goal: Improve EXP13 by selecting high-signal layers and combining optimally.
Innovation:
    - Target 5 strategic layers: embedding, early (layer 0), middle (layer 16), late (layer 30), output head
    - Use correlation analysis to weight layers
    - Ensemble with rank-based combination
Usage: python EXP22_selective_layer_ensemble.py or copy to Kaggle
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
from scipy.stats import rankdata
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

class SelectiveLayerGradientAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, 'max_length', 2048)
        self.selected_layers = self._select_strategic_layers()

    def _select_strategic_layers(self) -> Dict[str, torch.nn.Module]:
        """Select 5 strategic layers for gradient analysis"""
        layers = {}
        
        # 1. Embedding layer
        layers['embedding'] = self.model.get_input_embeddings()
        
        # 2-5. Transformer blocks at strategic positions
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            
            # Early layer (first 10%)
            layers['early'] = self.model.model.layers[0]
            
            # Middle layer (50%)
            layers['middle'] = self.model.model.layers[num_layers // 2]
            
            # Late layer (85%)
            layers['late'] = self.model.model.layers[int(num_layers * 0.85)]
            
            print(f"[*] Selected layers: 0, {num_layers//2}, {int(num_layers*0.85)} (Total: {num_layers})")
        
        # 5. Output head
        layers['head'] = self.model.get_output_embeddings()
        
        return layers

    @property
    def name(self):
        return "selective_layer_grad"

    def compute_layer_grads(self, text: str) -> Dict[str, float]:
        """Compute gradient norms for selected layers"""
        if not text:
            return {k: np.nan for k in self.selected_layers.keys()}
        
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
            
            layer_norms = {}
            for layer_name, layer_module in self.selected_layers.items():
                grad_norms = []
                for param in layer_module.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm(2).item())
                
                if grad_norms:
                    # RMS of all parameter gradients in this layer
                    layer_norms[layer_name] = np.sqrt(np.mean(np.square(grad_norms)))
                else:
                    layer_norms[layer_name] = np.nan
            
            self.model.zero_grad()
            return layer_norms
            
        except Exception:
            self.model.zero_grad()
            return {k: np.nan for k in self.selected_layers.keys()}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} scores...")
        all_layer_norms = []
        
        for text in tqdm(texts, desc="Selective Layer Gradient"):
            norms = self.compute_layer_grads(text)
            all_layer_norms.append(norms)
        
        df = pd.DataFrame(all_layer_norms)
        
        # Negate norms (lower norm = member = higher score)
        for col in df.columns:
            df[f"{col}_score"] = -df[col]
        
        # Rank-based ensemble
        # Convert each layer score to percentile
        rank_scores = {}
        for col in self.selected_layers.keys():
            score_col = f"{col}_score"
            if score_col in df.columns:
                ranks = rankdata(df[score_col].fillna(df[score_col].min()), method='average')
                rank_scores[col] = ranks / len(ranks)
        
        # Weighted combination (empirical weights based on expected signal strength)
        weights = {
            'embedding': 0.25,  # High signal
            'early': 0.10,      # Low signal
            'middle': 0.15,     # Moderate signal
            'late': 0.30,       # High signal (semantic)
            'head': 0.20        # Moderate-high signal
        }
        
        ensemble_score = np.zeros(len(df))
        for layer_name, weight in weights.items():
            if layer_name in rank_scores:
                ensemble_score += weight * rank_scores[layer_name]
        
        df['ensemble_score'] = ensemble_score
        
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
        attacker = SelectiveLayerGradientAttack(self.args, self.model, self.tokenizer)
        layer_df = attacker.compute_scores(df['content'].tolist())
        
        # Merge
        df = pd.concat([df.reset_index(drop=True), layer_df.reset_index(drop=True)], axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.args.model_name.replace('/', '_')
        fname = f"EXP22_{model_name_safe}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Saved to {fname}")
        
        # Evaluate
        try:
            # Individual layer AUCs
            print(f"\n{'='*50}")
            print(f"EXP22 - Selective Layer Gradient Ensemble")
            print(f"{'='*50}")
            
            for layer in attacker.selected_layers.keys():
                score_col = f"{layer}_score"
                if score_col in df.columns:
                    auc = roc_auc_score(df['is_member'], df[score_col].fillna(-999))
                    print(f"{layer:12s}: AUC = {auc:.4f}")
            
            # Ensemble AUC
            ensemble_auc = roc_auc_score(df['is_member'], df['ensemble_score'])
            print(f"{'='*50}")
            print(f"{'ENSEMBLE':12s}: AUC = {ensemble_auc:.4f}")
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

    print(f"[EXP22] Model: {Args.model_name}")
    Experiment(Args).run()
