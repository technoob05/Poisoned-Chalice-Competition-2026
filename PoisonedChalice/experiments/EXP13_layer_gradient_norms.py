
"""
EXPERIMENT 13: Layer-wise Gradient Norm Analysis (White-Box)
Method: Calculate the L2 Norm of Gradients for specific layers (blocks) across the model.
Goal: Map the "Optimization Trajectory" to see where the memorization resides.
      - Member data: Low gradients in semantic/late layers.
      - Non-member: High gradients throughout.
Reference: "Gradient-based Membership Inference".
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
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
    for param in model.parameters():
        param.requires_grad = True
        
    return model, tokenizer

# ============================================================================
# Layer-wise Gradient Norm Attack
# ============================================================================

class LayerGradientAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.layer_selection = self._identify_layers()

    def _identify_layers(self) -> Dict[str, torch.nn.Module]:
        """
        Identifies key layers to monitor based on StarCoder2 architecture (Llama-like).
        """
        layers = {}
        # 1. Input Embeddings
        layers['embedding'] = self.model.get_input_embeddings()
        
        # 2. Transformer Blocks
        # StarCoder2 has many layers. Let's pick 4 evenly spaced ones.
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            indices = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
            for idx in indices:
                layers[f'layer_{idx}'] = self.model.model.layers[idx]
        
        # 3. Final Head
        layers['head'] = self.model.get_output_embeddings()
        
        return layers

    @property
    def name(self):
        return "layer_grads"

    def compute_layer_norms(self, text: str) -> Dict[str, float]:
        if not text: return {}
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            
            norms = {}
            for name, layer in self.layer_selection.items():
                # Accumulate gradient norm for all parameters in this layer cluster
                layer_grad_norms = []
                for p in layer.parameters():
                    if p.grad is not None:
                        layer_grad_norms.append(p.grad.norm(2).item())
                
                if layer_grad_norms:
                    # Root mean square of norms or just sum? 
                    # Let's take the norm of the vector of norms.
                    norms[name] = np.sqrt(np.sum(np.square(layer_grad_norms)))
                else:
                    norms[name] = np.nan
            
            self.model.zero_grad()
            return norms
        except Exception:
            self.model.zero_grad()
            return {name: np.nan for name in self.layer_selection.keys()}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"Computing {self.name} norms...")
        all_norms = []
        
        for text in tqdm(texts, desc="Layer Gradient Analysis"):
            norms = self.compute_layer_norms(text)
            all_norms.append(norms)
            
        return pd.DataFrame(all_norms)

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
        attacker = LayerGradientAttack(self.args, self.model, self.tokenizer)
        norms_df = attacker.compute_scores(df['content'].tolist())
        
        # For the ranking, we can use the sum of negative norms as a heuristic
        # but the real value is the individual layer features for the ensemble.
        df = pd.concat([df.reset_index(drop=True), norms_df.reset_index(drop=True)], axis=1)
        
        # Heuristic score for current table: Average Negative Norms (higher is member)
        df[f"{attacker.name}_score"] = -norms_df.mean(axis=1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP13_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {fname}")
        
        try:
            auc = roc_auc_score(df['is_member'], df[f"{attacker.name}_score"].fillna(-999))
            print(f"AUC ({attacker.name} Heuristic): {auc:.4f}")
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
        sample_fraction = 0.05
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP13] Model: {Args.model_name}")
    Experiment(Args).run()
