
"""
EXPERIMENT 14: Internal State Entropy (White-Box) - RESEARCHER MODE
Method: Attention Entropy & Early Exit Loss Analysis.
Goal: Highlight "Token-level Confidence" through Heatmaps.
New in Researcher Mode:
      - Token-level Activation Heatmaps (Visualizing code importance).
      - Per-language AUC breakdown.
      - Integrated Analytics & Plotting.
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ============================================================================
# Kaggle & Environment Setup
# ============================================================================

def setup_environment():
    print("\n" + "="*50)
    print("      RESEARCHER MODE: ENVIRONMENT SETUP")
    print("="*50)
    try:
        import transformers
        import datasets
        import matplotlib
        import seaborn
    except ImportError:
        print("Installing research dependencies...")
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub matplotlib seaborn")
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except: pass
    print("--- Environment Ready ---\n")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # We need attentions for entropy calculation
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        dtype=dtype, 
        device_map="auto",
        output_attentions=True
    )
    model.eval() 
    return model, tokenizer

# ============================================================================
# Internal Entropy Attack
# ============================================================================

class InternalEntropyAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "internal_entropy"

    def compute_entropy_and_meta(self, text: str) -> Tuple[float, List[float], List[str]]:
        if not text: return np.nan, [], []
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Compute Attention Entropy
            # attentions: tuple of (batch, head, seq, seq)
            last_layer_attn = outputs.attentions[-1] # Focus on the last layer
            # Softmax is already applied in Transformer. 
            # Entropy = - sum(p * log(p))
            # We take the entropy of the attention distribution for the last token across all previous tokens
            seq_len = last_layer_attn.shape[-1]
            last_token_attn = last_layer_attn[0, :, -1, :] # (heads, seq)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-12
            entropy = -torch.sum(last_token_attn * torch.log(last_token_attn + epsilon), dim=-1)
            mean_entropy = entropy.mean().item()
            
            # Token-level info for heatmap (tokens and their attention-based confidence)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            token_confidences = last_token_attn.mean(dim=0).cpu().numpy().tolist()
            
            # Score logic: Members have LOW Entropy (Very sharp/focused attention)
            # So Score = -Mean_Entropy
            return -mean_entropy, token_confidences, tokens
            
        except Exception:
            return np.nan, [], []

    def compute_scores(self, texts: List[str]) -> Tuple[List[float], List[Dict]]:
        print(f"[*] Probing internal states of {len(texts)} samples...")
        scores = []
        metadata = []
        for text in tqdm(texts, desc="[ENTROPY] Analysis"):
            score, conf, tokens = self.compute_entropy_and_meta(text)
            scores.append(score)
            metadata.append({"conf": conf, "tokens": tokens})
        return scores, metadata

# ============================================================================
# Analytics & Plotting
# ============================================================================

class VisualAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_token_heatmap(self, tokens: List[str], confs: List[float], filename: str):
        # Truncate for visibility if needed
        limit = 50
        tokens = tokens[-limit:]
        confs = confs[-limit:]
        
        plt.figure(figsize=(15, 3))
        data = np.array(confs).reshape(1, -1)
        ax = sns.heatmap(data, annot=False, cmap="YlGnBu", cbar=True, yticklabels=False)
        plt.xticks(np.arange(len(tokens)) + 0.5, tokens, rotation=90, fontsize=8)
        plt.title("Token-level Confidence (Last Layer Attention)")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=200)
        plt.close()

    def plot_roc_curves(self, df: pd.DataFrame, score_col: str):
        plt.figure(figsize=(8, 6))
        y_true = df['is_member']
        y_score = df[score_col].fillna(df[score_col].min())
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"Overall (AUC={auc:.4f})", linewidth=3, color='black')
        
        subsets = df['subset'].unique()
        for subset in subsets:
            sub_df = df[df['subset'] == subset]
            if len(sub_df['is_member'].unique()) > 1:
                y_sub = sub_df['is_member']
                y_sub_score = sub_df[score_col].fillna(sub_df[score_col].min())
                f, t, _ = roc_curve(y_sub, y_sub_score)
                a = roc_auc_score(y_sub, y_sub_score)
                plt.plot(f, t, label=f"{subset} (AUC={a:.4f})", alpha=0.6)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: Internal Entropy Attack')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves.png", dpi=200)
        plt.close()

# ============================================================================
# Experiment Runners
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir) / f"EXP14_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)
        self.analyzer = VisualAnalyzer(self.output_dir)

    def load_data(self):
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        for subset in subsets:
            if is_local:
                path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(path): continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            sub_df = ds.to_pandas()
            sub_df['subset'] = subset
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)
        df['is_member'] = df['membership'].apply(lambda x: 1 if x == 'member' else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
        return df

    def run(self):
        df = self.load_data()
        attacker = InternalEntropyAttack(self.args, self.model, self.tokenizer)
        scores, meta = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        # Save Parquet
        df.to_parquet(self.output_dir / "results.parquet", index=False)
        
        # --- Analytics ---
        print("\n" + "="*50)
        print("          RESEARCHER PERFORMANCE REPORT (EXP14)")
        print("="*50)
        print(f"OVERALL AUC: {roc_auc_score(df['is_member'], df[f'{attacker.name}_score'].fillna(-99)):.4f}")
        
        # Generate representative heatmaps for top hits
        members = df[df['is_member'] == 1].sort_values(f"{attacker.name}_score", ascending=False).head(3)
        for i, idx in enumerate(members.index):
            m = meta[df.index.get_loc(idx)]
            if m["tokens"]:
                self.analyzer.plot_token_heatmap(m["tokens"], m["conf"], f"heatmap_member_{i}.png")
        
        self.analyzer.plot_roc_curves(df, f"{attacker.name}_score")
        print(f"[*] Visuals saved to {self.output_dir}")

if __name__ == "__main__":
    setup_environment()
    class Args:
        model_name = "bigcode/starcoder2-3b"
        dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        sample_fraction = 0.02 # Heavy attention analysis
        output_dir = "research_results"
        max_length = 1024
        seed = 42
    Experiment(Args).run()
