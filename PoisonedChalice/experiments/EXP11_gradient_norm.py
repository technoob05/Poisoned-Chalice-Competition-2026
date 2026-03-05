
"""
EXPERIMENT 11: Gradient Norm Analysis (White-Box) - RESEARCHER MODE v2.0
Method: Calculate the L2 Norm of the Gradients w.r.t Embedding Layer.
Goal: Detect "Flat Minima" and analyze memorization across languages.
      - Member data (memorized) -> Optimization converged -> Low Gradient Norm.
      - Non-member data -> High Gradient Norm.
New in Researcher Mode:
      - Per-subset (Language) performance breakdown.
      - Automated Visualization (ROC Curves, Score Distributions).
      - Detailed JSON Reporting for post-hoc analysis.
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

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
        print("[HF] Authenticated successfully.")
    except Exception as e:
        print(f"[HF] Note: {e}")

    print("--- Environment Ready ---\n")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"[*] Loading model: {model_path}")
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
        
    print(f"[*] Model loaded on {model.device} with {dtype} precision.")
    return model, tokenizer

# ============================================================================
# Gradient Norm Attack
# ============================================================================

class GradientNormAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "gradient_norm"

    def compute_grad_norm(self, text: str) -> float:
        if not text: return np.nan
        try:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True).to(self.model.device)
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
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

    def compute_scores(self, texts: List[str]) -> List[float]:
        print(f"[*] Analyzing {len(texts)} samples using {self.name}...")
        scores = []
        for text in tqdm(texts, desc="[GRAD] Batch Processing"):
            grad_norm = self.compute_grad_norm(text)
            if np.isnan(grad_norm):
                scores.append(np.nan)
            else:
                # Members: Low Norm -> High Score.
                scores.append(-grad_norm) 
        return scores

# ============================================================================
# Analytics & Plotting
# ============================================================================

class VisualAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_roc_curves(self, df: pd.DataFrame, score_col: str, title: str):
        plt.figure(figsize=(8, 6))
        
        # Overall ROC
        y_true = df['is_member']
        y_score = df[score_col].fillna(df[score_col].min())
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"Overall (AUC={auc:.4f})", linewidth=3, color='black')
        
        # Per-Subset ROC
        subsets = df['subset'].unique()
        colors = sns.color_palette("husl", len(subsets))
        for i, subset in enumerate(subsets):
            sub_df = df[df['subset'] == subset]
            if len(sub_df['is_member'].unique()) > 1:
                y_sub = sub_df['is_member']
                y_sub_score = sub_df[score_col].fillna(sub_df[score_col].min())
                f, t, _ = roc_curve(y_sub, y_sub_score)
                a = roc_auc_score(y_sub, y_sub_score)
                plt.plot(f, t, label=f"{subset} (AUC={a:.4f})", alpha=0.7, color=colors[i])
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {title}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve_detailed.png", dpi=200)
        plt.close()

    def plot_score_distributions(self, df: pd.DataFrame, score_col: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=score_col, hue='membership', bins=50, kde=True, element="step")
        plt.title("Score Distribution: Members vs Non-Members")
        plt.xlabel("MIA Score (Negative Grad Norm)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_distribution.png", dpi=200)
        plt.close()

# ============================================================================
# Experiment Runners
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir) / f"EXP11_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)
        self.analyzer = VisualAnalyzer(self.output_dir)

    def load_data(self):
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"[*] Loading data from {self.args.dataset}...")
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
            print(f"[*] Sampled {len(df)} rows for research analysis.")
        return df

    def run(self):
        df = self.load_data()
        attacker = GradientNormAttack(self.args, self.model, self.tokenizer)
        scores = attacker.compute_scores(df['content'].tolist())
        df[f"{attacker.name}_score"] = scores
        
        # Save Parquet
        fname = "results.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        
        # --- Deep Analytics ---
        print("\n" + "="*50)
        print("          RESEARCHER PERFORMANCE REPORT")
        print("="*50)
        
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.args.model_name,
            "overall_auc": 0.0,
            "subset_aucs": {}
        }
        
        # Overall AUC
        overall_auc = roc_auc_score(df['is_member'], df[f"{attacker.name}_score"].fillna(df[f"{attacker.name}_score"].min()))
        report["overall_auc"] = float(overall_auc)
        print(f"OVERALL AUC: {overall_auc:.4f}")
        
        # Subset AUCs
        print("\nPer-Language Performance:")
        print(f"{'Subset':<10} | {'AUC':<10} | {'Samples':<10}")
        print("-"*35)
        for subset in df['subset'].unique():
            sub_df = df[df['subset'] == subset]
            if len(sub_df['is_member'].unique()) > 1:
                y_true = sub_df['is_member']
                y_score = sub_df[f"{attacker.name}_score"].fillna(sub_df[f"{attacker.name}_score"].min())
                auc = roc_auc_score(y_true, y_score)
                report["subset_aucs"][subset] = float(auc)
                print(f"{subset:<10} | {auc:.4f}{' ':<5} | {len(sub_df):<10}")
        
        # --- Visuals ---
        print("\n[*] Generating Visualizations...")
        self.analyzer.plot_roc_curves(df, f"{attacker.name}_score", f"EXP11: {self.args.model_name}")
        self.analyzer.plot_score_distributions(df, f"{attacker.name}_score")
        
        # Save JSON Report
        with open(self.output_dir / "analytics_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        print("\n" + "="*50)
        print(f"[*] All outputs saved to:\n    {self.output_dir.absolute()}")
        print("="*50)

if __name__ == "__main__":
    setup_environment()
    
    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.05 
        output_dir = "research_results"
        max_length = 2048
        seed = 42

    print(f"[EXPERIMENT START] Target: {Args.model_name}")
    Experiment(Args).run()
