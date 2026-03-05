"""
NOVEL EXPERIMENT 02: OrthoFuse-MIA — Gradient × Attention Orthogonal Fusion

NOVELTY: First multiplicative fusion of gradient magnitude and attention
    concentration for MIA. Exploits the proven orthogonality between these
    two signal families (EXP11: gradient 0.6472, EXP43: attention 0.6642).

Core Idea:
    EXP27 showed that multiplying two ORTHOGONAL signals (grad × jsd) yields
    AUC 0.6484 > either standalone. But JSD was weak (0.4371).
    
    Attention concentration (conc_mean 0.6508) is a STRONG signal from a
    fundamentally different information channel than gradient norm (0.6472).
    Gradient measures loss-surface geometry; attention measures information
    routing patterns. They are orthogonal by construction.

    Product: score = -(grad_norm × attn_concentration)
    Both signals: lower = member. Product amplifies agreement, suppresses noise.

    Additionally, we apply per-language Z-normalization (EXP41 insight: +0.012)
    to BOTH signals before fusion.

    Key insight from EXP40: products only help when signals are ORTHOGONAL.
    grad×loss failed (correlated); grad×attention should succeed (orthogonal).

Builds on Insights:
    - Insight 8: gradient ceiling ~0.65
    - Insight 16: attention breaks gradient ceiling (0.6642)
    - EXP27: product of orthogonal signals > either alone
    - EXP40: products of correlated signals fail
    - EXP41: per-language Z-norm adds +0.012

Compute: 1 forward + 1 backward pass per sample.
    attn_implementation=eager (required for attention weights).
Expected runtime: ~12-15 min on A100.
Expected AUC: 0.67-0.72 (product of two strong orthogonal signals).
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL02: OrthoFuse-MIA — Gradient × Attention Fusion")
    print("  Novelty: Multiplicative fusion of orthogonal signal families")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
        attn_implementation="eager",  # required for attention weights
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}, attn=eager, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class OrthoFuseScorer:
    """Extract gradient norm + attention concentration + their product."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

    def extract(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 30:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 10:
                return result

            # --- Forward pass with attention + gradient ---
            self.model.zero_grad()
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(input_ids).detach().requires_grad_(True)

            outputs = self.model(
                inputs_embeds=embeds, labels=input_ids,
                output_attentions=True,
            )
            loss = outputs.loss
            loss.backward()

            # 1. Gradient norm of embedding
            grad_norm = embeds.grad.float().norm(2).item()
            result["grad_norm"] = grad_norm
            result["neg_grad_norm"] = -grad_norm

            # 2. Loss
            result["neg_mean_loss"] = -loss.float().item()

            # 3. Attention features
            # outputs.attentions: tuple of (batch, n_heads, seq, seq) per layer
            attentions = outputs.attentions

            kl_concentrations = []
            max_attentions = []
            entropy_values = []
            bary_drifts = []
            prev_bary = None

            for layer_idx, attn in enumerate(attentions):
                # attn shape: (1, n_heads, seq, seq)
                attn_mean = attn[0].float().mean(dim=0)  # (seq, seq) avg over heads
                T_attn = attn_mean.shape[0]

                # KL concentration: KL(attn || uniform)
                uniform = torch.ones_like(attn_mean) / T_attn
                kl = F.kl_div(uniform.log(), attn_mean, reduction='batchmean').item()
                kl_concentrations.append(kl)

                # Max attention (how peaked)
                max_attn = attn_mean.max(dim=-1).values.mean().item()
                max_attentions.append(max_attn)

                # Entropy of attention
                attn_clamp = attn_mean.clamp(min=1e-10)
                entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean().item()
                entropy_values.append(entropy)

                # Barycenter (weighted average position)
                positions = torch.arange(T_attn, device=attn_mean.device, dtype=torch.float32)
                bary = (attn_mean * positions.unsqueeze(0)).sum(dim=-1)  # (seq,)
                if prev_bary is not None:
                    drift = (bary - prev_bary).abs().mean().item()
                    bary_drifts.append(drift)
                prev_bary = bary

            # Attention summary features
            kl_arr = np.array(kl_concentrations)
            result["attn_conc_mean"] = float(kl_arr.mean())
            result["attn_conc_late"] = float(kl_arr[-5:].mean()) if len(kl_arr) >= 5 else float(kl_arr.mean())
            result["attn_max_mean"] = float(np.mean(max_attentions))
            result["attn_entropy_mean"] = float(np.mean(entropy_values))

            if bary_drifts:
                bary_arr = np.array(bary_drifts)
                result["bary_drift_mean"] = float(bary_arr.mean())
                result["bary_drift_std"] = float(bary_arr.std())

            # --- FUSION: Product of orthogonal signals ---
            # Both grad_norm and attn features: lower = member
            # So -grad_norm is positive for member, and attn_conc is complex
            # Use: -(grad_norm) and -(attn_entropy) → both higher = member
            # Product: neg_grad_norm × neg_attn_entropy
            result["neg_attn_entropy"] = -result["attn_entropy_mean"]

            result["product_grad_attn_conc"] = -(grad_norm * result["attn_conc_mean"])
            result["product_grad_attn_entropy"] = result["neg_grad_norm"] * result["neg_attn_entropy"]
            result["product_grad_bary_drift"] = -(grad_norm * result.get("bary_drift_std", 0.0))

            # Rank-based fusion (convert to ranks within batch later, use raw for now)
            result["sum_grad_conc"] = result["neg_grad_norm"] + result["attn_conc_mean"]

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL02 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self) -> pd.DataFrame:
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
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
            sub_df = ds.to_pandas()
            sub_df["subset"] = subset
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        scorer = OrthoFuseScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL02] Extracting grad + attention features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL02]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Per-language Z-normalization (EXP41 insight) ---
        z_cols = ["neg_grad_norm", "attn_conc_mean", "neg_attn_entropy", "bary_drift_std"]
        for col in z_cols:
            if col not in df.columns:
                continue
            z_col = f"{col}_z_lang"
            df[z_col] = np.nan
            for lang in df["subset"].unique():
                mask = df["subset"] == lang
                vals = df.loc[mask, col].dropna()
                if len(vals) > 10:
                    mu, sigma = vals.mean(), vals.std()
                    if sigma > 1e-10:
                        df.loc[mask, z_col] = (df.loc[mask, col] - mu) / sigma

            # Per-language products
            if col == "neg_grad_norm" and "attn_conc_mean_z_lang" in df.columns:
                df["product_z_grad_conc"] = df["neg_grad_norm_z_lang"] * df["attn_conc_mean_z_lang"]

        # If we have Z-normalized grad and attn, create the Z-product
        if "neg_grad_norm_z_lang" in df.columns and "neg_attn_entropy_z_lang" in df.columns:
            df["product_z_grad_entropy"] = df["neg_grad_norm_z_lang"] * df["neg_attn_entropy_z_lang"]

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL02: OrthoFuse-MIA RESULTS")
        print("=" * 70)

        score_cols = [c for c in df.columns if c not in ["seq_len", "content", "membership",
                      "is_member", "subset", "idx"] and df[c].dtype in [np.float64, np.float32, float]]
        aucs = {}
        for col in sorted(score_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= auc_neg else "-"
            aucs[col] = (best, d)
            tag = ""
            if "product" in col:
                tag = " <-- FUSION"
            elif col == "neg_grad_norm":
                tag = " (grad baseline)"
            elif col == "attn_conc_mean":
                tag = " (attn baseline)"
            print(f"  {d}{col:<40} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")

            grad_auc = aucs.get("neg_grad_norm", (0, ""))[0]
            attn_auc = aucs.get("attn_conc_mean", (0, ""))[0]
            print(f"  Grad alone:  {grad_auc:.4f}")
            print(f"  Attn alone:  {attn_auc:.4f}")
            print(f"  vs EXP50 memTrace: 0.6908")

            # Per-subset
            best_col, (_, best_dir) = best_sig
            print(f"\n{'Subset':<10} | {best_col:<30} | Grad     | Attn     | N")
            print("-" * 75)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                scores = {}
                for sc in [best_col, "neg_grad_norm", "attn_conc_mean"]:
                    v = sub.dropna(subset=[sc])
                    if not v.empty and len(v["is_member"].unique()) > 1:
                        ap = roc_auc_score(v["is_member"], v[sc])
                        an = roc_auc_score(v["is_member"], -v[sc])
                        scores[sc] = max(ap, an)
                    else:
                        scores[sc] = float("nan")
                print(f"  {subset:<10} | {scores.get(best_col, float('nan')):.4f}"
                      f"                        | {scores.get('neg_grad_norm', float('nan')):.4f}"
                      f"   | {scores.get('attn_conc_mean', float('nan')):.4f}"
                      f"   | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL02_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL02] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        max_length = 512
        output_dir = "results"
        seed = 42

    print(f"[NOVEL02] OrthoFuse: Gradient × Attention")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
