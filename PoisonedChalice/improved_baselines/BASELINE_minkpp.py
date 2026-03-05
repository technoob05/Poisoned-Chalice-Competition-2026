"""
BASELINE_minkpp.py — Min-K%++ Reimplementation for Poisoned Chalice

Clean reimplementation of Min-K%++ (Zhang et al., ICLR 2025 Spotlight).
This serves as the PROPER UNSUPERVISED BASELINE to compare all methods against.

Method (from paper Eq. 3-4):
    For each token position t:
        z_t = (log p(x_t | x<t) - μ_{·|x<t}) / σ_{·|x<t}
    where:
        μ_{·|x<t} = E_z[log p(z | x<t)]  (mean log-prob over vocab)
        σ_{·|x<t} = std_z[log p(z | x<t)]  (std of log-prob over vocab)
    
    Final score = mean of min-k% of z_t values (lowest k% z-scores)
    Higher score → more likely member

Properties:
    - Unsupervised (no labels needed)
    - Zero overhead (only 1 forward pass + algebraic ops on logits)
    - Reference-free (no extra model needed)
    - SOTA on WikiMIA (+6-10% over Min-K%), competitive on MIMIR

Compute: 1 forward pass only
Expected runtime: ~5-10 min on A100 (10% sample)
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
    print("  BASELINE: Min-K%++ Reimplementation")
    print("  (Zhang et al., ICLR 2025 Spotlight)")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class MinKPPExtractor:
    """
    Min-K%++ scoring from a single forward pass.
    
    For each token t:
        z_t = (log p(x_t|x<t) - μ) / σ
    where μ, σ are mean/std of log p(z|x<t) over the full vocabulary.
    
    Score = mean of the lowest k% of z_t values.
    """

    def __init__(self, model, tokenizer, max_length: int = 512, k_percent: float = 0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_percent = k_percent
        self._err_count = 0

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        features = {}
        if not text or len(text) < 20:
            return features

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                return features

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # (1, T, V)

            # Shift: predict position t from position t-1
            shift_logits = logits[0, :-1, :].float()  # (T-1, V)
            shift_labels = input_ids[0, 1:]             # (T-1,)
            T = shift_logits.shape[0]
            V = shift_logits.shape[1]

            if T < 3:
                return features

            # ── Core Min-K%++ computation ──────────────────────────────────
            # Softmax probabilities and log-probabilities over vocab
            probs = F.softmax(shift_logits, dim=-1)       # (T, V)
            log_probs = F.log_softmax(shift_logits, dim=-1)  # (T, V)

            # Token log-probability for actual next tokens
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (T,)

            # μ = E_z[log p(z|x<t)] = sum(p * log_p) over vocab
            mu = (probs * log_probs).sum(dim=-1)  # (T,)

            # σ² = E_z[(log p(z|x<t))²] - μ²
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)  # (T,)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()  # (T,)

            # z_t = (log p(x_t|x<t) - μ) / σ  [Eq. 3]
            z_scores = (token_log_probs - mu) / sigma  # (T,)

            z_np = z_scores.cpu().numpy()

            # ── Min-K% aggregation [Eq. 4] ────────────────────────────────
            for k in [0.1, 0.2, 0.3, 0.5, 1.0]:
                n_select = max(1, int(T * k))
                sorted_z = np.sort(z_np)[:n_select]  # lowest k%
                features[f"minkpp_k{int(k*100)}"] = float(np.mean(sorted_z))

            # ── Additional signals from the same forward pass ─────────────
            # Raw loss (baseline)
            features["neg_mean_loss"] = float(token_log_probs.mean().item())

            # Raw Min-K% (without calibration, for comparison)
            token_lp_np = token_log_probs.cpu().numpy()
            for k in [0.2]:
                n_select = max(1, int(T * k))
                sorted_lp = np.sort(token_lp_np)[:n_select]
                features[f"mink_k{int(k*100)}"] = float(np.mean(sorted_lp))

            # Entropy of predicted distributions (mean)
            entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
            features["entropy_mean"] = float(entropy.mean().item())
            features["neg_entropy_mean"] = float(-entropy.mean().item())

            # Sequence length (control variable)
            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[MinKPP WARN] {type(e).__name__}: {e}")
            self._err_count += 1

        return features


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
            try:
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
            except Exception as e:
                print(f"  [WARN] {subset}: {e}")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = MinKPPExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            k_percent=self.args.k_percent,
        )

        print(f"\n[Min-K%++] Extracting scores for {len(df)} samples...")
        print(f"  k values: [10%, 20%, 30%, 50%, 100%]")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[MinK++]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── AUC Results ───────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   Min-K%++ UNSUPERVISED AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        results = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= auc_neg else "-"
            results[col] = (best, d)
            print(f"  {d}{col:<40} AUC = {best:.4f}")

        # ── Per-subset breakdown for best signal ──────────────────────────
        best_signal = max(results.items(), key=lambda x: x[1][0])
        best_col = best_signal[0]
        best_dir = best_signal[1][1]
        print(f"\n  BEST SIGNAL: {best_dir}{best_col} = {best_signal[1][0]:.4f}")

        print(f"\n{'Subset':<10} | {'MinK++20':<10} | {'Loss':<10} | N")
        print("-" * 50)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v1 = sub.dropna(subset=[best_col])
            v2 = sub.dropna(subset=["neg_mean_loss"])
            r1 = roc_auc_score(v1["is_member"], v1[best_col]) if len(v1) > 10 else float("nan")
            r2 = roc_auc_score(v2["is_member"], v2["neg_mean_loss"]) if len(v2) > 10 else float("nan")
            print(f"{subset:<10} | {r1:.4f}     | {r2:.4f}     | {len(sub)}")

        # ── Comparison ────────────────────────────────────────────────────
        print(f"\n  --- Comparison ---")
        print(f"  Min-K%++ (this):   {best_signal[1][0]:.4f}")
        print(f"  Loss baseline:     {results.get('neg_mean_loss', (0, ''))[0]:.4f}")
        print(f"  EXP50 memTrace RF: 0.6908 (supervised)")
        print(f"  LUMIA-fast:        0.7805 (supervised)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"BASELINE_minkpp_{timestamp}.parquet", index=False)
        print(f"\n[Min-K%++] Results saved.")


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
        k_percent = 0.20

    print(f"[Min-K%++] {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, k={Args.k_percent*100:.0f}%")
    Experiment(Args).run()
