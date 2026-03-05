"""
EXPERIMENT 42: Full Dataset Gradient Norm — Test Ceiling Artifact

Motivation (Insight 8):
    All gradient experiments used 5-10% samples (5K-10K).
    The ~0.65 ceiling might be a SAMPLING ARTIFACT:
    - Small samples have higher variance → AUC estimation noisy
    - Probe-based methods (EXP30, EXP35, EXP38) suffer from small N

    THIS EXPERIMENT runs -grad_embed on the FULL 100% dataset (100K samples)
    to determine the TRUE gradient ceiling. Also includes -mean_loss and
    product_score for comparison.

    If AUC stays ~0.65: ceiling is real → information-theoretic limit.
    If AUC rises to 0.68+: ceiling was sampling artifact → bigger samples help.

Architecture:
    - 100% dataset (all ~100K samples)
    - Single forward + backward pass per sample
    - Compute: -grad_embed, -mean_loss, product_score
    - NO probe phase (pure unsupervised signals)
    - Batch processing with progress tracking

Expected AUC: 0.64-0.68 (testing whether ceiling changes with full data)

IMPORTANT: This requires Kaggle P100/T4 (16GB) or A100 (40GB).
           ~16 hours on T4, ~2 hours on A100 at full dataset.
           Set sample_fraction to 1.0 for full run, or 0.5 for faster test.
"""
import os
import json
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
    print("\n" + "=" * 65)
    print("  EXP42: FULL DATASET GRADIENT NORM (Ceiling Test)")
    print("=" * 65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}")
    return model, tokenizer


class MinimalExtractor:
    """Fastest possible extractor: only -grad_embed, -loss, product."""
    def __init__(self, model, tokenizer, max_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.embed_params = [n for n, _ in model.named_parameters() if "embed_tokens" in n]

    def extract(self, text: str) -> Dict[str, float]:
        r = {"grad_embed": np.nan, "mean_loss": np.nan}
        if not text or len(text) < 20:
            return r
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)

            self.model.zero_grad()
            out = self.model(**inputs, labels=inputs["input_ids"])
            r["mean_loss"] = out.loss.float().item()

            out.loss.backward()
            pd_ = {n: p for n, p in self.model.named_parameters()}
            norms = []
            for pn in self.embed_params:
                p = pd_.get(pn)
                if p is not None and p.grad is not None:
                    norms.append(p.grad.float().norm(2).item())
            if norms:
                r["grad_embed"] = float(np.sqrt(np.mean(np.square(norms))))
            self.model.zero_grad()
            return r
        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP42 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return r


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self):
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
        ext = MinimalExtractor(self.model, self.tokenizer, self.args.max_length)

        checkpoint_interval = 5000
        all_results = []

        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="[EXP42] Full Dataset")):
            all_results.append(ext.extract(row["content"]))

            if (i + 1) % checkpoint_interval == 0:
                partial = pd.DataFrame(all_results)
                valid = partial["grad_embed"].notna()
                if valid.sum() > 100:
                    partial_df = df.iloc[:len(partial)].copy()
                    partial_df["neg_grad_embed"] = -partial["grad_embed"].values
                    v = partial_df.dropna(subset=["neg_grad_embed"])
                    if len(v["is_member"].unique()) > 1:
                        auc = roc_auc_score(v["is_member"], v["neg_grad_embed"])
                        print(f"\n  [Checkpoint {i+1}/{len(df)}] Running AUC = {auc:.4f} "
                              f"(valid: {valid.sum()}/{len(partial)})")

        feat = pd.DataFrame(all_results)
        df["grad_embed"] = feat["grad_embed"].values
        df["mean_loss"] = feat["mean_loss"].values
        df["neg_grad_embed"] = -df["grad_embed"]
        df["neg_mean_loss"] = -df["mean_loss"]
        df["product_grad_loss"] = -(df["grad_embed"] * df["mean_loss"])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP42_{timestamp}.parquet", index=False)

        print("\n" + "=" * 65)
        print("   EXP42: FULL DATASET GRADIENT — CEILING TEST REPORT")
        print("=" * 65)
        print(f"  Total samples: {len(df)}")
        print(f"  Valid: {df['grad_embed'].notna().sum()}")
        if ext._err_count > 0:
            print(f"  Errors: {ext._err_count}")

        for sc, label in [
            ("neg_grad_embed", "-Grad Embed [PRIMARY]"),
            ("neg_mean_loss", "-Mean Loss"),
            ("product_grad_loss", "-(grad * loss)"),
        ]:
            v = df.dropna(subset=[sc])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[sc])
                print(f"  {label:<35} AUC = {auc:.4f}")

        print(f"\nComparison with 10% sample baselines:")
        print(f"  EXP11 -grad_embed (10%):  0.6472")
        print(f"  EXP27 product    (5%):    0.6484")
        v = df.dropna(subset=["neg_grad_embed"])
        full_auc = roc_auc_score(v["is_member"], v["neg_grad_embed"])
        print(f"  EXP42 -grad_embed ({self.args.sample_fraction*100:.0f}%): {full_auc:.4f}")
        delta = full_auc - 0.6472
        print(f"  Delta vs 10% baseline: {delta:+.4f}")
        if abs(delta) < 0.005:
            print(f"  -> CEILING IS REAL: ~0.65 is information-theoretic limit")
        elif delta > 0.01:
            print(f"  -> CEILING WAS ARTIFACT: more data helps!")
        else:
            print(f"  -> MARGINAL IMPROVEMENT: ceiling is approximately real")

        print(f"\n{'Subset':<10} | {'N':<8} | {'GradEmbed':<11} | {'Loss':<8} | {'Product':<9}")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["neg_grad_embed", "neg_mean_loss", "product_grad_loss"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {len(sub):<8} | {r.get('neg_grad_embed', float('nan')):.4f}      "
                  f"| {r.get('neg_mean_loss', float('nan')):.4f}   "
                  f"| {r.get('product_grad_loss', float('nan')):.4f}")

        print(f"\nGradient statistics (full dataset):")
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for feat_name in ["grad_embed", "mean_loss"]:
            print(f"  {feat_name:<15} M: {m[feat_name].mean():.4f} +/- {m[feat_name].std():.4f}  "
                  f"NM: {nm[feat_name].mean():.4f} +/- {nm[feat_name].std():.4f}  "
                  f"ratio: {m[feat_name].mean()/nm[feat_name].mean():.3f}")
        print("=" * 65)


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.50
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP42] Sample: {Args.sample_fraction*100:.0f}% (set to 1.0 for full run)")
    Experiment(Args).run()
