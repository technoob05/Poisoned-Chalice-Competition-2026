"""
EXPERIMENT 41: Per-Language Calibrated Gradient MIA

Motivation (Insight 7):
    Go consistently achieves AUC 0.70+ while Rust sits at ~0.58.
    This gap is STABLE across all signal families.

    A single global threshold misses language-specific baselines.
    Go code has lower gradient norms overall (idiomatic, regular syntax),
    so the member/non-member gap is clearer. Rust has higher variance.

    THIS EXPERIMENT:
    1. Compute -grad_embed for all samples
    2. Z-normalize PER LANGUAGE: score = (x - mean_lang) / std_lang
    3. This removes the language-specific baseline and focuses on
       within-language relative position
    4. Also test: per-language percentile rank, probe-based calibration

    Additionally: compute CAMIA + gradient PRODUCT per-language.

Expected AUC: 0.66-0.70 (removes language noise that hurts global threshold)
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
from scipy.stats import rankdata, zscore
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP41: PER-LANGUAGE CALIBRATED GRADIENT MIA")
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


class GradientExtractor:
    def __init__(self, model, tokenizer, max_length=2048, n_blocks=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_blocks = n_blocks
        self._err_count = 0
        self.embed_params = [n for n, _ in model.named_parameters() if "embed_tokens" in n]

    def extract(self, text: str) -> Dict[str, float]:
        r = {"grad_embed": np.nan, "mean_loss": np.nan, "std_loss": np.nan,
             "camia_mdm": np.nan, "camia_tvar": np.nan}
        if not text or len(text) < 20:
            return r
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            ids = inputs["input_ids"]

            self.model.zero_grad()
            out = self.model(**inputs, labels=ids)

            logits = out.logits
            sl = logits[:, :-1, :].contiguous()
            lab = ids[:, 1:].contiguous()
            ptl = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lab.view(-1), reduction="none"
            ).float().detach().cpu().numpy()

            r["mean_loss"] = float(ptl.mean())
            r["std_loss"] = float(ptl.std())

            nt = len(ptl)
            bs = max(1, nt // self.n_blocks)
            bl = []
            for b in range(self.n_blocks):
                s, e = b*bs, min((b+1)*bs, nt)
                if s < nt:
                    bl.append(float(ptl[s:e].mean()))
            if len(bl) >= 2:
                bla = np.array(bl)
                r["camia_mdm"] = float(bla.max() - bla.min())
                r["camia_tvar"] = float(bla.std())

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
                print(f"\n[EXP41 WARN] {type(e).__name__}: {e}")
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
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        ext = GradientExtractor(self.model, self.tokenizer, self.args.max_length)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP41] Extract"):
            rows.append(ext.extract(row["content"]))
        feat = pd.DataFrame(rows)
        for col in feat.columns:
            df[col] = feat[col].values

        df["neg_grad_embed"] = -df["grad_embed"]
        df["neg_mean_loss"] = -df["mean_loss"]
        df["product_grad_loss"] = -(df["grad_embed"] * df["mean_loss"])

        df["neg_grad_z_global"] = np.nan
        valid = df["grad_embed"].notna()
        if valid.sum() > 10:
            vals = df.loc[valid, "grad_embed"].values
            df.loc[valid, "neg_grad_z_global"] = -zscore(vals)

        df["neg_grad_z_lang"] = np.nan
        df["neg_loss_z_lang"] = np.nan
        df["product_z_lang"] = np.nan
        df["neg_grad_pctile_lang"] = np.nan

        for lang in df["subset"].unique():
            mask = (df["subset"] == lang) & df["grad_embed"].notna()
            if mask.sum() < 10:
                continue
            vals = df.loc[mask, "grad_embed"].values
            z = zscore(vals)
            df.loc[mask, "neg_grad_z_lang"] = -z

            pctile = rankdata(vals) / len(vals)
            df.loc[mask, "neg_grad_pctile_lang"] = 1.0 - pctile

            loss_mask = mask & df["mean_loss"].notna()
            if loss_mask.sum() > 10:
                loss_vals = df.loc[loss_mask, "mean_loss"].values
                loss_z = zscore(loss_vals)
                df.loc[loss_mask, "neg_loss_z_lang"] = -loss_z

                grad_z_lang = -zscore(df.loc[loss_mask, "grad_embed"].values)
                df.loc[loss_mask, "product_z_lang"] = grad_z_lang * (-loss_z)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP41_{timestamp}.parquet", index=False)

        print("\n" + "=" * 65)
        print("   EXP41: PER-LANGUAGE CALIBRATED GRADIENT — REPORT")
        print("=" * 65)

        score_cols = [
            ("neg_grad_embed", "-Grad Embed (raw, global)"),
            ("neg_grad_z_global", "-Grad Z-score (global)"),
            ("neg_grad_z_lang", "-Grad Z-score (per-lang) [KEY]"),
            ("neg_grad_pctile_lang", "-Grad Percentile (per-lang)"),
            ("neg_mean_loss", "-Mean Loss (raw)"),
            ("neg_loss_z_lang", "-Loss Z-score (per-lang)"),
            ("product_grad_loss", "-(grad * loss) raw"),
            ("product_z_lang", "-(grad_z * loss_z) per-lang [KEY]"),
        ]
        aucs = {}
        for sc, label in score_cols:
            if sc not in df.columns:
                continue
            v = df.dropna(subset=[sc])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[sc])
                aucs[sc] = auc
                print(f"  {label:<45} AUC = {auc:.4f}")

        print(f"\n{'Subset':<10} | {'Raw':<8} | {'Z-global':<10} | {'Z-lang':<8} | {'Pctile':<8} | {'Prod-Z':<8}")
        print("-" * 65)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["neg_grad_embed", "neg_grad_z_global", "neg_grad_z_lang",
                        "neg_grad_pctile_lang", "product_z_lang"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('neg_grad_embed', float('nan')):.4f}   "
                  f"| {r.get('neg_grad_z_global', float('nan')):.4f}     "
                  f"| {r.get('neg_grad_z_lang', float('nan')):.4f}   "
                  f"| {r.get('neg_grad_pctile_lang', float('nan')):.4f}   "
                  f"| {r.get('product_z_lang', float('nan')):.4f}")

        print(f"\nPer-language gradient stats:")
        for lang in sorted(df["subset"].unique()):
            sub = df[df["subset"] == lang]
            m = sub[sub["is_member"] == 1]["grad_embed"].mean()
            nm = sub[sub["is_member"] == 0]["grad_embed"].mean()
            ratio = m / nm if nm > 0 else float("nan")
            print(f"  {lang:<10} M: {m:.4f}  NM: {nm:.4f}  ratio: {ratio:.3f}")

        print("=" * 65)


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        output_dir = "results"
        max_length = 2048
        seed = 42

    Experiment(Args).run()
