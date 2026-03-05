"""
EXPERIMENT 40: Generalized Product Scores — Multiplicative Signal Combination

Motivation (Insight 1, 3):
    EXP27 discovered that -(grad_embed × jsd_early) = 0.6484 beats both
    grad_embed alone (0.6480) and rank-avg (0.5845). This was our BEST result.

    But EXP27 only tested one product: grad × JSD.
    JSD_early is a weak signal (0.4371). What if we multiply grad with
    STRONGER signals? Or combine multiple products?

    This experiment systematically tests ALL pairwise products:
    - grad × loss (both strong signals, ~0.58-0.65)
    - grad × SURP
    - grad × minkpp
    - grad × CAMIA sub-signals
    - grad_L28 × grad_embed (cross-layer product)
    - Three-way: grad × loss × CAMIA

    The intuition: product works because it creates a NON-LINEAR decision boundary.
    When both signals agree (high grad AND high loss → non-member), the product
    amplifies the signal. When they disagree, it dampens noise.

Expected AUC: 0.65-0.69 (product of two ~0.60 signals can exceed either)
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
from scipy.stats import rankdata
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP40: GENERALIZED PRODUCT SCORES")
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


class ProductScoreExtractor:
    def __init__(self, model, tokenizer, max_length=2048, n_blocks=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_blocks = n_blocks
        self._err_count = 0
        self.bookend = self._find_bookend()

    def _find_bookend(self):
        params = {}
        for tag, pat in [("embed", "embed_tokens"), ("head", "lm_head")]:
            params[tag] = [n for n, _ in self.model.named_parameters() if pat in n]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            nl = len(self.model.model.layers)
            for tag, idx in [("L28", min(28, nl-1)), ("L29", min(29, nl-1))]:
                params[tag] = [n for n, _ in self.model.named_parameters()
                               if n.startswith(f"model.layers.{idx}.")]
        return params

    def extract(self, text: str) -> Dict[str, float]:
        r = {k: np.nan for k in [
            "grad_embed", "grad_L28", "grad_L29", "grad_head",
            "mean_loss", "std_loss", "surp", "minkpp",
            "camia_mdm", "camia_tvar",
        ]}
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
            r["surp"] = r["mean_loss"] - r["std_loss"]
            k = max(1, int(0.2 * len(ptl)))
            r["minkpp"] = float(np.sort(ptl)[:k].mean())

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
            for tag, pns in self.bookend.items():
                norms = []
                for pn in pns:
                    p = pd_.get(pn)
                    if p is not None and p.grad is not None:
                        norms.append(p.grad.float().norm(2).item())
                if norms:
                    r[f"grad_{tag}"] = float(np.sqrt(np.mean(np.square(norms))))
            self.model.zero_grad()
            return r
        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP40 WARN] {type(e).__name__}: {e}")
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
        ext = ProductScoreExtractor(self.model, self.tokenizer,
                                    max_length=self.args.max_length, n_blocks=8)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP40] Extract"):
            rows.append(ext.extract(row["content"]))
        feat = pd.DataFrame(rows)

        feat["neg_grad_embed"] = -feat["grad_embed"]
        feat["neg_grad_L28"] = -feat["grad_L28"]
        feat["neg_grad_L29"] = -feat["grad_L29"]
        feat["neg_mean_loss"] = -feat["mean_loss"]

        products = {
            "prod_grad_loss": ("grad_embed", "mean_loss"),
            "prod_grad_surp": ("grad_embed", "surp"),
            "prod_grad_minkpp": ("grad_embed", "minkpp"),
            "prod_grad_mdm": ("grad_embed", "camia_mdm"),
            "prod_grad_tvar": ("grad_embed", "camia_tvar"),
            "prod_L28_L29": ("grad_L28", "grad_L29"),
            "prod_embed_L29": ("grad_embed", "grad_L29"),
            "prod_loss_surp": ("mean_loss", "surp"),
        }
        for name, (a, b) in products.items():
            feat[name] = -(feat[a] * feat[b])

        feat["prod_triple_grad_loss_mdm"] = -(feat["grad_embed"] * feat["mean_loss"] * feat["camia_mdm"])

        feat["rank_grad_loss"] = (
            rankdata(-feat["grad_embed"].fillna(0).values) +
            rankdata(-feat["mean_loss"].fillna(0).values)
        ) / (2 * len(feat))

        for col in feat.columns:
            if col not in df.columns:
                df[col] = feat[col].values

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP40_{timestamp}.parquet", index=False)

        print("\n" + "=" * 65)
        print("   EXP40: GENERALIZED PRODUCT SCORES — REPORT")
        print("=" * 65)

        score_cols = [
            ("neg_grad_embed", "-Grad Embed (baseline)"),
            ("neg_mean_loss", "-Mean Loss"),
            ("prod_grad_loss", "-(grad * loss) [KEY]"),
            ("prod_grad_surp", "-(grad * surp)"),
            ("prod_grad_minkpp", "-(grad * minkpp)"),
            ("prod_grad_mdm", "-(grad * camia_mdm)"),
            ("prod_grad_tvar", "-(grad * camia_tvar)"),
            ("prod_L28_L29", "-(L28 * L29)"),
            ("prod_embed_L29", "-(embed * L29)"),
            ("prod_loss_surp", "-(loss * surp)"),
            ("prod_triple_grad_loss_mdm", "-(grad*loss*mdm)"),
            ("rank_grad_loss", "Rank-avg(grad, loss)"),
        ]

        aucs = {}
        for sc, label in score_cols:
            if sc not in df.columns:
                continue
            v = df.dropna(subset=[sc])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[sc])
                aucs[sc] = auc
                best_tag = " <-- BEST" if auc == max(aucs.values()) else ""
                print(f"  {label:<40} AUC = {auc:.4f}{best_tag}")

        best_sc = max(aucs, key=aucs.get)
        print(f"\n  BEST PRODUCT: {best_sc} = {aucs[best_sc]:.4f}")
        print(f"  vs EXP27 product_score: 0.6484")
        print(f"  vs EXP11 -grad_embed:   0.6472")

        print(f"\n{'Subset':<10} | {'GradEmbed':<11} | {'Grad*Loss':<11} | {'Grad*SURP':<11} | {'Best Product'}")
        print("-" * 65)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["neg_grad_embed", "prod_grad_loss", "prod_grad_surp"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            best_sub = max(r, key=r.get)
            print(f"{subset:<10} | {r.get('neg_grad_embed', float('nan')):.4f}      "
                  f"| {r.get('prod_grad_loss', float('nan')):.4f}      "
                  f"| {r.get('prod_grad_surp', float('nan')):.4f}      "
                  f"| {best_sub}")
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
