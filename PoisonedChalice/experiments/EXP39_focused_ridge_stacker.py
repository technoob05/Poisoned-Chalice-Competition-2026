"""
EXPERIMENT 39: Focused Ridge Stacker — Minimal Features, Large Probe, Linear Model

Fixes ALL EXP38 failure modes (Insight 15):
    1. EXP38 had 18 features / 400 probe → overfitting. Fix: 6 features / 1000 probe.
    2. seq_len dominated importance → spurious. Fix: excluded entirely.
    3. XGBoost too flexible for small probe. Fix: Ridge Logistic Regression (linear, L2-regularized).
    4. Auto-flip on 400 unreliable. Fix: 1000 probe + only include signals with known direction.

Feature set (6 features, all with KNOWN direction from prior experiments):
    1. -grad_embed (EXP11/27: AUC 0.64-0.65)
    2. -grad_L28 (EXP30/34: AUC ~0.63)
    3. -grad_L29 (EXP30/34: AUC ~0.64)
    4. -mean_loss (EXP01/36: AUC ~0.58)
    5. surp_score = mean_loss - std_loss (EXP16: AUC 0.5884)
    6. -minkpp_score (EXP02: AUC 0.577)

    Engineered:
    7. product_score = -(grad_embed × mean_loss) (EXP27 formula generalized)

Architecture:
    - Probe: 1000 balanced (500M + 500NM) from 10% sample
    - Model: Ridge LogisticRegression (C=0.1, high regularization)
    - 5-fold CV on probe, final model on full probe
    - All features pre-negated (no auto-flip needed)

Expected AUC: 0.66-0.70 (linear model, robust to small probe)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP39: FOCUSED RIDGE STACKER (Fix EXP38)")
    print("=" * 65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


class FocusedExtractor:
    def __init__(self, model, tokenizer, max_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.bookend_params = self._find_bookend_params()

    def _find_bookend_params(self):
        params = {}
        for tag, pattern in [
            ("embed", "embed_tokens"),
            ("head", "lm_head"),
        ]:
            params[tag] = [n for n, _ in self.model.named_parameters() if pattern in n]

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            n = len(self.model.model.layers)
            for tag, idx in [("L28", min(28, n - 1)), ("L29", min(29, n - 1))]:
                prefix = f"model.layers.{idx}."
                params[tag] = [n for n, _ in self.model.named_parameters() if n.startswith(prefix)]
        return params

    def extract(self, text: str) -> Dict[str, float]:
        result = {k: np.nan for k in [
            "neg_grad_embed", "neg_grad_L28", "neg_grad_L29", "neg_grad_head",
            "neg_mean_loss", "surp_score", "neg_minkpp", "product_grad_loss",
        ]}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]

            self.model.zero_grad()
            outputs = self.model(**inputs, labels=input_ids)
            loss_val = outputs.loss

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).float().detach().cpu().numpy()

            mean_loss = float(per_token_loss.mean())
            std_loss = float(per_token_loss.std())
            result["neg_mean_loss"] = -mean_loss
            result["surp_score"] = mean_loss - std_loss

            k_pct = max(1, int(0.2 * len(per_token_loss)))
            result["neg_minkpp"] = -float(np.sort(per_token_loss)[:k_pct].mean())

            loss_val.backward()
            param_dict = {n: p for n, p in self.model.named_parameters()}
            grad_norms = {}
            for tag, param_names in self.bookend_params.items():
                norms = []
                for pn in param_names:
                    p = param_dict.get(pn)
                    if p is not None and p.grad is not None:
                        norms.append(p.grad.float().norm(2).item())
                if norms:
                    grad_norms[tag] = float(np.sqrt(np.mean(np.square(norms))))

            self.model.zero_grad()

            for tag in ["embed", "L28", "L29", "head"]:
                if tag in grad_norms:
                    result[f"neg_grad_{tag}"] = -grad_norms[tag]

            if "embed" in grad_norms:
                result["product_grad_loss"] = -(grad_norms["embed"] * mean_loss)

            return result

        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP39 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class FocusedRidgeAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.probe_size = getattr(args, "probe_size", 1000)
        self.n_folds = getattr(args, "n_folds", 5)
        self.C = getattr(args, "ridge_C", 0.1)
        self.extractor = FocusedExtractor(
            model, tokenizer, max_length=getattr(args, "max_length", 2048),
        )

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        print(f"\n[EXP39] Extracting 7 focused features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP39] Extract"):
            rows.append(self.extractor.extract(row["content"]))

        feat_df = pd.DataFrame(rows)
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feat_df.columns:
            med = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(med if not np.isnan(med) else 0.0)

        feat_cols = list(feat_df.columns)
        print(f"[EXP39] Features: {feat_cols}")

        n_each = self.probe_size // 2
        n_m = min(n_each, (df["is_member"] == 1).sum())
        n_nm = min(n_each, (df["is_member"] == 0).sum())
        members = df[df["is_member"] == 1].sample(n_m, random_state=self.args.seed)
        nonmembers = df[df["is_member"] == 0].sample(n_nm, random_state=self.args.seed)
        probe_idx = pd.concat([members, nonmembers]).index.tolist()
        print(f"[EXP39] Probe: {len(probe_idx)} samples ({n_m}M + {n_nm}NM)")

        X_probe = feat_df.loc[probe_idx, feat_cols].values
        y_probe = df.loc[probe_idx, "is_member"].values

        print(f"\n[EXP39] {self.n_folds}-fold CV with Ridge LR (C={self.C})...")
        cv_aucs = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.args.seed)
        for fold, (tr, va) in enumerate(skf.split(X_probe, y_probe)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_probe[tr])
            X_va = scaler.transform(X_probe[va])
            clf = LogisticRegression(C=self.C, penalty="l2", max_iter=1000, solver="lbfgs")
            clf.fit(X_tr, y_probe[tr])
            auc = roc_auc_score(y_probe[va], clf.predict_proba(X_va)[:, 1])
            cv_aucs.append(auc)
            print(f"  Fold {fold+1}: AUC = {auc:.4f}")
        print(f"  CV Mean: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

        scaler_final = StandardScaler()
        X_probe_s = scaler_final.fit_transform(X_probe)
        final_clf = LogisticRegression(C=self.C, penalty="l2", max_iter=1000, solver="lbfgs")
        final_clf.fit(X_probe_s, y_probe)

        coefs = final_clf.coef_[0]
        print(f"\n[EXP39] Ridge coefficients:")
        for fname, c in sorted(zip(feat_cols, coefs), key=lambda x: -abs(x[1])):
            print(f"  {fname:<25} {c:+.4f}")

        X_all = feat_df[feat_cols].values
        X_all_s = scaler_final.transform(X_all)
        all_preds = final_clf.predict_proba(X_all_s)[:, 1]

        feat_df["ridge_score"] = all_preds

        n_valid = feat_df["ridge_score"].notna().sum()
        print(f"\n[EXP39] Valid: {n_valid}/{len(df)}")
        if self.extractor._err_count > 0:
            print(f"[EXP39] Errors: {self.extractor._err_count}")
        return feat_df


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
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        attacker = FocusedRidgeAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df)
        for col in scores_df.columns:
            if col not in df.columns:
                df[col] = scores_df[col].values

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP39_{timestamp}.parquet", index=False)

        print("\n" + "=" * 65)
        print("   EXP39: FOCUSED RIDGE STACKER — REPORT")
        print("=" * 65)
        for sc, label in [
            ("ridge_score", "Ridge Stacker [PRIMARY]"),
            ("neg_grad_embed", "-Grad Embed (single)"),
            ("product_grad_loss", "Product -(grad*loss)"),
            ("neg_mean_loss", "-Mean Loss"),
            ("surp_score", "SURP"),
        ]:
            if sc not in df.columns:
                continue
            v = df.dropna(subset=[sc])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[sc])
                tag = " <-- PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<40} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'Ridge':<8} | {'GradEmbed':<11} | {'Product':<9} | N")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["ridge_score", "neg_grad_embed", "product_grad_loss"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('ridge_score', float('nan')):.4f}   "
                  f"| {r.get('neg_grad_embed', float('nan')):.4f}      "
                  f"| {r.get('product_grad_loss', float('nan')):.4f}    "
                  f"| {len(sub)}")
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
        probe_size = 1000
        n_folds = 5
        ridge_C = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP39] Probe: {Args.probe_size}, C={Args.ridge_C}, Features: 7")
    Experiment(Args).run()
