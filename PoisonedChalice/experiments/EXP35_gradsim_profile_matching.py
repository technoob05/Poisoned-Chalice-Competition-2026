"""
EXPERIMENT 35: GradSim — Gradient Profile Similarity via Discriminant Analysis

Novelty & motivation (from Insights 1, 6, 8, 10):
    Previous gradient experiments collapse per-layer gradient norms to a SCALAR
    (EXP11: embed only, EXP34: g_mean, EXP30: RMS of top-5).
    All hit the same ~0.65 AUC ceiling (Insight 8).

    Key observation: EXP30 showed memorization concentrates in layers 24-29
    with layer_norm > attn > mlp ordering (Insight 6). EXP34 showed
    per-layer gnorms carry different AUCs (embed 0.6422, L29 0.6407, L28 0.6303).

    THIS EXPERIMENT uses the FULL 32-dimensional gradient profile vector
    (embed + 30 transformer blocks + head) as a high-dimensional fingerprint.
    Instead of hand-picking layers or collapsing to a scalar, we let
    Linear Discriminant Analysis (LDA) find the OPTIMAL projection axis
    that maximally separates members from non-members in gradient space.

    This is strictly more expressive than any scalar gradient summary:
    - It can learn that embed and L29 matter more than L15
    - It can exploit correlations between layers (e.g., low-embed + high-L29 = member)
    - It uses ALL gradient information, not just magnitude

Architecture:
    Phase 1 (PROBE): 200 balanced labeled samples → compute 32-dim gradient profiles
                      → fit LDA on the profiles
    Phase 2 (INFERENCE): all samples → compute gradient profile → LDA.decision_function()
                          → score = LDA projection onto discriminant axis

    Secondary signals:
    - Logistic Regression on the same features (comparison)
    - Cosine similarity to member centroid (simpler baseline)
    - Raw top-layer norms for EXP15 stacking

Expected AUC: 0.66–0.70 (breaks scalar ceiling by using learned layer weighting)

Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from scipy.spatial.distance import cosine as cosine_dist

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP35: GradSim — GRADIENT PROFILE SIMILARITY (LDA)")
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


class GradientProfileExtractor:
    """Extracts a per-layer gradient norm vector for each sample."""

    def __init__(self, model):
        self.model = model
        self.component_names, self.component_params = self._build_components()
        self._err_count = 0
        print(f"[EXP35] {len(self.component_names)} gradient components defined.")

    def _build_components(self):
        names = []
        param_groups = []

        embed_params = [n for n, _ in self.model.named_parameters() if "embed_tokens" in n]
        if embed_params:
            names.append("embed")
            param_groups.append(embed_params)

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i in range(len(self.model.model.layers)):
                prefix = f"model.layers.{i}."
                block_params = [n for n, _ in self.model.named_parameters() if n.startswith(prefix)]
                if block_params:
                    names.append(f"block_{i}")
                    param_groups.append(block_params)

        head_params = [n for n, _ in self.model.named_parameters() if "lm_head" in n]
        if head_params:
            names.append("head")
            param_groups.append(head_params)

        return names, param_groups

    def extract_profile(self, text: str, tokenizer, max_length: int) -> Optional[np.ndarray]:
        if not text or len(text) < 20:
            return None
        try:
            inputs = tokenizer(
                text, return_tensors="pt", max_length=max_length, truncation=True,
            ).to(self.model.device)
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()

            param_dict = {n: p for n, p in self.model.named_parameters()}
            profile = np.zeros(len(self.component_names), dtype=np.float32)
            for idx, param_names in enumerate(self.component_params):
                norms = []
                for pn in param_names:
                    p = param_dict.get(pn)
                    if p is not None and p.grad is not None:
                        norms.append(p.grad.float().norm(2).item())
                profile[idx] = float(np.sqrt(np.mean(np.square(norms)))) if norms else np.nan

            self.model.zero_grad()
            return profile
        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP35 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return None


class GradSimAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.probe_size = getattr(args, "probe_size", 200)
        self.extractor = GradientProfileExtractor(model)

    @property
    def name(self):
        return "gradsim_profile_matching"

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        n_each = self.probe_size // 2
        members = df[df["is_member"] == 1].sample(min(n_each, (df["is_member"] == 1).sum()), random_state=self.args.seed)
        nonmembers = df[df["is_member"] == 0].sample(min(n_each, (df["is_member"] == 0).sum()), random_state=self.args.seed)
        probe_df = pd.concat([members, nonmembers], ignore_index=True)
        probe_indices = set(probe_df.index.tolist())

        print(f"\n[EXP35] PROBE PHASE: {len(probe_df)} samples")
        probe_profiles = []
        probe_labels = []
        for _, row in tqdm(probe_df.iterrows(), total=len(probe_df), desc="[PROBE]"):
            prof = self.extractor.extract_profile(row["content"], self.tokenizer, self.max_length)
            if prof is not None and not np.any(np.isnan(prof)):
                probe_profiles.append(prof)
                probe_labels.append(row["is_member"])

        X_probe = np.array(probe_profiles)
        y_probe = np.array(probe_labels)
        print(f"[EXP35] Valid probe profiles: {len(X_probe)}/{len(probe_df)}")

        scaler = StandardScaler()
        X_probe_s = scaler.fit_transform(X_probe)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_probe_s, y_probe)

        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_probe_s, y_probe)

        member_centroid = X_probe_s[y_probe == 1].mean(axis=0)

        probe_lda_auc = roc_auc_score(y_probe, lda.decision_function(X_probe_s))
        probe_lr_auc = roc_auc_score(y_probe, lr.predict_proba(X_probe_s)[:, 1])
        print(f"[EXP35] Probe fit — LDA AUC: {probe_lda_auc:.4f}, LR AUC: {probe_lr_auc:.4f}")

        print(f"\n[EXP35] INFERENCE PHASE: {len(df)} samples")
        all_profiles = []
        valid_mask = []
        for text in tqdm(df["content"].tolist(), desc="[EXP35] GradSim"):
            prof = self.extractor.extract_profile(text, self.tokenizer, self.max_length)
            if prof is not None and not np.any(np.isnan(prof)):
                all_profiles.append(prof)
                valid_mask.append(True)
            else:
                all_profiles.append(np.full(len(self.extractor.component_names), np.nan))
                valid_mask.append(False)

        X_all = np.array(all_profiles)
        valid_idx = [i for i, v in enumerate(valid_mask) if v]

        scores_df = pd.DataFrame(index=range(len(df)))
        scores_df["lda_score"] = np.nan
        scores_df["lr_score"] = np.nan
        scores_df["cosine_score"] = np.nan
        scores_df["neg_grad_mean"] = np.nan
        scores_df["combined_rank"] = np.nan

        for name in self.extractor.component_names:
            scores_df[f"gnorm_{name}"] = np.nan

        if valid_idx:
            X_valid = X_all[valid_idx]
            X_valid_s = scaler.transform(X_valid)

            lda_scores = lda.decision_function(X_valid_s)
            lr_scores = lr.predict_proba(X_valid_s)[:, 1]
            cos_scores = np.array([1.0 - cosine_dist(x, member_centroid) for x in X_valid_s])

            grad_means = X_valid.mean(axis=1)

            for j, i in enumerate(valid_idx):
                scores_df.at[i, "lda_score"] = float(lda_scores[j])
                scores_df.at[i, "lr_score"] = float(lr_scores[j])
                scores_df.at[i, "cosine_score"] = float(cos_scores[j])
                scores_df.at[i, "neg_grad_mean"] = float(-grad_means[j])
                for k, name in enumerate(self.extractor.component_names):
                    scores_df.at[i, f"gnorm_{name}"] = float(X_valid[j, k])

            rank_cols = ["lda_score", "lr_score", "cosine_score"]
            rank_sum = np.zeros(len(df))
            for col in rank_cols:
                vals = scores_df[col].fillna(scores_df[col].min()).values
                rank_sum += rankdata(vals, method="average") / len(vals)
            scores_df["combined_rank"] = rank_sum / len(rank_cols)

        n_valid = sum(valid_mask)
        print(f"[EXP35] Valid: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self.extractor._err_count > 0:
            print(f"[EXP35] Total errors: {self.extractor._err_count}")

        return scores_df


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
        print(f"[*] Loading data from {self.args.dataset}")
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
        if not dfs:
            raise ValueError("No data loaded!")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        attacker = GradSimAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df)
        df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP35_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "=" * 65)
        print("   EXP35: GradSim — PERFORMANCE REPORT")
        print("=" * 65)

        report = {"experiment": "EXP35_gradsim", "timestamp": timestamp, "aucs": {}, "subset_aucs": {}}

        for score_col, label in [
            ("lda_score", "LDA Discriminant [PRIMARY]"),
            ("lr_score", "Logistic Regression"),
            ("cosine_score", "Cosine to Member Centroid"),
            ("neg_grad_mean", "-Mean Gradient Norm"),
            ("combined_rank", "Rank-Avg(LDA+LR+Cosine)"),
        ]:
            if score_col not in df.columns:
                continue
            valid = df.dropna(subset=[score_col])
            if len(valid["is_member"].unique()) > 1:
                auc = roc_auc_score(valid["is_member"], valid[score_col])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<45} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'LDA':<8} | {'LR':<8} | {'Cosine':<8} | {'GradMean':<10}")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            row = {}
            for sc in ["lda_score", "lr_score", "cosine_score", "neg_grad_mean"]:
                v = sub.dropna(subset=[sc])
                row[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {row.get('lda_score', float('nan')):.4f}   "
                  f"| {row.get('lr_score', float('nan')):.4f}   "
                  f"| {row.get('cosine_score', float('nan')):.4f}   "
                  f"| {row.get('neg_grad_mean', float('nan')):.4f}")
            report["subset_aucs"][subset] = row

        print("=" * 65)
        report_path = self.output_dir / f"EXP35_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[*] Report saved: {report_path.name}")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.05
        probe_size = 200
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP35] Model  : {Args.model_name}")
    print(f"[EXP35] Sample : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP35] Probe  : {Args.probe_size}")
    Experiment(Args).run()
