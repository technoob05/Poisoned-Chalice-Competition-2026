"""
EXPERIMENT 55: Document-Level Perplexity MIA with Histogram Features

Paper: "Did the neurons read your book? Document-level membership inference
        for large language models"
       Meeus, Jain, Rei, de Montjoye (USENIX Security 2024)

Survey reference: Wu & Cao (arXiv:2503.19338v3, Aug 2025), Section 4.1 [40]

Core idea:
    Instead of using a single aggregate (mean loss, perplexity) as the
    membership signal, extract the DISTRIBUTION of per-token probabilities
    and encode it as HISTOGRAM features + aggregate statistics.
    Then train a Random Forest classifier on these features.

    The insight: the SHAPE of the probability distribution across tokens
    is more informative than just the mean. Members may have a characteristic
    bimodal distribution (some tokens very high probability = memorized
    patterns, others moderate) while non-members have a more uniform
    distribution.

    Feature extraction for each document:
    1. Compute per-token log-probabilities: log p(x_i | x_{<i})
    2. Normalize to z-scores (optional)
    3. Extract AGGREGATE features: mean, std, min, max, percentiles
    4. Extract HISTOGRAM features: bin the log-probs into fixed bins
       and use bin counts/fractions as features
    5. Additional: entropy of the histogram, skewness, kurtosis

    Train Random Forest on (aggregate + histogram) features.

How this differs from our existing experiments:
    - EXP01 uses only MEAN loss → single scalar
    - EXP16 (SURP) uses mean - std → still just two scalars
    - EXP36 (CodeMIF) uses confidence percentage + anchor-body gap
    - EXP55 uses the FULL DISTRIBUTION of token probabilities as
      histogram features → captures distribution SHAPE

    The paper reports document-level MIA works better than sentence-level
    for LLMs, which aligns with our setup (code files = documents).

Compute: 1 forward pass per sample. 10% sample.
Expected runtime: ~5-10 min on A100 (just forward passes + lightweight RF)
Expected AUC: 0.55-0.65 (histogram captures richer information than mean;
    RF can learn non-linear patterns in distribution shape)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP55: Document-Level Histogram MIA")
    print("  Paper: Meeus et al. (USENIX Security 2024)")
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
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}")
    return model, tokenizer


class HistogramExtractor:
    """Extract aggregate + histogram features from per-token log-probabilities."""

    # Fixed bins for log-probability histogram (covers range [-15, 0])
    HIST_BINS = np.array([-15, -12, -10, -8, -6, -5, -4, -3, -2.5, -2,
                          -1.5, -1, -0.7, -0.5, -0.3, -0.1, 0.0])

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_bins = len(self.HIST_BINS) - 1
        self._err_count = 0

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 10:
                return result

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, :-1, :].float()
            labels = input_ids[0, 1:]
            T = logits.shape[0]

            log_probs = F.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            ll_np = token_ll.cpu().numpy()  # shape (T,)

            # === AGGREGATE FEATURES ===
            result["agg_mean"] = float(ll_np.mean())
            result["agg_std"] = float(ll_np.std())
            result["agg_min"] = float(ll_np.min())
            result["agg_max"] = float(ll_np.max())
            result["agg_median"] = float(np.median(ll_np))

            # Percentiles
            for p in [5, 10, 25, 75, 90, 95]:
                result[f"agg_p{p}"] = float(np.percentile(ll_np, p))

            # Range and IQR
            result["agg_range"] = float(ll_np.max() - ll_np.min())
            result["agg_iqr"] = float(np.percentile(ll_np, 75) - np.percentile(ll_np, 25))

            # Higher-order moments
            if T >= 4:
                result["agg_skew"] = float(scipy_stats.skew(ll_np))
                result["agg_kurtosis"] = float(scipy_stats.kurtosis(ll_np))
            else:
                result["agg_skew"] = 0.0
                result["agg_kurtosis"] = 0.0

            # Fraction of "high confidence" tokens (log_prob > -0.5 = prob > 0.6)
            result["frac_confident"] = float(np.mean(ll_np > -0.5))
            # Fraction of "low confidence" tokens (log_prob < -5 = prob < 0.007)
            result["frac_uncertain"] = float(np.mean(ll_np < -5.0))

            # SURP-like: mean of bottom K%
            k_pct = 0.20
            k = max(1, int(T * k_pct))
            sorted_ll = np.sort(ll_np)
            result["bottom_20_mean"] = float(sorted_ll[:k].mean())
            result["top_20_mean"] = float(sorted_ll[-k:].mean())

            # === HISTOGRAM FEATURES ===
            hist_counts, _ = np.histogram(ll_np, bins=self.HIST_BINS)
            hist_fracs = hist_counts / (T + 1e-10)

            for i in range(self.n_bins):
                result[f"hist_bin_{i}"] = float(hist_fracs[i])

            # Histogram entropy (how spread out the distribution is)
            hist_probs = hist_fracs + 1e-10
            hist_probs = hist_probs / hist_probs.sum()
            result["hist_entropy"] = float(-np.sum(hist_probs * np.log(hist_probs)))

            # Mode bin (most frequent)
            result["hist_mode_bin"] = float(np.argmax(hist_counts))

            # Bimodality coefficient (high = more bimodal = potential memorization pattern)
            n = len(ll_np)
            if n >= 3 and result["agg_std"] > 1e-10:
                skew = result["agg_skew"]
                kurt = result["agg_kurtosis"]
                bc = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))
                result["bimodality_coeff"] = float(bc)
            else:
                result["bimodality_coeff"] = 0.0

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP55 WARN] {type(e).__name__}: {e}")
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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed,
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = HistogramExtractor(
            self.model, self.tokenizer, max_length=self.args.max_length,
        )

        print(f"\n[EXP55] Extracting histogram features for {len(df)} samples...")
        print(f"  Histogram bins: {extractor.n_bins}")
        print(f"  1 forward pass per sample")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP55]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[EXP55] Valid: {n_valid}/{len(df)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised signal AUCs ---
        print("\n" + "=" * 70)
        print("   EXP55: UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        unsup_aucs = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            vals = v[col].values
            if np.std(vals) < 1e-15:
                continue
            auc = roc_auc_score(v["is_member"], vals)
            auc_neg = roc_auc_score(v["is_member"], -vals)
            best = max(auc, auc_neg)
            direction = "+" if auc >= auc_neg else "-"
            unsup_aucs[col] = (best, direction)

        top_features = sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True)[:15]
        print("\nTop 15 features (unsupervised):")
        for col, (auc, direction) in top_features:
            print(f"  {direction}{col:<35} AUC = {auc:.4f}")

        # --- Random Forest (5-fold CV) ---
        print("\n" + "=" * 70)
        print("   EXP55: RANDOM FOREST CLASSIFIER (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy()
        y_all = df.loc[valid_mask, "is_member"].values

        X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)
        feature_names = list(X_all.columns)
        X_np = X_all.values.astype(np.float64)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)
        fold_aucs = []
        all_scores = np.full(len(X_np), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_np, y_all)):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced",
                random_state=self.args.seed, n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)

            proba = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fold_aucs.append(auc)
            all_scores[test_idx] = proba

            print(f"  Fold {fold_idx+1}/{n_folds}: AUC = {auc:.4f}")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  RF CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        df.loc[valid_mask, "hist_rf_score"] = all_scores

        # Feature importance
        print("\n--- Top 15 Feature Importances (last fold RF) ---")
        importances = clf.feature_importances_
        imp_idx = np.argsort(importances)[::-1][:15]
        for rank, idx in enumerate(imp_idx):
            print(f"  {rank+1:2d}. {feature_names[idx]:<35} imp = {importances[idx]:.4f}")

        # --- Comparison ---
        print("\n" + "=" * 70)
        print("   EXP55: COMPARISON")
        print("=" * 70)
        agg_mean_auc = unsup_aucs.get("agg_mean", (0.5, "+"))[0]
        print(f"  Histogram RF (5-fold CV): {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  agg_mean alone (=loss):   {agg_mean_auc:.4f}")
        print(f"  vs EXP01 raw loss:        0.5807")
        print(f"  vs EXP16 SURP:            0.5884")
        print(f"  vs EXP41 -grad_z_lang:    0.6539 (current best)")

        if mean_auc > agg_mean_auc:
            print(f"\n  Histogram features ADD value: RF {mean_auc:.4f} > mean_loss {agg_mean_auc:.4f}")
        else:
            print(f"\n  Histogram features do NOT help: RF {mean_auc:.4f} <= mean_loss {agg_mean_auc:.4f}")

        # Per-subset
        print(f"\n{'Subset':<10} | {'RF_AUC':<10} | N")
        print("-" * 35)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["hist_rf_score"])
            if not v.empty and len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v["hist_rf_score"])
            else:
                auc = float("nan")
            print(f"  {subset:<10} | {auc:.4f}    | {len(sub)}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP55_{timestamp}.parquet", index=False)
        print(f"\n[EXP55] Results saved.")


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

    print(f"[EXP55] Document-Level Histogram MIA: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  1 fwd pass/sample, lightweight histogram + RF")
    Experiment(Args).run()
