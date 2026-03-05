"""
EXPERIMENT 52: Semantic MIA (SMIA) — Embedding-Calibrated Perturbation Attack

Paper: "Semantic Membership Inference Attack against Large Language Models"
       Mozaffari & Marathe, NeurIPS Safe GenAI Workshop 2024

Survey reference: Wu & Cao, "Membership Inference Attacks on Large-Scale
    Models: A Survey" (arXiv:2503.19338v3, Aug 2025), Section 4.3 [45]

Core idea:
    Unlike basic neighborhood comparison (EXP03) which only measures
    loss difference, SMIA also incorporates the SEMANTIC DISTANCE of the
    perturbation using an EXTERNAL embedding model. The attack model is
    a neural network that learns the relationship between:
      (1) loss_diff = L(f, x) - L(f, x')     [loss change from perturbation]
      (2) sem_dist = ||embed(x) - embed(x')||  [how much meaning changed]
    and outputs P(member).

    Key insight: Members show CONSISTENT low loss even under large semantic
    perturbation, while non-members show loss change proportional to
    perturbation magnitude. The relationship between (loss_diff, sem_dist)
    is more discriminative than either alone.

How SMIA differs from our failed perturbation experiments:
    - EXP03 (Neighborhood): Only used loss difference, no semantic calibration
    - EXP07 (Adv Stability): Only used stability score, no semantic distance
    - EXP28 (Delta JSD): Used variable renaming (too weak perturbation)
    - EXP31/32 (Embedding Noise): Measured loss/gradient change, not
      semantically-calibrated. NO EXTERNAL embedding model.
    SMIA's novelty: uses an EXTERNAL code embedding model to measure HOW MUCH
    the perturbation changed the meaning, then a TRAINED classifier learns
    the (loss_change, semantic_distance) → membership mapping.

Adaptation for Poisoned Chalice:
    - External embedding: Use a code-specialized model (e.g., the tokenizer
      embedding layer or a small sentence-transformer) for semantic distance.
      We use the target model's own embedding layer (first hidden state) as
      a pragmatic alternative since no external code embedding model is
      available on Kaggle without extra downloads.
    - Perturbation: Token-level substitution (replace random tokens with
      random vocabulary tokens) at multiple rates [5%, 10%, 20%].
    - Attack model: Logistic Regression (simpler than paper's NN, but more
      robust with limited probe data).

    For each sample x, generate K perturbations at each rate:
      features = [loss_diff_1, sem_dist_1, ..., loss_diff_K, sem_dist_K,
                  mean_loss_diff, mean_sem_dist, slope(loss/sem)]
    → Logistic Regression classifier

Compute: (1 + K*num_rates) forward passes per sample. 10% sample.
Expected runtime: ~15-25 min on A100 (K=3, 3 rates = 10 fwd/sample)
Expected AUC: 0.52-0.62 (perturbation experiments have historically been
    weak in our setup, but semantic calibration is the new element)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP52: Semantic MIA (SMIA)")
    print("  Paper: Mozaffari & Marathe (NeurIPS 2024)")
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
    print(f"  Loaded. dtype={dtype}, vocab_size={tokenizer.vocab_size}")
    return model, tokenizer


class SMIAExtractor:
    """Semantic MIA: perturbation + semantic distance + loss difference."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 perturbation_rates: List[float] = None,
                 n_perturbations: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.perturbation_rates = perturbation_rates or [0.05, 0.10, 0.20]
        self.n_perturbations = n_perturbations
        self.vocab_size = tokenizer.vocab_size
        self._err_count = 0

    def _perturb_tokens(self, input_ids: torch.Tensor, rate: float) -> torch.Tensor:
        """Replace random tokens with random vocabulary tokens."""
        perturbed = input_ids.clone()
        seq_len = input_ids.shape[1]
        n_replace = max(1, int(seq_len * rate))
        positions = random.sample(range(1, seq_len), min(n_replace, seq_len - 1))
        for pos in positions:
            perturbed[0, pos] = random.randint(0, self.vocab_size - 1)
        return perturbed

    @torch.no_grad()
    def _get_loss_and_embedding(self, input_ids: torch.Tensor):
        """Get mean loss and mean embedding (first hidden state) for input."""
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits  # (1, T, V)
        # Use embedding layer output as semantic representation
        embedding = outputs.hidden_states[0]  # (1, T, D) — embedding layer

        seq_len = input_ids.shape[1]

        # Mean loss
        shift_logits = logits[0, :-1, :].float()
        shift_labels = input_ids[0, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mean_loss = -token_ll.mean().item()

        # Mean-pooled embedding
        mean_embed = embedding[0, :seq_len, :].float().mean(dim=0).cpu().numpy()

        return mean_loss, mean_embed

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract SMIA features: loss_diff + semantic_distance per perturbation."""
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

            # Original loss and embedding
            orig_loss, orig_embed = self._get_loss_and_embedding(input_ids)
            result["orig_loss"] = orig_loss

            all_loss_diffs = []
            all_sem_dists = []

            for rate in self.perturbation_rates:
                rate_loss_diffs = []
                rate_sem_dists = []
                r_tag = f"r{int(rate*100)}"

                for k in range(self.n_perturbations):
                    perturbed_ids = self._perturb_tokens(input_ids, rate)
                    pert_loss, pert_embed = self._get_loss_and_embedding(perturbed_ids)

                    loss_diff = pert_loss - orig_loss
                    sem_dist = float(np.linalg.norm(pert_embed - orig_embed))

                    rate_loss_diffs.append(loss_diff)
                    rate_sem_dists.append(sem_dist)

                mean_ld = np.mean(rate_loss_diffs)
                mean_sd = np.mean(rate_sem_dists)
                all_loss_diffs.append(mean_ld)
                all_sem_dists.append(mean_sd)

                result[f"loss_diff_{r_tag}"] = mean_ld
                result[f"sem_dist_{r_tag}"] = mean_sd

                # Ratio: how much loss changes per unit of semantic change
                if mean_sd > 1e-10:
                    result[f"loss_per_sem_{r_tag}"] = mean_ld / mean_sd
                else:
                    result[f"loss_per_sem_{r_tag}"] = 0.0

            # Cross-rate aggregates
            arr_ld = np.array(all_loss_diffs)
            arr_sd = np.array(all_sem_dists)
            result["loss_diff_mean"] = float(arr_ld.mean())
            result["loss_diff_std"] = float(arr_ld.std())
            result["sem_dist_mean"] = float(arr_sd.mean())
            result["sem_dist_std"] = float(arr_sd.std())

            # Slope: how loss_diff scales with perturbation rate
            rates = np.array(self.perturbation_rates)
            if len(rates) >= 2 and arr_ld.std() > 1e-10:
                slope = np.polyfit(rates, arr_ld, 1)[0]
                result["loss_slope"] = float(slope)
            else:
                result["loss_slope"] = 0.0

            # Semantic slope: how embedding distance scales with rate
            if len(rates) >= 2 and arr_sd.std() > 1e-10:
                sem_slope = np.polyfit(rates, arr_sd, 1)[0]
                result["sem_slope"] = float(sem_slope)
            else:
                result["sem_slope"] = 0.0

            # Normalized robustness: loss_diff / sem_dist aggregated
            if arr_sd.mean() > 1e-10:
                result["robustness_ratio"] = float(arr_ld.mean() / arr_sd.mean())
            else:
                result["robustness_ratio"] = 0.0

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP52 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
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
        extractor = SMIAExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            perturbation_rates=self.args.perturbation_rates,
            n_perturbations=self.args.n_perturbations,
        )

        n_fwd = 1 + len(self.args.perturbation_rates) * self.args.n_perturbations
        print(f"\n[EXP52] Extracting SMIA features for {len(df)} samples...")
        print(f"  Perturbation rates: {self.args.perturbation_rates}")
        print(f"  Perturbations per rate: {self.args.n_perturbations}")
        print(f"  Forward passes per sample: {n_fwd}")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP52]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[EXP52] Valid: {n_valid}/{len(df)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised signal AUCs ---
        print("\n" + "=" * 70)
        print("   EXP52: SMIA — UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns]
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

        for col, (auc, direction) in sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True):
            print(f"  {direction}{col:<40} AUC = {auc:.4f}")

        # --- Trained classifier (5-fold CV) ---
        print("\n" + "=" * 70)
        print("   EXP52: SMIA — LOGISTIC REGRESSION (5-fold CV)")
        print("=" * 70)

        clf_features = [c for c in feat_df.columns if c != "orig_loss"]
        valid_mask = feat_df[clf_features].dropna(how="any").index
        X_all = feat_df.loc[valid_mask, clf_features].copy()
        y_all = df.loc[valid_mask, "is_member"].values

        X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)
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

            clf = LogisticRegression(
                C=0.1, max_iter=500, solver="saga",
                class_weight="balanced", random_state=self.args.seed,
            )
            clf.fit(X_train_s, y_train)

            proba = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fold_aucs.append(auc)
            all_scores[test_idx] = proba

            print(f"  Fold {fold_idx+1}/{n_folds}: AUC = {auc:.4f} "
                  f"(train={len(train_idx)}, test={len(test_idx)})")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  SMIA LR CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        df.loc[valid_mask, "smia_lr_score"] = all_scores

        # Feature coefficients
        print("\n--- LR Feature Coefficients (last fold) ---")
        coefs = clf.coef_[0]
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"  {rank+1:2d}. {clf_features[idx]:<35} coef = {coefs[idx]:+.4f}")

        # --- Comparison ---
        print("\n" + "=" * 70)
        print("   EXP52: COMPARISON")
        print("=" * 70)
        orig_loss_auc = unsup_aucs.get("orig_loss", (0.5, "+"))[0]
        print(f"  SMIA LR (5-fold CV):   {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"  -orig_loss (unsup):    {orig_loss_auc:.4f}")
        print(f"  vs EXP03 Neighborhood: 0.5633")
        print(f"  vs EXP07 Adv Stability:0.5365")
        print(f"  vs EXP41 -grad_z_lang: 0.6539 (current best)")

        if mean_auc > orig_loss_auc:
            print(f"\n  Semantic calibration HELPS: {mean_auc:.4f} > {orig_loss_auc:.4f} (raw loss)")
        else:
            print(f"\n  Semantic calibration does NOT help: {mean_auc:.4f} <= {orig_loss_auc:.4f}")

        # Per-subset
        print(f"\n{'Subset':<10} | {'SMIA_LR':<10} | N")
        print("-" * 35)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["smia_lr_score"])
            if not v.empty and len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v["smia_lr_score"])
            else:
                auc = float("nan")
            print(f"  {subset:<10} | {auc:.4f}    | {len(sub)}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP52_{timestamp}.parquet", index=False)
        print(f"\n[EXP52] Results saved.")


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
        perturbation_rates = [0.05, 0.10, 0.20]
        n_perturbations = 3
        output_dir = "results"
        seed = 42

    n_fwd = 1 + len(Args.perturbation_rates) * Args.n_perturbations
    print(f"[EXP52] SMIA: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  rates={Args.perturbation_rates}, K={Args.n_perturbations}")
    print(f"  {n_fwd} fwd passes/sample")
    Experiment(Args).run()
