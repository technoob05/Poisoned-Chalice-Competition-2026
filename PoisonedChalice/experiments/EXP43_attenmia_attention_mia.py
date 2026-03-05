"""
EXPERIMENT 43: AttenMIA — Attention-Based Membership Inference Attack

Paper: "AttenMIA: LLM Membership Inference Attack through Attention Signals"
       Zaree et al. (arXiv:2601.18110, Jan 2026)

Key paper claims:
    - Attention patterns encode memorization: members show more concentrated,
      structured attention flows across layers compared to non-members.
    - Transitional features (consistency, barycenter drift between adjacent layers)
      and perturbation features (attention shift under token drop/replace/prefix)
      together achieve up to 0.996 AUC on WikiMIA-32.
    - On code (GitHub subset of MIMIR): 0.99-1.00 AUC with Pythia models.
    - Deduplication does NOT reduce attention-based membership signals.

Our adaptations for StarCoder2-3b + Poisoned Chalice (code domain):
    - attn_implementation="eager" to extract raw attention weights
    - max_length=512 (memory: 32 layers × 32 heads × 512² ≈ 1 GB attention)
    - Feature aggregation: per-head → head-mean → layer-pair stats → 29 features
    - Ridge LR classifier (Insight 15: linear model + large probe = robust)
    - Probe 1000 (500M + 500NM)

Feature families (29 total):
    A. Transitional (15): consistency {corr, frob, KL} + barycenter {mean, var}
       between adjacent layers, aggregated as mean/std/late across layer pairs
    B. Concentration (5): KL(attention || uniform) per layer, aggregated stats
    C. Perturbation (9): concentration shift under 3 perturbation types
       (token drop, token replace, prefix insertion) — 3 stats each

Compute: 4 forward passes per sample (1 orig + 3 perturbations), forward-only
Expected runtime: ~30-60 min on A100 (5% sample)
Expected AUC: 0.55-0.75 (new signal family, untested on code LLMs)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TRANSITIONAL_COLS = [
    "trans_corr_mean", "trans_corr_std", "trans_corr_late",
    "trans_frob_mean", "trans_frob_std", "trans_frob_late",
    "trans_kl_mean", "trans_kl_std", "trans_kl_late",
    "trans_bary_mean_mean", "trans_bary_mean_std", "trans_bary_mean_late",
    "trans_bary_var_mean", "trans_bary_var_std", "trans_bary_var_late",
]
CONCENTRATION_COLS = [
    "conc_mean", "conc_std", "conc_max", "conc_late_mean", "conc_early_late_diff",
]
PERTURBATION_COLS = [
    "pert_drop_shift_mean", "pert_drop_shift_std", "pert_drop_shift_max",
    "pert_replace_shift_mean", "pert_replace_shift_std", "pert_replace_shift_max",
    "pert_prefix_shift_mean", "pert_prefix_shift_std", "pert_prefix_shift_max",
]
ALL_FEATURE_COLS = TRANSITIONAL_COLS + CONCENTRATION_COLS + PERTURBATION_COLS


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP43: AttenMIA — Attention-Based MIA")
    print("  Paper: Zaree et al. (arXiv:2601.18110, Jan 2026)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, attn_impl: str = "eager"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto", attn_implementation=attn_impl,
        )
    except Exception as e:
        print(f"[WARN] attn_implementation='{attn_impl}' failed ({e}), trying default")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()
    print(f"[*] Model loaded. dtype={dtype}, attn_impl={attn_impl}")
    return model, tokenizer


class AttenMIAExtractor:
    """Extract attention-based features following the AttenMIA framework."""

    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self._n_layers = None
        self._n_heads = None

    @torch.no_grad()
    def _get_attentions(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.model(input_ids=input_ids, output_attentions=True)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attention weights")
        attentions = [a.squeeze(0).float().cpu() for a in outputs.attentions]
        if self._n_layers is None:
            self._n_layers = len(attentions)
            self._n_heads = attentions[0].shape[0]
            print(f"[AttenMIA] Detected {self._n_layers} layers, {self._n_heads} heads")
        return attentions

    def _kl_concentration(self, attentions: List[torch.Tensor]) -> np.ndarray:
        """KL(attention_row || uniform) per (layer, head), averaged across tokens.
        Higher = more concentrated. Shape: (n_layers, n_heads)."""
        result = []
        for A in attentions:
            H, T, _ = A.shape
            if T < 2:
                result.append(np.full(H, np.nan))
                continue
            eps = 1e-10
            # KL(row || U_T) = sum(row * log(row)) + log(T)
            kl_per_row = (A * torch.log(A + eps)).sum(dim=-1) + np.log(T)  # (H, T)
            skip = max(1, T // 10)
            kl_per_head = kl_per_row[:, skip:].mean(dim=-1)  # (H,)
            result.append(kl_per_head.numpy())
        return np.array(result)

    def _transitional_features(self, attentions: List[torch.Tensor]) -> Dict[str, float]:
        n_layers = len(attentions)
        metrics = {k: [] for k in ["corr", "frob", "kl", "bary_mean", "bary_var"]}

        for l in range(n_layers - 1):
            A_l = attentions[l]
            A_l1 = attentions[l + 1]
            H, T, _ = A_l.shape

            # Consistency-Frob
            diff = A_l1 - A_l
            frob = (diff ** 2).sum(dim=(1, 2)).sqrt() / (T * T)
            metrics["frob"].append(frob.mean().item())

            # Consistency-KL
            eps = 1e-10
            kl = (A_l * torch.log((A_l + eps) / (A_l1 + eps))).sum(dim=-1).mean(dim=-1)
            metrics["kl"].append(kl.mean().item())

            # Barycenter drift
            positions = torch.arange(T, dtype=torch.float32).view(1, 1, T)
            bary_l = (A_l * positions).sum(dim=-1)
            bary_l1 = (A_l1 * positions).sum(dim=-1)
            drift = (bary_l1 - bary_l).abs()
            metrics["bary_mean"].append(drift.mean().item())
            metrics["bary_var"].append(drift.var().item())

            # Consistency-Corr (vectorized across heads)
            flat_l = A_l.reshape(H, -1)
            flat_l1 = A_l1.reshape(H, -1)
            m_l = flat_l.mean(dim=-1, keepdim=True)
            m_l1 = flat_l1.mean(dim=-1, keepdim=True)
            c_l = flat_l - m_l
            c_l1 = flat_l1 - m_l1
            cov = (c_l * c_l1).sum(dim=-1)
            corr = cov / (c_l.norm(dim=-1) * c_l1.norm(dim=-1) + 1e-10)
            metrics["corr"].append(corr.mean().item())

        features = {}
        n_late_start = max(0, n_layers - 11)
        for name in ["corr", "frob", "kl", "bary_mean", "bary_var"]:
            vals = np.array(metrics[name])
            valid = vals[np.isfinite(vals)]
            features[f"trans_{name}_mean"] = float(valid.mean()) if len(valid) > 0 else np.nan
            features[f"trans_{name}_std"] = float(valid.std()) if len(valid) > 1 else np.nan
            late = vals[n_late_start:]
            late_valid = late[np.isfinite(late)]
            features[f"trans_{name}_late"] = float(late_valid.mean()) if len(late_valid) > 0 else np.nan

        return features

    def _concentration_features(self, conc: np.ndarray) -> Dict[str, float]:
        head_avg = np.nanmean(conc, axis=1)
        valid = head_avg[np.isfinite(head_avg)]
        features = {}
        if len(valid) > 0:
            features["conc_mean"] = float(valid.mean())
            features["conc_std"] = float(valid.std())
            features["conc_max"] = float(valid.max())
            features["conc_late_mean"] = float(valid[-10:].mean()) if len(valid) >= 10 else float(valid.mean())
            if len(valid) >= 20:
                features["conc_early_late_diff"] = float(valid[-10:].mean() - valid[:10].mean())
            else:
                features["conc_early_late_diff"] = np.nan
        else:
            for k in CONCENTRATION_COLS:
                features[k] = np.nan
        return features

    def _perturb_input(self, input_ids: torch.Tensor, method: str) -> torch.Tensor:
        ids = input_ids.clone()
        T = ids.shape[1]

        if method == "drop":
            n_drop = min(7, max(1, T // 8))
            if T <= n_drop + 4:
                return ids
            positions = np.linspace(1, T - 2, n_drop, dtype=int)
            mask = torch.ones(T, dtype=torch.bool)
            for p in positions:
                mask[p] = False
            ids = ids[:, mask]

        elif method == "replace":
            n_rep = min(7, max(1, T // 8))
            positions = np.linspace(1, T - 2, n_rep, dtype=int)
            vocab_size = getattr(self.tokenizer, "vocab_size", 49152) or 49152
            for p in positions:
                ids[0, p] = random.randint(0, vocab_size - 1)

        elif method == "prefix":
            prefix_text = "/* unrelated placeholder comment for testing purposes */\n"
            prefix_ids = self.tokenizer(
                prefix_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            ids = torch.cat([prefix_ids.to(ids.device), ids], dim=1)
            ids = ids[:, :self.max_length]

        return ids

    def _perturbation_features(self, orig_conc: np.ndarray, input_ids: torch.Tensor) -> Dict[str, float]:
        features = {}
        for method in ["drop", "replace", "prefix"]:
            try:
                pert_ids = self._perturb_input(input_ids, method)
                pert_attns = self._get_attentions(pert_ids.to(self.model.device))
                pert_conc = self._kl_concentration(pert_attns)
                del pert_attns

                n_common = min(orig_conc.shape[0], pert_conc.shape[0])
                eps = 1e-10
                shift = (pert_conc[:n_common] - orig_conc[:n_common]) / (np.abs(orig_conc[:n_common]) + eps)
                valid = shift[np.isfinite(shift)]
                if len(valid) > 0:
                    features[f"pert_{method}_shift_mean"] = float(valid.mean())
                    features[f"pert_{method}_shift_std"] = float(valid.std())
                    features[f"pert_{method}_shift_max"] = float(valid.max())
                else:
                    for stat in ["mean", "std", "max"]:
                        features[f"pert_{method}_shift_{stat}"] = np.nan
            except Exception:
                for stat in ["mean", "std", "max"]:
                    features[f"pert_{method}_shift_{stat}"] = np.nan

        return features

    def extract(self, text: str) -> Dict[str, float]:
        null_result = {k: np.nan for k in ALL_FEATURE_COLS}
        if not text or len(text) < 20:
            return null_result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 10:
                return null_result

            attentions = self._get_attentions(input_ids)
            conc = self._kl_concentration(attentions)

            features = {}
            features.update(self._transitional_features(attentions))
            features.update(self._concentration_features(conc))
            del attentions
            torch.cuda.empty_cache()

            features.update(self._perturbation_features(conc, input_ids))
            return features

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP43 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return null_result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name, args.attn_impl)

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

    def _train_and_eval(self, feat_df, df, feat_cols, label):
        """Train Ridge LR on probe, evaluate on all data."""
        X = feat_df[feat_cols].values
        y = df["is_member"].values

        X_clean = X.copy()
        for j in range(X_clean.shape[1]):
            col = X_clean[:, j]
            col[~np.isfinite(col)] = np.nan
            med = np.nanmedian(col)
            col[np.isnan(col)] = med if np.isfinite(med) else 0.0
            X_clean[:, j] = col

        n_each = self.args.probe_size // 2
        n_m = min(n_each, (y == 1).sum())
        n_nm = min(n_each, (y == 0).sum())
        m_idx = np.where(y == 1)[0]
        nm_idx = np.where(y == 0)[0]
        rng = np.random.RandomState(self.args.seed)
        probe_idx = np.concatenate([
            rng.choice(m_idx, n_m, replace=False),
            rng.choice(nm_idx, n_nm, replace=False),
        ])

        X_probe = X_clean[probe_idx]
        y_probe = y[probe_idx]
        print(f"\n  [{label}] Probe: {len(probe_idx)} ({n_m}M+{n_nm}NM), Features: {len(feat_cols)}")

        cv_aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
        for fold, (tr, va) in enumerate(skf.split(X_probe, y_probe)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_probe[tr])
            X_va = scaler.transform(X_probe[va])
            clf = LogisticRegression(C=self.args.ridge_C, penalty="l2", max_iter=1000, solver="lbfgs")
            clf.fit(X_tr, y_probe[tr])
            auc = roc_auc_score(y_probe[va], clf.predict_proba(X_va)[:, 1])
            cv_aucs.append(auc)
            print(f"    Fold {fold+1}: AUC = {auc:.4f}")
        print(f"    CV Mean: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

        scaler_final = StandardScaler()
        X_all_s = scaler_final.fit_transform(X_clean)
        X_probe_s = X_all_s[probe_idx]
        final_clf = LogisticRegression(C=self.args.ridge_C, penalty="l2", max_iter=1000, solver="lbfgs")
        final_clf.fit(X_probe_s, y_probe)

        coefs = final_clf.coef_[0]
        print(f"    Ridge coefficients (top 5):")
        sorted_coefs = sorted(zip(feat_cols, coefs), key=lambda x: -abs(x[1]))
        for fname, c in sorted_coefs[:5]:
            print(f"      {fname:<35} {c:+.4f}")

        scores = final_clf.predict_proba(X_all_s)[:, 1]
        return scores

    def run(self):
        df = self.load_data()
        extractor = AttenMIAExtractor(
            self.model, self.tokenizer, max_length=self.args.max_length,
        )

        print(f"\n[EXP43] Extracting AttenMIA features for {len(df)} samples...")
        print(f"  (4 forward passes per sample: 1 original + 3 perturbations)")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP43] Extract"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df[ALL_FEATURE_COLS[0]].notna().sum()
        print(f"\n[EXP43] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[EXP43] Errors: {extractor._err_count}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Individual feature AUCs ---
        print("\n" + "=" * 70)
        print("   EXP43: AttenMIA — INDIVIDUAL FEATURE AUCs")
        print("=" * 70)
        indiv_aucs = {}
        for col in ALL_FEATURE_COLS:
            if col not in df.columns:
                continue
            v = df.dropna(subset=[col])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[col])
                indiv_aucs[col] = auc
                tag = " <--" if auc > 0.55 or auc < 0.45 else ""
                print(f"  {col:<40} AUC = {auc:.4f}{tag}")
        if indiv_aucs:
            best_ind = max(indiv_aucs, key=indiv_aucs.get)
            worst_ind = min(indiv_aucs, key=indiv_aucs.get)
            print(f"\n  Best individual:  {best_ind} = {indiv_aucs[best_ind]:.4f}")
            print(f"  Worst individual: {worst_ind} = {indiv_aucs[worst_ind]:.4f}")

        # --- Classifier 1: Transitional-only ---
        print("\n" + "=" * 70)
        print("   CLASSIFIER 1: AttenMIA (Transitional Only)")
        print("=" * 70)
        trans_cols = TRANSITIONAL_COLS + CONCENTRATION_COLS
        scores_trans = self._train_and_eval(feat_df, df, trans_cols, "Transitional")
        df["attn_trans_score"] = scores_trans

        # --- Classifier 2: Full (Transitional + Perturbation) ---
        print("\n" + "=" * 70)
        print("   CLASSIFIER 2: AttenMIA (Full: Transitional + Perturbation)")
        print("=" * 70)
        scores_full = self._train_and_eval(feat_df, df, ALL_FEATURE_COLS, "Full")
        df["attn_full_score"] = scores_full

        # --- Summary report ---
        print("\n" + "=" * 70)
        print("   EXP43: AttenMIA — FINAL REPORT")
        print("=" * 70)
        for sc, label in [
            ("attn_trans_score", "AttenMIA (Transitional)"),
            ("attn_full_score", "AttenMIA (Full: Trans+Pert)"),
        ]:
            v = df.dropna(subset=[sc])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[sc])
                print(f"  {label:<45} AUC = {auc:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang: 0.6539 (current best)")
        print(f"  vs EXP39 Ridge stacker: 0.6490")
        print(f"  vs EXP11 -grad_embed:   0.6472")

        print(f"\n{'Subset':<10} | {'Transitional':<14} | {'Full':<14} | N")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["attn_trans_score", "attn_full_score"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('attn_trans_score', float('nan')):.4f}         "
                  f"| {r.get('attn_full_score', float('nan')):.4f}         "
                  f"| {len(sub)}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP43_{timestamp}.parquet", index=False)
        print(f"\n[EXP43] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.05
        probe_size = 1000
        ridge_C = 0.1
        output_dir = "results"
        max_length = 512
        attn_impl = "eager"
        seed = 42

    print(f"[EXP43] AttenMIA: {Args.sample_fraction*100:.0f}% sample, "
          f"max_len={Args.max_length}, probe={Args.probe_size}")
    Experiment(Args).run()
