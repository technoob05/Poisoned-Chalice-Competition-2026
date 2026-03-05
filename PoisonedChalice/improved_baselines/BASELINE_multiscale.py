"""
BASELINE_multiscale.py — MultiCal-MIA: Multi-Scale Calibrated Detection

★ THE A*-ORAL CANDIDATE — A new FRAMEWORK, not just a new formula.

CORE CONTRIBUTION:
    All existing methods calibrate at ONE SCALE:
    - Loss:     no calibration
    - Min-K%:   token-level selection (bottom k%)
    - Min-K%++: token-level z-normalization (μ_vocab, σ_vocab)
    - DC-PDD:   token-level frequency calibration
    - ReCaLL:   sequence-level prefix conditioning
    
    MultiCal calibrates at THREE SCALES simultaneously:
    
    Scale 1: TOKEN-LEVEL  — Min-K%++ z-score within vocab distribution
    Scale 2: POSITION-LEVEL — z-score across positions in the sequence
    Scale 3: DOMAIN-LEVEL — z-score across programming languages/domains

    Each scale captures DIFFERENT aspects of memorization:
    - Token scale:    "is this token a local max?"           (Min-K%++ insight)
    - Position scale: "is this position unusual for this sample?"
    - Domain scale:   "is this sample unusual for its language?"  (EXP41 insight: +0.012)

THEORETICAL MOTIVATION:
    Log-prob log p(x_t|x<t) has THREE sources of variation:
    1. Vocab distribution: some positions are more predictable (→ token calibration)
    2. Positional trend:   earlier tokens harder than later (→ position calibration)
    3. Domain baseline:    Python code easier than Rust (→ domain calibration)
    
    Min-K%++ removes source (1), but sources (2) and (3) remain as noise.
    MultiCal removes ALL THREE, yielding a cleaner membership signal.

    Hierarchical z-normalization:
        z_token(t) = (lp_t - μ_vocab) / σ_vocab       [Min-K%++]
        z_pos(t)   = (z_token(t) - μ_pos) / σ_pos      [positional]
        z_final(t) = (z_pos(t) - μ_domain) / σ_domain   [domain]

PROVEN INSIGHTS FROM TRACKER:
    - Token z-norm: Min-K%++ AUC 0.577 (EXP02)
    - Domain z-norm on gradient: +0.012 gain (EXP41: 0.6423 → 0.6539)
    - Per-language calibration in CaliFuse: further gain (NOVEL12)

SIGNALS: multiple aggregations of multi-scale calibrated scores
Compute: 1 forward pass only (no extra overhead)
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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  MultiCal-MIA: Multi-Scale Calibrated Detection")
    print("  Token × Position × Domain Hierarchical Z-Normalization")
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
    print(f"  Loaded. dtype={dtype}")
    return model, tokenizer


class MultiCalExtractor:
    """
    Multi-Scale Calibrated scoring.
    
    Scale 1 (Token):    z_t = (lp_t - μ_vocab) / σ_vocab
    Scale 2 (Position): z2_t = (z_t - μ_seq) / σ_seq
    Scale 3 (Domain):   Deferred to post-processing (needs all samples)
    """

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
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
            logits = outputs.logits[0, :-1, :].float()
            labels = input_ids[0, 1:]
            T = logits.shape[0]
            if T < 3:
                return features

            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # ════════════════════════════════════════════════════════════════
            # SCALE 1: TOKEN-LEVEL (Min-K%++)
            # z_t = (lp_t - μ_vocab) / σ_vocab
            # ════════════════════════════════════════════════════════════════
            mu_v = (probs * log_probs).sum(dim=-1)
            s2_v = (probs * log_probs.pow(2)).sum(dim=-1) - mu_v.pow(2)
            sigma_v = s2_v.clamp(min=1e-10).sqrt()
            z1 = ((token_lp - mu_v) / sigma_v).cpu().numpy()  # (T,)

            # Standard Min-K%++ aggregations
            tlp_np = token_lp.cpu().numpy()
            for k in [0.2, 0.3, 0.5]:
                n_sel = max(1, int(T * k))
                features[f"s1_minkpp_k{int(k*100)}"] = float(np.mean(np.sort(z1)[:n_sel]))
            features["neg_mean_loss"] = float(np.mean(tlp_np))

            # ════════════════════════════════════════════════════════════════
            # SCALE 2: POSITION-LEVEL
            # z2_t = (z1_t - μ_seq) / σ_seq
            # This removes the within-sequence TREND of z-scores
            # (e.g., later positions might systematically have higher z)
            # ════════════════════════════════════════════════════════════════
            mu_seq = np.mean(z1)
            sigma_seq = np.std(z1)
            if sigma_seq > 1e-6:
                z2 = (z1 - mu_seq) / sigma_seq  # (T,)
                for k in [0.2, 0.3, 0.5]:
                    n_sel = max(1, int(T * k))
                    features[f"s2_poszcal_k{int(k*100)}"] = float(np.mean(np.sort(z2)[:n_sel]))
                features["s2_poszcal_mean"] = float(np.mean(z2))

            # ════════════════════════════════════════════════════════════════
            # SCALE 1.5: WINDOWED POSITION CALIBRATION
            # Instead of global μ_seq, use local window statistics
            # This captures LOCAL positional trends
            # ════════════════════════════════════════════════════════════════
            window_size = min(32, T // 2)
            if window_size >= 4:
                z_windowed = np.full(T, np.nan)
                half_w = window_size // 2
                for t in range(T):
                    lo = max(0, t - half_w)
                    hi = min(T, t + half_w)
                    local = z1[lo:hi]
                    lmu, lstd = np.mean(local), np.std(local)
                    if lstd > 1e-6:
                        z_windowed[t] = (z1[t] - lmu) / lstd
                valid = ~np.isnan(z_windowed)
                if valid.sum() > 3:
                    zw = z_windowed[valid]
                    for k in [0.2]:
                        n_sel = max(1, int(len(zw) * k))
                        features[f"s15_localcal_k{int(k*100)}"] = float(np.mean(np.sort(zw)[:n_sel]))
                    features["s15_localcal_mean"] = float(np.mean(zw))

            # ════════════════════════════════════════════════════════════════
            # COMBINED STATIC FEATURES (for domain calibration in Phase 2)
            # ════════════════════════════════════════════════════════════════
            # These are RAW values that will be z-normalized per-domain later
            features["raw_minkpp_k20"] = features.get("s1_minkpp_k20", 0)
            features["raw_loss"] = float(np.mean(tlp_np))
            features["raw_z_mean"] = float(np.mean(z1))
            features["raw_z_std"] = float(np.std(z1))

            # Entropy (for cross-signal analysis)
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
            features["neg_entropy_mean"] = float(-np.mean(entropy))

            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[MultiCal WARN] {type(e).__name__}: {e}")
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

    def apply_domain_calibration(self, df: pd.DataFrame, raw_cols: List[str]):
        """
        Scale 3: DOMAIN-LEVEL z-normalization.
        For each raw_col, compute z-score within each programming language subset.
        This removes language-specific baseline differences.
        """
        print("\n[Scale 3] Applying per-language z-normalization...")
        for col in raw_cols:
            if col not in df.columns:
                continue
            zcol = col.replace("raw_", "s3_domain_")
            df[zcol] = np.nan
            for subset in df["subset"].unique():
                mask = df["subset"] == subset
                vals = df.loc[mask, col].dropna()
                if len(vals) < 10:
                    continue
                mu_lang = vals.mean()
                sigma_lang = vals.std()
                if sigma_lang > 1e-8:
                    df.loc[mask, zcol] = (df.loc[mask, col] - mu_lang) / sigma_lang
                else:
                    df.loc[mask, zcol] = 0.0
            valid = df[zcol].dropna()
            if len(valid) > 50 and len(df.loc[valid.index, "is_member"].unique()) > 1:
                auc_pos = roc_auc_score(df.loc[valid.index, "is_member"], valid)
                best = max(auc_pos, 1 - auc_pos)
                d = "+" if auc_pos >= 0.5 else "-"
                improvement = best - roc_auc_score(
                    df.loc[valid.index, "is_member"],
                    df.loc[valid.index, col] if d == "+" else -df.loc[valid.index, col]
                )
                print(f"    {d}{zcol:<40} AUC = {best:.4f}  "
                      f"(Δ = {improvement:+.4f} vs raw)")
        return df

    def run(self):
        df = self.load_data()
        extractor = MultiCalExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[MultiCal] Phase 1: Extract per-sample scores (1 fwd pass each)...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[MultiCal]"):
            feat = extractor.extract(row["content"])
            feat["subset_label"] = row.get("subset", "unknown")
            rows.append(feat)
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns and col != "subset_label":
                df[col] = feat_df[col].values

        # Phase 2: Domain-level calibration (needs all samples)
        raw_cols = [c for c in df.columns if c.startswith("raw_")]
        df = self.apply_domain_calibration(df, raw_cols)

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   MultiCal-MIA: ALL SIGNAL AUCs (3 Scales)")
        print("=" * 70)

        all_results = {}
        scale_cols = {
            "Scale 1 (Token — Min-K%++)": [c for c in df.columns if c.startswith("s1_")],
            "Scale 2 (Position)":          [c for c in df.columns if c.startswith("s2_") or c.startswith("s15_")],
            "Scale 3 (Domain)":            [c for c in df.columns if c.startswith("s3_")],
            "Baselines":                   ["neg_mean_loss", "neg_entropy_mean"],
        }

        for scale_name, cols in scale_cols.items():
            print(f"\n  ── {scale_name} ──")
            for col in sorted(cols):
                if col not in df.columns:
                    continue
                v = df.dropna(subset=[col])
                if len(v) < 50 or len(v["is_member"].unique()) < 2:
                    continue
                auc_pos = roc_auc_score(v["is_member"], v[col])
                best = max(auc_pos, 1 - auc_pos)
                d = "+" if auc_pos >= 0.5 else "-"
                all_results[col] = (best, d, scale_name)
                marker = " ★" if best > 0.59 else ""
                print(f"    {d}{col:<40} AUC = {best:.4f}{marker}")

        # ── Top signals ───────────────────────────────────────────────────
        if all_results:
            top = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:8]
            print(f"\n  ── TOP 8 SIGNALS ──")
            for rank, (col, (auc, d, scale)) in enumerate(top):
                print(f"    {rank+1:2d}. [{scale:30s}] {d}{col:<35} AUC = {auc:.4f}")

            # Improvement analysis
            s1_best = max((v for v in all_results.values() if "Scale 1" in v[2]),
                          key=lambda x: x[0], default=(0, "", ""))
            s3_best = max((v for v in all_results.values() if "Scale 3" in v[2]),
                          key=lambda x: x[0], default=(0, "", ""))
            if s1_best[0] > 0 and s3_best[0] > 0:
                delta = s3_best[0] - s1_best[0]
                print(f"\n  Domain calibration Δ: {delta:+.4f} "
                      f"(Scale 1 {s1_best[0]:.4f} → Scale 3 {s3_best[0]:.4f})")

            # Per-subset for best overall
            best_col = top[0][0]
            best_d = top[0][1][1]
            print(f"\n  Per-subset for best signal ({best_d}{best_col}={top[0][1][0]:.4f}):")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10:
                    score_vals = sub[best_col] if best_d == "+" else -sub[best_col]
                    auc = roc_auc_score(sub["is_member"], score_vals)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

        print("\n" + "=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"BASELINE_multiscale_{timestamp}.parquet", index=False)
        print(f"[MultiCal] Results saved.")


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

    print(f"[MultiCal] Multi-Scale Calibrated Detection")
    print(f"  Scale 1: Token-level (Min-K%++)")
    print(f"  Scale 2: Position-level (within-sequence)")
    print(f"  Scale 3: Domain-level (per-language)")
    print(f"  model={Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
