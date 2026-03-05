"""
NOVEL_LogitSig.py — LogitSignature: The Ultimate Logit-Only Baseline ⭐⭐

THE GOAL: Create a logit-only baseline so comprehensive that it becomes the
standard starting point for ALL future MIA work. Like Min-K%++ but covering
EVERY angle we discovered across 55+ experiments.

INSIGHT SYNTHESIS (all from OUR tracker, zero-overhead logit signals):

From EXP02:  Min-K%++ z-score = static calibration
From EXP16:  SURP (mean - std) = confidence interval
From EXP55:  Histogram shape = distribution structure
From EXP36:  Anchor vs body = holistic memorization
From EXP37:  Loss trajectory derivatives weak but directionally right
From EXP41:  Per-language z-norm = +0.012 consistently

THE LOGIT SIGNATURE — 6 signal families from 1 forward pass:

┌──────────────────┬───────────────────────────────────────────────────────┐
│ Family           │ Signals                                              │
├──────────────────┼───────────────────────────────────────────────────────┤
│ 1. CALIBRATED    │ Min-K%++ z-scores (k=10..100%)                      │
│ 2. DISTRIBUTIONAL│ Gini coefficient, entropy, effective support         │
│ 3. TRAJECTORY    │ Loss slope, curvature, first/second half gap         │
│ 4. TAIL          │ Top-K mass, mode gap, worst-token calibrated score   │
│ 5. CONSISTENCY   │ Z-score variance, surprise jitter, rank stability    │
│ 6. POSITIONAL    │ Early/mid/late loss, loss acceleration               │
└──────────────────┴───────────────────────────────────────────────────────┘

Total: ~30 signals, ALL from 1 forward pass, ALL unsupervised, zero overhead.

This is designed to be THE reference baseline that everyone cites.
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
    print("  LogitSignature: The Ultimate Logit-Only Baseline")
    print("  30 signals × 1 forward pass × zero overhead")
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
    return model, tokenizer


class LogitSigExtractor:
    """
    Comprehensive logit-based feature extraction — 6 signal families.
    All from 1 forward pass, all unsupervised.
    """

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        f = {}
        if not text or len(text) < 20:
            return f
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 8:
                return f

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, :-1, :].float()  # (T, V)
            labels = input_ids[0, 1:]
            T, V = logits.shape
            if T < 5:
                return f

            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            tlp = token_lp.cpu().numpy()

            # ══════════════════════════════════════════════════════════════
            # F1: CALIBRATED — Min-K%++ z-scores
            # ══════════════════════════════════════════════════════════════
            mu = (probs * log_probs).sum(dim=-1)
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()
            z = ((token_lp - mu) / sigma).cpu().numpy()

            for k in [0.1, 0.2, 0.3, 0.5, 1.0]:
                n = max(1, int(T * k))
                f[f"cal_minkpp_k{int(k*100)}"] = float(np.mean(np.sort(z)[:n]))

            f["cal_z_mean"] = float(np.mean(z))

            # SURP: mean + std (from EXP16)
            f["cal_surp"] = float(np.mean(tlp) - np.std(tlp))

            # ══════════════════════════════════════════════════════════════
            # F2: DISTRIBUTIONAL — shape of p(·|x<t)
            # ══════════════════════════════════════════════════════════════
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()
            f["dist_neg_entropy_mean"] = float(-np.mean(entropy))

            # Gini coefficient (from our TopoCal)
            probs_np = probs.cpu().numpy()
            sorted_p = np.sort(probs_np, axis=-1)
            cumsum = np.cumsum(sorted_p, axis=-1)
            gini = 1.0 - 2.0 * np.mean(cumsum, axis=-1)
            f["dist_gini_mean"] = float(np.mean(gini))

            # Effective support = 1/Σp²
            sum_p_sq = (probs ** 2).sum(dim=-1).cpu().numpy()
            eff_support = 1.0 / np.clip(sum_p_sq, 1e-12, None)
            f["dist_neg_eff_support"] = float(-np.mean(eff_support))

            # Top-1 mass (how confident is the model?)
            top1_mass = probs_np.max(axis=-1)
            f["dist_top1_mass"] = float(np.mean(top1_mass))

            # ══════════════════════════════════════════════════════════════
            # F3: TRAJECTORY — loss dynamics across positions
            # ══════════════════════════════════════════════════════════════
            # Loss slope
            positions = np.arange(T, dtype=np.float64)
            pm, lm = positions.mean(), tlp.mean()
            cov = np.sum((positions - pm) * (tlp - lm))
            var_p = np.sum((positions - pm) ** 2)
            if var_p > 1e-10:
                slope = cov / var_p
                f["traj_loss_slope"] = float(slope)

            # First half vs second half loss gap (from EXP37)
            mid = T // 2
            if mid > 2:
                f["traj_loss_gap"] = float(np.mean(tlp[mid:]) - np.mean(tlp[:mid]))
                # Positive = loss improves in second half = model recognizes pattern

            # Thirds: early-mid-late
            third = T // 3
            if third > 2:
                f["traj_loss_early"] = float(np.mean(tlp[:third]))
                f["traj_loss_mid"] = float(np.mean(tlp[third:2*third]))
                f["traj_loss_late"] = float(np.mean(tlp[2*third:]))
                # Acceleration: is the improvement accelerating or decelerating?
                d1 = np.mean(tlp[third:2*third]) - np.mean(tlp[:third])
                d2 = np.mean(tlp[2*third:]) - np.mean(tlp[third:2*third])
                f["traj_acceleration"] = float(d2 - d1)

            # ══════════════════════════════════════════════════════════════
            # F4: TAIL — extreme token analysis
            # ══════════════════════════════════════════════════════════════
            # Mode gap: log p(1st) - log p(2nd) for each position
            sorted_lp = np.sort(probs_np, axis=-1)[:, ::-1]
            mode_gap = np.log(np.clip(sorted_lp[:, 0], 1e-12, None)) - \
                       np.log(np.clip(sorted_lp[:, 1], 1e-12, None))
            f["tail_mode_gap_mean"] = float(np.mean(mode_gap))

            # Min-K% of mode gap — sharpness at WORST positions
            for k in [0.2]:
                n = max(1, int(T * k))
                f[f"tail_mode_gap_mink{int(k*100)}"] = float(np.mean(np.sort(mode_gap)[:n]))

            # Top-1 accuracy (fraction where true token = predicted mode)
            predicted = logits.argmax(dim=-1).cpu()
            actual = labels.cpu()
            f["tail_top1_acc"] = float((predicted == actual).float().mean().item())

            # Worst token surprisal (highest loss tokens)
            surprisals = -tlp
            f["tail_max_surprisal"] = float(np.max(surprisals))
            for k in [0.1, 0.2]:
                n = max(1, int(T * k))
                f[f"tail_worst_k{int(k*100)}"] = float(np.mean(np.sort(surprisals)[-n:]))

            # ══════════════════════════════════════════════════════════════
            # F5: CONSISTENCY — how stable are the signals across positions?
            # ══════════════════════════════════════════════════════════════
            f["con_neg_z_std"] = float(-np.std(z))
            f["con_neg_surprise_std"] = float(-np.std(surprisals))

            # Surprise jitter: |s_{t+1} - s_t| (from CDD)
            if T > 3:
                jitter = np.abs(np.diff(surprisals))
                f["con_neg_jitter"] = float(-np.mean(jitter))

            # Entropy consistency
            f["con_neg_entropy_std"] = float(-np.std(entropy))

            # Z-score signal-to-noise ratio
            z_std = np.std(z)
            if z_std > 1e-6:
                f["con_z_snr"] = float(np.mean(z) / z_std)

            # ══════════════════════════════════════════════════════════════
            # F6: BASELINE — canonical signals
            # ══════════════════════════════════════════════════════════════
            f["base_neg_loss"] = float(np.mean(tlp))
            f["base_neg_loss_std"] = float(-np.std(tlp))
            f["base_neg_loss_p10"] = float(-np.percentile(surprisals, 90))

            f["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[LogitSig WARN] {type(e).__name__}: {e}")
            self._err_count += 1
        return f


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
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def apply_lang_calibration(self, df, raw_cols):
        """Per-language z-normalization (proven +0.012 from EXP41)."""
        print("\n[LangCal] Applying per-language z-normalization...")
        for col in raw_cols:
            if col not in df.columns:
                continue
            zcol = col.replace("cal_", "lcal_").replace("base_", "lcal_")
            if zcol == col:
                zcol = f"lcal_{col}"
            df[zcol] = np.nan
            for subset in df["subset"].unique():
                mask = df["subset"] == subset
                vals = df.loc[mask, col].dropna()
                if len(vals) < 10:
                    continue
                mu, sig = vals.mean(), vals.std()
                if sig > 1e-8:
                    df.loc[mask, zcol] = (df.loc[mask, col] - mu) / sig
        return df

    def run(self):
        df = self.load_data()
        ext = LogitSigExtractor(self.model, self.tokenizer, self.args.max_length)

        print(f"\n[LogitSig] Extracting 6 signal families (~30 features)...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[LogitSig]"):
            rows.append(ext.extract(row["content"]))
        feat_df = pd.DataFrame(rows)
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # Per-language calibration on key signals
        key_cols = ["cal_minkpp_k20", "base_neg_loss", "cal_surp", "cal_z_mean"]
        df = self.apply_lang_calibration(df, key_cols)

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   LogitSignature: THE ULTIMATE LOGIT-ONLY BASELINE")
        print("=" * 70)

        feature_cols = [c for c in df.columns
                        if c not in ["content", "membership", "is_member", "subset", "seq_len"]
                        and c in feat_df.columns or c.startswith("lcal_")]

        families = {
            "F1: CALIBRATED":    [c for c in feature_cols if c.startswith("cal_")],
            "F2: DISTRIBUTIONAL": [c for c in feature_cols if c.startswith("dist_")],
            "F3: TRAJECTORY":    [c for c in feature_cols if c.startswith("traj_")],
            "F4: TAIL":          [c for c in feature_cols if c.startswith("tail_")],
            "F5: CONSISTENCY":   [c for c in feature_cols if c.startswith("con_")],
            "F6: BASELINE":      [c for c in feature_cols if c.startswith("base_")],
            "F7: LANG-CALIBRATED": [c for c in df.columns if c.startswith("lcal_")],
        }

        all_results = {}
        for fam, cols in families.items():
            print(f"\n  ── {fam} ──")
            for col in sorted(cols):
                if col not in df.columns:
                    continue
                v = df.dropna(subset=[col])
                if len(v) < 50 or len(v["is_member"].unique()) < 2:
                    continue
                auc_pos = roc_auc_score(v["is_member"], v[col])
                best = max(auc_pos, 1 - auc_pos)
                d = "+" if auc_pos >= 0.5 else "-"
                all_results[col] = (best, d, fam)
                marker = " ★" if best > 0.59 else ""
                print(f"    {d}{col:<40} AUC = {best:.4f}{marker}")

        if all_results:
            # Top 15 overall
            top = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:15]
            print(f"\n  ═══ TOP 15 LOGIT-ONLY SIGNALS ═══")
            for rank, (col, (auc, d, fam)) in enumerate(top):
                print(f"    {rank+1:2d}. [{fam:22s}] {d}{col:<35} AUC = {auc:.4f}")

            # Family comparison
            print(f"\n  ═══ FAMILY CHAMPION COMPARISON ═══")
            for fam in families:
                fam_results = [(c, v) for c, v in all_results.items() if v[2] == fam]
                if fam_results:
                    best_in_fam = max(fam_results, key=lambda x: x[1][0])
                    print(f"    {fam:25s} → {best_in_fam[1][1]}{best_in_fam[0]:<30} = {best_in_fam[1][0]:.4f}")

            # Per-subset for overall best
            best_col = top[0][0]
            best_d = top[0][1][1]
            print(f"\n  Per-subset ({best_d}{best_col}={top[0][1][0]:.4f}):")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10 and len(sub["is_member"].unique()) > 1:
                    sv = sub[best_col] if best_d == "+" else -sub[best_col]
                    auc = roc_auc_score(sub["is_member"], sv)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

            # Comparison with known results
            best_overall = top[0][1][0]
            print(f"\n  ═══ COMPARISON ═══")
            print(f"  LogitSig best:     {best_overall:.4f} (this, logit-only)")
            print(f"  Min-K%++ (EXP02):  0.5770 (logit-only)")
            print(f"  Loss (EXP01):      0.5807 (logit-only)")
            print(f"  SURP (EXP16):      0.5884 (logit-only)")
            print(f"  Gradient (EXP41):  0.6539 (requires backward)")
            print(f"  AttenMIA (EXP43):  0.6642 (requires attention)")
            print(f"  memTrace (EXP50):  0.6908 (requires hidden states)")
            print(f"  LUMIA-fast:        0.7805 (supervised + hidden states)")

        print("=" * 70)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL_logitsig_{ts}.parquet", index=False)
        print(f"[LogitSig] Results saved.")


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

    Experiment(Args).run()
