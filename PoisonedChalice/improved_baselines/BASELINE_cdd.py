"""
BASELINE_cdd.py — CDD-MIA: Conditional Distribution Dynamics for MIA

NOVEL UNSUPERVISED METHOD — goes beyond Min-K%++ by capturing HOW the
model's conditional distribution EVOLVES across token positions.

NOVELTY vs Min-K%++:
    Min-K%++ examines STATIC properties: "is this token a local maximum?"
    CDD-MIA  examines DYNAMIC properties: "how does the prediction trajectory evolve?"

    This is analogous to the difference between checking a snapshot vs watching a video.
    Memorized sequences produce SMOOTH, CONSISTENT prediction trajectories.
    Non-memorized sequences produce VOLATILE, INCONSISTENT trajectories.

THEORETICAL MOTIVATION:
    MLE training optimizes: max Σ_t log p(x_t | x<t)
    
    For memorized sequences, this optimization converges deeply, creating:
    1. Consistently high z-scores across ALL positions (not just some)
       → Low variance of z_t  ("z-consistency")
    2. Stable prediction distributions across consecutive positions
       → High cosine similarity of logit vectors  ("logit stability")
    3. Smoothly decreasing entropy as the model "recognizes" the pattern
       → Negative entropy slope  ("entropy trajectory")
    4. Consistently low rank for the true token (often rank-1 = mode)
       → Low mean normalized rank  ("rank calibration")
    
    For non-memorized sequences, optimization was incomplete:
    1. Volatile z-scores (some tokens fit, others don't)
    2. Unstable predictions (the model is "guessing")
    3. Fluctuating entropy
    4. Variable ranks (sometimes correct, sometimes way off)
    
    These TRAJECTORY features are a fundamentally different signal dimension
    from the STATIC per-token features that Min-K%++ captures.

SIGNALS (all from single forward pass, zero overhead):
    Static (Min-K%++ style):
        1. minkpp_k{20,30,50} — standard z-calibrated min-k% scores
    
    Dynamic (NEW — trajectory/consistency features):
        2. z_consistency      — negative std of per-token z-scores
        3. logit_stability    — mean cosine sim of consecutive logit vectors
        4. entropy_slope      — slope of entropy across positions (linear fit)
        5. rank_calibrated    — mean (1 - normalized_rank) of true tokens
        6. surprise_smoothness — negative std of token surprisals
    
    Cross-family (NEW — combining static × dynamic):
        7. z_rank_product     — minkpp × rank signal
        8. consistency_stability — z_consistency × logit_stability

Compute: 1 forward pass only (same as Min-K%++)
Expected runtime: ~5-10 min on A100 (10% sample)
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
    print("  CDD-MIA: Conditional Distribution Dynamics")
    print("  Beyond Min-K%++: Static Calibration → Dynamic Trajectories")
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
    print(f"  Loaded. dtype={dtype}, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class CDDExtractor:
    """
    Conditional Distribution Dynamics — extract both static (Min-K%++) and
    dynamic (trajectory) features from a single forward pass.
    
    All features are UNSUPERVISED: each produces a scalar score per sample.
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

            if seq_len < 8:
                return features

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, :-1, :].float()  # (T, V)
            labels = input_ids[0, 1:]                     # (T,)
            T = logits.shape[0]

            if T < 5:
                return features

            # ════════════════════════════════════════════════════════════════
            # PHASE 1: STATIC FEATURES (Min-K%++ baseline)
            # ════════════════════════════════════════════════════════════════
            probs = F.softmax(logits, dim=-1)           # (T, V)
            log_probs = F.log_softmax(logits, dim=-1)   # (T, V)

            # Per-token log probability of actual tokens
            token_lp = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)  # (T,)

            # Min-K%++ z-scores: z_t = (log p(x_t) - μ) / σ
            mu = (probs * log_probs).sum(dim=-1)  # (T,)
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()
            z_scores = (token_lp - mu) / sigma  # (T,)

            z_np = z_scores.cpu().numpy()
            tlp_np = token_lp.cpu().numpy()

            # Standard Min-K%++ scores at various k
            for k in [0.2, 0.3, 0.5]:
                n_sel = max(1, int(T * k))
                features[f"minkpp_k{int(k*100)}"] = float(np.mean(np.sort(z_np)[:n_sel]))

            # Raw loss baseline
            features["neg_mean_loss"] = float(np.mean(tlp_np))

            # ════════════════════════════════════════════════════════════════
            # PHASE 2: DYNAMIC FEATURES (novel — trajectory analysis)
            # ════════════════════════════════════════════════════════════════

            # ── 2a. Z-consistency: how consistent are the z-scores? ────────
            # Members: consistently high z → low std
            # Non-members: volatile z → high std
            z_std = float(np.std(z_np))
            features["neg_z_std"] = -z_std  # higher = more consistent = member
            features["z_mean"] = float(np.mean(z_np))

            # Combined: z_mean / z_std (signal-to-noise ratio of membership)
            if z_std > 1e-6:
                features["z_snr"] = float(np.mean(z_np) / z_std)

            # ── 2b. Logit stability: cosine sim between consecutive logits ─
            # Members: model predictions are stable across positions
            # Non-members: each token surprises the model → big logit shifts
            logit_norms = logits.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            logits_normed = logits / logit_norms
            cos_sims = (logits_normed[:-1] * logits_normed[1:]).sum(dim=-1)  # (T-1,)
            cos_np = cos_sims.cpu().numpy()

            features["logit_stability_mean"] = float(np.mean(cos_np))
            features["logit_stability_min"] = float(np.min(cos_np))
            features["neg_logit_stability_std"] = float(-np.std(cos_np))

            # ── 2c. Entropy trajectory: how does certainty evolve? ─────────
            # Members: entropy decreases as model "recognizes" the sequence
            entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
            ent_np = entropy.cpu().numpy()

            features["neg_entropy_mean"] = float(-np.mean(ent_np))

            # Entropy slope via linear regression (position vs entropy)
            positions = np.arange(T, dtype=np.float64)
            if T > 3:
                # Slope of entropy across positions
                p_mean = positions.mean()
                e_mean = ent_np.mean()
                cov = np.sum((positions - p_mean) * (ent_np - e_mean))
                var_p = np.sum((positions - p_mean) ** 2)
                if var_p > 1e-10:
                    slope = cov / var_p
                    features["neg_entropy_slope"] = float(-slope)
                    # Negative slope = entropy decreasing = model getting more certain
                    # Members should have more negative slope (model recognizes pattern)

            # Entropy in first vs second half
            mid = T // 2
            if mid > 2:
                ent_first = np.mean(ent_np[:mid])
                ent_second = np.mean(ent_np[mid:])
                features["entropy_drop"] = float(ent_first - ent_second)
                # Positive = entropy dropped (model got more certain) = member signal

            # ── 2d. Rank calibration: normalized rank of true tokens ───────
            # Members: true token usually has low rank (often rank 1 = mode)
            # Non-members: true token has higher rank
            # This is distribution-shape-invariant (unlike probability)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)  # (T, V)
            V = logits.shape[1]

            # Fast rank computation: find position of true token in sorted order
            # For each position t, what rank does label[t] have?
            ranks = torch.zeros(T, device=logits.device)
            for t in range(T):
                # Find where label[t] appears in the sorted order
                rank_mask = sorted_indices[t] == labels[t]
                rank_pos = rank_mask.nonzero(as_tuple=True)[0]
                if len(rank_pos) > 0:
                    ranks[t] = rank_pos[0].float()

            # Normalized rank: 0 = top token, 1 = bottom token
            norm_ranks = (ranks / V).cpu().numpy()

            features["neg_rank_mean"] = float(-np.mean(norm_ranks))
            # Higher (less negative) = lower rank = more member-like

            # Rank at mode (fraction of tokens where true token IS the mode)
            top1_acc = float((ranks == 0).float().mean().item())
            features["top1_accuracy"] = top1_acc

            # Min-K% of ranks (worst-ranked tokens)
            for k in [0.2, 0.3]:
                n_sel = max(1, int(T * k))
                worst_ranks = np.sort(norm_ranks)[-n_sel:]  # highest = worst
                features[f"neg_worst_rank_k{int(k*100)}"] = float(-np.mean(worst_ranks))

            # ── 2e. Surprise smoothness: variability of token surprisals ───
            # Surprisal = -log p(x_t | x<t)
            surprisals = -tlp_np
            features["neg_surprise_std"] = float(-np.std(surprisals))
            # Members: consistent surprisal → low std → less negative

            # Surprise gradient (how much surprisal changes between tokens)
            if T > 3:
                surprise_diffs = np.abs(np.diff(surprisals))
                features["neg_surprise_jitter"] = float(-np.mean(surprise_diffs))
                # Members: smooth surprisal → low jitter → less negative

            # ════════════════════════════════════════════════════════════════
            # PHASE 3: CROSS-FAMILY SIGNALS (static × dynamic)
            # ════════════════════════════════════════════════════════════════

            # Min-K%++ score × rank signal → amplify consistent membership
            minkpp20 = features.get("minkpp_k20", 0)
            neg_rank = features.get("neg_rank_mean", 0)
            features["z_rank_product"] = minkpp20 * neg_rank

            # Consistency × stability → double-dynamic fusion
            neg_z_std = features.get("neg_z_std", 0)
            stab_mean = features.get("logit_stability_mean", 0)
            features["consistency_stability"] = neg_z_std * stab_mean

            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[CDD WARN] {type(e).__name__}: {e}")
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

    def run(self):
        df = self.load_data()
        extractor = CDDExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[CDD-MIA] Extracting static + dynamic features...")
        print(f"  Static: Min-K%++ z-scores (k=20,30,50)")
        print(f"  Dynamic: z-consistency, logit stability, entropy trajectory,")
        print(f"           rank calibration, surprise smoothness")
        print(f"  Cross: z×rank product, consistency×stability")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[CDD-MIA]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   CDD-MIA: ALL UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]

        # Categorize features
        static_cols = [c for c in feature_cols if "minkpp" in c or "mink_" in c
                       or c == "neg_mean_loss" or c == "neg_entropy_mean"]
        dynamic_cols = [c for c in feature_cols if c not in static_cols
                        and c not in ["seq_len"]
                        and "product" not in c and "consistency_stability" not in c]
        cross_cols = [c for c in feature_cols if "product" in c or "consistency_stability" in c]

        all_results = {}

        for category, cols, label in [
            ("STATIC (Min-K%++ style)", static_cols, "static"),
            ("DYNAMIC (trajectory — NOVEL)", dynamic_cols, "dynamic"),
            ("CROSS-FAMILY (static × dynamic — NOVEL)", cross_cols, "cross"),
        ]:
            print(f"\n  ── {category} ──")
            for col in sorted(cols):
                v = df.dropna(subset=[col])
                if len(v) < 50 or len(v["is_member"].unique()) < 2:
                    continue
                auc_pos = roc_auc_score(v["is_member"], v[col])
                auc_neg = roc_auc_score(v["is_member"], -v[col])
                best = max(auc_pos, auc_neg)
                d = "+" if auc_pos >= auc_neg else "-"
                all_results[col] = (best, d, label)
                marker = " ★" if best > 0.60 else ""
                print(f"    {d}{col:<40} AUC = {best:.4f}{marker}")

        # ── Top 10 overall ────────────────────────────────────────────────
        print(f"\n  ── TOP 10 SIGNALS ──")
        top10 = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:10]
        for rank, (col, (auc, d, cat)) in enumerate(top10):
            tag = "STATIC" if cat == "static" else ("DYNAMIC★" if cat == "dynamic" else "CROSS★")
            print(f"    {rank+1:2d}. [{tag:8s}] {d}{col:<36} AUC = {auc:.4f}")

        # Count how many dynamic features beat best static
        best_static = max((v for v in all_results.values() if v[2] == "static"),
                          key=lambda x: x[0], default=(0, "", ""))
        dynamic_beats = sum(1 for v in all_results.values()
                           if v[2] in ("dynamic", "cross") and v[0] > best_static[0])
        print(f"\n  Dynamic/Cross features beating best static: {dynamic_beats}")

        # ── Per-subset breakdown ──────────────────────────────────────────
        if top10:
            best_col = top10[0][0]
            best_d = top10[0][1][1]
            print(f"\n  Per-subset for best signal ({best_d}{best_col}):")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                v = sub.dropna(subset=[best_col])
                if len(v) > 10 and len(v["is_member"].unique()) > 1:
                    score_vals = v[best_col] if best_d == "+" else -v[best_col]
                    auc = roc_auc_score(v["is_member"], score_vals)
                else:
                    auc = float("nan")
                print(f"    {subset:<10} AUC = {auc:.4f}  (n={len(sub)})")

        # ── Key comparisons ───────────────────────────────────────────────
        print(f"\n  --- COMPARISON ---")
        minkpp_auc = all_results.get("minkpp_k20", (0, "", ""))[0]
        loss_auc = all_results.get("neg_mean_loss", (0, "", ""))[0]
        best_dynamic = max(
            (v for v in all_results.values() if v[2] in ("dynamic", "cross")),
            key=lambda x: x[0], default=(0, "", "")
        )
        best_overall = max(all_results.values(), key=lambda x: x[0], default=(0, "", ""))

        print(f"  Loss baseline:         {loss_auc:.4f}")
        print(f"  Min-K%++ (k=20):       {minkpp_auc:.4f}")
        print(f"  Best DYNAMIC signal:   {best_dynamic[0]:.4f}")
        print(f"  Best OVERALL:          {best_overall[0]:.4f}")
        print(f"  EXP50 memTrace RF:     0.6908 (supervised)")
        print(f"  LUMIA-fast:            0.7805 (supervised)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"BASELINE_cdd_{timestamp}.parquet", index=False)
        print(f"\n[CDD-MIA] Results saved.")


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

    print(f"[CDD-MIA] Conditional Distribution Dynamics")
    print(f"  model={Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  Signals: static (Min-K%++) + dynamic (trajectory) + cross-family")
    print(f"  1 forward pass only, zero overhead")
    Experiment(Args).run()
