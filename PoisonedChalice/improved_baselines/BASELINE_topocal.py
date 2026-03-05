"""
BASELINE_topocal.py — TopoCal-MIA: Topological Calibration for MIA

NOVEL — No prior work examines the SHAPE of the conditional distribution
for pre-training data detection. This method extracts GEOMETRIC FEATURES
of p(·|x<t) that go beyond mean/std (Min-K%++) or frequency (DC-PDD).

THEORETICAL MOTIVATION:
    MLE training pushes p(·|x<t) toward a peaked distribution at x_t for
    members. This creates specific TOPOLOGICAL signatures:
    
    1. LOW ENTROPY at member positions (model is certain)
    2. HIGH GINI coefficient (probability mass concentrated on few tokens)
    3. LOW EFFECTIVE SUPPORT (fewer tokens above threshold)
    4. HIGH MODE SHARPNESS (the mode is much higher than 2nd-highest)
    5. HEAVY TAIL (probability mass in top-1 vs rest is extreme)
    
    For non-members, the distribution is more DIFFUSE:
    - Higher entropy, lower Gini, wider effective support, flatter mode

    These SHAPE features are INVARIANT to which specific token is predicted
    — they only care about HOW PEAKED the distribution is, not WHAT the peak is.
    This makes them complementary to Min-K%++ (which cares about the value
    at the specific predicted token).

NOVELTY vs Prior Work:
    Min-K%++: z-score of token log-prob (value at a POINT)
    DC-PDD:   frequency-calibrated log-prob (value vs prior)
    TopoCal:  SHAPE of entire distribution (topology of the surface)
    
    Analogy: Min-K%++ asks "how tall is this peak?"
             TopoCal asks "how sharp is the landscape around this peak?"

SIGNALS (all from 1 forward pass, zero overhead):
    1. gini_mean/min_k    — Gini coefficient of p(·|x<t)
    2. neg_entropy_mean   — Shannon entropy (baseline)
    3. mode_gap_mean      — log p(mode) - log p(2nd mode): "sharpness"
    4. top1_mass_mean     — probability mass in top-1 token
    5. topK_mass_mean     — probability mass in top-K tokens (K=5,10)
    6. eff_support_mean   — effective support size (-1/sum(p^2))
    7. tail_weight        — mass in bottom 90% of vocab
    8. renyi_entropy      — Rényi entropy of order 2 (= log(eff_support))

Compute: 1 forward pass only
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
    print("  TopoCal-MIA: Topological Distribution Calibration")
    print("  Shape of p(·|x<t) as Membership Signal")
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
    print(f"  Loaded. dtype={dtype}, vocab={model.config.vocab_size}")
    return model, tokenizer


class TopoCalExtractor:
    """
    Extracts topological features of the conditional distribution p(·|x<t)
    at each position, then aggregates across positions.
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
            logits = outputs.logits[0, :-1, :].float()  # (T, V)
            labels = input_ids[0, 1:]
            T, V = logits.shape
            if T < 3:
                return features

            probs = F.softmax(logits, dim=-1)           # (T, V)
            log_probs = F.log_softmax(logits, dim=-1)

            # Per-token log-prob for Min-K%++ baseline
            token_lp = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
            tlp_np = token_lp.cpu().numpy()

            # Min-K%++ z-scores baseline
            mu = (probs * log_probs).sum(dim=-1)
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()
            z_scores = ((token_lp - mu) / sigma).cpu().numpy()
            for k in [0.2]:
                n_sel = max(1, int(T * k))
                features[f"minkpp_k{int(k*100)}"] = float(np.mean(np.sort(z_scores)[:n_sel]))

            features["neg_mean_loss"] = float(np.mean(tlp_np))

            # ════════════════════════════════════════════════════════════════
            # TOPOLOGICAL FEATURES OF p(·|x<t)
            # ════════════════════════════════════════════════════════════════

            probs_np = probs.cpu().numpy()  # (T, V)

            # Sort probabilities descending for each position
            sorted_probs = np.sort(probs_np, axis=-1)[:, ::-1]  # (T, V) descending

            # ── 1. Shannon Entropy: H = -Σ p log p ────────────────────────
            entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # (T,)
            features["neg_entropy_mean"] = float(-np.mean(entropy))
            features["neg_entropy_min_k20"] = float(-np.mean(np.sort(-entropy)[:max(1, int(T*0.2))]))

            # ── 2. Rényi Entropy (order 2) = -log(Σ p²) ──────────────────
            # Related to effective number of outcomes
            sum_p_sq = (probs ** 2).sum(dim=-1).cpu().numpy()  # (T,)
            renyi2 = -np.log(np.clip(sum_p_sq, 1e-12, None))
            features["neg_renyi2_mean"] = float(-np.mean(renyi2))

            # ── 3. Effective support = 1 / Σ p² (inverse participation) ───
            eff_support = 1.0 / np.clip(sum_p_sq, 1e-12, None)  # (T,)
            features["neg_eff_support_mean"] = float(-np.mean(eff_support))
            # Members: lower eff_support (more peaked) → neg makes higher = member

            # ── 4. Gini coefficient of the distribution ───────────────────
            # Gini = 1 - 2 * Σ(cumulative_prob * (1/V))
            # Higher Gini = more concentrated = more member-like
            cumsum = np.cumsum(sorted_probs[:, ::-1], axis=-1)  # cumsum ascending
            gini = 1.0 - 2.0 * np.mean(cumsum, axis=-1)  # (T,)
            features["gini_mean"] = float(np.mean(gini))
            for k in [0.2, 0.3]:
                n_sel = max(1, int(T * k))
                features[f"gini_min_k{int(k*100)}"] = float(np.mean(np.sort(gini)[:n_sel]))

            # ── 5. Mode gap: log p(1st) - log p(2nd) ─────────────────────
            # How much sharper is the mode compared to second-best?
            top1_lp = np.log(np.clip(sorted_probs[:, 0], 1e-12, None))
            top2_lp = np.log(np.clip(sorted_probs[:, 1], 1e-12, None))
            mode_gap = top1_lp - top2_lp  # (T,) — always >= 0
            features["mode_gap_mean"] = float(np.mean(mode_gap))
            features["mode_gap_min_k20"] = float(np.mean(np.sort(mode_gap)[:max(1, int(T*0.2))]))

            # ── 6. Top-K mass: what fraction concentrated in top K? ───────
            for K in [1, 5, 10]:
                topk_mass = sorted_probs[:, :K].sum(axis=-1)  # (T,)
                features[f"top{K}_mass_mean"] = float(np.mean(topk_mass))

            # ── 7. Tail weight: mass in bottom 90% of vocab ──────────────
            tail_90 = sorted_probs[:, int(V * 0.1):].sum(axis=-1)  # (T,)
            features["neg_tail90_mean"] = float(-np.mean(tail_90))
            # Members have LESS tail mass → neg_tail is higher → member signal

            # ── 8. Token rank (from CDD) ──────────────────────────────────
            # Rank of the true token in the sorted distribution
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            ranks = torch.zeros(T, device=logits.device)
            for t in range(T):
                rank_mask = sorted_indices[t] == labels[t]
                rank_pos = rank_mask.nonzero(as_tuple=True)[0]
                if len(rank_pos) > 0:
                    ranks[t] = rank_pos[0].float()
            norm_ranks = (ranks / V).cpu().numpy()
            features["neg_rank_mean"] = float(-np.mean(norm_ranks))

            # ── 9. Combined: Gini × z-score product ──────────────────────
            # Sharp distribution + high relative probability = strong member signal
            features["gini_z_product"] = features["gini_mean"] * features.get("minkpp_k20", 0)

            # ── 10. Shape consistency: std of Gini across positions ───────
            features["neg_gini_std"] = float(-np.std(gini))
            # Members: consistently peaked → low Gini std

            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[TopoCal WARN] {type(e).__name__}: {e}")
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
        extractor = TopoCalExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[TopoCal] Extracting distribution shape features...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[TopoCal]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   TopoCal-MIA: ALL SIGNAL AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        results = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            best = max(auc_pos, 1 - auc_pos)
            d = "+" if auc_pos >= 0.5 else "-"
            results[col] = (best, d)
            marker = " ★" if best > 0.59 else ""
            print(f"  {d}{col:<40} AUC = {best:.4f}{marker}")

        if results:
            best = max(results.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best[1][1]}{best[0]} = {best[1][0]:.4f}")
            
            # Per-subset
            best_col = best[0]
            print(f"\n  Per-subset for {best_col}:")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10:
                    auc = roc_auc_score(sub["is_member"], sub[best_col])
                    auc = max(auc, 1 - auc)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"BASELINE_topocal_{timestamp}.parquet", index=False)
        print(f"[TopoCal] Results saved.")


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

    print(f"[TopoCal] Topological Distribution Calibration")
    print(f"  model={Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
