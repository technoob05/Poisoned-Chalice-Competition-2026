"""
EXPERIMENT 31: Tail-Loss Delta under Embedding Noise  (Zero-Shot Trajectory)

Papers fused:
    1. "The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability
       Without Reference Models" (ICLR 2026)
       — Training samples undergo a massive loss DROP from high→low during training.
         They are memorized in SHARP, NARROW minima of the loss surface.

    2. EXP24 (Perturbation Gradient Stability) × EXP04 (Structural Divergence)
       — Combining noise perturbation with loss-surface analysis.

Key insight (Simulated Tail Trajectory):
    We cannot observe the training curve, but we can PROBE the local loss surface
    by adding small Gaussian noise to the input embeddings:

        Members   → lie in sharp/narrow minima (very low original loss).
                    A tiny perturbation of input embeddings causes a LARGE relative
                    loss spike  →  high Δ_rel = (L_perturbed - L_orig) / L_orig.

        Non-members → lie on a smooth, high-loss plateau.
                    A small perturbation produces a proportionally SMALL change.
                    →  low Δ_rel.

Two signals extracted per sample:

    A. Relative Loss Delta (primary, no backward needed):
            score_delta = (loss_perturbed - loss_orig) / (loss_orig + ε)
            Higher = sharper minima = more likely member.

    B. Input Embedding Gradient Norm (secondary, one backward):
            Compute d(loss)/d(input_embeds).  High norm = steep local gradient
            = sharp surface = member.

Combined score:
    rank_avg(score_delta, score_input_grad)

Compute notes:
    - Signal A: 2 forward passes (no backward) → fast.
    - Signal B: 1 additional forward+backward w.r.t. input embeddings.
    - noise_level = 0.1 (10% of embedding standard deviation per sequence).
    - sample_fraction=0.10 recommended (fast, ~2h on A100 for signal A only).
    - Multiple noise scales probed (0.05, 0.10, 0.20) to find best discriminator.

Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment():
    print("\n" + "="*65)
    print("  EXP31: TAIL-LOSS DELTA — EMBEDDING NOISE PERTURBATION")
    print("="*65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
    except Exception as e:
        print(f"[HF] Note: {e}")


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    # requires_grad=True for the input gradient pass
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# Tail-Loss Delta Attack
# ============================================================================

class TailLossDeltaAttack:
    """
    Probes the sharpness of the model's loss surface around each sample by
    adding Gaussian noise to the input embeddings and measuring the loss response.

    Memorized samples (members) sit in sharp, narrow minima → high loss sensitivity.
    Unseen samples (non-members) sit on flat plateaus → low loss sensitivity.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        # Noise scales to probe (relative to per-sequence embedding std)
        self.noise_levels = getattr(args, "noise_levels", [0.05, 0.10, 0.20])
        self.compute_input_grad = getattr(args, "compute_input_grad", True)
        self.embed_layer = model.get_input_embeddings()

    @property
    def name(self) -> str:
        return "tail_loss_delta"

    def _get_loss_from_embeds(
        self,
        inputs_embeds: torch.Tensor,   # (1, seq, hidden) — any dtype
        labels: torch.Tensor,           # (1, seq) — original token ids
    ) -> float:
        """Forward pass using pre-computed embeddings. Returns scalar loss. Uses model dtype."""
        try:
            model_dtype = next(self.model.parameters()).dtype
            emb = inputs_embeds.detach().to(dtype=model_dtype)
            with torch.no_grad():
                outputs = self.model(inputs_embeds=emb, labels=labels)
            return outputs.loss.item()
        except Exception:
            return np.nan

    def _get_loss_and_input_grad(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Forward + backward w.r.t. inputs_embeds.
        Returns (loss, grad_norm_of_inputs_embeds). Uses model native dtype (bfloat16).
        """
        try:
            self.model.zero_grad()
            model_dtype = next(self.model.parameters()).dtype
            embed_in = inputs_embeds.detach().to(dtype=model_dtype).requires_grad_(True)
            outputs = self.model(inputs_embeds=embed_in, labels=labels)
            outputs.loss.backward()
            grad_norm = embed_in.grad.float().norm(2).item() if embed_in.grad is not None else np.nan
            loss_val = outputs.loss.item()
            self.model.zero_grad()
            return loss_val, grad_norm
        except Exception:
            self.model.zero_grad()
            return np.nan, np.nan

    def compute_tail_features(self, text: str) -> Dict[str, float]:
        """
        Core computation for one sample:
          1. Tokenize → get input embeddings.
          2. Compute original loss (+ optionally input gradient norm).
          3. For each noise_level: add Gaussian noise → compute perturbed loss.
          4. Derive delta metrics.
        """
        result: Dict[str, float] = {"loss_orig": np.nan}
        for nl in self.noise_levels:
            result[f"delta_rel_{int(nl*100):03d}"] = np.nan
            result[f"delta_abs_{int(nl*100):03d}"] = np.nan
        if self.compute_input_grad:
            result["input_grad_norm"] = np.nan

        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)

            input_ids = inputs["input_ids"]  # (1, seq)
            labels = input_ids.clone()

            # ---- Get original embeddings in model dtype (bfloat16 on A100) ----
            with torch.no_grad():
                embeds_orig = self.embed_layer(input_ids)  # native dtype

            # ---- Signal A: Loss delta under embedding noise ----
            loss_orig = self._get_loss_from_embeds(embeds_orig, labels)
            result["loss_orig"] = loss_orig

            if np.isnan(loss_orig):
                return result

            # Noise scale from per-sequence std (use float for stable std)
            embed_std = embeds_orig.float().std().item()

            for noise_level in self.noise_levels:
                noise_scale = embed_std * noise_level
                noise = (torch.randn_like(embeds_orig.float()) * noise_scale).to(embeds_orig.dtype)
                embeds_noisy = embeds_orig + noise

                loss_noisy = self._get_loss_from_embeds(embeds_noisy, labels)
                if not np.isnan(loss_noisy):
                    delta_abs = loss_noisy - loss_orig
                    delta_rel = delta_abs / (loss_orig + 1e-6)
                    key = f"{int(noise_level*100):03d}"
                    result[f"delta_abs_{key}"] = float(delta_abs)
                    result[f"delta_rel_{key}"] = float(delta_rel)

            # ---- Signal B: Input embedding gradient norm (one backward pass) ----
            if self.compute_input_grad:
                # Use the original embeddings for the gradient computation
                # (separate from the noise runs; no_grad was used above)
                _, grad_norm = self._get_loss_and_input_grad(embeds_orig, labels)
                result["input_grad_norm"] = grad_norm

            return result

        except Exception:
            return result

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP31] Processing {len(texts)} samples…")
        print(f"[EXP31] Noise levels: {self.noise_levels}")
        print(f"[EXP31] Input grad: {self.compute_input_grad}")

        rows = []
        for text in tqdm(texts, desc="[EXP31] Tail-Loss Delta"):
            rows.append(self.compute_tail_features(text))

        df = pd.DataFrame(rows)

        # ---- Primary delta score: use the middle noise level (10%) ----
        # Members: high Δ_rel → high score
        primary_key = f"delta_rel_{int(0.10*100):03d}"
        if primary_key in df.columns:
            df["score_delta"] = df[primary_key]

        # ---- Input gradient score ----
        if "input_grad_norm" in df.columns:
            df["score_input_grad"] = df["input_grad_norm"]  # higher = member

        # ---- Mean delta across all noise levels (robust aggregation) ----
        delta_rel_cols = [c for c in df.columns if c.startswith("delta_rel_")]
        if delta_rel_cols:
            df["mean_delta_rel"] = df[delta_rel_cols].mean(axis=1)
            df["score_mean_delta"] = df["mean_delta_rel"]

        # ---- Combined rank score ----
        rank_sources = ["score_delta", "score_input_grad"]
        valid_rank_cols = [c for c in rank_sources if c in df.columns]
        if valid_rank_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_sum += ranks / len(ranks)
            df["combined_rank_score"] = rank_sum / len(valid_rank_cols)

        return df


# ============================================================================
# Experiment Runner
# ============================================================================

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
        print(f"[*] Loading data from {self.args.dataset}…")
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
        attacker = TailLossDeltaAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP31_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "="*65)
        print("   EXP31: TAIL-LOSS DELTA — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(Δ_rel + input_grad) [PRIMARY]",
            "score_delta": "Δ_rel at noise=10%",
            "score_mean_delta": "Mean Δ_rel across noise levels",
            "score_input_grad": "Input Embedding Gradient Norm",
        }
        report = {
            "experiment": "EXP31_tail_loss_delta",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "noise_levels": self.args.noise_levels,
            "compute_input_grad": self.args.compute_input_grad,
            "aucs": {},
            "subset_aucs": {},
        }

        for score_col, label in score_candidates.items():
            if score_col not in df.columns:
                continue
            valid = df.dropna(subset=[score_col])
            if len(valid["is_member"].unique()) > 1:
                auc = roc_auc_score(valid["is_member"], valid[score_col])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if score_col == "combined_rank_score" else ""
                print(f"  {label:<48} AUC = {auc:.4f}{tag}")

        print(f"\nPer-noise-level Δ_rel AUC:")
        for nl in self.args.noise_levels:
            col = f"delta_rel_{int(nl*100):03d}"
            if col in df.columns:
                valid = df.dropna(subset=[col])
                if len(valid["is_member"].unique()) > 1:
                    auc = roc_auc_score(valid["is_member"], valid[col])
                    m_delta = valid[valid["is_member"] == 1][col].mean()
                    nm_delta = valid[valid["is_member"] == 0][col].mean()
                    print(
                        f"  noise={nl:.2f}:  AUC={auc:.4f}  "
                        f"Δ(M)={m_delta:.4f}  Δ(NM)={nm_delta:.4f}"
                    )

        print(f"\nLoss statistics:")
        m_loss = df[df["is_member"] == 1]["loss_orig"].mean()
        nm_loss = df[df["is_member"] == 0]["loss_orig"].mean()
        print(f"  Mean original loss — Members: {m_loss:.4f}  Non-members: {nm_loss:.4f}")
        print(f"  (Members should have LOWER loss — confirm memorization hypothesis)")

        print(f"\n{'Subset':<10} | {'AUC':<8} | {'N':<6} | {'Δ_rel (M)':<12} | {'Δ_rel (NM)'}")
        print("-" * 55)
        primary_score = "combined_rank_score" if "combined_rank_score" in df.columns else "score_delta"
        delta_col = "score_delta" if "score_delta" in df.columns else None
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            valid_sub = sub.dropna(subset=[primary_score]) if primary_score in sub.columns else pd.DataFrame()
            if not valid_sub.empty and len(valid_sub["is_member"].unique()) > 1:
                auc = roc_auc_score(valid_sub["is_member"], valid_sub[primary_score])
                m_d = valid_sub[valid_sub["is_member"] == 1][delta_col].mean() if delta_col else float("nan")
                nm_d = valid_sub[valid_sub["is_member"] == 0][delta_col].mean() if delta_col else float("nan")
                print(f"{subset:<10} | {auc:.4f}   | {len(valid_sub):<6} | {m_d:<12.4f} | {nm_d:.4f}")
                report["subset_aucs"][subset] = {"auc": float(auc)}

        print("="*65)
        print("\nInterpretation:")
        print("  Δ_rel(Members) >> Δ_rel(Non-Members) → sharp minima → confirmed memorization")
        print("  Baseline sanity: Members should have lower loss_orig than non-members")

        report_path = self.output_dir / f"EXP31_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\n[*] Report saved: {report_path.name}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10     # Fast (2 fwd + 1 bwd per sample) → ~4h on A100
        output_dir = "results"
        max_length = 2048
        # Noise levels to probe (relative to per-sequence embedding std)
        noise_levels = [0.05, 0.10, 0.20]
        # Set to False if you want ONLY the fast loss-delta (no backward pass)
        compute_input_grad = True
        seed = 42

    print(f"[EXP31] Model        : {Args.model_name}")
    print(f"[EXP31] Sample       : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP31] Noise levels : {Args.noise_levels}")
    print(f"[EXP31] Input grad   : {Args.compute_input_grad}")
    Experiment(Args).run()
