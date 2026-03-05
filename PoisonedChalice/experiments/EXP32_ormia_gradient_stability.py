"""
EXPERIMENT 32: OR-MIA — Optimization & Robustness-Informed MIA
            (Gradient Norm Stability under Embedding Noise)

Paper inspiration:
    "Optimization and Robustness-Informed Membership Inference Attacks for LLMs"
    — Member data lie in DEEP, FLAT minima (low gradient norm + low sensitivity to noise).
    — Non-member data lie on steep loss surfaces (high norm + high noise sensitivity).

Two complementary OR-MIA signals:

    1. Absolute Gradient Norm  (G_orig)
       Forward + backward on original embeddings.
       Members → low G_orig  (flat minimum, "nằm lòng").

    2. Gradient Norm Stability  (ΔG)
       Add small Gaussian noise ε ~ N(0, σ²) to input embeddings.
       Run another forward + backward pass.
       ΔG = |G_noisy - G_orig| / (G_orig + eps)
       Members → tiny ΔG  (deep flat minimum is locally invariant to noise).
       Non-members → large ΔG  (steep slope, any nudge changes the gradient drastically).

Combined score (primary):
    rank_avg(-G_orig, -ΔG)

Why Gaussian on EMBEDDINGS (not token substitution):
    - Token substitution (EXP24) can break syntax and change semantics completely.
    - Gaussian embedding noise is a CONTINUOUS, infinitesimally small perturbation.
      It probes the CURVATURE of the loss surface without leaving the neighbourhood
      of the original input point — exactly what "flat minimum" theory requires.

Compute strategy:
    - 2 forward+backward passes per sample (G_orig + G_noisy).
    - noise_levels probed: [0.01, 0.05] × per-sequence embedding std.
    - Multiple noise trials averaged for robustness (n_trials=3, controllable).
    - All raw features saved for EXP15 XGBoost stacking.

Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
    print("  EXP32: OR-MIA — GRADIENT NORM STABILITY (Embedding Noise)")
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
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# OR-MIA Attack
# ============================================================================

class ORMIAAttack:
    """
    Probes BOTH the magnitude AND the stability of the embedding-layer gradient.
    Two forward+backward passes per noise level; results averaged over n_trials.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        # Noise levels relative to per-sequence embedding std
        self.noise_levels: List[float] = getattr(args, "noise_levels", [0.01, 0.05])
        self.n_trials: int = getattr(args, "n_trials", 3)
        self.embed_layer = model.get_input_embeddings()
        self._err_count = 0

    @property
    def name(self) -> str:
        return "ormia_grad_stability"

    def _embed_grad_norm(
        self,
        inputs_embeds: torch.Tensor,   # (1, seq, hidden) — any dtype/device
        labels: torch.Tensor,           # (1, seq)
    ) -> float:
        """
        One backward pass w.r.t. inputs_embeds. Returns embedding gradient L2 norm.

        Key fix: feed the model in its NATIVE dtype (bfloat16 / float16).
        We keep requires_grad=True on the native-dtype tensor; PyTorch will
        compute gradients in that dtype. We then read the norm as float32.
        """
        try:
            self.model.zero_grad()
            # Match the model's dtype (bfloat16 on A100) — do NOT cast to float32
            model_dtype = next(self.model.parameters()).dtype
            embed_in = inputs_embeds.detach().to(dtype=model_dtype).requires_grad_(True)
            outputs = self.model(inputs_embeds=embed_in, labels=labels)
            outputs.loss.backward()
            # Read norm in float32 for precision, regardless of gradient dtype
            norm = embed_in.grad.float().norm(2).item() if embed_in.grad is not None else np.nan
            self.model.zero_grad()
            return norm
        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP32 WARNING] _embed_grad_norm error: {type(e).__name__}: {e}")
            self._err_count += 1
            return np.nan

    def compute_or_features(self, text: str) -> Dict[str, float]:
        """
        Returns:
            grad_norm_orig      : G_orig (lower = member)
            delta_G_<nl>        : ΔG at noise level nl (lower = member)
            delta_G_rel_<nl>    : relative ΔG = |G_noisy - G_orig| / G_orig
        """
        result: Dict[str, float] = {"grad_norm_orig": np.nan}
        for nl in self.noise_levels:
            key = f"{int(nl*100):03d}"
            result[f"delta_G_{key}"] = np.nan
            result[f"delta_G_rel_{key}"] = np.nan

        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)
            labels = inputs["input_ids"].clone()

            # ---- Pass 1: original gradient norm ----
            # Keep embeddings in the model's native dtype (bfloat16 on A100).
            # _embed_grad_norm handles dtype internally; std() for noise scale
            # is computed in float32 to avoid underflow with bfloat16 precision.
            with torch.no_grad():
                embeds_orig = self.embed_layer(inputs["input_ids"])  # native dtype

            g_orig = self._embed_grad_norm(embeds_orig, labels)
            result["grad_norm_orig"] = g_orig

            if np.isnan(g_orig):
                return result

            # Use float32 for std computation (bfloat16 may underflow for small norms)
            embed_std = embeds_orig.float().std().item()

            # ---- Pass 2+: noisy gradient norms ----
            for noise_level in self.noise_levels:
                noise_scale = embed_std * noise_level
                trial_norms: List[float] = []

                for _ in range(self.n_trials):
                    # Generate noise in float32, cast to model dtype before adding
                    noise = (torch.randn_like(embeds_orig.float()) * noise_scale).to(embeds_orig.dtype)
                    g_noisy = self._embed_grad_norm(embeds_orig + noise, labels)
                    if not np.isnan(g_noisy):
                        trial_norms.append(g_noisy)

                if trial_norms:
                    g_noisy_mean = float(np.mean(trial_norms))
                    delta_abs = abs(g_noisy_mean - g_orig)
                    delta_rel = delta_abs / (g_orig + 1e-9)
                    key = f"{int(noise_level*100):03d}"
                    result[f"delta_G_{key}"] = delta_abs
                    result[f"delta_G_rel_{key}"] = delta_rel

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP32 WARNING] compute_or_features error: {type(e).__name__}: {e}")
            self._err_count += 1
            return result

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP32] Processing {len(texts)} samples…")
        print(f"[EXP32] noise_levels={self.noise_levels}  n_trials={self.n_trials}")
        rows = []

        for text in tqdm(texts, desc="[EXP32] OR-MIA Gradient Stability"):
            rows.append(self.compute_or_features(text))

        df = pd.DataFrame(rows)

        # ---- Member signals: lower values → more likely member → negate ----
        if "grad_norm_orig" in df.columns:
            df["signal_grad_norm"] = -df["grad_norm_orig"]

        # Use smallest noise level's ΔG as primary stability signal
        primary_delta_col = f"delta_G_{int(self.noise_levels[0]*100):03d}"
        if primary_delta_col in df.columns:
            df["signal_delta_G"] = -df[primary_delta_col]

        # Mean ΔG across all noise levels
        delta_cols = [c for c in df.columns if c.startswith("delta_G_rel_")]
        if delta_cols:
            df["mean_delta_G_rel"] = df[delta_cols].mean(axis=1)
            df["signal_mean_delta"] = -df["mean_delta_G_rel"]

        # ---- Combined rank score ----
        rank_sources = ["signal_grad_norm", "signal_delta_G"]
        valid_rank_cols = [c for c in rank_sources if c in df.columns]
        if valid_rank_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_sum += ranks / len(ranks)
            df["combined_rank_score"] = rank_sum / len(valid_rank_cols)

        n_valid = df["combined_rank_score"].notna().sum() if "combined_rank_score" in df.columns else 0
        print(f"[EXP32] Valid samples: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP32] Total errors: {self._err_count}")
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
        attacker = ORMIAAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP32_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")
        print("[*] All OR-MIA features saved for EXP15 XGBoost stacking.")

        print("\n" + "="*65)
        print("   EXP32: OR-MIA — GRADIENT STABILITY — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(G_orig + ΔG)  [PRIMARY]",
            "signal_grad_norm":    "-G_orig (gradient magnitude only)",
            "signal_delta_G":      "-ΔG (stability only, small noise)",
            "signal_mean_delta":   "-mean_ΔG_rel (stability across all noise)",
        }
        report = {
            "experiment": "EXP32_ormia_gradient_stability",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "noise_levels": self.args.noise_levels,
            "n_trials": self.args.n_trials,
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
                tag = " ← PRIMARY" if "combined" in score_col else ""
                print(f"  {label:<48} AUC = {auc:.4f}{tag}")

        print(f"\nPer-noise-level ΔG AUC:")
        for nl in self.args.noise_levels:
            col = f"delta_G_{int(nl*100):03d}"
            if col in df.columns:
                valid = df.dropna(subset=[col])
                if len(valid["is_member"].unique()) > 1:
                    auc = roc_auc_score(valid["is_member"], -valid[col])
                    m_dg = valid[valid["is_member"] == 1][col].mean()
                    nm_dg = valid[valid["is_member"] == 0][col].mean()
                    print(f"  noise={nl:.2f}:  AUC={auc:.4f}  "
                          f"ΔG(M)={m_dg:.4f}  ΔG(NM)={nm_dg:.4f}")

        print(f"\nGradient norm summary:")
        m_g = df[df["is_member"] == 1]["grad_norm_orig"].mean()
        nm_g = df[df["is_member"] == 0]["grad_norm_orig"].mean()
        print(f"  Mean G_orig — Members: {m_g:.4f}  Non-members: {nm_g:.4f}")
        print(f"  (Members should have LOWER G_orig — flat minimum hypothesis)")

        print(f"\n{'Subset':<10} | {'CombinedAUC':<13} | {'G_normAUC':<11} | {'ΔG AUC'}")
        print("-"*50)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["combined_rank_score", "signal_grad_norm", "signal_delta_G"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('combined_rank_score', float('nan')):.4f}        "
                  f"| {r.get('signal_grad_norm', float('nan')):.4f}      "
                  f"| {r.get('signal_delta_G', float('nan')):.4f}")
            report["subset_aucs"][subset] = r

        print("="*65)

        report_path = self.output_dir / f"EXP32_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[*] Report saved: {report_path.name}")


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
        # (1 + n_noise_levels * n_trials) backward passes per sample
        # = (1 + 2*3) = 7 backward passes → ~5h on A100 at 10%
        sample_fraction = 0.10
        output_dir = "results"
        max_length = 2048
        noise_levels = [0.01, 0.05]   # 1% and 5% of per-seq embedding std
        n_trials = 3                   # Noise trials to average per level
        seed = 42

    print(f"[EXP32] Model        : {Args.model_name}")
    print(f"[EXP32] Sample       : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP32] Noise levels : {Args.noise_levels}")
    print(f"[EXP32] Trials/level : {Args.n_trials}")
    Experiment(Args).run()
