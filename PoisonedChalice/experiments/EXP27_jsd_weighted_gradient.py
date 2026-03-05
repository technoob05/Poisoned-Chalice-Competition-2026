"""
EXPERIMENT 27: JSD-Weighted Selective Gradient Norm (White-Box)

Method:
    Combine two complementary white-box MIA signals in a SINGLE forward+backward pass:

    1. JSD Convergence Signal (from EXP26 lineage):
       Project intermediate hidden states via logit lens and compute JSD vs. final
       layer distributions. Members settle early → low JSD at shallow layers.

    2. Gradient Norm Signal (from EXP11/EXP22 lineage):
       Compute gradient L2-norms at the embedding, mid-transformer, and head layers.
       Members lie in flat minima → low gradient norms.

    Combined Score:
       Rank-average the two member signals:
           score = 0.5 * rank(-jsd_early) + 0.5 * rank(-grad_norm_embed)
       Also outputs a product variant:
           score_product = -jsd_early * grad_norm_embed
       And the full feature vector for downstream XGBoost stacking (EXP15).

Motivation:
    Early-settling tokens but abnormally HIGH gradient norms may be random noise.
    Early-settling tokens combined with LOW gradient norms at the embedding and head
    (EXP22 best performers: embedding 0.6423, head 0.6423) reinforce the memorization
    signal. This JSD-weighted combination is expected to reduce false positives.

Compute Strategy:
    - Hidden states captured via forward hooks (detached) during the SAME forward pass
      used for gradient computation → one pass total.
    - JSD computed on detached tensors after backward → no graph retention overhead.
    - Stride-based token sampling: max 64 tokens for JSD.
    - Outputs ALL features as a parquet for EXP15 XGBoost stacking.

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
    print("  EXP27: JSD-WEIGHTED SELECTIVE GRADIENT NORM (White-Box)")
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
    # requires_grad=True for backward pass
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# JSD Helper
# ============================================================================

def jsd_batch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    p, q : (T, V) float32.
    Returns : (T,) JSD values in [0, ln(2)].
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


# ============================================================================
# JSD-Weighted Gradient Attack
# ============================================================================

class JSDWeightedGradientAttack:
    """
    Single-pass combined JSD + gradient analysis.

    Forward hooks capture hidden states (detached) at sampled layers during
    the forward pass. The backward pass then computes gradient norms. JSD is
    computed from the detached hidden states afterwards (no extra pass needed).
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.max_jsd_tokens = getattr(args, "max_jsd_tokens", 64)

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError("Cannot find model.model.layers — check architecture.")
        if not hasattr(model.model, "norm"):
            raise RuntimeError("Cannot find model.model.norm — check architecture.")

        self.transformer_layers = model.model.layers
        self.norm_layer = model.model.norm
        self.lm_head = model.get_output_embeddings()
        self.embed_layer = model.get_input_embeddings()

        self.sampled_indices = self._choose_layer_indices()
        # Middle block for gradient norm (50th percentile)
        n = len(self.transformer_layers)
        self.mid_grad_idx = n // 2
        print(f"[EXP27] Sampled layers (JSD): {self.sampled_indices}")
        print(f"[EXP27] Mid-layer gradient: {self.mid_grad_idx}")

    def _choose_layer_indices(self) -> List[int]:
        n = len(self.transformer_layers)
        raw = [n // 6, n // 3, n // 2, 2 * n // 3, n - 1]
        seen, indices = set(), []
        for idx in raw:
            idx = max(0, min(idx, n - 1))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    @property
    def name(self) -> str:
        return "jsd_weighted_gradient"

    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply final norm + LM head to get vocab probability distribution.
        Aligns device and dtype of hidden states with the norm layer weights
        (hidden is detached to CPU in bfloat16; norm/head live on GPU in bfloat16).
        Returns float32 probabilities for numerically stable JSD computation.
        """
        with torch.no_grad():
            model_dtype = next(self.norm_layer.parameters()).dtype
            h = hidden.to(device=self.norm_layer.weight.device, dtype=model_dtype)
            normed = self.norm_layer(h)
            logits = self.lm_head(normed)
            return F.softmax(logits.float(), dim=-1)

    def _sample_token_indices(self, seq_len: int) -> List[int]:
        if seq_len <= self.max_jsd_tokens:
            return list(range(seq_len))
        stride = seq_len // self.max_jsd_tokens
        return list(range(0, seq_len, stride))[: self.max_jsd_tokens]

    def compute_joint_features(self, text: str) -> Dict[str, float]:
        """
        Single forward+backward pass to extract:
            - grad_norm_embed  : L2 norm of embedding layer gradient
            - grad_norm_head   : L2 norm of LM head gradient
            - grad_norm_mid    : RMS norm of mid-transformer block gradients
            - jsd_layer_{idx}  : per-sampled-layer mean JSD vs. final layer
            - mean_jsd_early   : mean JSD across first two sampled layers

        Returns a flat dict of float features, empty dict on failure.
        """
        if not text or len(text) < 20:
            return {}

        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                # Detach: keeps hidden states off-graph so backward is unaffected
                captured[layer_idx] = hs[0].detach().cpu()
            return hook_fn

        handles = [
            self.transformer_layers[idx].register_forward_hook(make_hook(idx))
            for idx in self.sampled_indices
        ]

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)

            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 4:
                for h in handles:
                    h.remove()
                return {}

            # ---- Forward + Backward ----
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()

            for h in handles:
                h.remove()

            features: Dict[str, float] = {}

            # ---- Gradient Norms ----
            if self.embed_layer.weight.grad is not None:
                features["grad_norm_embed"] = (
                    self.embed_layer.weight.grad.norm(2).item()
                )

            if (
                self.lm_head.weight.grad is not None
                and self.lm_head.weight.data_ptr()
                != self.embed_layer.weight.data_ptr()   # not tied weights
            ):
                features["grad_norm_head"] = (
                    self.lm_head.weight.grad.norm(2).item()
                )
            else:
                # Tied weights: reuse embed grad (same tensor)
                features["grad_norm_head"] = features.get("grad_norm_embed", np.nan)

            # Mid-block gradient (RMS of all parameter gradients)
            mid_layer = self.transformer_layers[self.mid_grad_idx]
            mid_g = [
                p.grad.norm(2).item()
                for p in mid_layer.parameters()
                if p.grad is not None
            ]
            features["grad_norm_mid"] = (
                float(np.sqrt(np.mean(np.square(mid_g)))) if mid_g else np.nan
            )

            self.model.zero_grad()

            # ---- JSD from captured hidden states ----
            tok_indices = self._sample_token_indices(seq_len)
            final_idx = self.sampled_indices[-1]

            if final_idx in captured:
                final_h = captured[final_idx][tok_indices].to(self.model.device)
                final_dist = self._logit_lens(final_h)

                jsd_early_vals = []
                for layer_idx in self.sampled_indices[:-1]:
                    if layer_idx not in captured:
                        features[f"jsd_layer_{layer_idx}"] = np.nan
                        continue
                    layer_h = captured[layer_idx][tok_indices].to(self.model.device)
                    layer_dist = self._logit_lens(layer_h)
                    jsd_vals = jsd_batch(layer_dist, final_dist)
                    mean_jsd = jsd_vals.mean().item()
                    features[f"jsd_layer_{layer_idx}"] = mean_jsd
                    if layer_idx in self.sampled_indices[:2]:
                        jsd_early_vals.append(mean_jsd)
                    del layer_h, layer_dist, jsd_vals

                features[f"jsd_layer_{final_idx}"] = 0.0
                features["mean_jsd_early"] = (
                    float(np.mean(jsd_early_vals)) if jsd_early_vals else np.nan
                )
                del final_h, final_dist

            captured.clear()
            return features

        except Exception as e:
            for h in handles:
                try: h.remove()
                except Exception: pass
            self.model.zero_grad()
            captured.clear()
            if not hasattr(self, '_err_count'):
                self._err_count = 0
            if self._err_count < 3:
                print(f"\n[EXP27 WARNING] compute_joint_features error "
                      f"(#{self._err_count+1}): {type(e).__name__}: {e}")
            self._err_count += 1
            return {}

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP27] Processing {len(texts)} samples…")
        self._err_count = 0
        rows = []

        for text in tqdm(texts, desc="[EXP27] JSD + Gradient"):
            feat = self.compute_joint_features(text)
            rows.append(feat if feat else {})

        df = pd.DataFrame(rows)
        n_valid = df["combined_rank_score"].notna().sum() if "combined_rank_score" in df.columns else 0
        print(f"[EXP27] Valid (non-NaN combined_rank_score): {n_valid}/{len(df)} "
              f"({100*n_valid/max(1,len(df)):.1f}%)")
        if hasattr(self, '_err_count') and self._err_count > 0:
            print(f"[EXP27] Total errors: {self._err_count}")

        # ---- Member Signals (higher = more likely member) ----
        if "grad_norm_embed" in df.columns:
            df["signal_grad_embed"] = -df["grad_norm_embed"]
        if "grad_norm_head" in df.columns:
            df["signal_grad_head"] = -df["grad_norm_head"]
        if "mean_jsd_early" in df.columns:
            df["signal_jsd_early"] = -df["mean_jsd_early"]

        # ---- Rank-based Combined Score ----
        rank_sources = ["signal_grad_embed", "signal_jsd_early"]
        rank_cols = [c for c in rank_sources if c in df.columns]
        if rank_cols:
            rank_scores = np.zeros(len(df))
            for col in rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_scores += ranks / len(ranks)
            df["combined_rank_score"] = rank_scores / len(rank_cols)
        else:
            df["combined_rank_score"] = np.nan

        # ---- Product Score (secondary, for comparison) ----
        if "grad_norm_embed" in df.columns and "mean_jsd_early" in df.columns:
            # Members: both small → product small → negate for member = high score
            df["product_score"] = -(
                df["grad_norm_embed"].fillna(df["grad_norm_embed"].max())
                * df["mean_jsd_early"].fillna(df["mean_jsd_early"].max())
            )

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
        attacker = JSDWeightedGradientAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP27_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")
        print(
            "[*] NOTE: Parquet contains ALL JSD + gradient features. "
            "Feed them into EXP15 (XGBoost Stacker) for best results.\n"
            "       Feature columns: grad_norm_embed, grad_norm_head, grad_norm_mid,\n"
            "                        jsd_layer_*, mean_jsd_early"
        )

        # ---- Evaluation ----
        print("\n" + "="*65)
        print("    EXP27: JSD-WEIGHTED GRADIENT NORM — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(Grad + JSD_Early)",
            "product_score": "Product Score -(grad × jsd_early)",
            "signal_grad_embed": "Gradient Norm Only (-embed)",
            "signal_jsd_early": "JSD Early Only (-mean_jsd_early)",
        }

        report = {
            "experiment": "EXP27_jsd_weighted_gradient",
            "model": self.args.model_name,
            "timestamp": timestamp,
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
                print(f"  {label:<45} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'CombinedRank':<14} | {'GradEmbed':<12} | {'JSDEarly'}")
        print("-" * 55)
        subset_aucs = {}
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            row_vals = {}
            for score_col in ["combined_rank_score", "signal_grad_embed", "signal_jsd_early"]:
                valid_sub = sub.dropna(subset=[score_col])
                if len(valid_sub["is_member"].unique()) > 1:
                    row_vals[score_col] = roc_auc_score(
                        valid_sub["is_member"], valid_sub[score_col]
                    )
                else:
                    row_vals[score_col] = float("nan")
            print(
                f"{subset:<10} | {row_vals.get('combined_rank_score', float('nan')):.4f}         "
                f"| {row_vals.get('signal_grad_embed', float('nan')):.4f}       "
                f"| {row_vals.get('signal_jsd_early', float('nan')):.4f}"
            )
            subset_aucs[subset] = row_vals

        report["subset_aucs"] = subset_aucs
        print("="*65)

        report_path = self.output_dir / f"EXP27_report_{timestamp}.json"
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
        sample_fraction = 0.05   # ~12 500 samples; comparable to EXP11 (~3.5h on A100)
        output_dir = "results"
        max_length = 2048
        max_jsd_tokens = 64
        seed = 42

    print(f"[EXP27] Model : {Args.model_name}")
    print(f"[EXP27] Sample: {Args.sample_fraction * 100:.0f}%  max_jsd_tokens={Args.max_jsd_tokens}")
    Experiment(Args).run()
