"""
EXPERIMENT 26: Early-Settling Ratio (ESR) - "Shallow-Thinking" Depth Signal for MIA

Method:
    Use the Logit Lens technique to project intermediate hidden states to vocabulary
    space via the final norm + LM head. Compute Jensen-Shannon Divergence (JSD) between
    each intermediate layer's distribution and the final layer's distribution.

    Early-Settling Ratio (ESR): A token "settles" early if JSD(p_L || p_l) is low at
    shallow layers, meaning the model committed to its final prediction without deep
    processing. Files with a high fraction of early-settling tokens are likely memorized.

Hypothesis (Inverted DTR from "Think Deep, Not Just Long"):
    - Members (memorized code):  Model converges early -> LOW JSD at shallow layers.
    - Non-members (unseen code): Model needs deep reasoning -> HIGH JSD until late layers.

Score:
    -mean_jsd_early  (lower early-layer JSD = more likely member = higher score)

Optimization Strategy:
    - JSD computed at 5 sampled layer checkpoints only (adaptive to model depth).
    - Stride-based token sampling: max 64 tokens uniformly spread across the sequence.
    - Pure forward pass (no backward needed) -> fast.
    - bfloat16 inference, float32 JSD computation (numerical stability).

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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment():
    print("\n" + "="*60)
    print("  EXP26: EARLY-SETTLING RATIO (ESR) — JSD DEPTH MIA")
    print("="*60)
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
    # No gradients needed for this experiment
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# JSD Helper
# ============================================================================

def jsd_batch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Jensen-Shannon Divergence between two batches of distributions.
    p, q : (T, V) float32 probability tensors.
    Returns : (T,) JSD values in [0, ln(2)].
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


# ============================================================================
# Early-Settling Ratio Attack
# ============================================================================

class EarlySettlingAttack:
    """
    Computes the Early-Settling Ratio (ESR) as a white-box MIA signal.

    For each sample, hooks are registered on a set of sampled transformer
    blocks. A single forward pass captures the hidden states at those layers.
    The "logit lens" (final-norm + lm_head) is then applied to project each
    intermediate hidden state into vocabulary probability space, and JSD vs.
    the final-layer distribution is computed per-token.

    The mean JSD at early layers is negated to produce the MIA score.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.max_jsd_tokens = getattr(args, "max_jsd_tokens", 64)

        # Validate architecture access
        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError(
                "Cannot find model.model.layers — check architecture compatibility."
            )
        if not hasattr(model.model, "norm"):
            raise RuntimeError(
                "Cannot find model.model.norm — check architecture compatibility."
            )

        self.transformer_layers = model.model.layers
        self.norm_layer = model.model.norm
        self.lm_head = model.get_output_embeddings()

        self.sampled_indices = self._choose_layer_indices()
        print(f"[ESR] Sampled layer indices (0-based): {self.sampled_indices}")
        print(f"[ESR] Early layers (used for score): {self.sampled_indices[:2]}")

    def _choose_layer_indices(self) -> List[int]:
        """
        Return 5 representative layer indices:
        ~1/6, ~1/3, ~1/2, ~2/3, final.
        These match the paper's recommended checkpoints.
        """
        n = len(self.transformer_layers)
        raw = [n // 6, n // 3, n // 2, 2 * n // 3, n - 1]
        # Clamp to valid range and deduplicate while preserving order
        seen = set()
        indices = []
        for idx in raw:
            idx = max(0, min(idx, n - 1))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    @property
    def name(self) -> str:
        return "early_settling_ratio"

    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Apply final norm + LM head to intermediate hidden states.
        hidden : (T, D)  — any dtype, arrives on CPU after detach
        Returns : (T, V) float32 probability distributions

        Device/dtype alignment: hidden states are detached to CPU in bfloat16.
        We must move them to the norm layer's device and cast to its dtype
        before the forward pass, then return float32 probabilities.
        """
        with torch.no_grad():
            model_dtype = next(self.norm_layer.parameters()).dtype
            h = hidden.to(device=self.norm_layer.weight.device, dtype=model_dtype)
            normed = self.norm_layer(h)
            logits = self.lm_head(normed)
            return F.softmax(logits.float(), dim=-1)

    def _sample_token_indices(self, seq_len: int) -> List[int]:
        """Stride-based token sampling to cap computation at max_jsd_tokens."""
        if seq_len <= self.max_jsd_tokens:
            return list(range(seq_len))
        stride = seq_len // self.max_jsd_tokens
        return list(range(0, seq_len, stride))[: self.max_jsd_tokens]

    def compute_jsd_trajectory(self, text: str) -> Optional[Dict[int, float]]:
        """
        Run a single forward pass, capture hidden states at sampled layers,
        and compute the per-layer mean JSD vs. the final-layer distribution.

        Returns:
            Dict mapping layer_index -> mean_jsd_vs_final,
            or None on failure.
        """
        if not text or len(text) < 20:
            return None

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)

            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 4:
                return None

            # ---- Forward hooks to capture hidden states ----
            captured: Dict[int, torch.Tensor] = {}

            def make_hook(layer_idx: int):
                def hook_fn(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # Detach immediately to free computation graph
                    captured[layer_idx] = hs[0].detach().cpu()
                return hook_fn

            handles = [
                self.transformer_layers[idx].register_forward_hook(make_hook(idx))
                for idx in self.sampled_indices
            ]

            with torch.no_grad():
                self.model(**inputs)

            for h in handles:
                h.remove()

            # ---- Compute JSD ----
            tok_indices = self._sample_token_indices(seq_len)
            final_idx = self.sampled_indices[-1]

            if final_idx not in captured:
                return None

            # Final-layer reference distribution (computed on GPU for speed)
            final_hidden = captured[final_idx][tok_indices].to(self.model.device)
            final_dist = self._logit_lens(final_hidden)   # (T_sampled, V)

            trajectory: Dict[int, float] = {final_idx: 0.0}

            for layer_idx in self.sampled_indices[:-1]:
                if layer_idx not in captured:
                    continue
                layer_hidden = captured[layer_idx][tok_indices].to(self.model.device)
                layer_dist = self._logit_lens(layer_hidden)
                jsd_vals = jsd_batch(layer_dist, final_dist)  # (T_sampled,)
                trajectory[layer_idx] = jsd_vals.mean().item()
                del layer_hidden, layer_dist, jsd_vals

            del final_hidden, final_dist, captured
            return trajectory

        except Exception as e:
            if not hasattr(self, '_err_count'):
                self._err_count = 0
            if self._err_count < 3:
                print(f"\n[EXP26 WARNING] JSD trajectory error "
                      f"(#{self._err_count+1}): {type(e).__name__}: {e}")
            self._err_count += 1
            return None

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[ESR] Processing {len(texts)} samples…")
        self._err_count = 0

        early_layer_cols = [f"jsd_layer_{i}" for i in self.sampled_indices[:2]]
        rows = []

        for text in tqdm(texts, desc="[ESR] JSD Convergence Depth"):
            traj = self.compute_jsd_trajectory(text)

            row: Dict = {}
            for idx in self.sampled_indices:
                row[f"jsd_layer_{idx}"] = traj.get(idx, np.nan) if traj else np.nan

            if traj is not None:
                early_vals = [traj[i] for i in self.sampled_indices[:2] if i in traj]
                mid_vals = [
                    traj[i] for i in self.sampled_indices[2:4] if i in traj
                ]
                row["mean_jsd_early"] = float(np.mean(early_vals)) if early_vals else np.nan
                row["mean_jsd_mid"] = float(np.mean(mid_vals)) if mid_vals else np.nan
                # Primary MIA score: lower early JSD = more likely member
                row["esr_score"] = -row["mean_jsd_early"]
            else:
                row.update(
                    mean_jsd_early=np.nan, mean_jsd_mid=np.nan, esr_score=np.nan
                )

            rows.append(row)

        df_out = pd.DataFrame(rows)
        n_valid = df_out["esr_score"].notna().sum()
        n_total = len(df_out)
        print(f"[EXP26] Valid (non-NaN) samples: {n_valid}/{n_total} "
              f"({100*n_valid/max(1,n_total):.1f}%)")
        if hasattr(self, '_err_count') and self._err_count > 0:
            print(f"[EXP26] Total errors: {self._err_count}")
        return df_out


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
        attacker = EarlySettlingAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP26_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        # ---- Evaluation ----
        print("\n" + "="*60)
        print("     EXP26: EARLY-SETTLING RATIO — PERFORMANCE REPORT")
        print("="*60)

        valid = df.dropna(subset=["esr_score"])
        overall_auc = 0.0
        if len(valid["is_member"].unique()) > 1:
            overall_auc = roc_auc_score(valid["is_member"], valid["esr_score"])
            print(f"OVERALL AUC (ESR Score): {overall_auc:.4f}")
        else:
            print("WARNING: Single class in data — cannot compute AUC.")

        print(f"\n{'Subset':<10} | {'AUC':<8} | {'N':<6} | {'JSD_early (M)':<16} | {'JSD_early (NM)'}")
        print("-" * 65)
        subset_report = {}
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset].dropna(subset=["esr_score"])
            if len(sub["is_member"].unique()) > 1:
                auc = roc_auc_score(sub["is_member"], sub["esr_score"])
                m_jsd = sub[sub["is_member"] == 1]["mean_jsd_early"].mean()
                nm_jsd = sub[sub["is_member"] == 0]["mean_jsd_early"].mean()
                print(
                    f"{subset:<10} | {auc:.4f}   | {len(sub):<6} | {m_jsd:<16.4f} | {nm_jsd:.4f}"
                )
                subset_report[subset] = {
                    "auc": float(auc),
                    "member_mean_jsd_early": float(m_jsd),
                    "nonmember_mean_jsd_early": float(nm_jsd),
                }

        print("\nPer-Layer JSD AUC (lower JSD → member → negated score):")
        for idx in attacker.sampled_indices:
            col = f"jsd_layer_{idx}"
            if col in df.columns:
                valid_l = df.dropna(subset=[col])
                if len(valid_l["is_member"].unique()) > 1:
                    auc_l = roc_auc_score(valid_l["is_member"], -valid_l[col])
                    label = "← EARLY" if idx in attacker.sampled_indices[:2] else ""
                    print(f"  Layer {idx:3d}: AUC = {auc_l:.4f}  {label}")

        print("="*60)

        report = {
            "experiment": "EXP26_early_settling_ratio",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "overall_auc": float(overall_auc),
            "sampled_layers": attacker.sampled_indices,
            "early_layers": attacker.sampled_indices[:2],
            "max_jsd_tokens": self.args.max_jsd_tokens,
            "hypothesis": (
                "Lower JSD at shallow layers (early-settling) signals memorization. "
                "Inverted DTR from 'Think Deep, Not Just Long' paper."
            ),
            "subset_report": subset_report,
        }
        report_path = self.output_dir / f"EXP26_report_{timestamp}.json"
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
        sample_fraction = 0.05   # ~12 500 samples; ~2h on A100
        output_dir = "results"
        max_length = 2048
        max_jsd_tokens = 64      # Stride-sampled tokens for JSD (speed/accuracy tradeoff)
        seed = 42

    print(f"[EXP26] Model : {Args.model_name}")
    print(f"[EXP26] Sample: {Args.sample_fraction * 100:.0f}%  max_jsd_tokens={Args.max_jsd_tokens}")
    Experiment(Args).run()
