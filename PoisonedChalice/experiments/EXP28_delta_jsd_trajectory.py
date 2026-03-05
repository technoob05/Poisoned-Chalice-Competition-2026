"""
EXPERIMENT 28: Delta JSD Trajectory Stability under Context Perturbation

Method:
    Measure how a context perturbation (variable renaming) disrupts the
    model's "early-settling" behaviour by computing the Delta JSD Trajectory:

        Delta_JSD_early = mean_jsd_early(perturbed) - mean_jsd_early(original)

    Two forward passes per sample (no backward/gradient needed):
        1. Original code  → JSD trajectory at sampled layers
        2. Perturbed code → JSD trajectory at sampled layers

Hypothesis (Inverted EXP07 × JSD Lens):
    - Members (memorized):    Original code settles early (low JSD at shallow layers).
                              Renaming breaks the exact memorized token sequence, forcing
                              the model to "think deeper" → JSD at early layers INCREASES.
                              Δ = jsd_perturbed_early - jsd_original_early > 0  (LARGE)

    - Non-members (unseen):   Original code already requires deep thinking (high JSD).
                              Renaming doesn't meaningfully change the processing depth.
                              Δ ≈ 0  (SMALL)

Score:
    delta_jsd_early  (higher = more likely member)

Perturbation Strategy:
    Deterministic variable renaming (no randomness):
        i   → idx_var,  j  → jdx_var,  k  → kdx_var
        x   → xvar,     y  → yvar,     n  → nvar
        s   → svar,     v  → vvar,     p  → pvar
        data   → _data_,   result  → _result_,  tmp → _tmp_
        val    → _val_,    res     → _res_,      err → _err_

    Applied to the first 50% of the file (context), matching EXP07's split.
    This preserves the second half (target) unchanged, isolating the context
    perturbation effect on the model's prediction trajectory.

Compute Strategy:
    - Two pure forward passes per sample (no gradient → 2× EXP26 speed).
    - Stride-based token sampling: max 64 tokens for JSD per pass.
    - bfloat16 inference, float32 JSD computation.
    - sample_fraction=0.05 recommended (~3.5h on A100 for full dataset).

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
    print("\n" + "="*65)
    print("  EXP28: DELTA JSD TRAJECTORY — PERTURBATION DEPTH ANALYSIS")
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
        p.requires_grad_(False)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# Perturbation
# ============================================================================

# Deterministic renaming map: covers common short identifiers in
# Python, Go, Java, Ruby, Rust.  Applied only to the context (first 50%).
# Uses word-boundary-like replacements to avoid partial matches.
_RENAME_PAIRS: List[Tuple[str, str]] = [
    # Single-letter loop variables (with surrounding space/bracket)
    (" i ", " idx_var "),   (" j ", " jdx_var "),   (" k ", " kdx_var "),
    (" x ", " xvar "),      (" y ", " yvar "),       (" n ", " nvar "),
    (" s ", " svar "),      (" v ", " vvar "),       (" p ", " pvar "),
    # With punctuation following
    (" i,", " idx_var,"),   (" j,", " jdx_var,"),   (" k,", " kdx_var,"),
    (" i)", " idx_var)"),   (" j)", " jdx_var)"),    (" n)", " nvar)"),
    (" i<", " idx_var<"),   (" i>", " idx_var>"),    (" i+", " idx_var+"),
    (" i-", " idx_var-"),   ("[i]", "[idx_var]"),    ("[j]", "[jdx_var]"),
    # Common word identifiers
    (" data ", " _data_ "),     (" result ", " _result_ "),
    (" tmp ", " _tmp_ "),       (" val ", " _val_ "),
    (" res ", " _res_ "),       (" err ", " _err_ "),
    (" buf ", " _buf_ "),       (" out ", " _out_ "),
    (" cur ", " _cur_ "),       (" prev ", " _prev_ "),
    (" next ", " _next_ "),     (" node ", " _node_ "),
    (" left ", " _left_ "),     (" right ", " _right_ "),
    # Tab-delimited (Go / Rust style)
    ("\ti\t", "\tidx_var\t"), ("\tj\t", "\tjdx_var\t"),
    ("\ti,", "\tidx_var,"),   ("\tn,", "\tnvar,"),
]


def rename_variables(code: str) -> str:
    """
    Deterministic variable renaming to break memorized token sequences.
    Applied only to the context half of the file (caller's responsibility).
    """
    for old, new in _RENAME_PAIRS:
        code = code.replace(old, new)
    return code


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
# Delta JSD Trajectory Attack
# ============================================================================

class DeltaJSDTrajectoryAttack:
    """
    For each code file:
        1. Run forward pass on original → JSD trajectory (early-layer mean)
        2. Rename variables in the first 50% of the file
        3. Run forward pass on perturbed → JSD trajectory (early-layer mean)
        4. Score = Δ = jsd_early(perturbed) - jsd_early(original)

    Members   → large positive Δ  (perturbation disrupts early settling)
    Non-members → Δ ≈ 0           (deep-thinking, unaffected by renaming)
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

        self.sampled_indices = self._choose_layer_indices()
        print(f"[EXP28] Sampled layers: {self.sampled_indices}")
        print(f"[EXP28] Early layers (for Δ): {self.sampled_indices[:2]}")

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
        return "delta_jsd_trajectory"

    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Final norm + LM head projection → (T, V) softmax distribution.
        Hidden states are cast to the model's native dtype to avoid
        float32/bfloat16 dtype mismatch at the norm & head layers.
        """
        with torch.no_grad():
            # Use the weight dtype of the norm layer to avoid dtype mismatch
            model_dtype = next(self.norm_layer.parameters()).dtype
            h = hidden.to(device=self.norm_layer.weight.device, dtype=model_dtype)
            normed = self.norm_layer(h)
            logits = self.lm_head(normed)
            return F.softmax(logits.float(), dim=-1)   # return float32 probs

    def _sample_token_indices(self, seq_len: int) -> List[int]:
        if seq_len <= self.max_jsd_tokens:
            return list(range(seq_len))
        stride = seq_len // self.max_jsd_tokens
        return list(range(0, seq_len, stride))[: self.max_jsd_tokens]

    # Error counter for throttled stderr logging
    _err_count: int = 0
    _err_max_log: int = 3   # Only print the first 3 unique errors

    def _jsd_trajectory_for_text(self, text: str) -> Optional[Dict[int, float]]:
        """
        Run one forward pass and return {layer_idx: mean_jsd_vs_final}.
        Returns None on failure.
        """
        if not text or len(text) < 20:
            return None

        captured: Dict[int, torch.Tensor] = {}
        handles = []

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

            def make_hook(layer_idx: int):
                def hook_fn(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # Keep on CPU to save GPU memory; device-move happens in _logit_lens
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
            handles = []

            tok_indices = self._sample_token_indices(seq_len)
            final_idx = self.sampled_indices[-1]

            if final_idx not in captured:
                captured.clear()
                return None

            # _logit_lens handles device + dtype alignment internally
            final_dist = self._logit_lens(captured[final_idx][tok_indices])

            trajectory: Dict[int, float] = {final_idx: 0.0}
            for layer_idx in self.sampled_indices[:-1]:
                if layer_idx not in captured:
                    continue
                layer_dist = self._logit_lens(captured[layer_idx][tok_indices])
                jsd_vals = jsd_batch(layer_dist, final_dist)
                trajectory[layer_idx] = jsd_vals.mean().item()
                del layer_dist, jsd_vals

            del final_dist
            captured.clear()
            return trajectory

        except Exception as e:
            # Remove any handles still registered
            for h in handles:
                try: h.remove()
                except Exception: pass
            captured.clear()
            # Throttled error logging — only print first N errors
            if self._err_count < self._err_max_log:
                print(f"\n[EXP28 WARNING] _jsd_trajectory_for_text error "
                      f"(sample #{self._err_count+1}): {type(e).__name__}: {e}")
            self._err_count += 1
            return None

    def _mean_jsd_early(self, trajectory: Optional[Dict[int, float]]) -> float:
        """Return mean JSD at the first two sampled layers (excluding final)."""
        if trajectory is None:
            return np.nan
        early_vals = [trajectory[i] for i in self.sampled_indices[:2] if i in trajectory]
        return float(np.mean(early_vals)) if early_vals else np.nan

    def compute_delta(self, text: str) -> Dict[str, float]:
        """
        Compute Delta JSD Trajectory for one sample.

        Returns dict with:
            delta_jsd_early   : Δ = jsd_early(perturbed) - jsd_early(original)
            jsd_early_orig    : baseline JSD (original file)
            jsd_early_perturb : JSD after variable renaming
            <per-layer deltas>
        """
        result: Dict[str, float] = {
            "delta_jsd_early": np.nan,
            "jsd_early_orig": np.nan,
            "jsd_early_perturb": np.nan,
        }
        for idx in self.sampled_indices:
            result[f"delta_jsd_layer_{idx}"] = np.nan

        if not text or len(text) < 50:
            return result

        # ---- Pass 1: Original ----
        orig_traj = self._jsd_trajectory_for_text(text)

        # ---- Perturb context (first 50% of file) ----
        split = len(text) // 2
        context_perturbed = rename_variables(text[:split])
        perturbed_text = context_perturbed + text[split:]

        # ---- Pass 2: Perturbed ----
        perturb_traj = self._jsd_trajectory_for_text(perturbed_text)

        # ---- Compute Delta ----
        jsd_orig_early = self._mean_jsd_early(orig_traj)
        jsd_perturb_early = self._mean_jsd_early(perturb_traj)

        result["jsd_early_orig"] = jsd_orig_early
        result["jsd_early_perturb"] = jsd_perturb_early
        result["delta_jsd_early"] = (
            jsd_perturb_early - jsd_orig_early
            if not np.isnan(jsd_orig_early) and not np.isnan(jsd_perturb_early)
            else np.nan
        )

        # Per-layer deltas
        for idx in self.sampled_indices:
            if orig_traj and perturb_traj and idx in orig_traj and idx in perturb_traj:
                result[f"delta_jsd_layer_{idx}"] = (
                    perturb_traj[idx] - orig_traj[idx]
                )

        return result

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP28] Processing {len(texts)} samples (2 passes each)…")
        rows = []
        for text in tqdm(texts, desc="[EXP28] Delta JSD Trajectory"):
            rows.append(self.compute_delta(text))
        df = pd.DataFrame(rows)
        n_valid = df["delta_jsd_early"].notna().sum()
        n_total = len(df)
        print(f"[EXP28] Valid (non-NaN) samples: {n_valid}/{n_total} "
              f"({100*n_valid/max(1,n_total):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP28] Total errors silenced: {self._err_count} "
                  f"(only first {self._err_max_log} printed above)")
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
        attacker = DeltaJSDTrajectoryAttack(self.args, self.model, self.tokenizer)

        # Run a quick single-sample debug before processing the full dataset
        sample_text = df["content"].iloc[0]
        debug_single_sample(self.model, self.tokenizer, text=sample_text[:500])

        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP28_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        # ---- Evaluation ----
        print("\n" + "="*65)
        print("   EXP28: DELTA JSD TRAJECTORY — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "delta_jsd_early": "Δ JSD Early (primary)",
            "jsd_early_orig": "JSD Early Original (-orig) [EXP26 baseline]",
        }

        report = {
            "experiment": "EXP28_delta_jsd_trajectory",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "aucs": {},
            "subset_aucs": {},
        }

        # delta_jsd_early: higher = more likely member
        for score_col, label in score_candidates.items():
            if score_col not in df.columns:
                continue
            use_col = score_col
            negate = (score_col == "jsd_early_orig")  # orig: lower = member
            vals = -df[use_col] if negate else df[use_col]
            valid_mask = vals.notna()
            if valid_mask.sum() > 0 and len(df.loc[valid_mask, "is_member"].unique()) > 1:
                auc = roc_auc_score(df.loc[valid_mask, "is_member"], vals[valid_mask])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if score_col == "delta_jsd_early" else ""
                print(f"  {label:<50} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'Delta AUC':<12} | {'N':<6} | "
              f"{'Δ (Members)':<14} | {'Δ (Non-Mbrs)'}")
        print("-" * 65)
        subset_aucs = {}
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset].dropna(subset=["delta_jsd_early"])
            if len(sub["is_member"].unique()) > 1:
                auc = roc_auc_score(sub["is_member"], sub["delta_jsd_early"])
                delta_m = sub[sub["is_member"] == 1]["delta_jsd_early"].mean()
                delta_nm = sub[sub["is_member"] == 0]["delta_jsd_early"].mean()
                print(
                    f"{subset:<10} | {auc:.4f}       | {len(sub):<6} | "
                    f"{delta_m:<14.4f} | {delta_nm:.4f}"
                )
                subset_aucs[subset] = {
                    "auc": float(auc),
                    "member_mean_delta": float(delta_m),
                    "nonmember_mean_delta": float(delta_nm),
                }

        report["subset_aucs"] = subset_aucs
        print("="*65)
        print("\nInterpretation:")
        print("  Δ(Members)    >> Δ(Non-Members)  ✓  Hypothesis confirmed")
        print("  Δ(Members)    ~= Δ(Non-Members)     Hypothesis not supported")

        report_path = self.output_dir / f"EXP28_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\n[*] Report saved: {report_path.name}")


# ============================================================================
# Debug Utility — Run this FIRST on Kaggle to validate pipeline
# ============================================================================

def debug_single_sample(model, tokenizer, text: str = None):
    """
    Quick sanity check: run JSD trajectory on one sample and print all
    intermediate values. Call this after loading the model if you suspect
    NaN issues.
    """
    import traceback
    if text is None:
        text = "def hello_world():\n    print('hello')\n    return 42\n"

    print("\n" + "="*60)
    print("  EXP28 DEBUG: Single-sample pipeline check")
    print("="*60)

    class _DummyArgs:
        max_length = 512
        max_jsd_tokens = 16
        seed = 42

    try:
        att = DeltaJSDTrajectoryAttack(_DummyArgs(), model, tokenizer)
        print(f"Sampled layers: {att.sampled_indices}")

        # Direct trajectory test
        traj = att._jsd_trajectory_for_text(text)
        print(f"Trajectory result: {traj}")

        delta = att.compute_delta(text)
        print(f"Delta result: {delta}")
        print("✓ DEBUG PASSED — pipeline working correctly")
    except Exception:
        print("✗ DEBUG FAILED:")
        traceback.print_exc()
    print("="*60 + "\n")


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
        sample_fraction = 0.05   # 2 forward passes per sample → ~3.5h on A100
        output_dir = "results"
        max_length = 2048
        max_jsd_tokens = 64      # Stride-sampled tokens per pass
        seed = 42

    print(f"[EXP28] Model : {Args.model_name}")
    print(f"[EXP28] Sample: {Args.sample_fraction * 100:.0f}%  max_jsd_tokens={Args.max_jsd_tokens}")
    print("[EXP28] Perturbation: deterministic variable renaming on context (first 50%)")
    Experiment(Args).run()
