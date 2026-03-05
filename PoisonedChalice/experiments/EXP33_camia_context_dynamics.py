"""
EXPERIMENT 33: CAMIA — Context-Aware Loss Drop Dynamics

Paper inspiration:
    "Context-Aware Membership Inference Attacks against Pre-trained Large Language Models"
    (EMNLP 2025) — LLM next-token prediction is intrinsically context-dependent.
    Averaging loss/gradient over the whole sequence discards the DYNAMICS of how
    the model's certainty accumulates as it reads more of the file.

Core insight for code:
    Source code is highly context-coupled: function B relies on variables defined
    in function A; class methods depend on class-level attributes; import statements
    unlock entire namespaces. For a MEMORIZED file (member), the model "recognizes"
    the pattern progressively and confidently — producing a characteristic
    LOSS TRAJECTORY with one or more sharp drops as each code block is consumed.

    A non-member file may have uniformly moderate loss or a gradual decline,
    but NOT the abrupt recognition drops of a memorized file.

Three statistical features extracted from the per-block loss trajectory:

    A. Min Drop Magnitude (MDM):
       max(loss[i] - loss[i+1])  — the single biggest loss cliff across blocks.
       Members typically have at least one extreme drop.

    B. Trajectory Variance (TVar):
       std(block_losses)  — global dispersion of the loss curve.
       Members: high variance (rapid swings); Non-members: smooth / flat.

    C. Normalized Area Under Curve Gain (AUCG):
       AUC of (max_loss - loss[i]) / max_loss  — how much the loss "recovered"
       from its initial high towards a low by the end.
       Members: large AUCG (curve dips down sharply).

Block design:
    - Tokenize file → N_BLOCKS non-overlapping chunks of BLOCK_TOKENS tokens each.
    - Max BLOCK_TOKENS = 256; max N_BLOCKS = 8  (matching EXP21 window size).
    - Each block is evaluated with its CUMULATIVE PREFIX as context:
          input = [block_0 … block_i]  (increasing context window)
          loss measured only on the LAST block's tokens (not the prefix).
      This is the true context-aware loss for block i.
    - Uses teacher-forcing with causal masking via position_ids + loss masking.

Primary score:
    rank_avg(MDM, TVar, AUCG)

Compute notes:
    - N_BLOCKS forward passes per sample (no backward needed — forward only).
    - Context grows with each block → memory usage bounded by max_length=2048.
    - sample_fraction=0.10 recommended (~2.5h on A100 at 8 blocks per sample).

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
    print("  EXP33: CAMIA — CONTEXT-AWARE LOSS DROP DYNAMICS")
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
# CAMIA Attack
# ============================================================================

class CAMIAAttack:
    """
    Measures the LOSS TRAJECTORY across code blocks with growing context.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.block_tokens = getattr(args, "block_tokens", 256)
        self.max_blocks = getattr(args, "max_blocks", 8)
        self._err_count = 0

    @property
    def name(self) -> str:
        return "camia_loss_dynamics"

    def _block_loss_with_context(
        self,
        prefix_ids: List[int],   # context tokens (prefix blocks 0..i-1)
        block_ids: List[int],    # current block tokens (block i)
    ) -> Optional[float]:
        """
        Compute average cross-entropy loss on `block_ids` given `prefix_ids`.

        Method: concatenate [prefix | block], run forward pass, mask out
        loss contributions from prefix positions — keep only block positions.

        Returns mean loss per target token, or None on error.
        """
        full_ids = prefix_ids + block_ids
        # Truncate from the LEFT if total exceeds max_length (keep as much
        # of the prefix + the full block as possible)
        if len(full_ids) > self.max_length:
            # Always keep the block; trim prefix from the beginning
            keep_prefix = self.max_length - len(block_ids)
            if keep_prefix < 0:
                # Block alone is too long — truncate block
                block_ids = block_ids[:self.max_length]
                full_ids = block_ids
                prefix_ids = []
                keep_prefix = 0
            else:
                prefix_ids = prefix_ids[-keep_prefix:]
                full_ids = prefix_ids + block_ids

        n_prefix = len(prefix_ids)
        n_total = len(full_ids)

        try:
            input_ids = torch.tensor([full_ids], dtype=torch.long).to(self.model.device)

            # Build labels: ignore (-100) all prefix tokens; keep block tokens
            labels = torch.full((1, n_total), fill_value=-100, dtype=torch.long).to(self.model.device)
            labels[0, n_prefix:] = input_ids[0, n_prefix:]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
            return outputs.loss.item()

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP33 WARNING] block_loss error: {type(e).__name__}: {e}")
            self._err_count += 1
            return None

    def compute_trajectory(self, text: str) -> Optional[List[float]]:
        """
        Tokenize text, split into blocks, compute context-aware loss per block.
        Returns list of per-block losses, or None if too short.
        """
        if not text or len(text) < 50:
            return None

        try:
            tokens = self.tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
        except Exception:
            return None

        n_tokens = len(tokens)
        if n_tokens < self.block_tokens:
            # Too short for even one full block — use the whole file as one block
            loss = self._block_loss_with_context([], tokens)
            return [loss] if loss is not None else None

        n_blocks = min(n_tokens // self.block_tokens, self.max_blocks)
        if n_blocks < 2:
            return None

        trajectory: List[float] = []
        prefix_ids: List[int] = []

        for i in range(n_blocks):
            start = i * self.block_tokens
            end = start + self.block_tokens
            block_ids = tokens[start:end]

            loss = self._block_loss_with_context(prefix_ids, block_ids)
            if loss is None:
                return None       # Abort trajectory on any failure
            trajectory.append(loss)
            prefix_ids = tokens[: end]  # Grow context

        return trajectory

    def _trajectory_features(self, traj: List[float]) -> Dict[str, float]:
        """
        Extract MDM, TVar, AUCG and auxiliary statistics from a loss trajectory.
        """
        t = np.array(traj, dtype=np.float32)
        n = len(t)

        # A. Min Drop Magnitude: biggest single-step drop (loss[i] - loss[i+1])
        if n >= 2:
            drops = t[:-1] - t[1:]          # positive when loss decreases
            mdm = float(np.max(drops))       # larger = more "recognition" events
            mean_drop = float(np.mean(drops))
        else:
            mdm = 0.0
            mean_drop = 0.0

        # B. Trajectory Variance
        tvar = float(np.std(t))

        # C. AUCG — Area Under the "gain" curve
        # gain[i] = (max_loss - loss[i]) / max_loss
        max_loss = float(np.max(t)) if np.max(t) > 0 else 1.0
        gain = (max_loss - t) / max_loss   # 0 at start, grows as loss drops
        aucg = float(np.trapz(gain) / max(n - 1, 1))  # normalised trapezoid area

        # Additional: final loss, initial loss, net drop
        return {
            "loss_initial":   float(t[0]),
            "loss_final":     float(t[-1]),
            "loss_net_drop":  float(t[0] - t[-1]),
            "loss_min":       float(np.min(t)),
            "loss_max":       float(np.max(t)),
            "tvar":           tvar,
            "mdm":            mdm,
            "mean_drop":      mean_drop,
            "aucg":           aucg,
            "n_blocks":       n,
        }

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP33] Processing {len(texts)} samples…")
        print(f"[EXP33] block_tokens={self.block_tokens}  max_blocks={self.max_blocks}")
        rows = []

        for text in tqdm(texts, desc="[EXP33] CAMIA Loss Dynamics"):
            traj = self.compute_trajectory(text)
            if traj is not None and len(traj) >= 2:
                feat = self._trajectory_features(traj)
                # Store the full trajectory for potential XGBoost stacking
                for i, v in enumerate(traj):
                    feat[f"block_loss_{i}"] = v
            else:
                feat = {
                    "loss_initial": np.nan, "loss_final": np.nan,
                    "loss_net_drop": np.nan, "loss_min": np.nan, "loss_max": np.nan,
                    "tvar": np.nan, "mdm": np.nan, "mean_drop": np.nan,
                    "aucg": np.nan, "n_blocks": 0,
                }
            rows.append(feat)

        df = pd.DataFrame(rows)

        # ---- Member signals (higher = more likely member) ----
        # Members: large drops, high variance, large area gain
        for col, sign in [("mdm", +1), ("tvar", +1), ("aucg", +1),
                          ("loss_net_drop", +1), ("loss_final", -1)]:
            if col in df.columns:
                df[f"signal_{col}"] = sign * df[col]

        # ---- Combined rank score ----
        rank_sources = ["signal_mdm", "signal_tvar", "signal_aucg"]
        valid_rank_cols = [c for c in rank_sources if c in df.columns]
        if valid_rank_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_sum += ranks / len(ranks)
            df["combined_rank_score"] = rank_sum / len(valid_rank_cols)

        n_valid = df["combined_rank_score"].notna().sum() if "combined_rank_score" in df.columns else 0
        print(f"[EXP33] Valid samples: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP33] Total errors: {self._err_count}")
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
        attacker = CAMIAAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP33_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "="*65)
        print("   EXP33: CAMIA LOSS DYNAMICS — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(MDM + TVar + AUCG)  [PRIMARY]",
            "signal_mdm":          "Min-Drop Magnitude (MDM)",
            "signal_tvar":         "Trajectory Variance (TVar)",
            "signal_aucg":         "Area-Under-Gain Curve (AUCG)",
            "signal_loss_final":   "-Final Block Loss",
        }
        report = {
            "experiment": "EXP33_camia_loss_dynamics",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "block_tokens": self.args.block_tokens,
            "max_blocks": self.args.max_blocks,
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

        print(f"\nLoss trajectory statistics:")
        for col in ["loss_initial", "loss_final", "mdm", "tvar", "aucg"]:
            if col in df.columns:
                m_val = df[df["is_member"] == 1][col].mean()
                nm_val = df[df["is_member"] == 0][col].mean()
                print(f"  {col:<18}: Member={m_val:.4f}  Non-member={nm_val:.4f}")

        print(f"\n{'Subset':<10} | {'CombinedAUC':<13} | {'MDM AUC':<10} | {'TVar AUC':<10} | N")
        print("-"*55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["combined_rank_score", "signal_mdm", "signal_tvar"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('combined_rank_score', float('nan')):.4f}        "
                  f"| {r.get('signal_mdm', float('nan')):.4f}     "
                  f"| {r.get('signal_tvar', float('nan')):.4f}     | {len(sub)}")
            report["subset_aucs"][subset] = r

        print("="*65)
        print("\nInterpretation:")
        print("  High MDM + High TVar → abrupt loss cliffs → memorized (member)")
        print("  Low MDM + Low TVar  → smooth loss curve → unseen (non-member)")

        report_path = self.output_dir / f"EXP33_report_{timestamp}.json"
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
        # N_BLOCKS forward passes per sample (forward only, no backward)
        # 8 passes × ~0.1s each = ~0.8s/sample → ~2.5h on A100 at 10%
        sample_fraction = 0.10
        output_dir = "results"
        max_length = 2048
        block_tokens = 256      # Tokens per code block
        max_blocks = 8          # Max blocks per file
        seed = 42

    print(f"[EXP33] Model       : {Args.model_name}")
    print(f"[EXP33] Sample      : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP33] Block size  : {Args.block_tokens} tokens")
    print(f"[EXP33] Max blocks  : {Args.max_blocks}")
    Experiment(Args).run()
