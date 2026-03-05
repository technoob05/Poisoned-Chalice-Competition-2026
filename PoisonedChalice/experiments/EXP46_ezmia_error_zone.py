"""
EXPERIMENT 46: EZ-MIA — Error Zone Membership Inference Attack

Paper: "Powerful Training-Free Membership Inference Against Autoregressive LMs"
       Ilic et al. (arXiv:2601.12104v1, Jan 2026)

Core insight:
    Memorization manifests most strongly at ERROR positions — tokens where
    the model predicts incorrectly yet still shows elevated probability
    for training examples compared to a pretrained reference.

EZ Score formula:
    delta(t) = log p_target(x_t | x<t) - log p_ref(x_t | x<t)
    E = {t : argmax_v p_target(v | x<t) != x_t}  (error positions)
    P = sum([delta(t)]+ for t in E)  (upward probability mass)
    N = sum(|[delta(t)]-| for t in E)  (downward probability mass)
    EZ(x) = P / N  (ratio: scale-invariant, higher = more likely member)

    Key: at error positions, fine-tuning elevates correct token probability
    for members even when the token isn't the top prediction.
    This "residual memorization signal" is invisible to aggregate loss metrics.

Paper results (fine-tuned models):
    - WikiText/GPT-2: AUC 0.984, TPR@1%FPR 66.3%
    - AG News/Llama-2-7B: AUC 0.961, TPR@1%FPR 46.7%
    - Code (Swallow-Code/StableCode-3B LoRA): AUC 0.893, TPR@1%FPR 38.8%

IMPORTANT: Paper targets FINE-TUNED models (strong memorization from small
datasets, multiple epochs). Our setup is PRE-TRAINING detection on StarCoder2-3b
(single epoch, massive corpus) — weaker memorization signal expected.
We use the pretrained model itself as both target AND reference is not possible.
Instead, we adapt: use per-token log-prob differences between the model's
prediction and a "calibrated" baseline (e.g., uniform, or temperature-scaled).

Adaptation for pre-training MIA on StarCoder2-3b:
    Since we don't have a separate "before fine-tuning" checkpoint, we adapt
    EZ-MIA's core insight in two ways:
    1. EZ-Surrogate: use a smaller model (e.g., bigcode/starcoderbase-1b) as
       reference, measuring what StarCoder2-3b learned beyond the smaller model.
    2. EZ-Self: use temperature scaling as pseudo-reference — at high temperature,
       the model's predictions approach uniform, simulating "before memorization."
    3. Also compute standard loss and Min-K%++ for comparison.

Compute: 2 forward passes per sample (target + reference), forward-only
Expected runtime: ~20-30 min on A100 (10% sample)
Expected AUC: 0.50-0.60 (pre-training detection is much harder than fine-tuning)
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
    print("  EXP46: EZ-MIA — Error Zone Membership Inference")
    print("  Paper: Ilic et al. (arXiv:2601.12104v1, Jan 2026)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, label: str = "target"):
    print(f"[*] Loading {label} model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}")
    return model, tokenizer


class EZMIAScorer:
    """EZ-MIA: Error Zone score using target vs reference model."""

    def __init__(self, target_model, ref_model, tokenizer, max_length=512):
        self.target = target_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self._has_ref = ref_model is not None

    @torch.no_grad()
    def _get_token_logprobs(self, model, input_ids):
        """Get per-token log-probabilities and top predictions."""
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        T = shift_labels.shape[1]
        if T == 0:
            return None, None, None

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (T,)
        top_preds = shift_logits.argmax(dim=-1).squeeze(0)  # (T,)
        return token_lp.cpu().numpy(), top_preds.cpu().numpy(), shift_labels.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract EZ-MIA features for a single sample."""
        result = {k: np.nan for k in [
            "ez_score", "ez_score_log", "mean_delta_error", "mean_delta_all",
            "error_fraction", "pos_fraction_error",
            "neg_mean_loss", "neg_mean_loss_ref", "loss_ratio",
        ]}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.target.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 10:
                return result

            # Target model forward pass
            target_lp, target_top, labels = self._get_token_logprobs(self.target, input_ids)
            if target_lp is None:
                return result

            result["neg_mean_loss"] = float(target_lp.mean())
            T = len(target_lp)

            # Reference model forward pass
            if self._has_ref:
                ref_input = input_ids.to(self.ref.device) if hasattr(self.ref, 'device') else input_ids
                ref_lp, _, _ = self._get_token_logprobs(self.ref, ref_input)
                if ref_lp is None:
                    return result
                result["neg_mean_loss_ref"] = float(ref_lp.mean())
            else:
                # Self-reference via temperature scaling (T=5.0 smooths predictions)
                outputs = self.target(input_ids=input_ids)
                logits_hot = outputs.logits.float()[:, :-1, :] / 5.0
                log_probs_hot = F.log_softmax(logits_hot, dim=-1)
                ref_lp = log_probs_hot.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().numpy()
                result["neg_mean_loss_ref"] = float(ref_lp.mean())

            # Delta: log-prob difference (target - reference)
            delta = target_lp - ref_lp  # (T,)

            # Loss ratio (reference-based calibration like Carlini et al.)
            result["loss_ratio"] = float(target_lp.mean() - ref_lp.mean())

            # All-position statistics
            result["mean_delta_all"] = float(delta.mean())

            # Error positions: where target's top prediction != ground truth
            error_mask = target_top != labels
            n_errors = error_mask.sum()
            result["error_fraction"] = float(n_errors / T)

            if n_errors < 3:
                # Too few errors → assign high EZ (model predicts almost everything correctly)
                result["ez_score"] = 100.0
                result["ez_score_log"] = np.log(100.0)
                result["pos_fraction_error"] = 1.0
                result["mean_delta_error"] = float(delta.mean()) if len(delta) > 0 else 0.0
                return result

            # EZ score computation on error positions
            delta_error = delta[error_mask]
            result["mean_delta_error"] = float(delta_error.mean())

            P = np.maximum(delta_error, 0).sum()  # upward mass
            N = np.abs(np.minimum(delta_error, 0)).sum()  # downward mass

            result["pos_fraction_error"] = float((delta_error > 0).sum() / len(delta_error))

            if N < 1e-10:
                result["ez_score"] = 100.0  # all movement upward → strong member signal
                result["ez_score_log"] = np.log(100.0)
            elif P < 1e-10:
                result["ez_score"] = 0.01
                result["ez_score_log"] = np.log(0.01)
            else:
                ez = P / N
                result["ez_score"] = float(ez)
                result["ez_score_log"] = float(np.log(max(ez, 1e-10)))

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP46 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)

        self.target_model, self.tokenizer = load_model(args.model_name, "target")

        self.ref_model = None
        if args.ref_model_name and args.ref_model_name != args.model_name:
            try:
                self.ref_model, _ = load_model(args.ref_model_name, "reference")
            except Exception as e:
                print(f"[WARN] Could not load reference model: {e}")
                print("  Falling back to temperature-scaled self-reference")

    def load_data(self) -> pd.DataFrame:
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
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
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed).reset_index(drop=True)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        scorer = EZMIAScorer(
            self.target_model, self.ref_model, self.tokenizer,
            max_length=self.args.max_length,
        )
        ref_label = self.args.ref_model_name if self.ref_model else "self-temp(T=5)"
        print(f"\n[EXP46] Reference: {ref_label}")
        print(f"[EXP46] Extracting EZ features for {len(df)} samples...")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP46]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df["ez_score"].notna().sum()
        print(f"\n[EXP46] Valid: {n_valid}/{len(df)}")
        if scorer._err_count > 0:
            print(f"[EXP46] Errors: {scorer._err_count}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   EXP46: EZ-MIA — REPORT")
        print("=" * 70)

        score_cols = [
            ("neg_mean_loss", "Loss (avg LL)"),
            ("neg_mean_loss_ref", "Loss-Ref"),
            ("loss_ratio", "Loss Ratio (target - ref)"),
            ("ez_score", "EZ Score (P/N) [PRIMARY]"),
            ("ez_score_log", "log(EZ Score)"),
            ("mean_delta_error", "Mean delta at errors"),
            ("mean_delta_all", "Mean delta (all positions)"),
            ("pos_fraction_error", "Positive fraction at errors"),
            ("error_fraction", "-Error fraction"),
        ]

        aucs = {}
        for col, label in score_cols:
            if col not in df.columns:
                continue
            vals = df[col].copy()
            if col == "error_fraction":
                vals = -vals  # fewer errors = more likely member
            v = df.dropna(subset=[col])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], vals[v.index])
                aucs[col] = auc
                tag = " <-- PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<40} AUC = {auc:.4f}{tag}")

        if aucs:
            best = max(aucs, key=aucs.get)
            print(f"\n  Best signal: {best} = {aucs[best]:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:    0.6472")

        # Error stats
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for col, label in [("error_fraction", "Error fraction"), ("ez_score", "EZ score"),
                           ("mean_delta_error", "Mean delta@error")]:
            m_val = m[col].dropna()
            nm_val = nm[col].dropna()
            if len(m_val) > 0 and len(nm_val) > 0:
                print(f"  {label}: M={m_val.mean():.4f} NM={nm_val.mean():.4f}")

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'Loss':<8} | {'EZ Score':<10} | {'LossRatio':<10} | N")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["neg_mean_loss", "ez_score", "loss_ratio"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('neg_mean_loss', float('nan')):.4f}   "
                  f"| {r.get('ez_score', float('nan')):.4f}     "
                  f"| {r.get('loss_ratio', float('nan')):.4f}     "
                  f"| {len(sub)}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP46_{timestamp}.parquet", index=False)
        print(f"\n[EXP46] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        ref_model_name = "bigcode/starcoderbase-1b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        max_length = 512
        output_dir = "results"
        seed = 42

    print(f"[EXP46] EZ-MIA: target={Args.model_name}, ref={Args.ref_model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    Experiment(Args).run()
