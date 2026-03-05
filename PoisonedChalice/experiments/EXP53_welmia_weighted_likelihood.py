"""
EXPERIMENT 53: WEL-MIA — Weight-Enhanced Likelihood Membership Inference

Paper: "Not All Tokens Are Equal: Membership Inference Attacks Against
        Fine-tuned Language Models"
       Song, Zhao, Xiang (ACSAC 2024)

Survey reference: Wu & Cao (arXiv:2503.19338v3, Aug 2025), Section 4.2 [59]

Core idea:
    Standard likelihood ratio MIA treats all tokens equally. WEL-MIA assigns
    per-token WEIGHTS to the likelihood ratio to amplify informative tokens
    and suppress noisy ones.

    For token x_i, the weight w_i is:
        w_i = log p_target(x_i | x_{<i}) * (-1)    [higher if hard for target]
            / log p_ref(x_i | x_{<i}) * (-1)        [lower if hard for reference]

    Simplified: w_i = NLL_target(x_i) / NLL_ref(x_i)

    Tokens that are HARD for the target model (high NLL) get higher weight
    because they're more likely to discriminate member/non-member.
    Tokens that are also HARD for the reference model get lower weight
    because the difficulty is inherent (not membership-related).

    Weighted score:
        WEL(x) = sum(w_i * [log p_target(x_i) - log p_ref(x_i)]) / sum(w_i)

    This is a per-token weighted version of the likelihood ratio.

Adaptation for Poisoned Chalice:
    - Original paper fine-tunes the pre-trained model as reference. We use
      starcoderbase-1b as a surrogate reference (same approach as EXP46/47).
    - We also test a simpler weighting: w_i = NLL_target(x_i) (weight by
      token difficulty for target only).
    - Compare: unweighted ratio, target-weighted, dual-weighted.

    2 forward passes per sample (target + reference). Forward-only, 10% sample.
    Reference: bigcode/starcoderbase-1b (~3.5 GB VRAM)

Expected runtime: ~15-20 min on A100
Expected AUC: 0.55-0.65 (likelihood ratio with smart weighting may boost
    over raw loss; reference model provides calibration)
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
    print("  EXP53: WEL-MIA — Weighted Enhanced Likelihood")
    print("  Paper: Song, Zhao, Xiang (ACSAC 2024)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, label: str = ""):
    print(f"[*] Loading {label}: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded {label}. dtype={dtype}")
    return model, tokenizer


def load_ref_model_with_fallback(model_path: str, self_temp: float = 2.0):
    """Try to load reference model; fall back to self-temperature scaling if gated."""
    try:
        model, _ = load_model(model_path, "reference")
        print(f"  [EXP53] Reference model loaded successfully.")
        return model, False
    except (OSError, Exception) as e:
        print(f"\n[EXP53] WARNING: Reference model unavailable ({type(e).__name__}).")
        print(f"  Likely gated repo: {model_path}")
        print(f"  Falling back to SELF-TEMPERATURE reference (T={self_temp}).")
        print(f"  Note: self-temp is a monotonic transform, not a true reference.")
        print(f"  Expected AUC degradation: same as EXP46/47 (~0.56 ceiling).")
        return None, True


class WELScorer:
    """Compute weighted and unweighted likelihood ratio scores."""

    def __init__(self, target_model, ref_model, tokenizer, max_length: int = 512,
                 self_temp: float = 2.0):
        self.target_model = target_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_self_temp = ref_model is None
        self.self_temp = self_temp
        self._err_count = 0

    @torch.no_grad()
    def _get_per_token_nll(self, model, input_ids: torch.Tensor) -> np.ndarray:
        """Get per-token negative log-likelihood."""
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, :-1, :].float()
        labels = input_ids[0, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        return -token_ll.cpu().numpy()  # NLL, shape (T-1,)

    @torch.no_grad()
    def _get_per_token_nll_self_temp(self, input_ids: torch.Tensor) -> np.ndarray:
        """Self-temperature-scaled reference: logits divided by self_temp."""
        outputs = self.target_model(input_ids=input_ids)
        logits = (outputs.logits[0, :-1, :].float() / self.self_temp)
        labels = input_ids[0, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        return -token_ll.cpu().numpy()

    def score(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            )
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 5:
                return result

            # Target model NLL
            input_ids_t = input_ids.to(self.target_model.device)
            nll_target = self._get_per_token_nll(self.target_model, input_ids_t)

            # Reference model NLL (true ref or self-temp fallback)
            if self.use_self_temp:
                nll_ref = self._get_per_token_nll_self_temp(input_ids_t)
            else:
                input_ids_r = input_ids.to(self.ref_model.device)
                nll_ref = self._get_per_token_nll(self.ref_model, input_ids_r)

            T = len(nll_target)
            if T < 3:
                return result

            # Clamp to avoid division issues
            nll_target_c = np.clip(nll_target, 1e-6, 50.0)
            nll_ref_c = np.clip(nll_ref, 1e-6, 50.0)

            # --- 1. Unweighted likelihood ratio (baseline) ---
            lr_per_token = nll_ref_c - nll_target_c  # positive = member signal
            result["lr_mean"] = float(lr_per_token.mean())

            # --- 2. Target-weighted: w_i = NLL_target(x_i) ---
            w_target = nll_target_c
            w_target_norm = w_target / (w_target.sum() + 1e-10)
            result["wel_target"] = float((w_target_norm * lr_per_token).sum())

            # --- 3. Dual-weighted (paper): w_i = NLL_target / NLL_ref ---
            w_dual = nll_target_c / nll_ref_c
            w_dual_norm = w_dual / (w_dual.sum() + 1e-10)
            result["wel_dual"] = float((w_dual_norm * lr_per_token).sum())

            # --- 4. Inverse-ref weighted: w_i = 1/NLL_ref ---
            # Tokens easy for reference (low NLL_ref) get high weight
            w_inv_ref = 1.0 / nll_ref_c
            w_inv_ref_norm = w_inv_ref / (w_inv_ref.sum() + 1e-10)
            result["wel_inv_ref"] = float((w_inv_ref_norm * lr_per_token).sum())

            # --- 5. Min-K%-style: average LR of tokens with highest target NLL ---
            k_pct = 0.20
            k = max(1, int(T * k_pct))
            top_k_idx = np.argsort(nll_target)[-k:]
            result["lr_topk20"] = float(lr_per_token[top_k_idx].mean())

            # --- 6. Raw signals for comparison ---
            result["neg_mean_loss"] = -float(nll_target.mean())
            result["neg_ref_loss"] = -float(nll_ref.mean())
            result["loss_ratio"] = float(nll_target.mean() / (nll_ref.mean() + 1e-10))

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP53 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)

        self.target_model, self.tokenizer = load_model(args.model_name, "target")
        self.ref_model, self.use_self_temp = load_ref_model_with_fallback(
            args.ref_model_name, getattr(args, "self_temp", 2.0)
        )

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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed,
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        ref_desc = f"self-temp(T={getattr(self.args, 'self_temp', 2.0)})" if self.use_self_temp else self.args.ref_model_name
        scorer = WELScorer(
            self.target_model, self.ref_model, self.tokenizer,
            max_length=self.args.max_length,
            self_temp=getattr(self.args, "self_temp", 2.0),
        )

        n_fwd = 1 if self.use_self_temp else 2
        print(f"\n[EXP53] Scoring {len(df)} samples...")
        print(f"  Reference: {ref_desc}")
        print(f"  {n_fwd} forward pass(es) per sample")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP53]"):
            rows.append(scorer.score(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[EXP53] Valid: {n_valid}/{len(df)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- AUC Results ---
        print("\n" + "=" * 70)
        print("   EXP53: WEL-MIA RESULTS")
        print("=" * 70)

        score_cols = ["lr_mean", "wel_target", "wel_dual", "wel_inv_ref",
                      "lr_topk20", "neg_mean_loss", "neg_ref_loss", "loss_ratio"]

        aucs = {}
        for col in score_cols:
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best = max(auc_pos, auc_neg)
            direction = "+" if auc_pos >= auc_neg else "-"
            aucs[col] = (best, direction)
            print(f"  {direction}{col:<25} AUC = {best:.4f}")

        best_signal = max(aucs.items(), key=lambda x: x[1][0])
        print(f"\n  BEST: {best_signal[1][1]}{best_signal[0]} = {best_signal[1][0]:.4f}")
        print(f"  vs EXP41 -grad_z_lang: 0.6539 (current best)")
        print(f"  vs EXP01 raw loss:     0.5807")

        # Per-subset for best signal
        best_col = best_signal[0]
        best_dir = best_signal[1][1]
        print(f"\n{'Subset':<10} | {best_col:<20} | N")
        print("-" * 45)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=[best_col])
            if not v.empty and len(v["is_member"].unique()) > 1:
                vals = v[best_col] if best_dir == "+" else -v[best_col]
                auc = roc_auc_score(v["is_member"], vals)
            else:
                auc = float("nan")
            print(f"  {subset:<10} | {auc:.4f}              | {len(sub)}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP53_{timestamp}.parquet", index=False)
        print(f"\n[EXP53] Results saved.")


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
        self_temp = 2.0  # fallback temperature if ref model is gated

    print(f"[EXP53] WEL-MIA")
    print(f"  target: {Args.model_name}")
    print(f"  ref:    {Args.ref_model_name} (self-temp T={Args.self_temp} fallback if gated)")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    Experiment(Args).run()
