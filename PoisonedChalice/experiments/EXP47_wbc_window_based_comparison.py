"""
EXPERIMENT 47: WBC — Window-Based Comparison Membership Inference Attack

Paper: "Window-based Membership Inference Attacks Against Fine-tuned LLMs"
       Chen, Du, Zhang, Kundu, Fleming, Ribeiro, Li
       Purdue University / Cisco Research (arXiv:2601.02751v1, Jan 2026)

Core insight:
    Global averaging of per-token loss differences dilutes sparse, localized
    memorization signals. Membership evidence is concentrated in SHORT windows
    of consecutive tokens, not spread uniformly across the sequence.

    WBC replaces global averaging with a sliding window approach:
    1. Compute per-token loss from BOTH target and reference model
    2. Slide windows of size w across the loss-difference sequence
    3. Each window casts a BINARY VOTE: is reference_loss > target_loss?
       (sign-based comparison — robust to long-tailed noise)
    4. Aggregate votes: Tsign(w) = fraction of windows voting "member"
    5. Ensemble over geometrically-spaced window sizes for robustness

    Theoretical grounding: sign test has infinite asymptotic relative
    efficiency over mean test under Cauchy-contaminated distributions.
    The breakdown point of 0.5 makes it immune to up to 50% outlier windows.

Algorithm (per sample):
    For each window size w in W = {2, 3, 4, 6, 9, 13, 18, 25, 32, 40}:
        count = 0
        For i = 1 to n-w+1:
            if sum(ℓ_R[i:i+w]) > sum(ℓ_T[i:i+w]):
                count += 1
        Tsign(w) = count / (n - w + 1)
    SWBC = mean(Tsign(w) for w in W)

    Accelerated via 1D convolution: cumsum + diff = O(n) per window size.

Paper results (fine-tuned models):
    - Khan Academy/Pythia-2.8B: AUC 0.837, TPR@1%FPR 14.6%
    - Stanford/Pythia-2.8B: AUC 0.854
    - Web Samples v2/Pythia-2.8B: AUC 0.843
    - Average across 11 datasets: AUC 0.839 vs best baseline 0.754

IMPORTANT: Paper targets FINE-TUNED models (target = fine-tuned, reference =
pre-trained base). Our setup is PRE-TRAINING detection on StarCoder2-3b (no
separate fine-tuned checkpoint). Adaptation:
    - Target: bigcode/starcoder2-3b (the model under test)
    - Reference: bigcode/starcoderbase-1b (smaller model as surrogate baseline)
    - Also test temperature-scaled self-reference (T=2.0)
    The signal is weaker than fine-tuning MIA because both models are
    independently pre-trained, but WBC's robustness to noise may help
    extract whatever differential signal exists.

Compute: 2 forward passes per sample (target + reference), forward-only
Expected runtime: ~15-25 min on A100 (10% sample)
Expected AUC: 0.55-0.65 (WBC may boost weak reference-based signal;
    paper's mismatched-reference result: AUC 0.692-0.774 with size mismatch)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    print("  EXP47: WBC — Window-Based Comparison MIA")
    print("  Paper: Chen et al. (arXiv:2601.02751v1, Jan 2026)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, label: str = "model"):
    print(f"[*] Loading {label}: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model, tokenizer


def geometric_window_sizes(w_min: int = 2, w_max: int = 40, num_sizes: int = 10) -> List[int]:
    """Generate geometrically-spaced window sizes (Equation 12 from paper)."""
    if num_sizes == 1:
        return [w_min]
    sizes = set()
    for k in range(num_sizes):
        w = round(w_min * (w_max / w_min) ** (k / (num_sizes - 1)))
        sizes.add(max(w_min, min(w, w_max)))
    return sorted(sizes)


class WBCScorer:
    """Window-Based Comparison attack scorer.

    Computes per-token losses from target and reference models, then applies
    the WBC sliding window + sign-based aggregation algorithm.
    """

    def __init__(self, target_model, ref_model, tokenizer,
                 max_length: int = 512,
                 w_min: int = 2, w_max: int = 40, num_window_sizes: int = 10,
                 use_self_ref_temp: Optional[float] = None):
        self.target = target_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_self_ref_temp = use_self_ref_temp
        self._err_count = 0
        self._has_ref = ref_model is not None

        self.window_sizes = geometric_window_sizes(w_min, w_max, num_window_sizes)
        print(f"  Window sizes ({len(self.window_sizes)}): {self.window_sizes}")

    @torch.no_grad()
    def _get_per_token_loss(self, model, input_ids, temperature: float = 1.0):
        """Get per-token negative log-likelihood (loss) for each position.

        Returns: numpy array of shape (T,) where T = seq_len - 1
        """
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.float()

        if temperature != 1.0:
            logits = logits / temperature

        shift_logits = logits[:, :-1, :]  # (1, T, V)
        shift_labels = input_ids[:, 1:]   # (1, T)

        T = shift_labels.shape[1]
        if T == 0:
            return None

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        token_loss = -token_ll  # per-token negative log-likelihood
        return token_loss.cpu().numpy()

    def _wbc_score(self, loss_target: np.ndarray, loss_ref: np.ndarray) -> Dict[str, float]:
        """Apply WBC algorithm: sliding window + sign-based aggregation.

        For each window size w:
            Tsign(w) = fraction of windows where sum(ℓ_R) > sum(ℓ_T)
            (higher = more windows where reference is "more surprised" = member signal)

        Final: SWBC = mean over all window sizes.
        """
        n = len(loss_target)
        result = {}

        cumsum_target = np.cumsum(np.concatenate(([0.0], loss_target)))
        cumsum_ref = np.cumsum(np.concatenate(([0.0], loss_ref)))

        window_scores = []

        for w in self.window_sizes:
            if w >= n:
                continue

            n_windows = n - w + 1
            # Window sums via cumulative sum difference (O(n) per window size)
            win_sum_target = cumsum_target[w:] - cumsum_target[:n_windows]
            win_sum_ref = cumsum_ref[w:] - cumsum_ref[:n_windows]

            # Sign-based comparison: fraction where reference loss > target loss
            t_sign = float(np.mean(win_sum_ref > win_sum_target))
            window_scores.append(t_sign)

            result[f"wbc_w{w}"] = t_sign

        if window_scores:
            result["wbc_ensemble"] = float(np.mean(window_scores))
        else:
            result["wbc_ensemble"] = np.nan

        return result

    def _baseline_scores(self, loss_target: np.ndarray, loss_ref: np.ndarray) -> Dict[str, float]:
        """Compute baseline reference-based scores for comparison."""
        avg_target = loss_target.mean()
        avg_ref = loss_ref.mean()
        diff = loss_ref - loss_target

        return {
            "neg_mean_loss": -float(avg_target),
            "loss_ratio": float(avg_ref / (avg_target + 1e-10)),
            "loss_diff": float(diff.mean()),
            "neg_mean_loss_ref": -float(avg_ref),
            "mean_sign_fraction": float(np.mean(diff > 0)),
        }

    def extract(self, text: str) -> Dict[str, float]:
        """Extract WBC + baseline features for a single sample."""
        all_keys = (
            ["wbc_ensemble", "neg_mean_loss", "loss_ratio", "loss_diff",
             "neg_mean_loss_ref", "mean_sign_fraction"]
            + [f"wbc_w{w}" for w in self.window_sizes]
        )
        result = {k: np.nan for k in all_keys}

        if not text or len(text) < 30:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.target.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 10:
                return result

            # Target model: per-token loss
            loss_target = self._get_per_token_loss(self.target, input_ids)
            if loss_target is None or len(loss_target) < 5:
                return result

            # Reference model: per-token loss
            if self._has_ref:
                ref_ids = input_ids.to(self.ref.device) if hasattr(self.ref, 'device') else input_ids
                loss_ref = self._get_per_token_loss(self.ref, ref_ids)
            elif self.use_self_ref_temp is not None:
                loss_ref = self._get_per_token_loss(
                    self.target, input_ids, temperature=self.use_self_ref_temp
                )
            else:
                return result

            if loss_ref is None:
                return result

            # Align lengths (different tokenizers may produce different lengths)
            min_len = min(len(loss_target), len(loss_ref))
            if min_len < 5:
                return result
            loss_target = loss_target[:min_len]
            loss_ref = loss_ref[:min_len]

            # WBC scores
            wbc_result = self._wbc_score(loss_target, loss_ref)
            result.update(wbc_result)

            # Baseline scores
            baseline_result = self._baseline_scores(loss_target, loss_ref)
            result.update(baseline_result)

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP47 WARN] {type(e).__name__}: {e}")
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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()

        self_ref_temp = self.args.self_ref_temp if self.ref_model is None else None
        scorer = WBCScorer(
            self.target_model, self.ref_model, self.tokenizer,
            max_length=self.args.max_length,
            w_min=self.args.w_min, w_max=self.args.w_max,
            num_window_sizes=self.args.num_window_sizes,
            use_self_ref_temp=self_ref_temp,
        )

        ref_label = self.args.ref_model_name if self.ref_model else f"self-temp(T={self.args.self_ref_temp})"
        print(f"\n[EXP47] Reference: {ref_label}")
        print(f"[EXP47] Extracting WBC features for {len(df)} samples...")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP47]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df["wbc_ensemble"].notna().sum()
        print(f"\n[EXP47] Valid: {n_valid}/{len(df)}")
        if scorer._err_count > 0:
            print(f"[EXP47] Errors: {scorer._err_count}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   EXP47: WBC — Window-Based Comparison REPORT")
        print("=" * 70)

        # Per-window-size AUCs
        print("\n--- Per-Window-Size AUCs ---")
        print(f"{'Window':<8} {'AUC':<8} {'N_windows (512-tok seq)'}")
        print("-" * 40)
        window_aucs = {}
        for w in scorer.window_sizes:
            col = f"wbc_w{w}"
            if col in df.columns:
                v = df.dropna(subset=[col])
                if len(v["is_member"].unique()) > 1:
                    auc = roc_auc_score(v["is_member"], v[col])
                    window_aucs[w] = auc
                    n_win = max(0, 512 - w)
                    print(f"  w={w:<4}  {auc:.4f}   ~{n_win} windows")

        if window_aucs:
            best_w = max(window_aucs, key=window_aucs.get)
            print(f"\n  Best single window: w={best_w} -> AUC {window_aucs[best_w]:.4f}")

        # Ensemble + baseline AUCs
        print("\n--- Ensemble & Baseline AUCs ---")
        score_cols = [
            ("wbc_ensemble", "WBC Ensemble (geometric) [PRIMARY]"),
            ("mean_sign_fraction", "Global sign fraction (w=1 equivalent)"),
            ("loss_diff", "Mean loss difference (ℓ_R - ℓ_T)"),
            ("loss_ratio", "Loss ratio (ℓ_R / ℓ_T)"),
            ("neg_mean_loss", "Negative mean loss (target only)"),
        ]

        aucs = {}
        for col, label in score_cols:
            if col not in df.columns:
                continue
            v = df.dropna(subset=[col])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[col])
                aucs[col] = auc
                tag = " <-- PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<50} AUC = {auc:.4f}{tag}")

        if aucs:
            best = max(aucs, key=aucs.get)
            print(f"\n  Best signal: {best} = {aucs[best]:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:    0.6472")
        print(f"  vs EXP46 EZ-MIA:         pending")

        # WBC advantage analysis
        if "wbc_ensemble" in aucs and "loss_diff" in aucs:
            gain = aucs["wbc_ensemble"] - aucs["loss_diff"]
            print(f"\n  WBC ensemble vs global diff: {gain:+.4f}")
            if gain > 0.005:
                print(f"  -> WBC IMPROVES over global averaging (+{gain:.4f})")
            elif gain < -0.005:
                print(f"  -> WBC HURTS vs global averaging ({gain:.4f})")
            else:
                print(f"  -> WBC ~= global averaging (delta within noise)")

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'WBC_ens':<10} | {'Loss_diff':<10} | {'Ratio':<10} | {'Loss':<8} | N")
        print("-" * 70)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["wbc_ensemble", "loss_diff", "loss_ratio", "neg_mean_loss"]:
                v = sub.dropna(subset=[sc])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    r[sc] = roc_auc_score(v["is_member"], v[sc])
                else:
                    r[sc] = float("nan")
            print(
                f"{subset:<10} | {r.get('wbc_ensemble', float('nan')):.4f}     "
                f"| {r.get('loss_diff', float('nan')):.4f}     "
                f"| {r.get('loss_ratio', float('nan')):.4f}     "
                f"| {r.get('neg_mean_loss', float('nan')):.4f}   "
                f"| {len(sub)}"
            )

        # Distribution statistics
        print("\n--- Distribution Statistics ---")
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for col, label in [
            ("wbc_ensemble", "WBC ensemble"),
            ("loss_diff", "Loss diff (ℓ_R-ℓ_T)"),
            ("mean_sign_fraction", "Sign fraction"),
        ]:
            m_val = m[col].dropna()
            nm_val = nm[col].dropna()
            if len(m_val) > 0 and len(nm_val) > 0:
                print(
                    f"  {label:<25} M={m_val.mean():.4f}+-{m_val.std():.4f}  "
                    f"NM={nm_val.mean():.4f}+-{nm_val.std():.4f}  "
                    f"delta={m_val.mean() - nm_val.mean():.4f}"
                )

        # Best window per subset
        print("\n--- Best Window Size Per Subset ---")
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            best_auc_sub = 0
            best_w_sub = -1
            for w in scorer.window_sizes:
                col = f"wbc_w{w}"
                if col in sub.columns:
                    v = sub.dropna(subset=[col])
                    if not v.empty and len(v["is_member"].unique()) > 1:
                        a = roc_auc_score(v["is_member"], v[col])
                        if a > best_auc_sub:
                            best_auc_sub = a
                            best_w_sub = w
            if best_w_sub > 0:
                print(f"  {subset:<10} best w={best_w_sub}, AUC={best_auc_sub:.4f}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP47_{timestamp}.parquet", index=False)
        print(f"\n[EXP47] Results saved.")


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
        # WBC-specific params
        w_min = 2
        w_max = 40
        num_window_sizes = 10
        self_ref_temp = 2.0  # fallback if reference model unavailable

    print(f"[EXP47] WBC Attack: target={Args.model_name}, ref={Args.ref_model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  windows: wmin={Args.w_min}, wmax={Args.w_max}, |W|={Args.num_window_sizes}")
    Experiment(Args).run()
