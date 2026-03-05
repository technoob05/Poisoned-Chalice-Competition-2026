"""
EXPERIMENT 44: EM-MIA — Expectation-Maximization Membership Inference Attack

Paper: "Detecting Training Data of Large Language Models via Expectation Maximization"
       Kim et al. (arXiv:2410.07582v3, Jan 2026)

Core idea:
    ReCaLL score = LL(x|p) / LL(x) where p = prefix, x = target.
    Non-member prefixes suppress member signals more than non-member signals.
    EM-MIA iteratively refines which prefixes are effective and which samples
    are members, WITHOUT needing labeled non-members.

    Algorithm:
    1. Init f(x) with Loss or Min-K%++ (any off-the-shelf MIA)
    2. E-step: r(p) = AUC-ROC of ReCaLL_p(x) against pseudo-labels from f
    3. M-step: f(x) = -r(x) (bad prefix → likely member)
    4. Repeat until convergence (~5-10 iterations)

    Paper results: 97-99% AUC on WikiMIA, outperforms ReCaLL without labels.
    On hard settings (OLMoMIA Hard/Random): ~50% (near random — same as all methods).

Adaptations for StarCoder2-3b + code domain:
    - max_length=256 per sample (pair=512) to keep O(D²) passes feasible
    - D_em=200 balanced (100M+100NM) for pairwise computation
    - Phase 2: top-10 EM-selected prefixes to score larger eval set
    - Forward-only (no backward), gray-box setting

Compute: O(D²) forward passes for pairwise matrix + O(D_eval × k) for Phase 2
Expected runtime: ~50-70 min on A100
Expected AUC: 0.55-0.65 (prefix-based, orthogonal to gradient signals)
"""
import os
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List

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
    print("  EXP44: EM-MIA — Expectation-Maximization MIA")
    print("  Paper: Kim et al. (arXiv:2410.07582v3, Jan 2026)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"[*] Model loaded. dtype={dtype}")
    return model, tokenizer


class EMMIAScorer:
    """EM-MIA: joint estimation of prefix effectiveness and membership scores."""

    def __init__(self, model, tokenizer, max_length=256, max_pair_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_pair_length = max_pair_length
        self._err_count = 0

    @torch.no_grad()
    def compute_sample_stats(self, input_ids: torch.Tensor) -> Tuple[float, float]:
        """Compute avg log-likelihood and Min-K%++ for a single sample."""
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits.float()

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        if shift_labels.shape[1] == 0:
            return np.nan, np.nan

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        avg_ll = token_ll.mean().item()

        # Min-K%++ (Zhang et al. 2024): z-normalize by per-position vocab stats
        mu = log_probs.mean(dim=-1).squeeze(0)
        sigma = log_probs.std(dim=-1).squeeze(0)
        z = (token_ll - mu) / (sigma + 1e-10)
        k = max(1, int(0.2 * len(z)))
        minkpp = z.sort()[0][:k].mean().item()

        return avg_ll, minkpp

    @torch.no_grad()
    def compute_conditional_ll(self, prefix_ids: torch.Tensor, target_ids_no_bos: torch.Tensor) -> float:
        """Compute avg log-likelihood of target tokens conditioned on prefix."""
        combined = torch.cat([prefix_ids, target_ids_no_bos], dim=1)
        if combined.shape[1] > self.max_pair_length:
            target_budget = self.max_pair_length - prefix_ids.shape[1]
            if target_budget <= 2:
                return np.nan
            combined = torch.cat([prefix_ids, target_ids_no_bos[:, :target_budget]], dim=1)

        try:
            outputs = self.model(input_ids=combined)
            logits = outputs.logits.float()

            prefix_len = prefix_ids.shape[1]
            target_len = combined.shape[1] - prefix_len
            if target_len <= 0:
                return np.nan

            target_logits = logits[:, prefix_len - 1:prefix_len + target_len - 1, :]
            target_labels = combined[:, prefix_len:prefix_len + target_len]

            min_len = min(target_logits.shape[1], target_labels.shape[1])
            if min_len == 0:
                return np.nan
            target_logits = target_logits[:, :min_len, :]
            target_labels = target_labels[:, :min_len]

            log_probs = F.log_softmax(target_logits, dim=-1)
            token_ll = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
            return token_ll.mean().item()

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP44 WARN] cond_ll: {type(e).__name__}: {e}")
            self._err_count += 1
            return np.nan

    def tokenize_sample(self, text: str):
        """Returns (ids_with_bos, ids_no_bos) on model device."""
        ids_with = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True
        )["input_ids"].to(self.model.device)
        ids_no = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length,
            truncation=True, add_special_tokens=False
        )["input_ids"].to(self.model.device)
        return ids_with, ids_no

    def compute_pairwise(self, texts: List[str], labels: np.ndarray):
        """Compute pairwise ReCaLL matrix for D samples. O(D²) forward passes."""
        D = len(texts)
        print(f"\n[Phase 1] Pairwise ReCaLL for D={D} samples ({D*D:,} forward passes)")

        ids_with = []
        ids_no = []
        for text in tqdm(texts, desc="  Tokenizing"):
            w, n = self.tokenize_sample(text)
            ids_with.append(w)
            ids_no.append(n)

        ll_uncond = np.zeros(D)
        minkpp = np.zeros(D)
        for i in tqdm(range(D), desc="  Unconditional LL"):
            ll_uncond[i], minkpp[i] = self.compute_sample_stats(ids_with[i])

        auc_loss = roc_auc_score(labels, ll_uncond) if len(np.unique(labels)) > 1 else 0.5
        auc_mkpp = roc_auc_score(labels, minkpp) if len(np.unique(labels)) > 1 else 0.5
        print(f"  Baselines — Loss AUC: {auc_loss:.4f}, MinK++ AUC: {auc_mkpp:.4f}")

        ll_cond = np.full((D, D), np.nan)
        import time
        t0 = time.time()
        for i in tqdm(range(D), desc="  Pairwise conditional LL"):
            for j in range(D):
                if i != j:
                    ll_cond[i, j] = self.compute_conditional_ll(ids_with[i], ids_no[j])
            if i == 5:
                elapsed = time.time() - t0
                est_total = elapsed / 6 * D
                print(f"  Estimated total time: {timedelta(seconds=int(est_total))}")

        eps = 1e-10
        recall = np.full((D, D), np.nan)
        for j in range(D):
            if abs(ll_uncond[j]) > eps:
                recall[:, j] = ll_cond[:, j] / ll_uncond[j]

        return ll_uncond, minkpp, recall, ids_with, ids_no

    def run_em(self, recall: np.ndarray, init_scores: np.ndarray,
               labels: np.ndarray = None, n_iter: int = 10):
        """Run EM iterations. Returns (membership_scores, prefix_scores)."""
        D = recall.shape[0]
        f = init_scores.copy()

        print(f"\n[EM] Running {n_iter} iterations on D={D} samples...")
        for it in range(n_iter):
            # E-step: compute prefix scores r(p) using pseudo-labels
            tau = np.median(f)
            pseudo = (f > tau).astype(int)

            r = np.full(D, 0.5)
            for i in range(D):
                scores = recall[i, :]
                mask = np.ones(D, dtype=bool)
                mask[i] = False
                valid = mask & np.isfinite(scores)
                if valid.sum() > 20 and len(np.unique(pseudo[valid])) > 1:
                    r[i] = roc_auc_score(pseudo[valid], scores[valid])

            # M-step: membership score = negative prefix score
            f_new = -r

            converged = np.corrcoef(f, f_new)[0, 1]
            f = f_new

            if labels is not None and len(np.unique(labels)) > 1:
                auc = roc_auc_score(labels, f)
                print(f"  Iter {it+1:2d}: AUC = {auc:.4f}  (corr with prev: {converged:.4f})")
            else:
                print(f"  Iter {it+1:2d}: corr with prev: {converged:.4f}")

            if converged > 0.999:
                print(f"  Converged at iteration {it+1}")
                break

        return f, r


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

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
        scorer = EMMIAScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            max_pair_length=self.args.max_pair_length,
        )

        # Select balanced EM subset
        D_em = self.args.d_em
        n_each = D_em // 2
        members = df[df["is_member"] == 1].sample(n=min(n_each, (df["is_member"] == 1).sum()),
                                                    random_state=self.args.seed)
        nonmembers = df[df["is_member"] == 0].sample(n=min(n_each, (df["is_member"] == 0).sum()),
                                                      random_state=self.args.seed)
        em_df = pd.concat([members, nonmembers]).sample(frac=1, random_state=self.args.seed).reset_index(drop=True)
        em_texts = em_df["content"].tolist()
        em_labels = em_df["is_member"].values
        print(f"\n[*] EM subset: {len(em_df)} samples ({(em_labels==1).sum()}M + {(em_labels==0).sum()}NM)")

        # Phase 1: Pairwise ReCaLL + EM
        ll_uncond, minkpp, recall, ids_with, ids_no = scorer.compute_pairwise(em_texts, em_labels)

        # Also compute simple baselines on EM subset
        # Avg: average ReCaLL across all prefixes for each target
        recall_avg = np.nanmean(recall, axis=0)
        # AvgP: average ReCaLL for each prefix across all targets → f = -AvgP
        recall_avgp = -np.nanmean(recall, axis=1)

        # Run EM with loss initialization
        f_loss, r_loss = scorer.run_em(recall, ll_uncond, em_labels, n_iter=self.args.n_iter)

        # Run EM with Min-K%++ initialization
        f_mkpp, r_mkpp = scorer.run_em(recall, minkpp, em_labels, n_iter=self.args.n_iter)

        # Phase 2: Score full eval set using top-k EM prefixes
        top_k = self.args.top_k_prefixes
        best_prefix_idx = np.argsort(-r_loss)[:top_k]
        print(f"\n[Phase 2] Scoring {len(df)} samples with top-{top_k} EM prefixes...")
        print(f"  Best prefix indices: {best_prefix_idx.tolist()}")
        print(f"  Prefix scores: {[f'{r_loss[i]:.4f}' for i in best_prefix_idx]}")
        print(f"  Prefix labels: {[int(em_labels[i]) for i in best_prefix_idx]}")

        # Pre-tokenized top-k prefixes (from EM subset)
        topk_prefix_ids = [ids_with[i] for i in best_prefix_idx]

        # Score all samples
        eval_ll = np.full(len(df), np.nan)
        eval_recall_topk = np.full(len(df), np.nan)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[Phase 2] Scoring"):
            text = row["content"]
            w, n = scorer.tokenize_sample(text)

            # Unconditional LL
            ll, _ = scorer.compute_sample_stats(w)
            eval_ll[idx] = ll

            if abs(ll) < 1e-10:
                continue

            # Conditional LL with each top-k prefix → ReCaLL → average
            recalls = []
            for prefix_ids in topk_prefix_ids:
                cll = scorer.compute_conditional_ll(prefix_ids, n)
                if np.isfinite(cll):
                    recalls.append(cll / ll)
            if recalls:
                eval_recall_topk[idx] = np.mean(recalls)

        df["em_loss_baseline"] = eval_ll
        df["em_recall_topk"] = eval_recall_topk

        # For EM subset, also store direct EM scores
        em_scores = pd.DataFrame(index=em_df.index)
        em_scores["em_mia_loss_init"] = f_loss
        em_scores["em_mia_mkpp_init"] = f_mkpp
        em_scores["recall_avg"] = recall_avg
        em_scores["recall_avgp"] = recall_avgp

        # --- Report ---
        print("\n" + "=" * 70)
        print("   EXP44: EM-MIA — REPORT")
        print("=" * 70)

        # EM subset results
        print(f"\n  === EM Subset ({len(em_df)} samples) ===")
        em_signals = {
            "Loss (LL)": ll_uncond,
            "Min-K%++": minkpp,
            "ReCaLL-Avg": recall_avg,
            "-AvgP (prefix score)": recall_avgp,
            "EM-MIA (Loss init)": f_loss,
            "EM-MIA (MinK++ init)": f_mkpp,
        }
        for name, scores in em_signals.items():
            valid = np.isfinite(scores)
            if valid.sum() > 0 and len(np.unique(em_labels[valid])) > 1:
                auc = roc_auc_score(em_labels[valid], scores[valid])
                tag = " <-- PRIMARY" if "EM-MIA" in name and auc == max(
                    roc_auc_score(em_labels[np.isfinite(s)], s[np.isfinite(s)])
                    for s in [f_loss, f_mkpp] if np.isfinite(s).sum() > 0
                ) else ""
                print(f"    {name:<30} AUC = {auc:.4f}{tag}")

        # Full eval results
        print(f"\n  === Full Eval ({len(df)} samples) ===")
        for name, col in [
            ("Loss (LL)", "em_loss_baseline"),
            ("ReCaLL top-k", "em_recall_topk"),
        ]:
            v = df.dropna(subset=[col])
            if len(v["is_member"].unique()) > 1:
                auc = roc_auc_score(v["is_member"], v[col])
                print(f"    {name:<30} AUC = {auc:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang: 0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:   0.6472")
        print(f"  vs EXP39 Ridge stacker:  0.6490")

        # Subset breakdown
        print(f"\n{'Subset':<10} | {'Loss':<8} | {'ReCaLL-topk':<13} | N")
        print("-" * 50)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["em_loss_baseline", "em_recall_topk"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('em_loss_baseline', float('nan')):.4f}   "
                  f"| {r.get('em_recall_topk', float('nan')):.4f}        "
                  f"| {len(sub)}")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP44_{timestamp}.parquet", index=False)
        print(f"\n[EXP44] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.02
        d_em = 200
        n_iter = 10
        top_k_prefixes = 10
        max_length = 256
        max_pair_length = 512
        output_dir = "results"
        seed = 42

    print(f"[EXP44] EM-MIA: {Args.sample_fraction*100:.0f}% sample, D_em={Args.d_em}, "
          f"max_len={Args.max_length}, top_k={Args.top_k_prefixes}")
    Experiment(Args).run()
