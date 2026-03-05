"""
EXPERIMENT 45: CON-RECALL — Contrastive Decoding for MIA

Paper: "CON-RECALL: Detecting Pre-training Data in LLMs via Contrastive Decoding"
       Wang et al. (arXiv:2409.03363v2, Jan 2025)

Core formula:
    Score(x) = [LL(x|P_nm) - gamma * LL(x|P_m)] / LL(x)

    Where:
    - LL(x) = unconditional avg log-likelihood (all negative)
    - LL(x|P_nm) = conditional LL with non-member prefix
    - LL(x|P_m) = conditional LL with member prefix
    - gamma = contrastive strength (search 0.0-1.0)
    - gamma=0 reduces to standard ReCaLL: LL(x|P_nm)/LL(x)

Key insight: Member prefixes barely affect member targets (already memorized)
but hurt non-member targets. Non-member prefixes hurt member targets more.
Contrasting BOTH effects amplifies the membership signal.

Paper: 95-98% AUC on WikiMIA (vs 88-91% ReCaLL, vs ~66% Loss)
Only needs gray-box access (token probabilities), no reference model.

Adaptations for StarCoder2-3b + code:
    - n_shots=5, max 128 tokens/shot → prefix ~640 tokens
    - Target max 256 tokens → total ~900 tokens (fits in 16K context)
    - 3 forward passes per sample (uncond + member-cond + nonmember-cond)
    - Gamma search: 0.0 to 1.0 in steps of 0.1 (cheap, just numpy)
    - Probe shots excluded from eval set for fairness

Compute: 3 forward passes x 10K samples = 30K passes, ~40-50 min A100
Expected AUC: 0.55-0.70 (contrastive prefix signal on code, orthogonal to gradient)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

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
    print("  EXP45: CON-RECALL — Contrastive Decoding MIA")
    print("  Paper: Wang et al. (arXiv:2409.03363v2, Jan 2025)")
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


class ConRecallScorer:
    """CON-RECALL: contrastive prefix-based membership inference."""

    def __init__(self, model, tokenizer, max_target_len=256, max_shot_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len
        self.max_shot_len = max_shot_len
        self._err_count = 0

    def build_prefix(self, texts: List[str], n_shots: int) -> torch.Tensor:
        """Concatenate n_shots code samples into a fixed prefix (with BOS on first)."""
        ids_list = []
        for i, text in enumerate(texts[:n_shots]):
            add_special = (i == 0)
            ids = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_shot_len,
                truncation=True, add_special_tokens=add_special,
            )["input_ids"]
            ids_list.append(ids)
            # newline separator between shots
            if i < n_shots - 1:
                nl = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False)["input_ids"]
                ids_list.append(nl)
        prefix = torch.cat(ids_list, dim=1).to(self.model.device)
        return prefix

    @torch.no_grad()
    def compute_ll(self, input_ids: torch.Tensor) -> float:
        """Avg log-likelihood of all tokens (unconditional)."""
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        if shift_labels.shape[1] == 0:
            return np.nan
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ll = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_ll.mean().item()

    @torch.no_grad()
    def compute_minkpp(self, input_ids: torch.Tensor) -> float:
        """Min-K%++ score (Zhang et al. 2024)."""
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        if shift_labels.shape[1] == 0:
            return np.nan
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).squeeze(0)
        mu = log_probs.mean(dim=-1).squeeze(0)
        sigma = log_probs.std(dim=-1).squeeze(0)
        z = (token_lp - mu) / (sigma + 1e-10)
        k = max(1, int(0.2 * len(z)))
        return z.sort()[0][:k].mean().item()

    @torch.no_grad()
    def compute_conditional_ll(self, prefix_ids: torch.Tensor,
                                target_ids_no_bos: torch.Tensor) -> float:
        """Avg log-likelihood of target tokens conditioned on prefix."""
        combined = torch.cat([prefix_ids, target_ids_no_bos], dim=1)
        max_ctx = getattr(self.model.config, "max_position_embeddings", 16384)
        if combined.shape[1] > max_ctx:
            budget = max_ctx - prefix_ids.shape[1]
            if budget <= 2:
                return np.nan
            combined = torch.cat([prefix_ids, target_ids_no_bos[:, :budget]], dim=1)

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

            log_probs = F.log_softmax(target_logits[:, :min_len, :], dim=-1)
            token_ll = log_probs.gather(2, target_labels[:, :min_len].unsqueeze(-1)).squeeze(-1)
            return token_ll.mean().item()
        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP45 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return np.nan


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
        scorer = ConRecallScorer(
            self.model, self.tokenizer,
            max_target_len=self.args.max_target_len,
            max_shot_len=self.args.max_shot_len,
        )
        n_shots = self.args.n_shots

        # Select prefix shots from dataset (balanced)
        members = df[df["is_member"] == 1].sample(n=n_shots, random_state=self.args.seed)
        nonmembers = df[df["is_member"] == 0].sample(n=n_shots, random_state=self.args.seed)
        shot_idx = set(members.index) | set(nonmembers.index)
        print(f"\n[*] Prefix shots: {n_shots} members + {n_shots} non-members = {len(shot_idx)} removed from eval")

        # Build fixed prefixes
        P_member = scorer.build_prefix(members["content"].tolist(), n_shots)
        P_nonmember = scorer.build_prefix(nonmembers["content"].tolist(), n_shots)
        print(f"  P_member: {P_member.shape[1]} tokens")
        print(f"  P_nonmember: {P_nonmember.shape[1]} tokens")

        # Remove shots from eval set
        eval_df = df[~df.index.isin(shot_idx)].reset_index(drop=True)
        print(f"  Eval set: {len(eval_df)} samples")

        # Pre-tokenize all targets
        print(f"\n[EXP45] Computing 3 log-likelihoods per sample...")
        ll_uncond = np.full(len(eval_df), np.nan)
        ll_member = np.full(len(eval_df), np.nan)
        ll_nonmember = np.full(len(eval_df), np.nan)
        minkpp_scores = np.full(len(eval_df), np.nan)

        for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="[EXP45]")):
            text = row["content"]
            # Skip null or too-short content
            if not isinstance(text, str) or len(text.strip()) < 20:
                continue
            # Tokenize target
            target_with_bos = self.tokenizer(
                text, return_tensors="pt", max_length=self.args.max_target_len, truncation=True,
            )["input_ids"].to(self.model.device)
            target_no_bos = self.tokenizer(
                text, return_tensors="pt", max_length=self.args.max_target_len,
                truncation=True, add_special_tokens=False,
            )["input_ids"].to(self.model.device)

            if target_with_bos.shape[1] < 5:
                continue

            # 1. Unconditional LL + Min-K%++
            ll_uncond[i] = scorer.compute_ll(target_with_bos)
            minkpp_scores[i] = scorer.compute_minkpp(target_with_bos)

            # 2. Conditional LL with member prefix
            ll_member[i] = scorer.compute_conditional_ll(P_member, target_no_bos)

            # 3. Conditional LL with non-member prefix
            ll_nonmember[i] = scorer.compute_conditional_ll(P_nonmember, target_no_bos)

        labels = eval_df["is_member"].values

        # Compute scores for various gamma values
        gammas = np.arange(0.0, 1.05, 0.1)
        eps = 1e-10

        recall_score = ll_nonmember / (ll_uncond + eps)  # gamma=0 = ReCaLL

        print("\n" + "=" * 70)
        print("   EXP45: CON-RECALL — GAMMA SEARCH")
        print("=" * 70)

        best_gamma = 0.0
        best_auc = 0.0
        gamma_aucs = {}

        for gamma in gammas:
            con_score = (ll_nonmember - gamma * ll_member) / (ll_uncond + eps)
            valid = np.isfinite(con_score)
            if valid.sum() > 0 and len(np.unique(labels[valid])) > 1:
                auc = roc_auc_score(labels[valid], con_score[valid])
                gamma_aucs[gamma] = auc
                tag = ""
                if auc > best_auc:
                    best_auc = auc
                    best_gamma = gamma
                    tag = " <-- BEST"
                print(f"  gamma={gamma:.1f}  AUC = {auc:.4f}{tag}")

        # Compute final best CON-RECALL scores
        con_recall_best = (ll_nonmember - best_gamma * ll_member) / (ll_uncond + eps)

        # Store in eval_df
        eval_df = eval_df.copy()
        eval_df["ll_uncond"] = ll_uncond
        eval_df["recall_score"] = recall_score
        eval_df["con_recall_score"] = con_recall_best
        eval_df["minkpp_score"] = minkpp_scores

        # --- Final Report ---
        print("\n" + "=" * 70)
        print("   EXP45: CON-RECALL — FINAL REPORT")
        print("=" * 70)

        signals = {
            "Loss (LL)": ll_uncond,
            "Min-K%++": minkpp_scores,
            f"ReCaLL (gamma=0)": recall_score,
            f"CON-RECALL (gamma={best_gamma:.1f})": con_recall_best,
        }
        for name, scores in signals.items():
            valid = np.isfinite(scores)
            if valid.sum() > 0 and len(np.unique(labels[valid])) > 1:
                auc = roc_auc_score(labels[valid], scores[valid])
                tag = " <-- PRIMARY" if "CON-RECALL" in name else ""
                print(f"  {name:<40} AUC = {auc:.4f}{tag}")

        print(f"\n  Best gamma: {best_gamma:.1f}")
        print(f"  CON-RECALL improvement over ReCaLL: {best_auc - gamma_aucs.get(0.0, 0):.4f}")
        print(f"\n  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:    0.6472")

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'Loss':<8} | {'ReCaLL':<8} | {'CON-ReCaLL':<12} | N")
        print("-" * 55)
        for subset in sorted(eval_df["subset"].unique()):
            sub = eval_df[eval_df["subset"] == subset]
            r = {}
            for sc, col in [("Loss", "ll_uncond"), ("ReCaLL", "recall_score"), ("CON", "con_recall_score")]:
                v = sub.dropna(subset=[col])
                r[sc] = roc_auc_score(v["is_member"], v[col]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('Loss', float('nan')):.4f}   "
                  f"| {r.get('ReCaLL', float('nan')):.4f}   "
                  f"| {r.get('CON', float('nan')):.4f}       "
                  f"| {len(sub)}")

        # Member vs non-member LL stats
        m_mask = labels == 1
        nm_mask = labels == 0
        valid_m = m_mask & np.isfinite(ll_uncond)
        valid_nm = nm_mask & np.isfinite(ll_uncond)
        if valid_m.sum() > 0 and valid_nm.sum() > 0:
            print(f"\n  LL stats:")
            print(f"    Unconditional:   M={ll_uncond[valid_m].mean():.3f}  NM={ll_uncond[valid_nm].mean():.3f}")
            print(f"    With M prefix:   M={ll_member[valid_m].mean():.3f}  NM={ll_member[valid_nm].mean():.3f}")
            print(f"    With NM prefix:  M={ll_nonmember[valid_m].mean():.3f}  NM={ll_nonmember[valid_nm].mean():.3f}")
            print(f"    ReCaLL ratio:    M={recall_score[valid_m].mean():.4f}  NM={recall_score[valid_nm].mean():.4f}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_df.to_parquet(self.output_dir / f"EXP45_{timestamp}.parquet", index=False)
        print(f"\n[EXP45] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        n_shots = 5
        max_target_len = 256
        max_shot_len = 128
        output_dir = "results"
        seed = 42

    print(f"[EXP45] CON-RECALL: {Args.n_shots} shots, "
          f"target={Args.max_target_len}, shot={Args.max_shot_len}, "
          f"sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
