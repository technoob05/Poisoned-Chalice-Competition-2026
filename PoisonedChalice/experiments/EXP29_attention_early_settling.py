"""
EXPERIMENT 29: Attention Early-Settling (AttenMIA × Early-Settling Ratio)

Papers fused:
    1. "AttenMIA: LLM Membership Inference Attack through Attention Signals" (2026)
       — Attention-based information flow reveals memorization patterns more
         robustly than output-logit methods.
    2. "Think Deep, Not Just Long" (DTR paper)
       — Memorized inputs cause early convergence of processing (Shallow-Thinking).

Novelty fusion:
    Instead of measuring JSD on *logit distributions* (EXP26), measure the
    early-settling behaviour of *attention maps* across transformer layers.

Two complementary signals are extracted at 5 sampled layer checkpoints:

    A. Attention Entropy (AttenESR):
       H_l = mean over (heads, query-tokens) of -sum(p * log(p))
       on the attention distribution at layer l.
       LOW entropy at early layers → focused, frozen attention → memorized (member).

    B. Attention Convergence (AttenConv):
       cosine_similarity(flatten(A_l), flatten(A_L))
       between each layer's attention matrix and the final layer's.
       HIGH convergence at early layers → settled early → member.

Primary score:
    rank_avg(-attn_entropy_early, attn_conv_early)

Compute notes:
    - Model loaded with attn_implementation="eager" to bypass flash attention
      and ensure attention weight tensors are available.
    - Sequence length capped at 512 tokens for the attention forward pass
      (full attention is O(n²) in memory; 512 gives 25 MB/layer vs 1 GB at 2048).
    - 5 sampled layer checkpoints, same adaptive formula as EXP26/27/28.
    - Pure forward pass (no backward) → fast, ~EXP26 speed.

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
    print("  EXP29: ATTENTION EARLY-SETTLING (AttenMIA × ESR)")
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
# Model Loading  (eager attention required for weight access)
# ============================================================================

def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    print("[*] Using attn_implementation='eager' to enable attention weight extraction.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # attn_implementation="eager" disables flash-attention → attention weights available
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
        )
    except TypeError:
        # Older transformers versions don't support attn_implementation
        print("[*] Note: attn_implementation not supported, loading without it.")
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
# Attention Early-Settling Attack
# ============================================================================

class AttentionEarlySettlingAttack:
    """
    Extracts attention entropy and convergence signals at sampled transformer
    layers to detect early-settling behaviour caused by memorization.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        # Attention matrices are O(seq²): cap to control memory
        self.max_seq_for_attn = getattr(args, "max_seq_for_attn", 512)

        if not (hasattr(model, "model") and hasattr(model.model, "layers")):
            raise RuntimeError("Cannot find model.model.layers — check architecture.")

        self.transformer_layers = model.model.layers
        self.sampled_indices = self._choose_layer_indices()
        print(f"[EXP29] Sampled layers: {self.sampled_indices}")
        print(f"[EXP29] Early layers (score): {self.sampled_indices[:2]}")

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
        return "attention_early_settling"

    def _attn_entropy(self, attn: torch.Tensor, eps: float = 1e-9) -> float:
        """
        Compute mean attention entropy over all heads and query positions.
        attn : (1, heads, seq, seq) — standard causal attention matrix.
        Returns scalar entropy value.
        """
        # attn shape: (batch=1, heads, seq_q, seq_k)
        a = attn[0].float().clamp(min=eps)  # (heads, seq_q, seq_k)
        # Entropy per (head, query) = -sum_k p_k * log(p_k)
        H = -(a * a.log()).sum(dim=-1)  # (heads, seq_q)
        return H.mean().item()

    def _attn_convergence(
        self, attn_l: torch.Tensor, attn_final: torch.Tensor
    ) -> float:
        """
        Cosine similarity between flattened attention matrices of layer l and final.
        High similarity = attention pattern settled early.
        """
        a_l = attn_l[0].float().reshape(-1)
        a_f = attn_final[0].float().reshape(-1)
        # Clamp to avoid divide-by-zero
        norm_l = a_l.norm(2).clamp(min=1e-9)
        norm_f = a_f.norm(2).clamp(min=1e-9)
        cosine = (a_l * a_f).sum() / (norm_l * norm_f)
        return cosine.item()

    def compute_attention_signals(self, text: str) -> Optional[Dict]:
        """
        Single forward pass with output_attentions=True.
        Captures attention matrices only at sampled layers.
        Returns dict with {entropy, convergence} per sampled layer.
        """
        if not text or len(text) < 20:
            return None

        try:
            # Tokenize — cap at max_seq_for_attn to control memory
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_seq_for_attn,
                truncation=True,
            ).to(self.model.device)

            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 4:
                return None

            # ---- Hook approach: capture only sampled layers ----
            captured_attn: Dict[int, torch.Tensor] = {}

            def make_attn_hook(layer_idx: int):
                def hook_fn(module, inp, out):
                    # out = (hidden_state, attn_weights, ...)  when output_attentions
                    # Some architectures put attn_weights at index 1
                    if isinstance(out, tuple) and len(out) > 1:
                        aw = out[1]
                        if aw is not None and isinstance(aw, torch.Tensor):
                            # aw: (batch, heads, seq, seq)
                            captured_attn[layer_idx] = aw.detach().cpu()
                return hook_fn

            handles = []
            for idx in self.sampled_indices:
                if hasattr(self.transformer_layers[idx], "self_attn"):
                    h = self.transformer_layers[idx].self_attn.register_forward_hook(
                        make_attn_hook(idx)
                    )
                else:
                    h = self.transformer_layers[idx].register_forward_hook(
                        make_attn_hook(idx)
                    )
                handles.append(h)

            with torch.no_grad():
                self.model(**inputs, output_attentions=True)

            for h in handles:
                h.remove()

            if not captured_attn:
                return None

            # If hooks didn't capture attn weights (some archs output differently),
            # fall back to full output_attentions pass and index manually
            if len(captured_attn) < len(self.sampled_indices):
                captured_attn.clear()
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                if outputs.attentions is not None:
                    for idx in self.sampled_indices:
                        if idx < len(outputs.attentions):
                            aw = outputs.attentions[idx]
                            if aw is not None:
                                captured_attn[idx] = aw.detach().cpu()

            if not captured_attn:
                return None

            # ---- Compute signals ----
            final_idx = self.sampled_indices[-1]
            final_attn = captured_attn.get(final_idx)

            results: Dict = {"attn_entropy_final": 0.0, "attn_convergence_final": 1.0}
            entropy_early_vals, conv_early_vals = [], []

            for layer_idx in self.sampled_indices:
                if layer_idx not in captured_attn:
                    continue

                attn_l = captured_attn[layer_idx]
                H = self._attn_entropy(attn_l)
                results[f"attn_entropy_layer_{layer_idx}"] = H

                if final_attn is not None and layer_idx != final_idx:
                    # Pad/crop if shapes differ (GQA might differ in num heads)
                    if attn_l.shape == final_attn.shape:
                        conv = self._attn_convergence(attn_l, final_attn)
                    else:
                        conv = np.nan
                    results[f"attn_conv_layer_{layer_idx}"] = conv

                    if layer_idx in self.sampled_indices[:2]:
                        entropy_early_vals.append(H)
                        if not np.isnan(conv):
                            conv_early_vals.append(conv)

            results["mean_attn_entropy_early"] = (
                float(np.mean(entropy_early_vals)) if entropy_early_vals else np.nan
            )
            results["mean_attn_conv_early"] = (
                float(np.mean(conv_early_vals)) if conv_early_vals else np.nan
            )

            captured_attn.clear()
            return results

        except Exception:
            captured_attn.clear() if "captured_attn" in dir() else None
            return None

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP29] Processing {len(texts)} samples…")
        rows = []

        for text in tqdm(texts, desc="[EXP29] Attention ESR"):
            signals = self.compute_attention_signals(text)
            if signals is None:
                row = {
                    "mean_attn_entropy_early": np.nan,
                    "mean_attn_conv_early": np.nan,
                }
                for idx in self.sampled_indices:
                    row[f"attn_entropy_layer_{idx}"] = np.nan
                    row[f"attn_conv_layer_{idx}"] = np.nan
            else:
                row = signals
            rows.append(row)

        df = pd.DataFrame(rows)

        # ---- Member signals (higher = more likely member) ----
        if "mean_attn_entropy_early" in df.columns:
            df["signal_entropy"] = -df["mean_attn_entropy_early"]  # low entropy = member
        if "mean_attn_conv_early" in df.columns:
            df["signal_conv"] = df["mean_attn_conv_early"]          # high conv = member

        # ---- Combined rank score ----
        rank_sources = ["signal_entropy", "signal_conv"]
        valid_rank_cols = [c for c in rank_sources if c in df.columns]
        if valid_rank_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_sum += ranks / len(ranks)
            df["combined_rank_score"] = rank_sum / len(valid_rank_cols)
        else:
            df["combined_rank_score"] = np.nan

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
        attacker = AttentionEarlySettlingAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())
        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP29_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "="*65)
        print("   EXP29: ATTENTION EARLY-SETTLING — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(Entropy + Conv) [PRIMARY]",
            "signal_entropy": "-mean_attn_entropy_early",
            "signal_conv": "mean_attn_conv_early",
        }
        report = {
            "experiment": "EXP29_attention_early_settling",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "sampled_layers": attacker.sampled_indices,
            "early_layers": attacker.sampled_indices[:2],
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
                print(f"  {label:<45} AUC = {auc:.4f}")

        print(f"\n{'Subset':<10} | {'Combined':<10} | {'Entropy':<10} | {'Conv':<10} | N")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            row_aucs = {}
            for sc in ["combined_rank_score", "signal_entropy", "signal_conv"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                if not v.empty and len(v["is_member"].unique()) > 1:
                    row_aucs[sc] = roc_auc_score(v["is_member"], v[sc])
                else:
                    row_aucs[sc] = float("nan")
            print(
                f"{subset:<10} | {row_aucs.get('combined_rank_score', float('nan')):.4f}     "
                f"| {row_aucs.get('signal_entropy', float('nan')):.4f}     "
                f"| {row_aucs.get('signal_conv', float('nan')):.4f}     "
                f"| {len(sub)}"
            )
            report["subset_aucs"][subset] = row_aucs

        print("="*65)
        print("\nInterpretation:")
        print("  Low attn entropy at early layers → attention froze early → member")
        print("  High attn convergence at early layers → pattern settled early → member")

        report_path = self.output_dir / f"EXP29_report_{timestamp}.json"
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
        sample_fraction = 0.05     # ~2h on A100 (forward only, but eager attention)
        output_dir = "results"
        max_length = 2048
        max_seq_for_attn = 512     # Cap for O(n²) attention memory
        seed = 42

    print(f"[EXP29] Model        : {Args.model_name}")
    print(f"[EXP29] Sample       : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP29] Max attn seq : {Args.max_seq_for_attn} tokens")
    Experiment(Args).run()
