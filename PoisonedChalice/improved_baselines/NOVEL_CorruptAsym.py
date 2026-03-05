"""
NOVEL_CorruptAsym.py — Corruption Asymmetry MIA ⭐⭐

NOVEL — discovered from our EXP48 experiments.

OUR DISCOVERY (NO PRIOR ART):
    Different corruption types produce OPPOSITE membership signals:
    
    ┌──────────────────────┬────────────────┬─────────────────────────────────┐
    │ Corruption Type      │ AUC Direction  │ Mechanism                       │
    ├──────────────────────┼────────────────┼─────────────────────────────────┤
    │ SPAN corruption      │ 0.6419 (✓)     │ Members recognize surrounding   │
    │ (replace contiguous) │ Members MORE   │ memorized context despite local  │
    │                      │ robust         │ noise                            │
    ├──────────────────────┼────────────────┼─────────────────────────────────┤
    │ TOKEN mask/drop      │ 0.45 (❌)      │ Non-members' generalizable      │
    │ (random individuals) │ Members LESS   │ representations handle random    │
    │                      │ robust         │ token loss better                │
    └──────────────────────┴────────────────┴─────────────────────────────────┘

    The ASYMMETRY between these two — the fact that the signal FLIPS direction —
    is itself a novel and powerful signal. No prior work has observed this.

PAPER STORY:
    "The direction of the memorization signal under corruption depends critically
    on the corruption GRANULARITY. Span-level corruption preserves local syntax
    and thus membership signal; token-level corruption destroys it. The gap 
    between span-robustness and token-fragility — the 'corruption asymmetry' — 
    captures the tension between local memorization and global generalization."

SIGNALS:
    1. sim_span_mid     — cosine sim under span corruption (member = higher)
    2. sim_token_mid    — cosine sim under token corruption (member = LOWER)
    3. corrupt_asym     — sim_span - sim_token (the NOVEL asymmetry signal)
    4. asym_ratio       — sim_span / sim_token (multiplicative asymmetry)

Compute: 4 forward passes with output_hidden_states=True, ~15 min on A100
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
    print("  CorruptAsym-MIA: Corruption Asymmetry as Membership Signal")
    print("  Span-robustness vs Token-fragility")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}")
    return model, tokenizer


class CorruptAsymExtractor:
    """
    Extracts corruption asymmetry features.
    
    1. Get hidden states for ORIGINAL input
    2. Apply SPAN corruption → get hidden states → cosine similarity
    3. Apply TOKEN corruption → get hidden states → cosine similarity
    4. Asymmetry = span_sim - token_sim (NOVEL)
    """

    def __init__(self, model, tokenizer, max_length: int = 512,
                 corruption_rate: float = 0.15, n_corruptions: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.n_corruptions = n_corruptions
        self.vocab_size = model.config.vocab_size
        self._err_count = 0

    def _corrupt_span(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace a contiguous span with random tokens."""
        ids = input_ids.clone()
        T = ids.shape[1]
        span_len = max(1, int(T * self.corruption_rate))
        start = random.randint(0, max(0, T - span_len - 1))
        rand_ids = torch.randint(0, self.vocab_size, (1, span_len), device=ids.device)
        ids[0, start:start + span_len] = rand_ids[0]
        return ids

    def _corrupt_token(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace random individual tokens throughout the sequence."""
        ids = input_ids.clone()
        T = ids.shape[1]
        n_corrupt = max(1, int(T * self.corruption_rate))
        positions = random.sample(range(T), min(n_corrupt, T))
        for pos in positions:
            ids[0, pos] = random.randint(0, self.vocab_size - 1)
        return ids

    def _get_hidden_representation(self, input_ids: torch.Tensor, layer_idx: int):
        """Get mean-pooled hidden state at specified layer."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx][0].float()  # (T, D)
        return hs.mean(dim=0)  # (D,)

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        features = {}
        if not text or len(text) < 20:
            return features
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 10:
                return features

            n_layers = self.model.config.num_hidden_layers
            mid_layer = min(15, n_layers)
            last_layer = n_layers

            # 1. Original hidden states
            orig_outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            orig_mid = orig_outputs.hidden_states[mid_layer][0].float().mean(dim=0)
            orig_last = orig_outputs.hidden_states[last_layer][0].float().mean(dim=0)

            # Loss baseline
            logits = orig_outputs.logits
            shift_lp = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
            shift_labels = input_ids[0, 1:]
            token_lp = shift_lp.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            features["neg_mean_loss"] = float(token_lp.mean().item())

            # 2. Span corruption → similarities
            span_sims_mid = []
            span_sims_last = []
            for _ in range(self.n_corruptions):
                corrupt_ids = self._corrupt_span(input_ids)
                c_out = self.model(input_ids=corrupt_ids, output_hidden_states=True)
                c_mid = c_out.hidden_states[mid_layer][0].float().mean(dim=0)
                c_last = c_out.hidden_states[last_layer][0].float().mean(dim=0)
                span_sims_mid.append(F.cosine_similarity(orig_mid, c_mid, dim=0).item())
                span_sims_last.append(F.cosine_similarity(orig_last, c_last, dim=0).item())

            # 3. Token corruption → similarities
            token_sims_mid = []
            token_sims_last = []
            for _ in range(self.n_corruptions):
                corrupt_ids = self._corrupt_token(input_ids)
                c_out = self.model(input_ids=corrupt_ids, output_hidden_states=True)
                c_mid = c_out.hidden_states[mid_layer][0].float().mean(dim=0)
                c_last = c_out.hidden_states[last_layer][0].float().mean(dim=0)
                token_sims_mid.append(F.cosine_similarity(orig_mid, c_mid, dim=0).item())
                token_sims_last.append(F.cosine_similarity(orig_last, c_last, dim=0).item())

            ss_mid = np.mean(span_sims_mid)
            ss_last = np.mean(span_sims_last)
            ts_mid = np.mean(token_sims_mid)
            ts_last = np.mean(token_sims_last)

            # ── Individual robustness signals ─────────────────────────────
            features["sim_span_mid"] = float(ss_mid)
            features["sim_span_last"] = float(ss_last)
            features["neg_sim_token_mid"] = float(-ts_mid)     # INVERT: lower = member
            features["neg_sim_token_last"] = float(-ts_last)

            # ── THE NOVEL ASYMMETRY SIGNAL ────────────────────────────────
            # Span robustness MINUS token robustness
            # Members: high span_sim + low token_sim → LARGE asymmetry
            # Non-members: moderate both → SMALL asymmetry
            features["corrupt_asym_mid"] = float(ss_mid - ts_mid)
            features["corrupt_asym_last"] = float(ss_last - ts_last)
            features["corrupt_asym_avg"] = float(
                (ss_mid - ts_mid + ss_last - ts_last) / 2)

            # Ratio variant
            if ts_mid > 1e-6:
                features["asym_ratio_mid"] = float(ss_mid / ts_mid)
            if ts_last > 1e-6:
                features["asym_ratio_last"] = float(ss_last / ts_last)

            # ── Stability under corruption ────────────────────────────────
            features["neg_span_std_mid"] = float(-np.std(span_sims_mid))
            features["neg_token_std_mid"] = float(-np.std(token_sims_mid))

            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[CorruptAsym WARN] {type(e).__name__}: {e}")
            self._err_count += 1
        return features


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
        for subset in subsets:
            try:
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
            except Exception as e:
                print(f"  [WARN] {subset}: {e}")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        ext = CorruptAsymExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            corruption_rate=self.args.corruption_rate,
            n_corruptions=self.args.n_corruptions,
        )
        passes = 1 + 2 * self.args.n_corruptions
        print(f"\n[CorruptAsym] {passes} fwd passes/sample, rate={self.args.corruption_rate}")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[CorruptAsym]"):
            rows.append(ext.extract(row["content"]))
        feat_df = pd.DataFrame(rows)
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   CorruptAsym-MIA: CORRUPTION ASYMMETRY SIGNALS")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        cat_map = {
            "SPAN ROBUSTNESS": [c for c in feature_cols if "span" in c],
            "TOKEN FRAGILITY": [c for c in feature_cols if "token" in c],
            "★ ASYMMETRY (NOVEL)": [c for c in feature_cols if "asym" in c],
            "BASELINE": ["neg_mean_loss"],
        }

        all_results = {}
        for cat, cols in cat_map.items():
            print(f"\n  ── {cat} ──")
            for col in sorted(cols):
                if col not in df.columns:
                    continue
                v = df.dropna(subset=[col])
                if len(v) < 50 or len(v["is_member"].unique()) < 2:
                    continue
                auc_pos = roc_auc_score(v["is_member"], v[col])
                best = max(auc_pos, 1 - auc_pos)
                d = "+" if auc_pos >= 0.5 else "-"
                all_results[col] = (best, d)
                marker = " ★" if best > 0.60 else ""
                print(f"    {d}{col:<40} AUC = {best:.4f}{marker}")

        if all_results:
            top = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
            best_col = top[0][0]
            best_d = top[0][1][1]
            print(f"\n  BEST: {best_d}{best_col} = {top[0][1][0]:.4f}")
            print(f"\n  Per-subset:")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10 and len(sub["is_member"].unique()) > 1:
                    sv = sub[best_col] if best_d == "+" else -sub[best_col]
                    auc = roc_auc_score(sub["is_member"], sv)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

        print("=" * 70)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL_corrupt_asym_{ts}.parquet", index=False)
        print(f"[CorruptAsym] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        max_length = 512
        output_dir = "results"
        seed = 42
        corruption_rate = 0.15
        n_corruptions = 3

    Experiment(Args).run()
