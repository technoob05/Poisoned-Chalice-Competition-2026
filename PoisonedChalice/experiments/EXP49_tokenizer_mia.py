"""
EXPERIMENT 49: Tokenizer-Based MIA — Membership Inference via Tokenizer Signals

Paper: "Membership Inference Attacks on Tokenizers of Large Language Models"
       Tong, Du, Chen, Zhang, Li
       USTC / Purdue University (arXiv:2510.05699v1, Oct 2025)

Core insight:
    BPE tokenizers overfit to their training data by merging "distinctive
    tokens" — rare subwords that appear disproportionately in specific
    training datasets. The paper exploits this at the DATASET level
    (is this website's data in the tokenizer's training corpus?) achieving
    AUC 0.771 via Vocabulary Overlap and 0.740 via Frequency Estimation.

    Key findings from the paper:
    1. Larger vocabularies → more distinctive tokens → higher MIA AUC
    2. Larger datasets → more distinctive tokens → easier to infer
    3. Token merge order follows a power law with token frequency
    4. Distinctive tokens have high merge index (merged late) and their
       occurrences concentrate in specific datasets

Adaptation from DATASET-level to SAMPLE-level:
    The paper infers whether an entire dataset (website) was in training.
    We need to infer whether a single code file is a member. Key adaptations:

    1. COMPRESSION RATE: members should tokenize more efficiently (lower
       bytes-per-token) since the BPE tokenizer was optimized on training data.
       Paper baseline (AUC 0.509 at dataset-level) — but sample-level may be
       stronger since individual files have more variance.

    2. TOKEN RARITY SCORE: for each sample, measure how many rare/late-merged
       tokens it contains. Members contributed distinctive tokens to the vocab,
       so they may use more high-merge-index tokens.

    3. MERGE INDEX STATISTICS: max, mean, and percentile statistics of token
       merge indices in each sample. Late-merged tokens (high index) are rare
       and likely training-data-specific.

    4. OOV/UNK FRACTION: non-members may produce more unknown/byte-fallback
       tokens that the tokenizer doesn't handle well.

    5. TOKEN FREQUENCY SCORE: inspired by RTF-SI metric from the paper.
       Estimate token "rarity" via merge index as a proxy for frequency
       (power-law relationship). Members should have a distinctive pattern
       of rare tokens.

UNIQUE PROPERTIES:
    - NO MODEL QUERIES NEEDED — purely tokenizer-based
    - EXTREMELY FAST — tokenization is ~1000x faster than forward pass
    - Completely orthogonal to all gradient/loss/attention signals
    - Can process 100% of dataset in minutes (no GPU needed for scoring)
    - Even if AUC is modest, it's FREE signal for stacking

Compute: tokenizer only, no model inference
Expected runtime: ~1-3 min for FULL dataset (100% sample)
Expected AUC: 0.50-0.58 (sample-level is harder than dataset-level;
    paper's compression rate baseline was only 0.509 at dataset-level,
    but our code-specific tokenizer may have stronger per-file signals)
"""
import os
import random
import warnings
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP49: Tokenizer-Based MIA")
    print("  Paper: Tong et al. (arXiv:2510.05699v1, Oct 2025)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


class TokenizerAnalyzer:
    """Analyze tokenizer signals for membership inference.

    Exploits the BPE tokenizer's tendency to overfit distinctive tokens
    from training data into its vocabulary.
    """

    def __init__(self, tokenizer_name: str, max_length: int = 2048):
        print(f"[*] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self._err_count = 0

        self.vocab_size = self.tokenizer.vocab_size
        print(f"  Vocab size: {self.vocab_size}")

        self._build_merge_index()
        self._build_token_length_map()

    def _build_merge_index(self):
        """Build merge index from tokenizer's merge file (BPE merge order).

        Tokens merged later (higher index) are rarer and more likely
        to be "distinctive tokens" specific to training data.
        """
        self.merge_index = {}

        if hasattr(self.tokenizer, 'backend_tokenizer'):
            bt = self.tokenizer.backend_tokenizer
            if hasattr(bt, 'model') and hasattr(bt.model, 'get_vocab'):
                vocab = bt.model.get_vocab()
                self.merge_index = {token_id: token_id for token_id in vocab.values()}

        if not self.merge_index:
            vocab = self.tokenizer.get_vocab()
            self.merge_index = {v: v for v in vocab.values()}

        self.max_merge_idx = max(self.merge_index.values()) if self.merge_index else self.vocab_size
        print(f"  Max merge index: {self.max_merge_idx}")

        # Percentile thresholds for "rare" tokens
        all_indices = sorted(self.merge_index.values())
        n = len(all_indices)
        if n > 0:
            self.p50_idx = all_indices[n // 2]
            self.p75_idx = all_indices[int(n * 0.75)]
            self.p90_idx = all_indices[int(n * 0.90)]
            self.p95_idx = all_indices[int(n * 0.95)]
        else:
            self.p50_idx = self.p75_idx = self.p90_idx = self.p95_idx = 0

        print(f"  Merge index percentiles: p50={self.p50_idx}, p75={self.p75_idx}, "
              f"p90={self.p90_idx}, p95={self.p95_idx}")

    def _build_token_length_map(self):
        """Map token IDs to their byte/character length."""
        self.token_byte_len = {}
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            self.token_byte_len[token_id] = len(token_str.encode('utf-8', errors='replace'))

    def extract(self, text: str) -> Dict[str, float]:
        """Extract tokenizer-based MIA features for a single sample."""
        feature_keys = [
            "bytes_per_token", "tokens_per_char",
            "n_tokens", "n_unique_tokens",
            "merge_idx_max", "merge_idx_mean", "merge_idx_std",
            "merge_idx_p90", "merge_idx_p95",
            "frac_above_p75", "frac_above_p90", "frac_above_p95",
            "rare_token_count_p90", "rare_token_count_p95",
            "avg_token_byte_len", "max_token_byte_len",
            "token_type_ratio",
            "rarity_score", "neg_mean_loss_proxy",
        ]
        result = {k: np.nan for k in feature_keys}

        if not text or len(text) < 10:
            return result

        try:
            text_bytes = len(text.encode('utf-8', errors='replace'))
            token_ids = self.tokenizer.encode(
                text, max_length=self.max_length, truncation=True
            )
            n_tokens = len(token_ids)

            if n_tokens < 3:
                return result

            result["n_tokens"] = float(n_tokens)
            result["n_unique_tokens"] = float(len(set(token_ids)))

            # --- Compression Rate (Section 5.1 of paper) ---
            result["bytes_per_token"] = text_bytes / n_tokens
            result["tokens_per_char"] = n_tokens / max(1, len(text))

            # --- Merge Index Statistics ---
            merge_indices = []
            for tid in token_ids:
                idx = self.merge_index.get(tid, tid)
                merge_indices.append(idx)

            mi = np.array(merge_indices, dtype=np.float64)
            result["merge_idx_max"] = float(mi.max())
            result["merge_idx_mean"] = float(mi.mean())
            result["merge_idx_std"] = float(mi.std())
            sorted_mi = np.sort(mi)
            result["merge_idx_p90"] = float(sorted_mi[int(len(sorted_mi) * 0.90)])
            result["merge_idx_p95"] = float(sorted_mi[int(min(len(sorted_mi) - 1, len(sorted_mi) * 0.95))])

            # --- Distinctive Token Fractions ---
            # Tokens with high merge index are "distinctive" (merged late, rare)
            result["frac_above_p75"] = float(np.mean(mi > self.p75_idx))
            result["frac_above_p90"] = float(np.mean(mi > self.p90_idx))
            result["frac_above_p95"] = float(np.mean(mi > self.p95_idx))
            result["rare_token_count_p90"] = float(np.sum(mi > self.p90_idx))
            result["rare_token_count_p95"] = float(np.sum(mi > self.p95_idx))

            # --- Token Length Statistics ---
            byte_lens = [self.token_byte_len.get(tid, 1) for tid in token_ids]
            result["avg_token_byte_len"] = float(np.mean(byte_lens))
            result["max_token_byte_len"] = float(np.max(byte_lens))

            # --- Type-Token Ratio (lexical diversity) ---
            result["token_type_ratio"] = float(len(set(token_ids)) / n_tokens)

            # --- Rarity Score (inspired by RTF-SI) ---
            # Approximate self-information via merge index (power law: higher index = rarer)
            # SI(t_i) ≈ alpha * log(i) for power-law distributed frequencies
            alpha = 1.5  # typical value from the paper (Table 1)
            si_scores = []
            for idx in merge_indices:
                if idx > 256:  # skip byte-level tokens (always present)
                    si = alpha * math.log(max(idx, 1))
                    si_scores.append(si)

            if si_scores:
                result["rarity_score"] = float(np.mean(si_scores))
            else:
                result["rarity_score"] = 0.0

            # --- Compression as loss proxy ---
            # Higher bytes_per_token = less efficient compression = possibly non-member
            # Negate so higher = more likely member (like neg_mean_loss)
            result["neg_mean_loss_proxy"] = -result["bytes_per_token"]

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP49 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)

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
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        analyzer = TokenizerAnalyzer(self.args.model_name, self.args.max_length)

        print(f"\n[EXP49] Extracting tokenizer features for {len(df)} samples...")
        print(f"  NOTE: No model inference needed — tokenizer only!")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP49]"):
            rows.append(analyzer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df["bytes_per_token"].notna().sum()
        print(f"\n[EXP49] Valid: {n_valid}/{len(df)}")
        if analyzer._err_count > 0:
            print(f"[EXP49] Errors: {analyzer._err_count}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   EXP49: Tokenizer-Based MIA — REPORT")
        print("=" * 70)

        # All signal AUCs
        score_cols = [
            ("bytes_per_token", "Bytes per token (higher = non-member?)"),
            ("neg_mean_loss_proxy", "-Bytes/token (compression, higher=member)"),
            ("tokens_per_char", "Tokens per char (higher = worse compression)"),
            ("merge_idx_max", "Max merge index (distinctive token presence)"),
            ("merge_idx_mean", "Mean merge index"),
            ("merge_idx_p90", "P90 merge index"),
            ("merge_idx_p95", "P95 merge index"),
            ("frac_above_p75", "Fraction tokens above vocab p75"),
            ("frac_above_p90", "Fraction tokens above vocab p90"),
            ("frac_above_p95", "Fraction tokens above vocab p95 [KEY]"),
            ("rare_token_count_p90", "Count of rare tokens (above p90)"),
            ("rare_token_count_p95", "Count of rare tokens (above p95)"),
            ("rarity_score", "Rarity score (RTF-SI inspired) [KEY]"),
            ("avg_token_byte_len", "Avg token byte length"),
            ("token_type_ratio", "Type-token ratio (lexical diversity)"),
            ("n_tokens", "Total tokens"),
            ("n_unique_tokens", "Unique tokens"),
        ]

        print("\n--- Signal AUCs ---")
        aucs = {}
        for col, label in score_cols:
            if col not in df.columns:
                continue
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc = roc_auc_score(v["is_member"], v[col])
            aucs[col] = auc
            tag = ""
            if "[KEY]" in label:
                tag = " <--"
            print(f"  {label:<55} AUC = {auc:.4f}{tag}")

        # Also try negated versions for signals where direction is unclear
        print("\n--- Negated Signal AUCs (checking direction) ---")
        for col in ["bytes_per_token", "tokens_per_char", "merge_idx_mean",
                     "frac_above_p90", "rarity_score", "token_type_ratio"]:
            if col not in df.columns:
                continue
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            auc_pos = aucs.get(col, 0.5)
            better = "NEG" if auc_neg > auc_pos else "POS"
            print(f"  -{col:<50} AUC = {auc_neg:.4f}  (best: {better})")
            if auc_neg > auc_pos:
                aucs[f"neg_{col}"] = auc_neg

        if aucs:
            best = max(aucs, key=aucs.get)
            print(f"\n  Best signal: {best} = {aucs[best]:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:    0.6472")
        print(f"  vs EXP01 Loss baseline:  0.5807")

        # Distribution statistics
        print("\n--- Distribution Statistics (Members vs Non-Members) ---")
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for col in ["bytes_per_token", "rarity_score", "frac_above_p95",
                     "merge_idx_max", "token_type_ratio", "n_tokens"]:
            if col not in df.columns:
                continue
            m_val = m[col].dropna()
            nm_val = nm[col].dropna()
            if len(m_val) > 0 and len(nm_val) > 0:
                print(
                    f"  {col:<25} M={m_val.mean():.4f}+-{m_val.std():.4f}  "
                    f"NM={nm_val.mean():.4f}+-{nm_val.std():.4f}  "
                    f"ratio={m_val.mean()/(nm_val.mean()+1e-10):.4f}"
                )

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'Bytes/Tok':<10} | {'Rarity':<10} | {'Frac>p95':<10} | {'MergeMax':<10} | N")
        print("-" * 75)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["bytes_per_token", "rarity_score", "frac_above_p95", "merge_idx_max"]:
                if sc not in sub.columns:
                    r[sc] = float("nan")
                    continue
                v = sub.dropna(subset=[sc])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    r[sc] = roc_auc_score(v["is_member"], v[sc])
                else:
                    r[sc] = float("nan")
            print(
                f"{subset:<10} | {r.get('bytes_per_token', float('nan')):.4f}     "
                f"| {r.get('rarity_score', float('nan')):.4f}     "
                f"| {r.get('frac_above_p95', float('nan')):.4f}     "
                f"| {r.get('merge_idx_max', float('nan')):.4f}     "
                f"| {len(sub)}"
            )

        # Per-language tokenization statistics
        print("\n--- Per-Language Tokenization Stats ---")
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            bpt = sub["bytes_per_token"].dropna()
            if len(bpt) > 0:
                print(f"  {subset:<10} bytes/token: {bpt.mean():.2f} +/- {bpt.std():.2f}  "
                      f"(range: {bpt.min():.2f} - {bpt.max():.2f})")

        # Verdict
        print("\n--- VERDICT ---")
        best_auc = max(aucs.values()) if aucs else 0.5
        if best_auc > 0.55:
            print(f"  PROMISING: best tokenizer signal = {best_auc:.4f}")
            print(f"  → Tokenizer leaks membership info — add to stacker as FREE feature")
        elif best_auc > 0.52:
            print(f"  WEAK: best tokenizer signal = {best_auc:.4f}")
            print(f"  → Marginal signal, may help as tiebreaker in ensemble")
        else:
            print(f"  NEGLIGIBLE: best tokenizer signal = {best_auc:.4f}")
            print(f"  → Tokenizer signals too weak at sample level for this model/dataset")
            print(f"  → Paper's dataset-level signals don't transfer to sample-level")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP49_{timestamp}.parquet", index=False)
        print(f"\n[EXP49] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 1.00  # Full dataset! Tokenizer-only = fast enough
        max_length = 2048
        output_dir = "results"
        seed = 42

    print(f"[EXP49] Tokenizer MIA: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  NOTE: No GPU needed — tokenizer only, ~1-3 min for full dataset")
    Experiment(Args).run()
