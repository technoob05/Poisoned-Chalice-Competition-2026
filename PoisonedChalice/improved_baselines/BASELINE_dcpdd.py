"""
BASELINE_dcpdd.py — DC-PDD: Divergence-based Calibration for Pre-Training Data Detection

Reimplementation of Zhang, Sun et al. (arXiv:2409.14781, 2024).
Outperforms Min-K%++ on some benchmarks (PatentMIA, partial MIMIR).

KEY IDEA (different from Min-K%++):
    Min-K%++: calibrates by VOCAB distribution  p(·|x<t)
    DC-PDD:   calibrates by TOKEN FREQUENCY distribution  p_freq(·)

    For each token position t:
        DC-PDD_t = log p(x_t|x<t) - log p_freq(x_t)
    
    where p_freq(x_t) is the unigram frequency of token x_t in a reference corpus.
    
    Intuition: If a token has high model probability BUT is also very frequent 
    (e.g., "the", "import"), that's NOT evidence of memorization. DC-PDD discounts 
    common tokens by subtracting their prior frequency.

    This is the CROSS-ENTROPY between model distribution and frequency distribution:
        D(p_model, p_freq) = -Σ p_model(z|x<t) · log p_freq(z)
    
    The divergence-from-randomness principle: training tokens should deviate MORE 
    from the random (frequency-based) baseline than non-training tokens.

SIGNALS:
    1. dcpdd_k{20,30,50} — DC-PDD min-k% scores
    2. minkpp_k20         — Min-K%++ baseline for comparison
    3. neg_mean_loss       — Raw loss baseline
    4. freq_calibrated_loss — Average frequency-calibrated log-prob

Properties:
    - Unsupervised (token frequencies computed on-the-fly from corpus)
    - Zero overhead (1 forward pass + frequency lookup)
    - Reference-free (frequency from the DATASET ITSELF, no extra model)

Compute: 1 forward pass only
Expected runtime: ~5-10 min on A100 (10% sample)
"""

import os
import random
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
    print("  DC-PDD: Divergence-based Calibration (Zhang et al., 2024)")
    print("  Calibration via Token Frequency, NOT Vocab Distribution")
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
    print(f"  Loaded. dtype={dtype}, vocab={model.config.vocab_size}")
    return model, tokenizer


class DCPDDExtractor:
    """
    DC-PDD: Divergence-based Calibrated Pre-training Data Detection.
    
    Calibrates token log-probability by subtracting the log frequency of that
    token in the corpus, rather than by the mean/std of the vocab distribution
    (which is what Min-K%++ does).
    """

    def __init__(self, model, tokenizer, max_length: int = 512,
                 log_freq: Optional[np.ndarray] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.log_freq = log_freq  # (V,) log-frequency of each token
        self._err_count = 0

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
            if seq_len < 5:
                return features

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, :-1, :].float()  # (T, V)
            labels = input_ids[0, 1:]                     # (T,)
            T = logits.shape[0]
            if T < 3:
                return features

            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Per-token log probability
            token_lp = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)  # (T,)
            tlp_np = token_lp.cpu().numpy()

            # ── Min-K%++ baseline ─────────────────────────────────────────
            mu = (probs * log_probs).sum(dim=-1)
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()
            z_scores = ((token_lp - mu) / sigma).cpu().numpy()

            for k in [0.2, 0.3]:
                n_sel = max(1, int(T * k))
                features[f"minkpp_k{int(k*100)}"] = float(np.mean(np.sort(z_scores)[:n_sel]))

            # ── DC-PDD: Frequency-calibrated scores ───────────────────────
            if self.log_freq is not None:
                labels_np = labels.cpu().numpy()
                # Per-token frequency calibration
                lf = self.log_freq[labels_np]  # (T,) log-frequency of each actual token
                # DC-PDD score: log p(x_t|x<t) - log p_freq(x_t)
                dcpdd_scores = tlp_np - lf  # (T,)

                for k in [0.2, 0.3, 0.5, 1.0]:
                    n_sel = max(1, int(T * k))
                    sorted_dc = np.sort(dcpdd_scores)[:n_sel]
                    features[f"dcpdd_k{int(k*100)}"] = float(np.mean(sorted_dc))

                # Also: z-normalized DC-PDD (combine both calibrations)
                dc_mean = np.mean(dcpdd_scores)
                dc_std = np.std(dcpdd_scores)
                if dc_std > 1e-8:
                    dc_z = (dcpdd_scores - dc_mean) / dc_std
                    for k in [0.2]:
                        n_sel = max(1, int(T * k))
                        features[f"dcpdd_z_k{int(k*100)}"] = float(np.mean(np.sort(dc_z)[:n_sel]))

                features["freq_calibrated_loss"] = float(np.mean(dcpdd_scores))

                # Cross-entropy between model dist and frequency dist
                # D(p_model, p_freq) = -Σ p(z|x<t) * log_freq(z)
                log_freq_tensor = torch.tensor(
                    self.log_freq, device=logits.device, dtype=logits.dtype
                ).unsqueeze(0).expand(T, -1)  # (T, V)
                cross_ent = -(probs * log_freq_tensor).sum(dim=-1)  # (T,)
                features["neg_cross_ent_freq"] = float(-cross_ent.mean().item())

            # ── Raw loss baseline ─────────────────────────────────────────
            features["neg_mean_loss"] = float(np.mean(tlp_np))
            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[DC-PDD WARN] {type(e).__name__}: {e}")
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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def build_token_frequencies(self, texts: list) -> np.ndarray:
        """Build unigram token frequency distribution from the dataset."""
        print("[*] Building token frequency distribution...")
        vocab_size = self.model.config.vocab_size
        counts = np.zeros(vocab_size, dtype=np.float64)

        for text in tqdm(texts[:min(len(texts), 50000)], desc="[Freq]"):
            if not text or len(text) < 10:
                continue
            toks = self.tokenizer(
                text, max_length=self.args.max_length, truncation=True
            )["input_ids"]
            for tok_id in toks:
                if 0 <= tok_id < vocab_size:
                    counts[tok_id] += 1

        # Laplace smoothing to avoid log(0)
        counts += 1.0
        total = counts.sum()
        freqs = counts / total
        log_freq = np.log(freqs).astype(np.float32)
        print(f"  Built frequency from {min(len(texts), 50000)} texts, "
              f"total tokens: {int(total)}, non-zero: {(counts > 1).sum()}/{vocab_size}")
        return log_freq

    def run(self):
        df = self.load_data()

        # Build frequency distribution from corpus
        log_freq = self.build_token_frequencies(df["content"].tolist())

        extractor = DCPDDExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            log_freq=log_freq,
        )

        print(f"\n[DC-PDD] Extracting scores...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[DC-PDD]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   DC-PDD: UNSUPERVISED AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        results = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            best = max(auc_pos, 1 - auc_pos)
            d = "+" if auc_pos >= 0.5 else "-"
            results[col] = (best, d)
            marker = " ★" if best > 0.59 else ""
            print(f"  {d}{col:<40} AUC = {best:.4f}{marker}")

        # Best signal
        if results:
            best = max(results.items(), key=lambda x: x[1][0])
            minkpp = results.get("minkpp_k20", (0, ""))
            loss = results.get("neg_mean_loss", (0, ""))
            print(f"\n  BEST SIGNAL: {best[1][1]}{best[0]} = {best[1][0]:.4f}")
            print(f"  Min-K%++ (k=20):   {minkpp[0]:.4f}")
            print(f"  Loss baseline:     {loss[0]:.4f}")

            # Per-subset
            best_col = best[0]
            print(f"\n  Per-subset for {best_col}:")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10:
                    auc = roc_auc_score(sub["is_member"], sub[best_col])
                    auc = max(auc, 1 - auc)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"BASELINE_dcpdd_{timestamp}.parquet", index=False)
        print(f"[DC-PDD] Results saved.")


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

    print(f"[DC-PDD] Divergence-based Calibration")
    print(f"  model={Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
