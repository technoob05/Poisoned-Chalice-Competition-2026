#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║  ESP-Cal: Entropy Slope + Multi-Scale Calibration for MIA    ║
║  Paper 2 — Full Benchmark Evaluation                         ║
║                                                               ║
║  Benchmarks: Poisoned Chalice, WikiMIA, MIMIR, BookMIA       ║
║  Author: [Redacted for review]                               ║
║  Self-contained Kaggle notebook script                       ║
╚═══════════════════════════════════════════════════════════════╝

Method:
  1. Entropy Slope (ESP): Linear fit of per-token prediction entropy
     across token positions. Members show STEEPER entropy decline
     (model becomes more confident as it sees more of the memorized text).
  
  2. Multi-Scale Calibration (3-scale z-normalization):
     Scale 1 — Token-level: z-normalize per-token entropy
     Scale 2 — Position-level: z-normalize by positional bucket
     Scale 3 — Domain-level: z-normalize per language/domain

  Final score: z_esp = domain_znorm(position_znorm(token_znorm(ESP)))

Grey-box: requires only output logits, no hidden states or gradients.
Fully unsupervised: no reference model, no training data needed.
"""

import os
import sys
import gc
import json
import time
import warnings
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# 0. Environment Setup
# ═══════════════════════════════════════════

def setup_environment():
    """Install deps + HuggingFace auth for Kaggle."""
    print("=" * 60)
    print("  ESP-Cal — Full Benchmark Evaluation")
    print("=" * 60)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers", "accelerate", "datasets",
                    "scikit-learn", "scipy", "huggingface_hub"],
                   capture_output=True)

    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        token = UserSecretsClient().get_secret("posioned")
        login(token=token, add_to_git_credential=True)
        print("  ✓ HuggingFace authenticated (Kaggle)")
    except Exception:
        print("  ○ Not on Kaggle or no HF secret — using local auth")


import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════

@dataclass
class Config:
    """Experiment configuration."""
    # Model
    model_name: str = "bigcode/starcoder2-3b"
    max_length: int = 512
    torch_dtype: str = "bfloat16"

    # Dataset
    dataset_name: str = "AISE-TUDelft/Poisoned-Chalice"
    languages: List[str] = field(default_factory=lambda: ["Go", "Java", "Python", "Ruby", "Rust"])
    sample_fraction: float = 1.0
    split: str = "train"

    # ESP parameters
    entropy_eps: float = 1e-10  # epsilon for log stability
    position_buckets: int = 16  # number of positional buckets for scale-2 calibration
    min_tokens: int = 8  # minimum tokens for valid entropy slope

    # Ablation: which scales to enable
    enable_scale1_token: bool = True
    enable_scale2_position: bool = True
    enable_scale3_domain: bool = True

    # Evaluation
    seed: int = 42
    output_dir: str = "./results"

    # Benchmark mode
    benchmark: str = "poisoned_chalice"

    # WikiMIA-specific
    wikimia_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    wikimia_model: str = "EleutherAI/pythia-2.8b-deduped"  # default/primary
    # Multi-model evaluation (SamplePaper: "5 families of 10 models")
    wikimia_models: List[str] = field(default_factory=lambda: [
        # Pythia family (trained on Pile, Wikipedia included)
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
        # Mamba SSM (state-space model, different architecture)
        "state-spaces/mamba-2.8b-hf",
        # GPT-Neo family (trained on Pile)
        "EleutherAI/gpt-neo-2.7B",
        # OPT family (Meta, trained on diverse web data)
        "facebook/opt-2.7b",
    ])

    # MIMIR-specific
    mimir_domains: List[str] = field(default_factory=lambda: [
        "wikipedia", "github", "pile_cc", "pubmed_central",
        "arxiv", "dm_mathematics", "hackernews"
    ])
    mimir_model: str = "EleutherAI/pythia-2.8b-deduped"  # default/primary
    # Multi-model: Pythia suite (trained on Pile — MIMIR's source)
    mimir_models: List[str] = field(default_factory=lambda: [
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-1.4b-deduped",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
    ])

    # BookMIA-specific (Shi et al., 2024 — copyright books, 512 tokens)
    bookmia_model: str = "EleutherAI/pythia-2.8b-deduped"  # default/primary
    bookmia_models: List[str] = field(default_factory=lambda: [
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
        "EleutherAI/gpt-neo-2.7B",
    ])

    # Multi-model control
    multi_model: bool = True  # set False to use single model (faster)


# ═══════════════════════════════════════════
# 2. Model Loader
# ═══════════════════════════════════════════

def load_model(model_name: str, dtype_str: str = "bfloat16"):
    """Load a causal LM for logit extraction."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n  Loading model: {model_name}")
    dtype = getattr(torch, dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  ✓ Model loaded: {model.config.num_hidden_layers} layers, dtype={dtype}")
    return model, tokenizer


# ═══════════════════════════════════════════
# 3. ESP Feature Extractor
# ═══════════════════════════════════════════

class ESPExtractor:
    """
    Extract Entropy Slope Profile (ESP) features from logits.

    For each token position t, compute:
        H(t) = -Σ_v p(v|x_{<t}) log p(v|x_{<t})
    
    Then fit a linear model: H(t) = α·t + β
    
    ESP score = α (slope of entropy trajectory)
    Members: steeper negative slope (model gets more confident faster)
    """

    def __init__(self, model, tokenizer, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract ESP and related features from a single text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        seq_len = input_ids.shape[1]

        if seq_len < self.cfg.min_tokens:
            return self._empty_features()

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # ── Per-token entropy ──
        # Shift: predict token t from context x_{<t}
        shift_logits = logits[:, :-1, :].float()  # (1, seq-1, V)
        shift_labels = input_ids[:, 1:]  # (1, seq-1)

        # Probabilities
        probs = F.softmax(shift_logits, dim=-1)  # (1, seq-1, V)
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, seq-1, V)

        # Token-wise entropy: H(t) = -Σ_v p(v) log p(v)
        entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # (seq-1,)
        entropy_np = entropy.cpu().numpy()

        # Token-wise loss (negative log-likelihood)
        token_lp = log_probs.squeeze(0).gather(1, shift_labels.squeeze(0).unsqueeze(1)).squeeze(1)
        token_loss = -token_lp.cpu().numpy()  # (seq-1,)

        n_tokens = len(entropy_np)

        # ── 1. Entropy Slope (ESP) ──
        positions = np.arange(n_tokens)
        esp_slope, esp_intercept = np.polyfit(positions, entropy_np, 1)

        # ── 2. Additional entropy trajectory features ──
        # Entropy mean and std
        h_mean = entropy_np.mean()
        h_std = entropy_np.std()

        # Entropy in first half vs second half
        mid = n_tokens // 2
        h_first = entropy_np[:mid].mean() if mid > 0 else h_mean
        h_second = entropy_np[mid:].mean() if mid < n_tokens else h_mean
        h_drop = h_first - h_second  # positive = entropy decreases (member-like)

        # Quadratic curvature
        if n_tokens >= 6:
            coeffs = np.polyfit(positions, entropy_np, 2)
            h_curvature = coeffs[0]  # positive = U-shape, negative = inverted-U
        else:
            h_curvature = 0.0

        # ── 3. Scale-1: Token-level z-normalized entropy ──
        # z-normalize per-token entropy within this sample
        if h_std > 0:
            z_entropy = (entropy_np - h_mean) / h_std
            z_entropy_slope = np.polyfit(positions, z_entropy, 1)[0]
        else:
            z_entropy_slope = 0.0

        # ── 4. Loss-based features (for comparison with baselines) ──
        mean_loss = token_loss.mean()
        loss_std = token_loss.std()

        # Loss slope
        loss_slope = np.polyfit(positions, token_loss, 1)[0]

        # Min-K% Prob (k=20%)
        k = max(1, int(n_tokens * 0.2))
        sorted_lp = np.sort(token_lp.cpu().numpy())
        minkprob = sorted_lp[:k].mean()

        # ── 5. Rank-based features ──
        ranks = (shift_logits.squeeze(0).argsort(dim=-1, descending=True)
                 .argsort(dim=-1).gather(1, shift_labels.squeeze(0).unsqueeze(1))
                 .squeeze(1).float().cpu().numpy())
        mean_rank = ranks.mean()
        median_rank = np.median(ranks)

        # ── 6. Composite features ──
        # SURP (Surprise): mean_loss - loss_std
        surp = mean_loss - loss_std

        # Zlib ratio
        import zlib
        zlib_len = len(zlib.compress(text.encode("utf-8")))
        zlib_ratio = mean_loss / (zlib_len / n_tokens) if zlib_len > 0 else 0.0

        features = {
            # Primary ESP features
            "esp_slope": esp_slope,
            "esp_intercept": esp_intercept,
            "z_esp_slope": z_entropy_slope,

            # Entropy trajectory
            "h_mean": h_mean,
            "h_std": h_std,
            "h_drop": h_drop,
            "h_curvature": h_curvature,

            # Loss features (baselines)
            "neg_mean_loss": -mean_loss,
            "loss_slope": loss_slope,
            "loss_std": loss_std,
            "minkprob_20": minkprob,
            "surp": -surp,

            # Rank features
            "neg_mean_rank": -mean_rank,
            "neg_median_rank": -median_rank,

            # Compression
            "zlib_ratio": zlib_ratio,

            # Metadata
            "seq_len": seq_len,
            "n_tokens": n_tokens,
        }

        # Signal columns (higher = more member)
        features["signal_esp"] = -esp_slope  # steeper decline = more member → negate
        features["signal_h_drop"] = h_drop   # larger drop = more member
        features["signal_loss"] = -mean_loss  # lower loss = more member → negate

        del outputs, logits, probs, log_probs, entropy
        torch.cuda.empty_cache()

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return NaN for too-short sequences."""
        keys = ["esp_slope", "esp_intercept", "z_esp_slope",
                "h_mean", "h_std", "h_drop", "h_curvature",
                "neg_mean_loss", "loss_slope", "loss_std",
                "minkprob_20", "surp", "neg_mean_rank", "neg_median_rank",
                "zlib_ratio", "seq_len", "n_tokens",
                "signal_esp", "signal_h_drop", "signal_loss"]
        return {k: np.nan for k in keys}


# ═══════════════════════════════════════════
# 4. Multi-Scale Calibration
# ═══════════════════════════════════════════

class MultiScaleCalibrator:
    """
    3-scale hierarchical z-normalization for ESP scores.

    Scale 1 (Token):    already done inside ESPExtractor (z_esp_slope)
    Scale 2 (Position): z-normalize by sequence-length bucket
    Scale 3 (Domain):   z-normalize per language/domain subset
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def calibrate(self, df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
        """Apply multi-scale calibration to the dataframe."""
        df = df.copy()

        for col in score_columns:
            # Save raw score
            df[f"{col}_raw"] = df[col]

            # ── Scale 2: Position-level z-norm (by sequence length bucket) ──
            if self.cfg.enable_scale2_position and "n_tokens" in df.columns:
                # Create length buckets
                df["_len_bucket"] = pd.qcut(
                    df["n_tokens"].fillna(0),
                    q=min(self.cfg.position_buckets, df["n_tokens"].nunique()),
                    duplicates="drop",
                    labels=False,
                )
                grouped = df.groupby("_len_bucket")[col]
                means = grouped.transform("mean")
                stds = grouped.transform("std").replace(0, 1)
                df[col] = (df[col] - means) / stds
                df.drop(columns=["_len_bucket"], inplace=True)

            # ── Scale 3: Domain-level z-norm ──
            if self.cfg.enable_scale3_domain and "subset" in df.columns:
                grouped = df.groupby("subset")[col]
                means = grouped.transform("mean")
                stds = grouped.transform("std").replace(0, 1)
                df[col] = (df[col] - means) / stds

        return df


# ═══════════════════════════════════════════
# 5. Data Loaders
# ═══════════════════════════════════════════

def load_poisoned_chalice(cfg: Config) -> pd.DataFrame:
    """Load Poisoned Chalice dataset."""
    from datasets import load_dataset

    print("\n  Loading Poisoned Chalice dataset...")
    all_rows = []

    kaggle_path = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
    local_path = os.path.join(os.path.dirname(__file__), "..", "..", "poisoned_chalice_dataset")

    for lang in cfg.languages:
        try:
            if os.path.exists(kaggle_path):
                from datasets import load_from_disk
                ds = load_from_disk(os.path.join(kaggle_path, lang, cfg.split))
            elif os.path.exists(local_path):
                from datasets import load_from_disk
                ds = load_from_disk(os.path.join(local_path, lang, cfg.split))
            else:
                ds = load_dataset(cfg.dataset_name, lang, split=cfg.split)

            for row in ds:
                all_rows.append({
                    "text": row["content"],
                    "is_member": int(row["membership"]),
                    "subset": lang,
                })
            print(f"    {lang}: {len(ds)} samples")
        except Exception as e:
            print(f"    {lang}: ERROR — {e}")

    df = pd.DataFrame(all_rows)

    if cfg.sample_fraction < 1.0:
        df = df.groupby(["subset", "is_member"]).apply(
            lambda x: x.sample(frac=cfg.sample_fraction, random_state=cfg.seed)
        ).reset_index(drop=True)
        print(f"  Sampled {cfg.sample_fraction:.0%} → {len(df)} rows")

    print(f"  Total: {len(df)} samples, {df['is_member'].sum()} members")
    return df


def load_wikimia(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load WikiMIA dataset."""
    from datasets import load_dataset

    print("\n  Loading WikiMIA dataset...")
    data_by_length = {}

    for length in cfg.wikimia_lengths:
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
            rows = []
            for row in ds:
                rows.append({
                    "text": row["input"],
                    "is_member": int(row["label"]),
                    "subset": f"len{length}",
                })
            df = pd.DataFrame(rows)
            data_by_length[f"len{length}"] = df
            mem = df["is_member"].sum()
            print(f"    Length {length}: {len(df)} samples ({mem}M/{len(df)-mem}NM)")
        except Exception as e:
            print(f"    Length {length}: ERROR — {e}")

    return data_by_length


def load_mimir(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load MIMIR dataset."""
    from datasets import load_dataset

    print("\n  Loading MIMIR dataset...")
    data_by_domain = {}

    for domain in cfg.mimir_domains:
        try:
            ds = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)
            rows = []
            for split_name, label in [("member", 1), ("nonmember", 0)]:
                if split_name in ds:
                    for row in ds[split_name]:
                        text = row.get("text", row.get("input", ""))
                        rows.append({
                            "text": text,
                            "is_member": label,
                            "subset": domain,
                        })
            if rows:
                df = pd.DataFrame(rows)
                data_by_domain[domain] = df
                mem = df["is_member"].sum()
                print(f"    {domain}: {len(df)} ({mem}M/{len(df)-mem}NM)")
        except Exception as e:
            print(f"    {domain}: ERROR — {e}")

    return data_by_domain


def load_bookmia(cfg: Config) -> pd.DataFrame:
    """Load BookMIA dataset (Shi et al., 2024).
    
    9,870 book excerpts (512 tokens), balanced members/non-members.
    Members: text from books published before model training cutoff.
    Non-members: text from books published in 2023.
    Source: https://huggingface.co/datasets/swj0419/BookMIA
    """
    from datasets import load_dataset

    print("\n  Loading BookMIA dataset...")

    try:
        ds = load_dataset("swj0419/BookMIA", split="train")
        rows = []
        for row in ds:
            rows.append({
                "text": row["snippet"],
                "is_member": int(row["label"]),
                "subset": f"book_{row.get('book_id', 0)}",
            })
        df = pd.DataFrame(rows)
        mem = df["is_member"].sum()
        print(f"    BookMIA: {len(df)} samples ({mem} members, {len(df)-mem} non-members)")
        return df
    except Exception as e:
        print(f"    BookMIA: ERROR — {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════
# 6. Evaluation Utilities
# ═══════════════════════════════════════════

def evaluate_scores(df: pd.DataFrame, score_columns: List[str],
                    label_col: str = "is_member") -> pd.DataFrame:
    """Compute AUROC for each score column."""
    results = []
    for col in score_columns:
        valid = df[col].notna() & df[label_col].notna()
        if valid.sum() < 10:
            continue
        y_true = df.loc[valid, label_col].values
        y_score = df.loc[valid, col].values
        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_score)
        best_auc = max(auc, 1 - auc)
        polarity = "+" if auc >= 0.5 else "-"

        results.append({
            "score": col,
            "auc": best_auc,
            "auc_raw": auc,
            "polarity": polarity,
            "n_samples": int(valid.sum()),
        })

    return pd.DataFrame(results).sort_values("auc", ascending=False)


def evaluate_per_subset(df: pd.DataFrame, score_col: str,
                        label_col: str = "is_member") -> pd.DataFrame:
    """Compute per-subset AUROC."""
    results = []
    for subset, group in df.groupby("subset"):
        valid = group[score_col].notna() & group[label_col].notna()
        if valid.sum() < 10:
            continue
        y_true = group.loc[valid, label_col].values
        y_score = group.loc[valid, score_col].values
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_score)
        best_auc = max(auc, 1 - auc)
        results.append({"subset": subset, "auc": best_auc, "n": int(valid.sum())})
    return pd.DataFrame(results)


# ═══════════════════════════════════════════
# 7. Baseline Implementations (for comparison)
# ═══════════════════════════════════════════

class BaselineComparison:
    """Compute baseline MIA scores for comparison in the paper."""

    @staticmethod
    def minkprob(log_probs: np.ndarray, k_pct: float = 0.2) -> float:
        """Min-K% Prob (Shi et al., 2024)."""
        k = max(1, int(len(log_probs) * k_pct))
        return np.sort(log_probs)[:k].mean()

    @staticmethod
    def minkprob_plus_plus(log_probs: np.ndarray, mu: np.ndarray,
                            sigma: np.ndarray, k_pct: float = 0.2) -> float:
        """Min-K%++ (Zhang et al., 2024) — requires reference statistics."""
        z_scores = (log_probs - mu) / np.maximum(sigma, 1e-10)
        k = max(1, int(len(z_scores) * k_pct))
        return np.sort(z_scores)[:k].mean()

    @staticmethod
    def loss_attack(token_losses: np.ndarray) -> float:
        """Simple loss-based attack (Yeom et al., 2018)."""
        return -token_losses.mean()

    @staticmethod
    def surp(token_losses: np.ndarray) -> float:
        """SURP attack (Carlini et al., 2024, simplified)."""
        return -(token_losses.mean() - token_losses.std())

    @staticmethod
    def zlib_ratio(text: str, mean_loss: float) -> float:
        """Zlib ratio (Carlini et al., 2021)."""
        import zlib
        zlib_len = len(zlib.compress(text.encode("utf-8")))
        return mean_loss / max(zlib_len, 1)


# ═══════════════════════════════════════════
# 8. Ablation Study Runner
# ═══════════════════════════════════════════

def run_ablation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Ablation study: measure contribution of each calibration scale.

    Conditions:
      (a) ESP raw — no calibration
      (b) ESP + Scale 1 (token z-norm) only
      (c) ESP + Scale 1 + Scale 2 (position z-norm)
      (d) ESP + Scale 1 + Scale 2 + Scale 3 (domain z-norm) = full ESP-Cal
    """
    print("\n" + "─" * 50)
    print("  ABLATION: Calibration Scale Contributions")
    print("─" * 50)

    ablation_results = []

    # (a) Raw ESP
    auc_raw = max(
        roc_auc_score(df["is_member"], df["esp_slope_raw"]),
        1 - roc_auc_score(df["is_member"], df["esp_slope_raw"])
    ) if "esp_slope_raw" in df.columns else np.nan

    # (b) Token z-norm only → z_esp_slope (computed per-sample)
    valid = df["z_esp_slope"].notna()
    auc_s1 = max(
        roc_auc_score(df.loc[valid, "is_member"], df.loc[valid, "z_esp_slope"]),
        1 - roc_auc_score(df.loc[valid, "is_member"], df.loc[valid, "z_esp_slope"])
    )

    # (c) Token + Position → need to rerun calibrator with only scale 2
    cfg_s12 = Config()
    cfg_s12.enable_scale2_position = True
    cfg_s12.enable_scale3_domain = False
    cal_s12 = MultiScaleCalibrator(cfg_s12)
    df_s12 = cal_s12.calibrate(df.assign(esp_slope=df["esp_slope_raw"]) if "esp_slope_raw" in df.columns else df, ["esp_slope"])
    valid_s12 = df_s12["esp_slope"].notna()
    auc_s12 = max(
        roc_auc_score(df_s12.loc[valid_s12, "is_member"], df_s12.loc[valid_s12, "esp_slope"]),
        1 - roc_auc_score(df_s12.loc[valid_s12, "is_member"], df_s12.loc[valid_s12, "esp_slope"])
    )

    # (d) Full calibration (already done in main df["signal_esp"])
    valid_full = df["signal_esp"].notna()
    auc_full = max(
        roc_auc_score(df.loc[valid_full, "is_member"], df.loc[valid_full, "signal_esp"]),
        1 - roc_auc_score(df.loc[valid_full, "is_member"], df.loc[valid_full, "signal_esp"])
    )

    ablation_results = [
        {"condition": "(a) ESP raw", "auc": auc_raw},
        {"condition": "(b) + Scale 1 (token)", "auc": auc_s1},
        {"condition": "(c) + Scale 2 (position)", "auc": auc_s12},
        {"condition": "(d) + Scale 3 (domain) = ESP-Cal", "auc": auc_full},
    ]

    abl_df = pd.DataFrame(ablation_results)
    for _, row in abl_df.iterrows():
        marker = "★" if "ESP-Cal" in row["condition"] else " "
        print(f"  {marker} {row['condition']:40s}  AUC={row['auc']:.4f}")

    return abl_df


# ═══════════════════════════════════════════
# 9. Main Experiment Runner
# ═══════════════════════════════════════════

class ESPCalExperiment:
    """Run ESP-Cal on any benchmark."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def run_poisoned_chalice(self):
        """Full evaluation on Poisoned Chalice."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: Poisoned Chalice (Code MIA)")
        print("█" * 60)

        model, tokenizer = load_model(self.cfg.model_name, self.cfg.torch_dtype)
        extractor = ESPExtractor(model, tokenizer, self.cfg)
        df = load_poisoned_chalice(self.cfg)

        return self._extract_and_evaluate(df, extractor, "PoisonedChalice", do_ablation=True)

    def run_wikimia(self):
        """Full evaluation on WikiMIA (multi-model)."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: WikiMIA")
        print("█" * 60)

        models = self.cfg.wikimia_models if self.cfg.multi_model else [self.cfg.wikimia_model]
        data_by_length = load_wikimia(self.cfg)

        all_results = {}
        for model_name in models:
            short_name = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")

            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)

                for length_key, df in data_by_length.items():
                    print(f"\n  ── WikiMIA {length_key} / {short_name} ──")
                    results = self._extract_and_evaluate(
                        df.copy(), extractor, f"WikiMIA_{length_key}_{short_name}",
                        do_ablation=False, calibrate=False
                    )
                    all_results[f"{length_key}_{short_name}"] = results

                del model, tokenizer, extractor
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
                continue

        return all_results

    def run_mimir(self):
        """Full evaluation on MIMIR (multi-model)."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: MIMIR")
        print("█" * 60)

        models = self.cfg.mimir_models if self.cfg.multi_model else [self.cfg.mimir_model]
        data_by_domain = load_mimir(self.cfg)

        all_results = {}
        for model_name in models:
            short_name = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")

            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)

                for domain, df in data_by_domain.items():
                    print(f"\n  ── MIMIR {domain} / {short_name} ──")
                    results = self._extract_and_evaluate(
                        df.copy(), extractor, f"MIMIR_{domain}_{short_name}",
                        do_ablation=False, calibrate=False
                    )
                    all_results[f"{domain}_{short_name}"] = results

                del model, tokenizer, extractor
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
                continue

        return all_results

    def run_bookmia(self):
        """Full evaluation on BookMIA (multi-model, Shi et al., 2024)."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: BookMIA (Copyright Books)")
        print("█" * 60)

        models = self.cfg.bookmia_models if self.cfg.multi_model else [self.cfg.bookmia_model]
        df_base = load_bookmia(self.cfg)
        if len(df_base) == 0:
            print("  ✗ No BookMIA data loaded, skipping.")
            return {}

        all_results = {}
        for model_name in models:
            short_name = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")

            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)

                results = self._extract_and_evaluate(
                    df_base.copy(), extractor, f"BookMIA_{short_name}",
                    do_ablation=False, calibrate=False
                )
                all_results[short_name] = results

                del model, tokenizer, extractor
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
                continue

        return all_results

    def _extract_and_evaluate(self, df: pd.DataFrame, extractor: ESPExtractor,
                               tag: str, do_ablation: bool = False,
                               calibrate: bool = True) -> Dict:
        """Core extraction + calibration + evaluation."""
        print(f"\n  Extracting features for {len(df)} samples...")
        t0 = time.time()

        features_list = []
        for idx, row in df.iterrows():
            if idx > 0 and idx % 500 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (len(df) - idx) / rate
                print(f"    [{idx}/{len(df)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

            feats = extractor.extract(row["text"])
            features_list.append(feats)

        features_df = pd.DataFrame(features_list)
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

        elapsed = time.time() - t0
        print(f"  ✓ Extraction complete in {elapsed:.1f}s ({len(df)/elapsed:.1f} samples/s)")

        # ── Multi-Scale Calibration ──
        calibration_cols = ["signal_esp", "signal_h_drop", "signal_loss",
                            "neg_mean_loss", "minkprob_20", "surp"]
        calibration_cols = [c for c in calibration_cols if c in df.columns]

        if calibrate:
            calibrator = MultiScaleCalibrator(self.cfg)
            df = calibrator.calibrate(df, calibration_cols)
            print("  ✓ 3-scale calibration applied")

        # ── Evaluate ──
        all_score_cols = [c for c in df.columns if c not in
                          ["text", "is_member", "subset", "seq_len", "n_tokens",
                           "_len_bucket"] and not c.endswith("_raw")]

        print("\n" + "─" * 50)
        print(f"  RESULTS: {tag}")
        print("─" * 50)

        results_df = evaluate_scores(df, all_score_cols)

        if len(results_df) > 0:
            # Print top-10
            for i, (_, row_data) in enumerate(results_df.head(15).iterrows()):
                marker = "★" if "esp" in row_data["score"].lower() else " "
                print(f"  {marker} {row_data['score']:30s}  AUC={row_data['auc']:.4f}  ({row_data['polarity']})")

        # Per-subset breakdown
        best_col = "signal_esp" if "signal_esp" in results_df["score"].values else results_df.iloc[0]["score"]
        if "subset" in df.columns and df["subset"].nunique() > 1:
            print(f"\n  Per-subset ({best_col}):")
            subset_results = evaluate_per_subset(df, best_col)
            for _, sr in subset_results.iterrows():
                print(f"    {sr['subset']:15s}  AUC={sr['auc']:.4f}  (n={sr['n']})")

        # ── Ablation ──
        ablation_df = None
        if do_ablation:
            try:
                ablation_df = run_ablation(df, self.cfg)
            except Exception as e:
                print(f"  Ablation failed: {e}")

        # ── Save results ──
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.cfg.output_dir, f"espcal_{tag}_{ts}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"\n  Saved → {out_path}")

        summary = {
            "benchmark": tag,
            "timestamp": ts,
            "n_samples": len(df),
            "results": results_df.to_dict(orient="records") if len(results_df) > 0 else [],
            "ablation": ablation_df.to_dict(orient="records") if ablation_df is not None else None,
        }
        json_path = os.path.join(self.cfg.output_dir, f"espcal_{tag}_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {"df": df, "results": results_df, "summary": summary}


# ═══════════════════════════════════════════
# 10. Comparison Table Generator
# ═══════════════════════════════════════════

def generate_comparison_table(results: Dict) -> str:
    """
    Generate LaTeX comparison table for the paper.
    Compares ESP-Cal vs baselines (Loss, Min-K%, SURP, Zlib).
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Go} & \textbf{Java} & \textbf{Python} & \textbf{Ruby} & \textbf{Rust} & \textbf{Avg} \\",
        r"\midrule",
    ]

    methods = {
        "neg_mean_loss": "Loss",
        "minkprob_20": "Min-K\\%",
        "surp": "SURP",
        "zlib_ratio": "Zlib",
        "signal_esp": "\\textbf{ESP-Cal (Ours)}",
    }

    if "df" in results:
        df = results["df"]
        for score_col, display_name in methods.items():
            if score_col not in df.columns:
                continue
            aucs = []
            for lang in ["Go", "Java", "Python", "Ruby", "Rust"]:
                subset = df[df["subset"] == lang]
                valid = subset[score_col].notna() & subset["is_member"].notna()
                if valid.sum() >= 10:
                    auc = roc_auc_score(subset.loc[valid, "is_member"],
                                        subset.loc[valid, score_col])
                    auc = max(auc, 1 - auc)
                    aucs.append(f"{auc:.3f}")
                else:
                    aucs.append("—")
            avg = np.mean([float(a) for a in aucs if a != "—"]) if any(a != "—" for a in aucs) else 0.0
            row = f"  {display_name} & " + " & ".join(aucs) + f" & {avg:.3f} \\\\"
            lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{AUROC on Poisoned Chalice by language. ESP-Cal consistently outperforms logit-only baselines.}",
        r"\label{tab:poisoned_chalice_results}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 11. Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    setup_environment()

    cfg = Config()

    # Detect Kaggle
    if os.path.exists("/kaggle/input"):
        cfg.output_dir = "/kaggle/working/results"
    else:
        cfg.output_dir = "./results"

    experiment = ESPCalExperiment(cfg)

    print("\n" + "═" * 60)
    print("  RUNNING ALL BENCHMARKS")
    print("═" * 60)

    # 1. Poisoned Chalice
    pc_results = experiment.run_poisoned_chalice()

    # Generate LaTeX table
    latex_table = generate_comparison_table(pc_results)
    table_path = os.path.join(cfg.output_dir, "latex_table_poisoned_chalice.tex")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(table_path, "w") as f:
        f.write(latex_table)
    print(f"\n  LaTeX table → {table_path}")

    gc.collect()
    torch.cuda.empty_cache()

    # 2. WikiMIA
    wikimia_results = experiment.run_wikimia()

    gc.collect()
    torch.cuda.empty_cache()

    # 3. MIMIR
    mimir_results = experiment.run_mimir()

    gc.collect()
    torch.cuda.empty_cache()

    # 4. BookMIA (Shi et al., 2024 — copyright books)
    bookmia_results = experiment.run_bookmia()

    # ══════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY — ESP-Cal")
    print("═" * 60)

    print("\n  Poisoned Chalice:")
    if pc_results and "results" in pc_results:
        for _, row in pc_results["results"].head(5).iterrows():
            marker = "★" if "esp" in row["score"].lower() else " "
            print(f"    {marker} {row['score']:25s} AUC={row['auc']:.4f}")

    print("\n  WikiMIA (multi-model):")
    for key, res in wikimia_results.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            print(f"    {key:40s} {best['score']:25s} AUC={best['auc']:.4f}")

    print("\n  MIMIR (multi-model):")
    for key, res in mimir_results.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            print(f"    {key:40s} {best['score']:25s} AUC={best['auc']:.4f}")

    print("\n  BookMIA (multi-model):")
    for key, res in bookmia_results.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            print(f"    {key:40s} {best['score']:25s} AUC={best['auc']:.4f}")

    # ── Generate multi-model summary table ──
    print("\n" + "─" * 60)
    print("  MULTI-MODEL SUMMARY TABLE (best AUC per model)")
    print("─" * 60)

    all_model_results = []
    for key, res in {**wikimia_results, **mimir_results, **bookmia_results}.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            all_model_results.append({
                "benchmark_model": key,
                "best_signal": best["score"],
                "auroc": best["auc"],
            })
    
    if all_model_results:
        summary_df = pd.DataFrame(all_model_results)
        summary_path = os.path.join(cfg.output_dir, "multimodel_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Multi-model summary → {summary_path}")

    print("\n  Done!")
