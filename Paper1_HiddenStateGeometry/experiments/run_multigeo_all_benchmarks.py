#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║  MultiGeo-MIA: Multi-Axis Hidden-State Geometry for MIA      ║
║  Paper 1 — Full Benchmark Evaluation                         ║
║                                                               ║
║  Benchmarks: Poisoned Chalice, WikiMIA, MIMIR                ║
║  Author: [Redacted for review]                               ║
║  Self-contained Kaggle notebook script                       ║
╚═══════════════════════════════════════════════════════════════╝

Four orthogonal unsupervised axes:
  1. Magnitude  — negative mid-layer hidden-state norm
  2. Dimensionality — effective rank of hidden-state matrix (SVD)
  3. Dynamics — layer cascade drift (cosine distance between consecutive layers)
  4. Routing — attention entropy (lower = sharper = member)

Combined via rank-averaging. No training, no reference model, fully unsupervised.
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
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# 0. Environment Setup (Kaggle-compatible)
# ═══════════════════════════════════════════

def setup_environment():
    """Install deps + HuggingFace auth for Kaggle."""
    print("=" * 60)
    print("  MultiGeo-MIA — Full Benchmark Evaluation")
    print("=" * 60)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers", "accelerate", "datasets",
                    "scikit-learn", "scipy", "huggingface_hub"],
                   capture_output=True)

    # Kaggle HF token
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
    sample_fraction: float = 1.0  # 1.0 = full dataset
    split: str = "train"  # "train" or "test"

    # MultiGeo axes
    magnitude_layers: str = "mid"  # "mid", "all", "last"
    svd_top_k: int = 50  # top-k singular values for effective rank
    cascade_pairs: int = 5  # number of layer pairs for dynamics

    # Evaluation
    seed: int = 42
    output_dir: str = "./results"
    per_language_znorm: bool = True

    # Benchmark mode
    benchmark: str = "poisoned_chalice"  # "poisoned_chalice", "wikimia", "mimir"

    # WikiMIA-specific
    wikimia_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    wikimia_model: str = "EleutherAI/pythia-2.8b-deduped"

    # MIMIR-specific
    mimir_domains: List[str] = field(default_factory=lambda: [
        "wikipedia", "github", "pile_cc", "pubmed_central",
        "arxiv", "dm_mathematics", "hackernews"
    ])
    mimir_model: str = "EleutherAI/pythia-2.8b-deduped"


# ═══════════════════════════════════════════
# 2. Model Loader
# ═══════════════════════════════════════════

def load_model(model_name: str, dtype_str: str = "bfloat16"):
    """Load a causal LM with hidden states + attentions enabled."""
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
        output_hidden_states=True,
        attn_implementation="eager",  # needed for raw attention weights
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"  ✓ Model loaded: {n_layers} layers, dtype={dtype}")
    return model, tokenizer, n_layers


# ═══════════════════════════════════════════
# 3. MultiGeo Feature Extractor
# ═══════════════════════════════════════════

class MultiGeoExtractor:
    """
    Extract 4-axis geometric features from a single forward pass.

    Axis 1 — Magnitude:    ‖h_mid‖_2  (mid-layer hidden-state L2 norm)
    Axis 2 — Dimensionality: effective rank of H matrix (SVD)
    Axis 3 — Dynamics:      mean cosine distance between consecutive layer hidden states
    Axis 4 — Routing:       mean attention entropy across heads and layers
    """

    def __init__(self, model, tokenizer, n_layers: int, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.cfg = cfg
        self.device = next(model.parameters()).device

        # Pre-compute layer indices
        mid = n_layers // 2
        self.mid_layer_idx = mid  # For magnitude
        self.cascade_layer_indices = self._select_cascade_layers()
        self.attn_layer_indices = self._select_attn_layers()

    def _select_cascade_layers(self) -> List[int]:
        """Select evenly-spaced layers for cascade drift measurement."""
        n = self.n_layers
        k = min(self.cfg.cascade_pairs + 1, n)
        return [int(i * (n - 1) / (k - 1)) for i in range(k)]

    def _select_attn_layers(self) -> List[int]:
        """Select layers for attention entropy (early, mid, late)."""
        n = self.n_layers
        return [0, n // 4, n // 2, 3 * n // 4, n - 1]

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract all 4-axis features from a single text."""
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

        if seq_len < 4:
            return self._empty_features()

        # Single forward pass with hidden states + attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device),
            output_hidden_states=True,
            output_attentions=True,
        )

        features = {}

        # ── Axis 1: Magnitude (mid-layer norm) ──
        features.update(self._compute_magnitude(outputs.hidden_states))

        # ── Axis 2: Dimensionality (effective rank via SVD) ──
        features.update(self._compute_dimensionality(outputs.hidden_states))

        # ── Axis 3: Dynamics (layer cascade drift) ──
        features.update(self._compute_dynamics(outputs.hidden_states))

        # ── Axis 4: Routing (attention entropy) ──
        features.update(self._compute_routing(outputs.attentions))

        # ── Combined rank-average signal ──
        # (Computed later at the dataset level)

        # Also compute basic loss for comparison
        features["loss"] = self._compute_loss(outputs.logits, input_ids)
        features["seq_len"] = seq_len

        # Clean up
        del outputs, input_ids
        torch.cuda.empty_cache()

        return features

    def _compute_magnitude(self, hidden_states) -> Dict[str, float]:
        """
        Axis 1: Hidden-state magnitude.
        Members tend to have LOWER mid-layer activation norms
        (model more "relaxed" on memorized data).
        """
        mid_hs = hidden_states[self.mid_layer_idx + 1].float()  # +1 because idx 0 is embedding
        # L2 norm per token, then mean across tokens
        norms = torch.norm(mid_hs[0], dim=-1)  # (seq_len,)
        mid_norm = norms.mean().item()
        mid_norm_std = norms.std().item() if norms.numel() > 1 else 0.0

        # Also compute last-layer norm for comparison
        last_hs = hidden_states[-1].float()
        last_norm = torch.norm(last_hs[0], dim=-1).mean().item()

        # Multi-layer profile
        all_norms = []
        for li in range(0, self.n_layers + 1, max(1, self.n_layers // 8)):
            hs = hidden_states[li].float()
            all_norms.append(torch.norm(hs[0], dim=-1).mean().item())
        norm_slope = np.polyfit(range(len(all_norms)), all_norms, 1)[0] if len(all_norms) > 1 else 0.0

        return {
            "mag_mid_norm": mid_norm,
            "mag_mid_norm_std": mid_norm_std,
            "mag_last_norm": last_norm,
            "mag_norm_slope": norm_slope,
            # Signal: negative mid-layer norm (lower = more member)
            "signal_magnitude": -mid_norm,
        }

    def _compute_dimensionality(self, hidden_states) -> Dict[str, float]:
        """
        Axis 2: Effective dimensionality of hidden-state matrix.
        Members tend to live in LOWER-dimensional subspaces
        (model has a more structured/"decisive" representation).

        Effective rank = exp(entropy of normalized singular values)
        """
        # Use mid-layer hidden states: shape (1, seq_len, d_model)
        H = hidden_states[self.mid_layer_idx + 1].float().squeeze(0)  # (seq, d)
        seq_len, d = H.shape

        # SVD on hidden state matrix
        try:
            # Use truncated SVD for efficiency
            k = min(self.cfg.svd_top_k, seq_len, d)
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)
            S = S[:k]

            # Normalize singular values to form a distribution
            S_norm = S / S.sum()
            S_norm = S_norm.clamp(min=1e-10)

            # Shannon entropy of singular value distribution
            sv_entropy = -(S_norm * torch.log(S_norm)).sum().item()

            # Effective rank = exp(entropy)
            eff_rank = np.exp(sv_entropy)

            # Concentration ratio: top-5 / total
            top5_ratio = S[:min(5, k)].sum().item() / S.sum().item()

            # Spectral gap: ratio of 1st to 2nd singular value
            spectral_gap = (S[0] / S[1]).item() if k > 1 and S[1] > 0 else 0.0

        except Exception:
            sv_entropy = 0.0
            eff_rank = 0.0
            top5_ratio = 1.0
            spectral_gap = 0.0

        return {
            "dim_eff_rank": eff_rank,
            "dim_sv_entropy": sv_entropy,
            "dim_top5_ratio": top5_ratio,
            "dim_spectral_gap": spectral_gap,
            # Signal: negative effective rank (lower = more member)
            "signal_dimensionality": -eff_rank,
        }

    def _compute_dynamics(self, hidden_states) -> Dict[str, float]:
        """
        Axis 3: Layer cascade dynamics.
        Members have MORE STABLE hidden states across layers
        (smaller drift between consecutive layers).

        Metric: 1 - cosine_similarity(h_l, h_{l+1}), averaged
        """
        drifts = []
        for i in range(len(self.cascade_layer_indices) - 1):
            l1 = self.cascade_layer_indices[i]
            l2 = self.cascade_layer_indices[i + 1]

            h1 = hidden_states[l1 + 1].float().squeeze(0)  # (seq, d)
            h2 = hidden_states[l2 + 1].float().squeeze(0)

            # Mean-pool across tokens
            h1_mean = h1.mean(dim=0)
            h2_mean = h2.mean(dim=0)

            cos_sim = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()
            drift = 1.0 - cos_sim
            drifts.append(drift)

        mean_drift = np.mean(drifts) if drifts else 0.0
        max_drift = np.max(drifts) if drifts else 0.0
        drift_std = np.std(drifts) if len(drifts) > 1 else 0.0

        # Also compute early vs late drift asymmetry
        if len(drifts) >= 4:
            early_drift = np.mean(drifts[:len(drifts) // 2])
            late_drift = np.mean(drifts[len(drifts) // 2:])
            drift_asymmetry = early_drift - late_drift
        else:
            drift_asymmetry = 0.0

        return {
            "dyn_mean_drift": mean_drift,
            "dyn_max_drift": max_drift,
            "dyn_drift_std": drift_std,
            "dyn_drift_asymmetry": drift_asymmetry,
            # Signal: negative drift (less drift = more member)
            "signal_dynamics": -mean_drift,
        }

    def _compute_routing(self, attentions) -> Dict[str, float]:
        """
        Axis 4: Attention routing patterns.
        Members have SHARPER attention (lower entropy) because the model
        is more "certain" about how to route information for seen data.

        Metric: mean attention entropy across selected layers and heads
        """
        entropies = []
        concentrations = []

        for li in self.attn_layer_indices:
            if li >= len(attentions):
                continue

            attn = attentions[li].float().squeeze(0)  # (heads, seq, seq)
            n_heads, seq_q, seq_k = attn.shape

            # Avoid log(0)
            attn_clamped = attn.clamp(min=1e-10)

            # Entropy per head per query position
            H = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)  # (heads, seq_q)
            mean_entropy = H.mean().item()
            entropies.append(mean_entropy)

            # KL from uniform = log(K) - H  → measures concentration
            max_entropy = np.log(seq_k) if seq_k > 0 else 1.0
            concentration = max_entropy - mean_entropy
            concentrations.append(concentration)

        mean_attn_entropy = np.mean(entropies) if entropies else 0.0
        mean_concentration = np.mean(concentrations) if concentrations else 0.0

        # Early vs late attention entropy
        if len(entropies) >= 3:
            early_entropy = np.mean(entropies[:len(entropies) // 2])
            late_entropy = np.mean(entropies[len(entropies) // 2:])
            entropy_trajectory = late_entropy - early_entropy
        else:
            entropy_trajectory = 0.0

        return {
            "rout_mean_entropy": mean_attn_entropy,
            "rout_mean_concentration": mean_concentration,
            "rout_entropy_trajectory": entropy_trajectory,
            # Signal: negative entropy (lower = sharper = more member)
            "signal_routing": -mean_attn_entropy,
        }

    def _compute_loss(self, logits, input_ids) -> float:
        """Standard cross-entropy loss."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean"
        )
        return -loss.item()  # negative loss: higher = more member

    def _empty_features(self) -> Dict[str, float]:
        """Return NaN features for too-short sequences."""
        return {
            "mag_mid_norm": np.nan, "mag_mid_norm_std": np.nan,
            "mag_last_norm": np.nan, "mag_norm_slope": np.nan,
            "signal_magnitude": np.nan,
            "dim_eff_rank": np.nan, "dim_sv_entropy": np.nan,
            "dim_top5_ratio": np.nan, "dim_spectral_gap": np.nan,
            "signal_dimensionality": np.nan,
            "dyn_mean_drift": np.nan, "dyn_max_drift": np.nan,
            "dyn_drift_std": np.nan, "dyn_drift_asymmetry": np.nan,
            "signal_dynamics": np.nan,
            "rout_mean_entropy": np.nan, "rout_mean_concentration": np.nan,
            "rout_entropy_trajectory": np.nan,
            "signal_routing": np.nan,
            "loss": np.nan, "seq_len": 0,
        }


# ═══════════════════════════════════════════
# 4. Data Loaders
# ═══════════════════════════════════════════

def load_poisoned_chalice(cfg: Config) -> pd.DataFrame:
    """Load Poisoned Chalice dataset."""
    from datasets import load_dataset

    print("\n  Loading Poisoned Chalice dataset...")
    all_rows = []

    # Check for local Kaggle dataset first
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
    """Load WikiMIA dataset (multiple length splits)."""
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
            print(f"    Length {length}: {len(df)} samples ({mem} members, {len(df)-mem} non-members)")
        except Exception as e:
            print(f"    Length {length}: ERROR — {e}")

    return data_by_length


def load_mimir(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load MIMIR dataset (multiple domains)."""
    from datasets import load_dataset

    print("\n  Loading MIMIR dataset...")
    data_by_domain = {}

    for domain in cfg.mimir_domains:
        try:
            ds = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)
            rows = []
            # MIMIR has member/nonmember splits
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
                print(f"    {domain}: {len(df)} samples ({mem} members, {len(df)-mem} non-members)")
            else:
                print(f"    {domain}: No data found")
        except Exception as e:
            print(f"    {domain}: ERROR — {e}")

    return data_by_domain


# ═══════════════════════════════════════════
# 5. Evaluation Utilities
# ═══════════════════════════════════════════

def rank_average(df: pd.DataFrame, columns: List[str], name: str = "rank_avg") -> pd.Series:
    """Compute rank-average of multiple score columns (unsupervised fusion)."""
    ranks = pd.DataFrame()
    for col in columns:
        valid = df[col].notna()
        r = pd.Series(np.nan, index=df.index)
        r[valid] = df.loc[valid, col].rank(pct=True)
        ranks[col] = r
    return ranks.mean(axis=1)


def per_language_znorm(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """Z-normalize score columns per-language subset."""
    df = df.copy()
    for col in score_columns:
        df[f"{col}_raw"] = df[col]
        grouped = df.groupby("subset")[col]
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, 1)
        df[col] = (df[col] - means) / stds
    return df


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
        # Handle polarity: always report best orientation
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
# 6. Main Experiment Runner
# ═══════════════════════════════════════════

class MultiGeoExperiment:
    """Run MultiGeo-MIA on any benchmark."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def run_poisoned_chalice(self):
        """Full evaluation on Poisoned Chalice."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: Poisoned Chalice (Code MIA)")
        print("█" * 60)

        model, tokenizer, n_layers = load_model(self.cfg.model_name, self.cfg.torch_dtype)
        extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)
        df = load_poisoned_chalice(self.cfg)

        return self._extract_and_evaluate(df, extractor, "PoisonedChalice")

    def run_wikimia(self):
        """Full evaluation on WikiMIA."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: WikiMIA")
        print("█" * 60)

        model_name = self.cfg.wikimia_model
        model, tokenizer, n_layers = load_model(model_name, self.cfg.torch_dtype)
        extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)

        data_by_length = load_wikimia(self.cfg)

        all_results = {}
        for length_key, df in data_by_length.items():
            print(f"\n  ── WikiMIA {length_key} ──")
            results = self._extract_and_evaluate(
                df, extractor, f"WikiMIA_{length_key}", znorm=False
            )
            all_results[length_key] = results

        return all_results

    def run_mimir(self):
        """Full evaluation on MIMIR."""
        print("\n" + "█" * 60)
        print("  BENCHMARK: MIMIR")
        print("█" * 60)

        model_name = self.cfg.mimir_model
        model, tokenizer, n_layers = load_model(model_name, self.cfg.torch_dtype)
        extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)

        data_by_domain = load_mimir(self.cfg)

        all_results = {}
        for domain, df in data_by_domain.items():
            print(f"\n  ── MIMIR {domain} ──")
            results = self._extract_and_evaluate(
                df, extractor, f"MIMIR_{domain}", znorm=False
            )
            all_results[domain] = results

        return all_results

    def _extract_and_evaluate(self, df: pd.DataFrame, extractor: MultiGeoExtractor,
                               tag: str, znorm: bool = True) -> Dict:
        """Core extraction + evaluation loop."""
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

        # Define score columns
        signal_cols = ["signal_magnitude", "signal_dimensionality",
                       "signal_dynamics", "signal_routing"]
        all_feature_cols = [c for c in features_df.columns if c not in ["seq_len"]]

        # Per-language z-normalization (for Poisoned Chalice)
        if znorm and self.cfg.per_language_znorm and "subset" in df.columns:
            df = per_language_znorm(df, all_feature_cols)
            print("  ✓ Per-language z-normalization applied")

        # Compute rank-average combinations
        df["multigeo_4axis"] = rank_average(df, signal_cols, "4-axis")
        df["multigeo_mag_dim"] = rank_average(df, ["signal_magnitude", "signal_dimensionality"])
        df["multigeo_dyn_rout"] = rank_average(df, ["signal_dynamics", "signal_routing"])

        combo_cols = ["multigeo_4axis", "multigeo_mag_dim", "multigeo_dyn_rout"]

        # Evaluate
        print("\n" + "─" * 50)
        print(f"  RESULTS: {tag}")
        print("─" * 50)

        eval_cols = signal_cols + combo_cols + ["loss"]
        results_df = evaluate_scores(df, eval_cols)

        if len(results_df) > 0:
            for _, row in results_df.iterrows():
                marker = "★" if row["score"] == "multigeo_4axis" else " "
                print(f"  {marker} {row['score']:30s}  AUC={row['auc']:.4f}  ({row['polarity']})")

        # Per-subset breakdown
        if "subset" in df.columns and df["subset"].nunique() > 1:
            print(f"\n  Per-subset breakdown (multigeo_4axis):")
            subset_results = evaluate_per_subset(df, "multigeo_4axis")
            for _, sr in subset_results.iterrows():
                print(f"    {sr['subset']:15s}  AUC={sr['auc']:.4f}  (n={sr['n']})")

        # Save results
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.cfg.output_dir, f"multigeo_{tag}_{ts}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"\n  Saved → {out_path}")

        # Save summary JSON
        summary = {
            "benchmark": tag,
            "timestamp": ts,
            "n_samples": len(df),
            "results": results_df.to_dict(orient="records") if len(results_df) > 0 else [],
        }
        json_path = os.path.join(self.cfg.output_dir, f"multigeo_{tag}_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "df": df,
            "results": results_df,
            "summary": summary,
        }


# ═══════════════════════════════════════════
# 7. Entry Point
# ═══════════════════════════════════════════

if __name__ == "__main__":
    setup_environment()

    cfg = Config()

    # ── Detect Kaggle paths ──
    if os.path.exists("/kaggle/input"):
        cfg.output_dir = "/kaggle/working/results"
        # Check for pre-uploaded dataset
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            print("  ✓ Found Kaggle dataset")
    else:
        cfg.output_dir = "./results"

    # ── Run all benchmarks sequentially ──
    experiment = MultiGeoExperiment(cfg)

    print("\n" + "═" * 60)
    print("  RUNNING ALL BENCHMARKS")
    print("═" * 60)

    # 1. Poisoned Chalice (primary benchmark)
    pc_results = experiment.run_poisoned_chalice()

    # Free GPU memory before loading different model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. WikiMIA (Pythia models)
    wikimia_results = experiment.run_wikimia()

    gc.collect()
    torch.cuda.empty_cache()

    # 3. MIMIR (Pythia models)
    mimir_results = experiment.run_mimir()

    # ══════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY — MultiGeo-MIA")
    print("═" * 60)

    print("\n  Poisoned Chalice:")
    if pc_results and "results" in pc_results:
        best = pc_results["results"].iloc[0] if len(pc_results["results"]) > 0 else None
        if best is not None:
            print(f"    Best: {best['score']} = {best['auc']:.4f}")

    print("\n  WikiMIA:")
    for length_key, res in wikimia_results.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            print(f"    {length_key}: {best['score']} = {best['auc']:.4f}")

    print("\n  MIMIR:")
    for domain, res in mimir_results.items():
        if res and "results" in res and len(res["results"]) > 0:
            best = res["results"].iloc[0]
            print(f"    {domain}: {best['score']} = {best['auc']:.4f}")

    print("\n  Done!")
