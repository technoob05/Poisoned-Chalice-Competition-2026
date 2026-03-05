"""
EXPERIMENT 34: Block-wise Privacy Leakage Signature (Gradient Resonance Across Layers)

Paper inspiration:
    "Membership and Memorization in LLM Knowledge Distillation" (EMNLP 2025)
    — Per-block privacy analysis framework: privacy leakage varies enormously
      across transformer blocks; memorization creates RESONANCE patterns where
      specific distant blocks simultaneously exhibit low gradient activity.

Problem with EXP13 / EXP22:
    - EXP13: simple mean of all layer gradient norms → destroys inter-layer structure.
    - EXP22: 5 hand-picked layers → ad-hoc, misses resonance between distant blocks.

Innovation (this experiment):
    Extract the FULL per-block gradient norm vector (one scalar per transformer layer)
    and derive structural features that capture JOINT behaviour across layers:

    Feature set extracted from the N-dimensional gradient norm vector G = [g_0, …, g_{N-1}]:

    1. Resonance Pair Score (RPS):
       The maximum pairwise Pearson correlation between the K lowest-norm blocks
       and any other block. Members have "co-resonating" distant blocks (e.g.,
       block 2 and block 30 both crash simultaneously). Non-members don't.

    2. Inter-Block Variance Ratio (IBVR):
       var(G_low_half) / var(G_high_half)  where low/high = bottom/top half by index.
       Members: high variance in early layers (shallow memorization resonance),
       low in late layers → ratio > 1.

    3. Entropy of the normalised gradient profile:
       H = -sum(p_i * log(p_i))  where p_i = g_i / sum(g)
       Members: gradient energy concentrates in a few layers → LOW entropy.

    4. Top-K Gradient Index Pattern (TKGIP):
       The POSITIONS of the K smallest gradient norms in the N-layer vector.
       Members tend to have consistent low-gradient positions in specific
       architecture sub-regions (we encode as mean of top-K positions
       normalised to [0,1] — a proxy for WHERE the memorization lives).

    5. Cross-Block Correlation (CBC):
       Mean of the off-diagonal elements of the inter-block Pearson correlation
       matrix of G. High mean CBC → many blocks co-resonate → member.

    All 5 features + raw per-layer norms are saved as columns for EXP15 XGBoost stacking.

Primary score:
    rank_avg(-entropy_grad, rps, -ibvr_inv, cbc)

Compute notes:
    - 1 backward pass per sample (same as EXP11/EXP22) — gradient norms for ALL layers.
    - All N transformer layers + embedding + head = N+2 components.
    - No memory overhead beyond what EXP13 already uses.
    - sample_fraction=0.10 recommended (~3.5h on A100).

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
from scipy.stats import rankdata, pearsonr
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
    print("  EXP34: BLOCK-WISE PRIVACY LEAKAGE SIGNATURE (Gradient Resonance)")
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
# Model Loading
# ============================================================================

def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# Block-wise Privacy Leakage Signature Attack
# ============================================================================

class BlockwisePrivacySignatureAttack:
    """
    Extracts the FULL per-layer gradient norm profile and computes structural
    features capturing resonance patterns that indicate memorization.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.top_k_resonance = getattr(args, "top_k_resonance", 5)  # K for RPS & TKGIP
        self._err_count = 0

        # Build ordered list of (name, module) for all gradient components
        self.components = self._build_components()
        print(f"[EXP34] Total gradient components: {len(self.components)}")

    def _build_components(self) -> List[Tuple[str, torch.nn.Module]]:
        """
        Returns ordered list: [embed, layer_0, layer_1, ..., layer_{N-1}, head]
        Each entry: (component_name, module).
        """
        comps: List[Tuple[str, torch.nn.Module]] = []

        # Embedding
        embed = self.model.get_input_embeddings()
        comps.append(("embed", embed))

        # All transformer layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                comps.append((f"layer_{i:02d}", layer))

        # LM Head
        head = self.model.get_output_embeddings()
        comps.append(("head", head))

        return comps

    @property
    def name(self) -> str:
        return "blockwise_privacy_signature"

    def _rms_grad_norm(self, module: torch.nn.Module) -> float:
        """RMS of parameter gradient L2 norms within a module."""
        norms = [p.grad.norm(2).item()
                 for p in module.parameters()
                 if p.grad is not None]
        return float(np.sqrt(np.mean(np.square(norms)))) if norms else np.nan

    def compute_gradient_profile(self, text: str) -> Optional[np.ndarray]:
        """
        One backward pass → returns N-dimensional gradient norm vector.
        Returns None on failure.
        """
        if not text or len(text) < 20:
            return None
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)

            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()

            profile = np.array([
                self._rms_grad_norm(module) for _, module in self.components
            ], dtype=np.float32)

            self.model.zero_grad()
            return profile

        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP34 WARNING] gradient_profile error: {type(e).__name__}: {e}")
            self._err_count += 1
            return None

    def _profile_features(self, G: np.ndarray) -> Dict[str, float]:
        """
        Compute structural features from the N-dim gradient profile vector G.
        """
        N = len(G)
        K = min(self.top_k_resonance, N // 2)
        eps = 1e-9

        # --- Replace NaN with median for robust computation ---
        median_g = float(np.nanmedian(G))
        G_clean = np.where(np.isnan(G), median_g, G)

        # Sorted indices: ascending norm (low-norm = potentially resonating)
        sorted_idx = np.argsort(G_clean)

        # 1. Entropy of normalised gradient profile
        p = G_clean / (G_clean.sum() + eps)
        p = np.clip(p, eps, 1.0)
        entropy_grad = float(-np.sum(p * np.log(p)))

        # 2. Resonance Pair Score (RPS)
        # Pearson correlation between each "low-norm block" and the rest
        low_k_idx = sorted_idx[:K]
        low_k_vals = G_clean[low_k_idx]
        rps = 0.0
        n_corr = 0
        for i in range(N):
            if i in low_k_idx:
                continue
            if N < 3:
                break
            # Build two paired vectors: [g_low_0, g_low_1, ...g_low_{K-1}]
            # vs [g_i, g_i, ...] won't work for Pearson. Instead correlate
            # the FULL low half vs high half across positions.
            # Simpler: correlation between G_clean[0:N//2] and G_clean[N//2:]
        # Better formulation: top-K low vs top-K high block
        high_k_idx = sorted_idx[-K:]
        high_k_vals = G_clean[high_k_idx]
        if K >= 3:
            try:
                r, _ = pearsonr(low_k_vals, high_k_vals)
                rps = float(r)
            except Exception:
                rps = 0.0

        # 3. Inter-Block Variance Ratio (IBVR)
        # Variance in the "early" vs "late" halves of the gradient profile
        # (excluding embed and head for cleaner transformer-only signal)
        transformer_G = G_clean[1:-1]  # exclude embed (idx 0) and head (idx -1)
        n_tr = len(transformer_G)
        if n_tr >= 4:
            half = n_tr // 2
            var_early = float(np.var(transformer_G[:half]))
            var_late = float(np.var(transformer_G[half:]))
            ibvr = var_early / (var_late + eps)
        else:
            ibvr = np.nan

        # 4. Top-K Gradient Index Pattern (TKGIP)
        # Mean of normalised positions of K lowest-gradient transformer layers
        tr_sorted = np.argsort(transformer_G) if n_tr >= K else np.arange(n_tr)
        tkgip = float(np.mean(tr_sorted[:K]) / max(n_tr - 1, 1))

        # 5. Cross-Block Correlation (CBC)
        # Mean off-diagonal of Pearson correlation matrix of G
        # (Only feasible if N is small-ish; cap at first 20 entries)
        G_sub = transformer_G[:min(n_tr, 20)]
        n_sub = len(G_sub)
        cbc = 0.0
        if n_sub >= 4:
            # Build correlation matrix via outer product of z-scores
            z = (G_sub - G_sub.mean()) / (G_sub.std() + eps)
            corr_mat = np.outer(z, z) / n_sub
            # Mean of upper triangle (off-diagonal)
            upper = corr_mat[np.triu_indices(n_sub, k=1)]
            cbc = float(np.mean(upper))

        # Raw aggregate stats
        return {
            "entropy_grad": entropy_grad,    # low = concentrated = member
            "rps":           rps,             # high positive = resonance = member
            "ibvr":          ibvr,            # >1 = early variance > late = member
            "tkgip":         tkgip,           # position of minimum (varies)
            "cbc":           cbc,             # high = co-resonating = member
            "g_mean":        float(np.mean(G_clean)),
            "g_std":         float(np.std(G_clean)),
            "g_min":         float(np.min(G_clean)),
            "g_max":         float(np.max(G_clean)),
            "g_cv":          float(np.std(G_clean) / (np.mean(G_clean) + eps)),  # coeff of variation
        }

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP34] Processing {len(texts)} samples…")
        print(f"[EXP34] Components: {len(self.components)}  "
              f"(embed + {len(self.components)-2} layers + head)")
        rows = []

        for text in tqdm(texts, desc="[EXP34] Block Privacy Signature"):
            G = self.compute_gradient_profile(text)
            if G is not None and not np.all(np.isnan(G)):
                feat = self._profile_features(G)
                # Store raw per-layer norms for XGBoost stacking
                for (cname, _), val in zip(self.components, G):
                    feat[f"gnorm_{cname}"] = float(val)
            else:
                feat = {
                    "entropy_grad": np.nan, "rps": np.nan, "ibvr": np.nan,
                    "tkgip": np.nan, "cbc": np.nan,
                    "g_mean": np.nan, "g_std": np.nan, "g_min": np.nan,
                    "g_max": np.nan, "g_cv": np.nan,
                }
            rows.append(feat)

        df = pd.DataFrame(rows)

        # ---- Member signals ----
        # entropy_grad: lower = concentrated = member → signal = -entropy
        # rps: higher positive correlation = member
        # ibvr: >1 = early > late variance = member
        # cbc: higher = co-resonation = member
        if "entropy_grad" in df.columns:
            df["signal_entropy"]  = -df["entropy_grad"]
        if "rps" in df.columns:
            df["signal_rps"]      = df["rps"]
        if "ibvr" in df.columns:
            df["signal_ibvr"]     = df["ibvr"]
        if "cbc" in df.columns:
            df["signal_cbc"]      = df["cbc"]
        if "g_mean" in df.columns:
            df["signal_g_mean"]   = -df["g_mean"]   # low mean grad = flat minimum = member

        # ---- Combined rank score ----
        rank_sources = ["signal_entropy", "signal_rps", "signal_cbc", "signal_g_mean"]
        valid_rank_cols = [c for c in rank_sources if c in df.columns]
        if valid_rank_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_rank_cols:
                vals = df[col].fillna(df[col].min())
                ranks = rankdata(vals, method="average")
                rank_sum += ranks / len(ranks)
            df["combined_rank_score"] = rank_sum / len(valid_rank_cols)

        n_valid = df["combined_rank_score"].notna().sum() if "combined_rank_score" in df.columns else 0
        print(f"[EXP34] Valid samples: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP34] Total errors: {self._err_count}")
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
        attacker = BlockwisePrivacySignatureAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP34_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")
        print("[*] NOTE: parquet contains ALL per-layer gnorm_* features for EXP15.")

        print("\n" + "="*65)
        print("  EXP34: BLOCK-WISE PRIVACY SIGNATURE — PERFORMANCE REPORT")
        print("="*65)

        score_candidates = {
            "combined_rank_score": "Rank-Avg(Entropy + RPS + CBC + G_mean)  [PRIMARY]",
            "signal_entropy":      "-Gradient Profile Entropy",
            "signal_rps":          "Resonance Pair Score (RPS)",
            "signal_cbc":          "Cross-Block Correlation (CBC)",
            "signal_g_mean":       "-Mean Gradient Norm",
            "signal_ibvr":         "Inter-Block Variance Ratio (IBVR)",
        }
        report = {
            "experiment": "EXP34_blockwise_privacy_signature",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "n_components": len(attacker.components),
            "top_k_resonance": self.args.top_k_resonance,
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
                tag = " ← PRIMARY" if "combined" in score_col else ""
                print(f"  {label:<52} AUC = {auc:.4f}{tag}")

        print(f"\nGradient profile statistics:")
        for col in ["entropy_grad", "rps", "cbc", "ibvr", "g_mean"]:
            if col in df.columns:
                m_val = df[df["is_member"] == 1][col].mean()
                nm_val = df[df["is_member"] == 0][col].mean()
                print(f"  {col:<18}: Member={m_val:.4f}  Non-member={nm_val:.4f}")

        # Find the single most discriminative per-layer gradient norm
        gnorm_cols = [c for c in df.columns if c.startswith("gnorm_")]
        if gnorm_cols:
            layer_aucs = {}
            for col in gnorm_cols:
                valid = df.dropna(subset=[col])
                if len(valid["is_member"].unique()) > 1:
                    auc = roc_auc_score(valid["is_member"], -valid[col])
                    layer_aucs[col] = auc
            if layer_aucs:
                top5 = sorted(layer_aucs, key=layer_aucs.get, reverse=True)[:5]
                print(f"\nTop-5 most discriminative individual layer gradients:")
                for c in top5:
                    print(f"  {c:<25}: AUC = {layer_aucs[c]:.4f}")
                report["top_individual_layers"] = {c: layer_aucs[c] for c in top5}

        print(f"\n{'Subset':<10} | {'CombinedAUC':<13} | {'Entropy':<10} | {'CBC':<10} | N")
        print("-"*55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["combined_rank_score", "signal_entropy", "signal_cbc"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('combined_rank_score', float('nan')):.4f}        "
                  f"| {r.get('signal_entropy', float('nan')):.4f}     "
                  f"| {r.get('signal_cbc', float('nan')):.4f}     | {len(sub)}")
            report["subset_aucs"][subset] = r

        print("="*65)
        print("\nInterpretation:")
        print("  Low entropy + High CBC → gradient energy concentrated + co-resonating blocks → member")
        print("  High entropy + Low CBC → gradient energy spread uniformly → non-member")

        report_path = self.output_dir / f"EXP34_report_{timestamp}.json"
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
        # 1 backward pass per sample — same speed as EXP11
        sample_fraction = 0.10
        output_dir = "results"
        max_length = 2048
        top_k_resonance = 5     # K lowest-gradient layers used for RPS & TKGIP
        seed = 42

    print(f"[EXP34] Model           : {Args.model_name}")
    print(f"[EXP34] Sample          : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP34] Top-K resonance : {Args.top_k_resonance}")
    Experiment(Args).run()
