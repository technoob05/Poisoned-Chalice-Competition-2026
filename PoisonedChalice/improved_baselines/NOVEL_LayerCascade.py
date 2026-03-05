"""
NOVEL_LayerCascade.py — Layer Cascade Signature MIA ⭐⭐

NOVEL — derived from our cross-experiment discovery that memorization
manifests DIFFERENTLY at different depths:

    ┌────────────────┬─────────────────────────┬───────────────────────┬──────┐
    │ Depth          │ Strongest Signal        │ Source                │ AUC  │
    ├────────────────┼─────────────────────────┼───────────────────────┼──────┤
    │ Early (L1-8)   │ LUMIA probes (0.70+)    │ EXP51, LUMIA-fast     │ 0.70 │
    │ Mid   (L12-18) │ Norm variance (0.6335)  │ EXP50, NOVEL12        │ 0.63 │
    │ Late  (L24-29) │ Gradient norm (0.6456)  │ EXP30 (PVC all late)  │ 0.65 │
    └────────────────┴─────────────────────────┴───────────────────────┴──────┘

    No existing paper traces HOW geometric properties CASCADE through all layers.

THE CASCADE SIGNATURE:
    For each layer l, compute: norm(l), velocity(l), cosine_drift(l), eff_rank(l)
    Then extract TRAJECTORY features: slope, curvature, convergence point, variance

    Members have:
    - SMOOTHER norm decrease across layers
    - EARLIER velocity convergence (representations "settle" faster)
    - HIGHER rank that INCREASES then plateaus
    - LOWER overall norm trajectory

Compute: 1 forward pass with output_hidden_states=True, ~5 min on A100
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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  LayerCascade-MIA: Layer-Depth Memorization Trajectory")
    print("  Tracing How Memorization Evolves Across Transformer Layers")
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
    return model, tokenizer


class LayerCascadeExtractor:

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

    def _trajectory_features(self, curve: np.ndarray, name: str) -> Dict[str, float]:
        """Extract trajectory features from a per-layer curve."""
        feats = {}
        n = len(curve)
        if n < 3:
            return feats

        feats[f"{name}_mean"] = float(np.mean(curve))
        feats[f"{name}_std"] = float(np.std(curve))
        feats[f"{name}_range"] = float(np.max(curve) - np.min(curve))

        # Slope via linear regression
        x = np.arange(n, dtype=np.float64)
        xm, ym = x.mean(), curve.mean()
        cov = np.sum((x - xm) * (curve - ym))
        var = np.sum((x - xm) ** 2)
        if var > 1e-10:
            slope = cov / var
            feats[f"{name}_slope"] = float(slope)
            # Residual std (curvature/nonlinearity)
            predicted = ym + slope * (x - xm)
            residuals = curve - predicted
            feats[f"{name}_curvature"] = float(np.std(residuals))

        # Early-mid-late breakdown
        third = n // 3
        if third > 0:
            feats[f"{name}_early"] = float(np.mean(curve[:third]))
            feats[f"{name}_mid"] = float(np.mean(curve[third:2*third]))
            feats[f"{name}_late"] = float(np.mean(curve[2*third:]))

        # Convergence: where does the curve plateau? (std of last 1/3 vs first 1/3)
        if third > 1:
            feats[f"{name}_convergence"] = float(
                np.std(curve[2*third:]) / max(np.std(curve[:third]), 1e-8))

        return feats

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        features = {}
        if not text or len(text) < 20:
            return features
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 5:
                return features

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            n_layers = len(hidden_states) - 1
            T = hidden_states[0].shape[1]

            # ── Per-layer curves ──────────────────────────────────────────
            norm_curve = []        # mean activation norm at each layer
            norm_std_curve = []    # std of activation norm across tokens
            velocity_curve = []    # ||h_{l+1} - h_l|| (inter-layer change)
            cos_drift_curve = []   # 1 - cosine_sim(h_l, h_{l+1})

            for l in range(n_layers + 1):
                hs = hidden_states[l][0].float()  # (T, D)
                per_token_norm = hs.norm(dim=-1)   # (T,)
                norm_curve.append(per_token_norm.mean().item())
                norm_std_curve.append(per_token_norm.std().item())

                if l > 0:
                    h_prev = hidden_states[l - 1][0].float()
                    diff = hs - h_prev
                    vel = diff.norm(dim=-1).mean().item()
                    velocity_curve.append(vel)

                    cos = torch.nn.functional.cosine_similarity(
                        hs.mean(dim=0, keepdim=True),
                        h_prev.mean(dim=0, keepdim=True), dim=-1
                    ).item()
                    cos_drift_curve.append(1.0 - cos)

            # ── SVD rank at sampled layers ────────────────────────────────
            rank_curve = []
            sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 6)))
            for l in sample_layers:
                hs = hidden_states[l][0].float()
                hs_c = hs - hs.mean(dim=0, keepdim=True)
                n_tok = min(T, 128)
                try:
                    sv = torch.linalg.svdvals(hs_c[:n_tok])
                    sv_np = sv.cpu().numpy()
                    sv_np = sv_np[sv_np > 1e-10]
                    if len(sv_np) > 0:
                        sv_sq = sv_np ** 2
                        p = sv_sq / sv_sq.sum()
                        er = np.exp(-np.sum(p * np.log(p + 1e-12)))
                        rank_curve.append(er)
                except Exception:
                    pass

            # ── Extract trajectory features ───────────────────────────────
            features.update(self._trajectory_features(
                np.array(norm_curve), "neg_norm"))
            # Negate: lower norm = member → neg makes higher = member
            for k in list(features.keys()):
                if k.startswith("neg_norm_"):
                    features[k] = -features[k]

            features.update(self._trajectory_features(
                np.array(norm_std_curve), "norm_std"))

            if velocity_curve:
                features.update(self._trajectory_features(
                    np.array(velocity_curve), "neg_vel"))
                for k in list(features.keys()):
                    if k.startswith("neg_vel_"):
                        features[k] = -features[k]

            if cos_drift_curve:
                features.update(self._trajectory_features(
                    np.array(cos_drift_curve), "neg_drift"))
                for k in list(features.keys()):
                    if k.startswith("neg_drift_"):
                        features[k] = -features[k]

            if len(rank_curve) > 2:
                features.update(self._trajectory_features(
                    np.array(rank_curve), "rank"))

            # ── Cross-trajectory features ─────────────────────────────────
            # Norm-velocity correlation: do norms and velocities move together?
            if len(velocity_curve) >= 3 and len(norm_curve) >= 4:
                nc = np.array(norm_curve[1:])[:len(velocity_curve)]
                vc = np.array(velocity_curve)[:len(nc)]
                if len(nc) == len(vc) and len(nc) > 2:
                    corr = np.corrcoef(nc, vc)[0, 1]
                    if not np.isnan(corr):
                        features["norm_vel_corr"] = float(corr)

            # Loss baseline
            logits = outputs.logits
            shift_lp = torch.nn.functional.log_softmax(logits[0, :-1, :].float(), dim=-1)
            labels = inputs["input_ids"][0, 1:]
            token_lp = shift_lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            features["neg_mean_loss"] = float(token_lp.mean().item())

            features["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[LayerCascade WARN] {type(e).__name__}: {e}")
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
        ext = LayerCascadeExtractor(self.model, self.tokenizer, self.args.max_length)

        print(f"\n[LayerCascade] Extracting layer-depth trajectory features...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[LayerCascade]"):
            rows.append(ext.extract(row["content"]))
        feat_df = pd.DataFrame(rows)
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # ── Results ───────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("   LayerCascade-MIA: TRAJECTORY SIGNAL AUCs")
        print("=" * 70)

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        traj_families = {
            "NORM TRAJECTORY":     [c for c in feature_cols if "norm" in c and "vel" not in c and "drift" not in c],
            "VELOCITY TRAJECTORY": [c for c in feature_cols if "vel" in c],
            "DRIFT TRAJECTORY":    [c for c in feature_cols if "drift" in c],
            "RANK TRAJECTORY":     [c for c in feature_cols if "rank" in c],
            "CROSS-TRAJECTORY":    [c for c in feature_cols if "corr" in c],
            "BASELINE":            ["neg_mean_loss"],
        }

        all_results = {}
        for family, cols in traj_families.items():
            print(f"\n  ── {family} ──")
            for col in sorted(cols):
                if col not in df.columns:
                    continue
                v = df.dropna(subset=[col])
                if len(v) < 50 or len(v["is_member"].unique()) < 2:
                    continue
                auc_pos = roc_auc_score(v["is_member"], v[col])
                best = max(auc_pos, 1 - auc_pos)
                d = "+" if auc_pos >= 0.5 else "-"
                all_results[col] = (best, d, family)
                marker = " ★" if best > 0.60 else ""
                print(f"    {d}{col:<40} AUC = {best:.4f}{marker}")

        if all_results:
            top = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)[:10]
            print(f"\n  ── TOP 10 ──")
            for rank, (col, (auc, d, fam)) in enumerate(top):
                print(f"    {rank+1:2d}. [{fam:20s}] {d}{col:<35} AUC = {auc:.4f}")

            best_col = top[0][0]
            best_d = top[0][1][1]
            print(f"\n  Per-subset ({best_d}{best_col}={top[0][1][0]:.4f}):")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10 and len(sub["is_member"].unique()) > 1:
                    sv = sub[best_col] if best_d == "+" else -sub[best_col]
                    auc = roc_auc_score(sub["is_member"], sv)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

        print("=" * 70)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL_layer_cascade_{ts}.parquet", index=False)
        print(f"[LayerCascade] Results saved.")


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

    Experiment(Args).run()
