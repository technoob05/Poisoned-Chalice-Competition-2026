"""
NOVEL EXPERIMENT 03: RepDen-MIA — k-NN Density in Hidden State Space

NOVELTY: First use of k-nearest-neighbor density estimation in the model's
    HIDDEN STATE space for membership inference. Prior k-NN MIA works
    (e.g., RMIA/neighbor-based) operate in LOSS space or OUTPUT space.
    Operating directly in the learned representation space is novel.

Core Idea:
    Member samples SHAPED the model's internal representation space during
    training. Therefore, their hidden state representations should lie in
    DENSER regions of that space — surrounded by other training examples
    that collectively formed the local geometry.

    Non-member samples were never used to update the model → their
    representations are mapped to sparser, less "practiced" regions of
    hidden state space.

    Algorithm:
    1. Select a probe set of known M/NM samples (e.g., 500M + 500NM)
    2. Extract mean-pooled hidden states at mid-layer (L15) for all probes
    3. For each test sample, compute k-NN distances to probe members
    4. Score = -mean_knn_distance (closer to members = more likely member)
    5. Also compute: distance to NM probes, ratio M_dist/NM_dist

    This is FUNDAMENTALLY DIFFERENT from:
    - EXP50 memTrace: uses STATISTICS of hidden states (norms, variance)
    - EXP35 GradSim: uses gradient profiles, not hidden state positions
    - EXP03 RMIA: uses loss perturbation, not representation geometry
    - SMIA (EXP52): measures drift under perturbation, not absolute position

Builds on Insights:
    - Insight 22: hidden states at mid-layer (L15) are most discriminative
    - Insight 15: probe-based methods need ≥1000 samples
    - EXP50: hidden state space encodes membership (0.6908)

Compute: 1 forward pass per sample (output_hidden_states=True).
    k-NN computation on mean-pooled vectors (hidden_dim ≈ 3072).
Expected runtime: ~8-12 min on A100 (forward passes dominate).
Expected AUC: 0.62-0.70
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL03: RepDen-MIA — k-NN Density in Hidden Space")
    print("  Novelty: Density estimation in representation manifold")
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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded. dtype={dtype}, layers={n_layers}")
    return model, tokenizer


class RepresentationExtractor:
    """Extract mean-pooled hidden states at specified layers."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 layers: List[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        self.layer_map = {
            "early": self.n_layers // 4,
            "mid": self.n_layers // 2,
            "late": 3 * self.n_layers // 4,
            "last": self.n_layers - 1,
        }
        self.extract_layers = layers or ["mid", "last"]
        print(f"  Extract layers: {[(k, self.layer_map[k]) for k in self.extract_layers]}")

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, np.ndarray]:
        """Return mean-pooled hidden states at specified layers + loss."""
        result = {"loss": np.nan}
        if not text or len(text) < 30:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 10:
                return result

            outputs = self.model(
                input_ids=input_ids, output_hidden_states=True, labels=input_ids,
            )
            result["loss"] = outputs.loss.float().item()

            for layer_name in self.extract_layers:
                layer_idx = self.layer_map[layer_name]
                hs = outputs.hidden_states[layer_idx + 1]  # (1, seq, dim)
                pooled = hs.float().mean(dim=1).squeeze(0).cpu().numpy()  # (dim,)
                result[f"rep_{layer_name}"] = pooled

            result["seq_len"] = input_ids.shape[1]
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL03 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


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
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = RepresentationExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            layers=self.args.extract_layers,
        )

        # Phase 1: Extract representations for ALL samples
        print(f"\n[NOVEL03] Phase 1: Extracting representations for {len(df)} samples...")
        all_reps = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL03 extract]"):
            all_reps.append(extractor.extract(row["content"]))

        # Collect representations into matrices
        losses = np.array([r.get("loss", np.nan) for r in all_reps])
        seq_lens = np.array([r.get("seq_len", 0) for r in all_reps])
        df["neg_mean_loss"] = -losses

        rep_matrices = {}
        for layer_name in self.args.extract_layers:
            key = f"rep_{layer_name}"
            vecs = []
            valid_mask = []
            for r in all_reps:
                if key in r and isinstance(r[key], np.ndarray):
                    vecs.append(r[key])
                    valid_mask.append(True)
                else:
                    vecs.append(np.zeros(self.model.config.hidden_size))
                    valid_mask.append(False)
            rep_matrices[layer_name] = np.stack(vecs)  # (N, dim)

        # Phase 2: Split into probe and evaluation sets
        probe_size = min(self.args.probe_size, len(df) // 2)
        probe_per_class = probe_size // 2

        members = df[df["is_member"] == 1].index.tolist()
        nonmembers = df[df["is_member"] == 0].index.tolist()
        random.shuffle(members)
        random.shuffle(nonmembers)
        probe_m = members[:probe_per_class]
        probe_nm = nonmembers[:probe_per_class]
        probe_idx = set(probe_m + probe_nm)
        eval_idx = [i for i in range(len(df)) if i not in probe_idx]

        print(f"\n[NOVEL03] Phase 2: Probe={len(probe_idx)} (M={len(probe_m)}, NM={len(probe_nm)})")
        print(f"  Eval={len(eval_idx)}")

        # Phase 3: k-NN density estimation
        print(f"\n[NOVEL03] Phase 3: k-NN density estimation (k={self.args.k_neighbors})...")

        for layer_name in self.args.extract_layers:
            reps = rep_matrices[layer_name]

            # Standardize
            scaler = StandardScaler()
            reps_scaled = scaler.fit_transform(reps)

            # Build k-NN index on probe members
            probe_m_reps = reps_scaled[probe_m]
            probe_nm_reps = reps_scaled[probe_nm]

            knn_m = NearestNeighbors(n_neighbors=self.args.k_neighbors, metric="cosine")
            knn_m.fit(probe_m_reps)

            knn_nm = NearestNeighbors(n_neighbors=self.args.k_neighbors, metric="cosine")
            knn_nm.fit(probe_nm_reps)

            # Also build k-NN on ALL probes
            probe_all = list(probe_m) + list(probe_nm)
            probe_labels = [1] * len(probe_m) + [0] * len(probe_nm)
            knn_all = NearestNeighbors(n_neighbors=self.args.k_neighbors, metric="cosine")
            knn_all.fit(reps_scaled[probe_all])

            # Score evaluation samples
            eval_reps = reps_scaled[eval_idx]

            dist_m, _ = knn_m.kneighbors(eval_reps)  # (N_eval, k)
            dist_nm, _ = knn_nm.kneighbors(eval_reps)
            dist_all, idx_all = knn_all.kneighbors(eval_reps)

            # Features
            mean_dist_m = dist_m.mean(axis=1)  # mean distance to member probes
            mean_dist_nm = dist_nm.mean(axis=1)
            density_ratio = mean_dist_nm / (mean_dist_m + 1e-10)  # higher = closer to M

            # k-NN vote: fraction of k neighbors that are members
            probe_labels_arr = np.array(probe_labels)
            knn_vote = np.array([
                probe_labels_arr[idx_all[i]].mean() for i in range(len(eval_idx))
            ])

            # Assign scores
            col_prefix = f"knn_{layer_name}"
            for i, eval_i in enumerate(eval_idx):
                df.loc[eval_i, f"{col_prefix}_neg_dist_m"] = -mean_dist_m[i]
                df.loc[eval_i, f"{col_prefix}_neg_dist_nm"] = -mean_dist_nm[i]
                df.loc[eval_i, f"{col_prefix}_density_ratio"] = density_ratio[i]
                df.loc[eval_i, f"{col_prefix}_vote"] = knn_vote[i]
                df.loc[eval_i, f"{col_prefix}_dist_diff"] = mean_dist_nm[i] - mean_dist_m[i]

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL03: RepDen-MIA — k-NN Density RESULTS")
        print("=" * 70)

        eval_df = df.iloc[eval_idx]
        score_cols = [c for c in df.columns if c.startswith("knn_") or c == "neg_mean_loss"]
        aucs = {}
        for col in sorted(score_cols):
            v = eval_df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= auc_neg else "-"
            aucs[col] = (best, d)
            tag = " <-- PRIMARY" if "density_ratio" in col else ""
            print(f"  {d}{col:<40} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")
            print(f"  vs EXP50 memTrace RF:  0.6908")
            print(f"  vs EXP41 -grad_z_lang: 0.6539")

            # Per-subset
            best_col, (_, best_dir) = best_sig
            print(f"\n{'Subset':<10} | {best_col:<30} | N")
            print("-" * 55)
            for subset in sorted(eval_df["subset"].unique()):
                sub = eval_df[eval_df["subset"] == subset]
                v = sub.dropna(subset=[best_col])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    vals = v[best_col] if best_dir == "+" else -v[best_col]
                    auc = roc_auc_score(v["is_member"], vals)
                else:
                    auc = float("nan")
                print(f"  {subset:<10} | {auc:.4f}                     | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL03_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL03] Results saved.")


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
        extract_layers = ["mid", "last"]
        probe_size = 1000  # 500M + 500NM
        k_neighbors = 10

    print(f"[NOVEL03] RepDen-MIA: k-NN in Hidden Space")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    print(f"  probe={Args.probe_size}, k={Args.k_neighbors}, layers={Args.extract_layers}")
    Experiment(Args).run()
