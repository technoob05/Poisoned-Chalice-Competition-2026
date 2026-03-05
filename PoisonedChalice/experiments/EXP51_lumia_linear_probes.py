"""
EXPERIMENT 51: LUMIA — Linear Probing for Membership Inference via Internal LLM States

Paper: "LUMIA: Linear probing for Unimodal and MultiModal Membership Inference
        Attacks leveraging internal LLM states"
       Ibáñez-Lissen, Gonzalez-Manzano, de Fuentes, Anciaux, Garcia-Alfaro
       (arXiv:2411.19876, Nov 2024)

Survey reference: Wu & Cao, "Membership Inference Attacks on Large-Scale
    Models: A Survey" (arXiv:2503.19338v3, Aug 2025), Section 4.4 [27]

Core idea:
    Attach lightweight LINEAR PROBES to the hidden representations at each
    transformer layer. For each layer l, compute the average hidden activation
    across all token positions:  A_l(X) = (1/N) * sum(a_l(x_i))
    Then feed A_l(X) into a per-layer linear probe (logistic regression)
    trained to classify membership. The layer with highest AUC is the
    decision point.

    Key findings from paper:
    1. Middle and deeper layers reveal MORE membership information
    2. Simple linear probes suffice — no need for complex classifiers
    3. Works on both LLMs (Pythia, GPT-Neo) and VLMs (LLaVA)
    4. Membership AUC > 0.6 reported as decision threshold

How LUMIA differs from our existing experiments:
    - vs EXP50 (memTrace): memTrace extracts CROSS-LAYER features (transitions,
      confidence variance, etc.) → Random Forest. LUMIA uses RAW mean hidden
      activation at EACH layer independently → per-layer linear probe.
      Simpler, more interpretable, and identifies WHICH layers leak most.
    - vs EXP11/13/34 (gradient norms): Those measure gradient MAGNITUDE.
      LUMIA uses the hidden STATE VECTOR itself (direction + magnitude).
    - vs EXP14/29/43 (attention-based): Those analyze attention PATTERNS.
      LUMIA analyzes hidden ACTIVATIONS (pre-attention or post-FFN outputs).

Implementation:
    1. Forward pass each sample with output_hidden_states=True
    2. For each of 33 layers (embedding + 32 transformer blocks):
       Extract mean-pooled hidden state vector (dim=3072 for StarCoder2-3b)
    3. Split into probe set (1000 samples) and scoring set
    4. For each layer: train LogisticRegression on probe → score rest → AUC
    5. Report per-layer AUC curve (expect peak at middle layers)
    6. Also try: concatenate top-K layer activations → single classifier

Compute: 1 forward pass per sample (output_hidden_states=True), 10% sample.
    Linear probes train in seconds. Very lightweight.
Expected runtime: ~10-15 min on A100 (dominated by forward passes)
Expected AUC: 0.55-0.70 (raw activations carry strong signal per paper;
    similar to gradient norm but from a different information source)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP51: LUMIA — Linear Probes on Hidden Activations")
    print("  Paper: Ibáñez-Lissen et al. (arXiv:2411.19876)")
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
    hidden_dim = model.config.hidden_size
    print(f"  Loaded. dtype={dtype}, layers={n_layers}, hidden_dim={hidden_dim}")
    return model, tokenizer


class LUMIAExtractor:
    """Extract per-layer mean-pooled hidden state vectors."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self._err_count = 0

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, np.ndarray]:
        """Extract mean-pooled hidden state at each layer.

        Returns dict mapping "layer_{i}" -> np.ndarray of shape (hidden_dim,)
        Plus "seq_len" scalar.
        """
        result = {}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 3:
                return result

            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors, each (1, T, D)

            for layer_idx, hs in enumerate(hidden_states):
                mean_activation = hs[0, :seq_len, :].float().mean(dim=0).cpu().numpy()
                result[f"layer_{layer_idx}"] = mean_activation

            result["seq_len"] = np.array([float(seq_len)])
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP51 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)
        self.n_layers = self.model.config.num_hidden_layers

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
                frac=self.args.sample_fraction, random_state=self.args.seed,
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = LUMIAExtractor(
            self.model, self.tokenizer, max_length=self.args.max_length,
        )

        total_layers = self.n_layers + 1  # embedding + transformer layers

        print(f"\n[EXP51] Extracting per-layer mean activations for {len(df)} samples...")
        print(f"  Layers: {total_layers} (embedding + {self.n_layers} transformer blocks)")
        print(f"  Hidden dim: {extractor.hidden_dim}")

        all_activations = {f"layer_{i}": [] for i in range(total_layers)}
        valid_indices = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="[EXP51]"):
            feats = extractor.extract(row["content"])
            if f"layer_0" in feats:
                for i in range(total_layers):
                    all_activations[f"layer_{i}"].append(feats[f"layer_{i}"])
                valid_indices.append(idx)

        n_valid = len(valid_indices)
        print(f"\n[EXP51] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[EXP51] Errors: {extractor._err_count}")

        y = df.loc[valid_indices, "is_member"].values
        subsets = df.loc[valid_indices, "subset"].values

        # Stack activations into matrices
        layer_matrices = {}
        for i in range(total_layers):
            layer_matrices[i] = np.stack(all_activations[f"layer_{i}"], axis=0)
            # shape: (n_valid, hidden_dim)

        del all_activations  # free memory

        # --- Per-layer linear probe (5-fold CV) ---
        print("\n" + "=" * 70)
        print("   EXP51: LUMIA — PER-LAYER LINEAR PROBE RESULTS (5-fold CV)")
        print("=" * 70)

        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)

        layer_aucs = {}
        best_layer = -1
        best_auc = 0.0

        # Also store per-sample scores from each layer for later analysis
        per_layer_scores = {}

        for layer_idx in range(total_layers):
            X = layer_matrices[layer_idx]

            fold_aucs = []
            all_scores = np.full(n_valid, np.nan)

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(
                    C=0.1, max_iter=500, solver="saga",
                    class_weight="balanced", random_state=self.args.seed,
                )
                clf.fit(X_train_s, y_train)

                proba = clf.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)
                all_scores[test_idx] = proba

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            layer_aucs[layer_idx] = (mean_auc, std_auc)
            per_layer_scores[layer_idx] = all_scores

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_layer = layer_idx

            layer_name = "embed" if layer_idx == 0 else f"block_{layer_idx - 1}"
            marker = " <<<" if layer_idx == best_layer else ""
            print(f"  Layer {layer_idx:2d} ({layer_name:>10}): "
                  f"AUC = {mean_auc:.4f} +/- {std_auc:.4f}{marker}")

        print(f"\n  BEST LAYER: {best_layer} with AUC = {best_auc:.4f}")

        # --- Concatenation of top-K layers ---
        print("\n" + "=" * 70)
        print("   EXP51: LUMIA — TOP-K LAYER CONCATENATION")
        print("=" * 70)

        sorted_layers = sorted(layer_aucs.items(), key=lambda x: x[1][0], reverse=True)
        top_k_configs = [1, 3, 5, 8]

        concat_aucs = {}
        best_concat_scores = None
        best_concat_auc = 0.0

        for k in top_k_configs:
            top_layers = [l[0] for l in sorted_layers[:k]]
            X_concat = np.concatenate([layer_matrices[l] for l in top_layers], axis=1)

            fold_aucs = []
            all_scores = np.full(n_valid, np.nan)

            for train_idx, test_idx in skf.split(X_concat, y):
                X_train, X_test = X_concat[train_idx], X_concat[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(
                    C=0.01, max_iter=500, solver="saga",
                    class_weight="balanced", random_state=self.args.seed,
                )
                clf.fit(X_train_s, y_train)

                proba = clf.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)
                all_scores[test_idx] = proba

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            concat_aucs[k] = (mean_auc, std_auc)

            layer_desc = ", ".join([f"L{l}" for l in top_layers])
            print(f"  Top-{k} ({layer_desc}): "
                  f"AUC = {mean_auc:.4f} +/- {std_auc:.4f} "
                  f"(dim={X_concat.shape[1]})")

            if mean_auc > best_concat_auc:
                best_concat_auc = mean_auc
                best_concat_scores = all_scores

        # --- Norm-based baseline (scalar per layer) ---
        print("\n" + "=" * 70)
        print("   EXP51: NORM-BASED UNSUPERVISED BASELINE")
        print("=" * 70)

        norm_aucs = {}
        for layer_idx in range(total_layers):
            norms = np.linalg.norm(layer_matrices[layer_idx], axis=1)
            auc_pos = roc_auc_score(y, norms)
            auc_neg = roc_auc_score(y, -norms)
            best = max(auc_pos, auc_neg)
            direction = "+" if auc_pos >= auc_neg else "-"
            norm_aucs[layer_idx] = (best, direction)

        sorted_norm = sorted(norm_aucs.items(), key=lambda x: x[1][0], reverse=True)
        print("\nTop 10 layers by activation norm AUC (unsupervised):")
        for layer_idx, (auc, direction) in sorted_norm[:10]:
            layer_name = "embed" if layer_idx == 0 else f"block_{layer_idx - 1}"
            print(f"  {direction}norm_L{layer_idx} ({layer_name}): AUC = {auc:.4f}")

        # Write scores to df
        use_scores = best_concat_scores if best_concat_auc > best_auc else per_layer_scores[best_layer]
        df["lumia_score"] = np.nan
        df.loc[valid_indices, "lumia_score"] = use_scores

        overall_best = max(best_auc, best_concat_auc)

        # --- Comparison ---
        print("\n" + "=" * 70)
        print("   EXP51: COMPARISON")
        print("=" * 70)
        print(f"  LUMIA best single layer:  {best_auc:.4f} (layer {best_layer})")
        print(f"  LUMIA best concatenation: {best_concat_auc:.4f}")
        print(f"  LUMIA overall best:       {overall_best:.4f}")
        print(f"  vs EXP50 memTrace RF:     (pending)")
        print(f"  vs EXP41 -grad_z_lang:    0.6539 (current best)")
        print(f"  vs EXP39 Ridge stacker:   0.6490")
        print(f"  vs EXP11 -grad_embed:     0.6472")

        if overall_best > 0.6539:
            print(f"\n  NEW BEST! LUMIA beats -grad_z_lang ({overall_best:.4f} > 0.6539)")
        elif overall_best > 0.6472:
            print(f"\n  STRONG: LUMIA beats -grad_embed ({overall_best:.4f} > 0.6472)")
        elif overall_best > 0.60:
            print(f"\n  PROMISING: LUMIA shows signal ({overall_best:.4f})")
        else:
            print(f"\n  WEAK: LUMIA = {overall_best:.4f}")

        # Per-subset breakdown using best scores
        print(f"\n{'Subset':<10} | {'LUMIA':<10} | N")
        print("-" * 35)
        for subset in sorted(np.unique(subsets)):
            mask = subsets == subset
            sub_y = y[mask]
            sub_scores = use_scores[mask]
            valid_mask = ~np.isnan(sub_scores)
            if valid_mask.sum() > 10 and len(np.unique(sub_y[valid_mask])) > 1:
                auc = roc_auc_score(sub_y[valid_mask], sub_scores[valid_mask])
            else:
                auc = float("nan")
            print(f"  {subset:<10} | {auc:.4f}    | {mask.sum()}")

        # Layer AUC curve summary (for paper figure)
        print(f"\n--- Layer AUC Curve ---")
        print(f"  Layer | Probe AUC | Norm AUC")
        print(f"  ------|-----------|--------")
        for i in range(total_layers):
            probe_auc = layer_aucs[i][0]
            norm_auc = norm_aucs[i][0]
            layer_name = "embed" if i == 0 else f"block_{i-1}"
            marker = " <<<" if i == best_layer else ""
            print(f"  {i:2d} ({layer_name:>10}) | {probe_auc:.4f}    | {norm_auc:.4f}{marker}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP51_{timestamp}.parquet", index=False)
        print(f"\n[EXP51] Results saved.")


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

    print(f"[EXP51] LUMIA: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  1 fwd pass/sample (output_hidden_states=True)")
    print(f"  Per-layer linear probes + top-K concatenation")
    Experiment(Args).run()
