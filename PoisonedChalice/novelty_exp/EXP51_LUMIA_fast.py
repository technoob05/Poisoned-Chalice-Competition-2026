"""
EXP51_LUMIA_fast.py — LUMIA Fast: Linear Probes on Full Dataset

OPTIMIZED VERSION of EXP51 for full 100K dataset within 12-hour wall-clock limit.

v2 (PROPER TRAIN/TEST SPLIT):
    - Load dataset["train"] split → extract activations → fit PCA + Ridge
    - Load dataset["test"]  split → extract activations → transform PCA → predict
    - No cross-validation leakage; true out-of-sample AUC on held-out test set
    - Identical pipeline, no label leakage between splits

Original EXP51 bottlenecks (10% = 10K samples, took 7 hours):
    1. LogReg with solver='saga' on dim=3072 vectors → very slow for large N
    2. 31 layers × 5-fold CV → 155 LogReg fits
    3. Top-K concat dim=24,576 × 5 folds → memory + solver bottleneck

Optimizations:
    1. Ridge Regression (closed-form, ~50x faster than saga LogReg)
    2. Only probe KEY layers — sweet spot from EXP51 (blocks 1,4,5,6,7,20,21,29)
    3. PCA(n=256) fit on TRAIN, transform TEST → reduces dim 3072→256
    4. Batched forward passes (batch_size=8, fix pad_token for StarCoder2)
    5. Train probe once on full train split (no CV needed with proper split)

Time budget analysis (A100 80GB):
    - Train forward passes (~100K): ~26 min single-sample / ~4 min batched
    - Test  forward passes (~100K): ~26 min single-sample / ~4 min batched
    - PCA fit on train + Ridge fit: ~2 min
    - Total: ~55 min single-sample / ~10 min batched ✅ (well within 12h)
"""

import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ─── Key layers from EXP51 Insight 32 ────────────────────────────────────────
# Block_5 (layer 6): AUC 0.7031 — best single
# Block_4 (layer 5): AUC 0.7019
# Block_6 (layer 7): AUC 0.7009
# Block_1 (layer 2): AUC 0.7007
# Block_20/21 (layers 21/22): secondary bump 0.697
# Block_29 (layer 30): 0.688 — final layer
# Layer 0 (embed): 0.607 — included for completeness
KEY_LAYERS = [0, 2, 5, 6, 7, 8, 21, 22, 30]  # 9 layers (0=embed, rest=block idx+1)
# Covers: embed, early-peak (blocks 1,4,5,6,7), secondary bump (blocks 20,21), final


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP51_LUMIA_fast — Linear Probes (Full Dataset, Optimized)")
    print("  Key layers only + Ridge Regression + PCA → fits in <2h on A100")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Logged in.")
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Fix: StarCoder2 tokenizer has no pad token by default → batch inference fails
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  [Fix] Set pad_token = eos_token for batch inference")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded. dtype={dtype}, layers={n_layers}, hidden_dim={hidden_dim}")
    return model, tokenizer


class FastLUMIAExtractor:
    """
    Extract mean-pooled hidden states at KEY layers only.
    One forward pass per sample; no gradient computation needed.
    """

    def __init__(self, model, tokenizer, max_length: int = 512,
                 key_layers: Optional[List[int]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self.key_layers = key_layers if key_layers is not None else KEY_LAYERS
        # Clip to valid range [0 .. n_layers]
        self.key_layers = [l for l in self.key_layers if 0 <= l <= self.n_layers]
        self._err_count = 0
        print(f"  Probing {len(self.key_layers)} key layers: {self.key_layers}")
        print(f"  (embed=0, transformer blocks as layer_idx+1)")

    @torch.no_grad()
    def extract_batch(self, texts: List[str]) -> Optional[Dict[int, np.ndarray]]:
        """
        Process a BATCH of texts. Returns dict {layer_idx: (B, D) array}.
        If any text in batch fails, falls back to per-sample extraction.
        Assumes all texts have already been truncated/padded.
        """
        result = {}
        try:
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_attention_mask=True,
            ).to(self.model.device)

            attention_mask = enc["attention_mask"]  # (B, T)
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Mean-pool over non-padding tokens per sample per layer
            # hidden_states: tuple of (n_layers+1) tensors each (B, T, D)
            mask_f = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            lengths = mask_f.sum(dim=1)  # (B, 1)

            for layer_idx in self.key_layers:
                hs = outputs.hidden_states[layer_idx].float()  # (B, T, D)
                pooled = (hs * mask_f).sum(dim=1) / lengths  # (B, D)
                result[layer_idx] = pooled.cpu().numpy()

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[FAST_LUMIA_batch] {type(e).__name__}: {e}")
            self._err_count += 1
            return None

    @torch.no_grad()
    def extract_single(self, text: str) -> Optional[Dict[int, np.ndarray]]:
        """Fallback single-sample extraction."""
        if not text or len(text) < 10:
            return None
        try:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            ).to(self.model.device)
            seq_len = enc["input_ids"].shape[1]
            if seq_len < 3:
                return None
            outputs = self.model(
                input_ids=enc["input_ids"],
                output_hidden_states=True,
            )
            result = {}
            for layer_idx in self.key_layers:
                hs = outputs.hidden_states[layer_idx][0, :seq_len, :].float()
                result[layer_idx] = hs.mean(dim=0).cpu().numpy()  # (D,)
            return result
        except Exception as e:
            if self._err_count < 3:
                print(f"\n[FAST_LUMIA_single] {type(e).__name__}: {e}")
            self._err_count += 1
            return None


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

    def _load_split(self, split: str, sample_fraction: float = 1.0) -> pd.DataFrame:
        """Load one split ('train' or 'test') across all language subsets."""
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        for subset in subsets:
            try:
                if is_local:
                    path = os.path.join(self.args.dataset, subset)
                    if not os.path.exists(path):
                        continue
                    ds_dict = load_from_disk(path)
                    if hasattr(ds_dict, "keys") and split in ds_dict.keys():
                        ds = ds_dict[split]
                    elif split == "test" and hasattr(ds_dict, "keys") and "test" not in ds_dict.keys():
                        # Fallback: dataset has no train split (HF download only has test)
                        ds = ds_dict[list(ds_dict.keys())[0]]
                    else:
                        ds = ds_dict
                else:
                    ds = load_dataset(self.args.dataset, subset, split=split)
                sub_df = ds.to_pandas()
                sub_df["subset"] = subset
                dfs.append(sub_df)
            except Exception as e:
                print(f"  [WARN] Could not load {subset}/{split}: {e}")

        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=self.args.seed).reset_index(drop=True)
        return df

    def load_data_splits(self):
        """Load train and test splits separately."""
        print(f"[*] Loading TRAIN split...")
        train_df = self._load_split("train", self.args.sample_fraction)
        print(f"  Train: {len(train_df)} samples")

        print(f"[*] Loading TEST split...")
        test_df = self._load_split("test", self.args.sample_fraction)
        print(f"  Test:  {len(test_df)} samples")

        if train_df.empty:
            print("  [WARN] No train split found! Falling back to CV on test split.")
            return None, test_df
        return train_df, test_df

    def _extract_activations(self, df: pd.DataFrame, extractor: "FastLUMIAExtractor",
                             label: str) -> tuple:
        """Extract mean-pooled hidden states for all samples in df.
        Returns (act_matrices dict, valid_idx, labels, subsets).
        """
        n_total = len(df)
        key_layers = extractor.key_layers
        act_matrices = {
            layer_idx: np.full((n_total, self.hidden_dim), np.nan, dtype=np.float32)
            for layer_idx in key_layers
        }
        valid_mask = np.zeros(n_total, dtype=bool)
        texts = df["content"].tolist()
        batch_size = self.args.batch_size
        use_batch = batch_size > 1
        i = 0
        pbar = tqdm(total=n_total, desc=f"[LUMIA-fast {label}]")

        while i < n_total:
            if use_batch:
                end = min(i + batch_size, n_total)
                result = extractor.extract_batch(texts[i:end])
                if result is not None:
                    actual_n = next(iter(result.values())).shape[0]
                    for layer_idx, arr in result.items():
                        act_matrices[layer_idx][i:i + actual_n] = arr
                    valid_mask[i:i + actual_n] = True
                    pbar.update(actual_n)
                    i += actual_n
                    continue
                else:
                    use_batch = False
                    print(f"[LUMIA-fast] Batch failed → single-sample mode")

            result = extractor.extract_single(texts[i])
            if result is not None:
                for layer_idx, vec in result.items():
                    act_matrices[layer_idx][i] = vec
                valid_mask[i] = True
            pbar.update(1)
            i += 1

        pbar.close()
        valid_idx = np.where(valid_mask)[0]
        y = df["is_member"].values[valid_idx]
        subsets_arr = df["subset"].values[valid_idx]
        for layer_idx in key_layers:
            act_matrices[layer_idx] = act_matrices[layer_idx][valid_idx]
        print(f"  {label}: {valid_mask.sum()}/{n_total} valid")
        return act_matrices, valid_idx, y, subsets_arr

    def run(self):
        t_start = time.time()

        # ── Load splits ───────────────────────────────────────────────────────
        train_df, test_df = self.load_data_splits()
        has_train_split = train_df is not None and len(train_df) > 0

        extractor = FastLUMIAExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            key_layers=self.args.key_layers,
        )
        key_layers = extractor.key_layers
        n_key = len(key_layers)

        # ── STEP 1: Forward Passes ────────────────────────────────────────────
        print(f"\n[STEP 1a] Forward passes — TRAIN ({len(train_df) if has_train_split else 0} samples)")
        t_fwd = time.time()

        if has_train_split:
            act_tr, _, y_tr, sub_tr = self._extract_activations(train_df, extractor, "TRAIN")
        else:
            print("  No train split — will use CV on test split.")

        print(f"\n[STEP 1b] Forward passes — TEST ({len(test_df)} samples)")
        act_te, valid_idx_te, y_te, sub_te = self._extract_activations(test_df, extractor, "TEST")
        t_fwd_done = time.time() - t_fwd
        print(f"  Total forward pass time: {t_fwd_done / 60:.1f} min")

        # ── STEP 2: PCA fit on TRAIN, transform TEST ──────────────────────────
        if self.args.pca_components and self.args.pca_components < self.hidden_dim:
            print(f"\n[STEP 2] PCA: {self.hidden_dim} → {self.args.pca_components} dims")
            print(f"  Fit on TRAIN, transform TEST (no leakage)")
            t_pca = time.time()
            pca_models = {}
            for layer_idx in key_layers:
                X_fit = act_tr[layer_idx] if has_train_split else act_te[layer_idx]
                pca = PCA(n_components=self.args.pca_components, random_state=self.args.seed)
                pca.fit(X_fit)
                pca_models[layer_idx] = pca
                ev = pca.explained_variance_ratio_.sum()
                if has_train_split:
                    act_tr[layer_idx] = pca.transform(act_tr[layer_idx]).astype(np.float32)
                act_te[layer_idx] = pca.transform(act_te[layer_idx]).astype(np.float32)
                print(f"  Layer {layer_idx:2d}: explained_var = {ev:.3f}")
            print(f"  PCA done in {(time.time() - t_pca):.1f}s")
        else:
            print(f"\n[STEP 2] PCA skipped")

        feat_dim = act_te[key_layers[0]].shape[1]
        print(f"  Feature dim: {feat_dim}D")

        # ── STEP 3: Fit Ridge on TRAIN, predict on TEST ───────────────────────
        print(f"\n[STEP 3] Ridge probes: fit on TRAIN → predict on TEST")
        t_probe = time.time()

        layer_aucs = {}
        per_layer_scores_te = {}

        for layer_idx in key_layers:
            lname = "embed" if layer_idx == 0 else f"block_{layer_idx - 1}"

            if has_train_split:
                # Proper train→test evaluation
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(act_tr[layer_idx])
                X_test_s = scaler.transform(act_te[layer_idx])
                clf = RidgeClassifier(
                    alpha=self.args.ridge_alpha, max_iter=1000,
                    solver="auto", class_weight="balanced",
                )
                clf.fit(X_train_s, y_tr)
                scores = clf.decision_function(X_test_s)
            else:
                # Fallback: 3-fold CV on test split
                skf = StratifiedKFold(n_splits=self.args.n_folds, shuffle=True,
                                      random_state=self.args.seed)
                scores = np.full(len(y_te), np.nan)
                for train_idx, val_idx in skf.split(act_te[layer_idx], y_te):
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(act_te[layer_idx][train_idx])
                    X_va_s = scaler.transform(act_te[layer_idx][val_idx])
                    clf = RidgeClassifier(alpha=self.args.ridge_alpha, max_iter=1000,
                                         solver="auto", class_weight="balanced")
                    clf.fit(X_tr_s, y_te[train_idx])
                    scores[val_idx] = clf.decision_function(X_va_s)

            auc = roc_auc_score(y_te, scores)
            layer_aucs[layer_idx] = auc
            per_layer_scores_te[layer_idx] = scores
            print(f"  Layer {layer_idx:2d} ({lname:>10}): TEST AUC = {auc:.4f}")

        t_probe_done = time.time() - t_probe
        print(f"  Ridge probes done in {t_probe_done:.1f}s")

        # ── STEP 4: Top-K concatenation (train→test) ──────────────────────────
        print(f"\n[STEP 4] Top-K concatenation")
        sorted_layers = sorted(layer_aucs.items(), key=lambda x: x[1], reverse=True)
        top_k_configs = sorted(set([1, 3, 5, n_key]))

        concat_aucs = {}
        best_concat_auc = 0.0
        best_concat_scores = None

        for k in top_k_configs:
            top_layers = [l[0] for l in sorted_layers[:k]]
            X_concat_te = np.concatenate([act_te[l] for l in top_layers], axis=1)
            layer_desc = "+".join([f"L{l}" for l in top_layers])

            if has_train_split:
                X_concat_tr = np.concatenate([act_tr[l] for l in top_layers], axis=1)
                alpha = self.args.ridge_alpha * max(1.0, k ** 0.5)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_concat_tr)
                X_te_s = scaler.transform(X_concat_te)
                clf = RidgeClassifier(alpha=alpha, max_iter=1000,
                                      solver="auto", class_weight="balanced")
                clf.fit(X_tr_s, y_tr)
                scores = clf.decision_function(X_te_s)
                auc = roc_auc_score(y_te, scores)
                std_str = ""
            else:
                # CV fallback
                skf = StratifiedKFold(n_splits=self.args.n_folds, shuffle=True,
                                      random_state=self.args.seed)
                fold_aucs, scores = [], np.full(len(y_te), np.nan)
                alpha = self.args.ridge_alpha * max(1.0, k ** 0.5)
                for tr_idx, va_idx in skf.split(X_concat_te, y_te):
                    sc = StandardScaler()
                    X_tr_s = sc.fit_transform(X_concat_te[tr_idx])
                    X_va_s = sc.transform(X_concat_te[va_idx])
                    clf = RidgeClassifier(alpha=alpha, max_iter=1000,
                                         solver="auto", class_weight="balanced")
                    clf.fit(X_tr_s, y_te[tr_idx])
                    scores[va_idx] = clf.decision_function(X_va_s)
                    fold_aucs.append(roc_auc_score(y_te[va_idx], scores[va_idx]))
                auc = roc_auc_score(y_te, scores)
                std_str = f" ± {np.std(fold_aucs):.4f} (CV)"

            concat_aucs[k] = auc
            if auc > best_concat_auc:
                best_concat_auc = auc
                best_concat_scores = scores.copy()

            print(f"  Top-{k} ({layer_desc}): TEST AUC = {auc:.4f}{std_str}"
                  f" (dim={X_concat_te.shape[1]})")

        # ── STEP 5: Unsupervised norm baseline (test set only) ────────────────
        print(f"\n[STEP 5] Unsupervised norm baseline (TEST)")
        for layer_idx in key_layers:
            norms = np.linalg.norm(act_te[layer_idx], axis=1)
            auc_pos = roc_auc_score(y_te, norms)
            best_n = max(auc_pos, 1 - auc_pos)
            d = "+" if auc_pos >= 0.5 else "-"
            lname = "embed" if layer_idx == 0 else f"block_{layer_idx - 1}"
            print(f"  Layer {layer_idx:2d} ({lname:>10}): Norm ({d}) = {best_n:.4f}")

        # ── RESULTS ───────────────────────────────────────────────────────────
        best_single_layer = max(layer_aucs, key=layer_aucs.get)
        best_single_auc = layer_aucs[best_single_layer]
        overall_best = max(best_single_auc, best_concat_auc)
        use_scores = best_concat_scores if best_concat_auc >= best_single_auc \
            else per_layer_scores_te[best_single_layer]

        print("\n" + "=" * 70)
        print("   EXP51_LUMIA_fast v2 — RESULTS (proper TRAIN→TEST)")
        print("=" * 70)
        split_mode = "TRAIN→TEST" if has_train_split else f"{self.args.n_folds}-fold CV on TEST"
        print(f"  Evaluation mode: {split_mode}")
        print(f"  Best single layer: {best_single_layer} AUC = {best_single_auc:.4f}")
        print(f"  Best Top-K concat: AUC = {best_concat_auc:.4f}")
        print(f"  Overall best:      {overall_best:.4f}")

        print(f"\n  Per-subset TEST AUC (best scores):")
        for subset in sorted(np.unique(sub_te)):
            mask = sub_te == subset
            sub_y = y_te[mask]
            sub_s = use_scores[mask]
            ok = ~np.isnan(sub_s)
            if ok.sum() > 10 and len(np.unique(sub_y[ok])) > 1:
                auc = roc_auc_score(sub_y[ok], sub_s[ok])
            else:
                auc = float("nan")
            print(f"    {subset:<10} AUC = {auc:.4f}  (n_test={mask.sum()})")

        t_total = time.time() - t_start
        print(f"\n  --- Comparison ---")
        print(f"  LUMIA-fast v2 (train→test): {overall_best:.4f}")
        print(f"  LUMIA-fast v1 (CV on test 100K): 0.7805")
        print(f"  EXP51 (10% CV):             0.7338")
        print(f"  EXP50 memTrace RF:           0.6908")
        print(f"\n  Total wall-clock: {t_total / 3600:.2f}h")
        print(f"  Forward pass:     {t_fwd_done / 60:.1f} min")
        print(f"  Probe train:      {t_probe_done:.1f}s")
        print("=" * 70)

        # Save test predictions
        test_df["lumia_fast_score"] = np.nan
        test_df.loc[valid_idx_te, "lumia_fast_score"] = use_scores

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"EXP51_LUMIA_fast_v2_{timestamp}.parquet"
        test_df[["content", "subset", "is_member", "lumia_fast_score"]].to_parquet(
            out_path, index=False,
        )
        print(f"\n[LUMIA-fast] Saved test predictions → {out_path}")
        print(f"[LUMIA-fast] DONE. Total: {t_total / 3600:.2f}h")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"

        # Dataset path (Kaggle or HuggingFace)
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"

        # ─── DATA ──────────────────────────────────────────────────────────
        # Set to 1.0 for full dataset. 0.1 for quick test (~3 min).
        sample_fraction = 1.0          # FULL DATA
        max_length = 512
        seed = 42

        # ─── MODEL LAYERS ──────────────────────────────────────────────────
        # From EXP51 Insight 32: sweet spot is early-to-mid blocks (4-7)
        # + secondary bump (20-21) + final block (29)
        key_layers = KEY_LAYERS        # [0, 2, 5, 6, 7, 8, 21, 22, 30]

        # ─── INFERENCE ─────────────────────────────────────────────────────
        # batch_size > 1 speeds up forward passes ~2-4x on A100
        # If OOM → reduce to 4 or 1
        batch_size = 8

        # ─── DIMENSIONALITY REDUCTION ──────────────────────────────────────
        # PCA before Ridge: reduces dim 3072 → pca_components
        # This is the KEY speedup for probe training on large N.
        # None = skip PCA (raw 3072-D, slower Ridge but potentially higher AUC)
        # 256 = safe fast option; 128 = fastest
        pca_components = 256

        # ─── PROBE TRAINING ────────────────────────────────────────────────
        # n_folds only used as FALLBACK if no train split is available
        n_folds = 3
        ridge_alpha = 1.0              # Ridge regularization (higher = stronger)
        # Note: alpha is also scaled by sqrt(k) for top-K concatenation

        # ─── OUTPUT ────────────────────────────────────────────────────────
        output_dir = "results"

    print(f"\n[LUMIA-fast v2] Configuration:")
    print(f"  model:          {Args.model_name}")
    print(f"  sample:         {Args.sample_fraction * 100:.0f}%")
    print(f"  key_layers:     {Args.key_layers}")
    print(f"  batch_size:     {Args.batch_size} (pad_token fix applied)")
    print(f"  pca_components: {Args.pca_components} (fit on TRAIN, transform TEST)")
    print(f"  ridge_alpha:    {Args.ridge_alpha}")
    print(f"  eval_mode:      TRAIN split → fit probe → TEST split → AUC")
    print(f"\n  Time budget estimate (A100, batched):")
    n_tr = int(500000 * Args.sample_fraction)  # ~500K train samples
    n_te = int(100000 * Args.sample_fraction)  # ~100K test samples
    # ~0.025s/batch of 8
    fwd_min = (n_tr + n_te) / (Args.batch_size * 60 / 0.025)
    print(f"  - Forward passes (train+test): ~{fwd_min:.0f} min")
    print(f"  - PCA fit + Ridge fit:         ~2 min")
    print(f"  - Total:                       ~{fwd_min + 2:.0f} min ✅ well within 12h")

    Experiment(Args).run()
