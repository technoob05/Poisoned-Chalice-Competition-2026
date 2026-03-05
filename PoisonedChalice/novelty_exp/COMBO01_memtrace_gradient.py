"""
COMBO01: memTrace + Gradient — Hidden State Geometry × Loss Landscape Fusion

Rationale:
    EXP50 memTrace (AUC 0.6908) captures hidden state geometry (norms,
    transitions, position similarity) using FORWARD-ONLY pass.
    EXP11 gradient norm (AUC 0.6472) captures loss landscape geometry
    using BACKWARD pass — members sit in flat minima (low gradient).

    These two signal families are ORTHOGONAL:
    - Hidden state norms measure REPRESENTATION magnitude at each layer
    - Gradient norms measure SENSITIVITY of loss to parameter perturbation
    - A sample can have similar hidden state norms but very different gradients
      (e.g., a non-member might have normal norms but high gradient)

    Key insight: memTrace best feature (hnorm_std_L15 = 0.6335) and
    gradient best feature (grad_embed = 0.6472) capture different aspects
    of memorization. Combining them in RF should break 0.70.

    Additional gradient features:
    - Embedding gradient norm (EXP11: 0.6472)
    - Late-layer gradient norms L28, L29 (EXP30: 0.6407, 0.6303)
    - Gradient sparsity (Hoyer metric) — how concentrated grad is
    - Gradient-to-state ratio — gradient norm / hidden state norm per layer

Compute: 1 forward pass (output_hidden_states) + 1 backward pass
Expected runtime: ~8-12 min on A100 (10% sample)
Expected AUC: 0.71-0.74 (orthogonal fusion of two strongest signal families)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  COMBO01: memTrace + Gradient Fusion")
    print("  Hidden State Geometry × Loss Landscape")
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
    print(f"  Loaded. dtype={dtype}, layers={n_layers}, hidden={hidden_dim}")
    return model, tokenizer


class MemTraceGradExtractor:
    """Extract memTrace features PLUS gradient features for maximum coverage.

    Feature families:
    A. memTrace (forward-only):
        1. Layer transition features (L2 surprise + cosine stability)
        2. Prediction confidence features (entropy, confidence variance)
        3. Hidden state norm statistics per layer
        4. Position-based features (first-last similarity)
        5. Context evolution (mean representation changes)
    B. Gradient (backward pass):
        6. Embedding gradient norm
        7. Late-layer gradient norms (L24, L28, L29, head)
        8. Gradient sparsity (Hoyer metric)
        9. Gradient-to-state ratio (gradient norm / hidden state norm)
    """

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    # ===== MEMTRACE FEATURES (from EXP50) =====

    def _compute_transition_features(self, hidden_states: List[torch.Tensor],
                                     seq_len: int) -> Dict[str, float]:
        features = {}
        n_transitions = len(hidden_states) - 1
        all_surprises_mean = []
        all_stability_mean = []

        for i in range(n_transitions):
            h_curr = hidden_states[i][0, :seq_len, :].float()
            h_next = hidden_states[i+1][0, :seq_len, :].float()

            diff = h_next - h_curr
            surprise = diff.norm(dim=-1)
            cos_sim = F.cosine_similarity(h_curr, h_next, dim=-1)

            s_mean = surprise.mean().item()
            c_mean = cos_sim.mean().item()
            all_surprises_mean.append(s_mean)
            all_stability_mean.append(c_mean)

            if i in [0, self.n_layers // 4, self.n_layers // 2,
                     3 * self.n_layers // 4, self.n_layers - 1]:
                tag = f"L{i}"
                features[f"surprise_mean_{tag}"] = s_mean
                features[f"surprise_std_{tag}"] = surprise.std().item()
                features[f"surprise_max_{tag}"] = surprise.max().item()
                features[f"stability_mean_{tag}"] = c_mean
                features[f"stability_std_{tag}"] = cos_sim.std().item()
                features[f"stability_min_{tag}"] = cos_sim.min().item()

        arr_s = np.array(all_surprises_mean)
        arr_c = np.array(all_stability_mean)
        features["surprise_global_mean"] = float(arr_s.mean())
        features["surprise_global_std"] = float(arr_s.std())
        features["surprise_global_max"] = float(arr_s.max())
        features["surprise_argmax"] = float(arr_s.argmax())
        features["stability_global_mean"] = float(arr_c.mean())
        features["stability_global_std"] = float(arr_c.std())
        features["stability_global_min"] = float(arr_c.min())
        features["stability_argmin"] = float(arr_c.argmin())

        mid_start = self.n_layers // 3
        mid_end = 2 * self.n_layers // 3
        if mid_end > mid_start:
            mid_surprise = arr_s[mid_start:mid_end].mean()
            edge_surprise = np.concatenate([arr_s[:mid_start], arr_s[mid_end:]]).mean()
            features["surprise_mid_edge_ratio"] = float(
                mid_surprise / (edge_surprise + 1e-10)
            )

        return features

    def _compute_confidence_features(self, logits: torch.Tensor,
                                     input_ids: torch.Tensor,
                                     seq_len: int) -> Dict[str, float]:
        features = {}
        shift_logits = logits[0, :seq_len - 1, :].float()
        shift_labels = input_ids[0, 1:seq_len]
        T = shift_logits.shape[0]
        if T < 3:
            return features

        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)

        entropy = -(probs * log_probs).sum(dim=-1)
        features["entropy_mean"] = entropy.mean().item()
        features["entropy_std"] = entropy.std().item()
        features["entropy_min"] = entropy.min().item()
        features["entropy_max"] = entropy.max().item()

        confidence = probs.max(dim=-1).values
        features["confidence_mean"] = confidence.mean().item()
        features["confidence_std"] = confidence.std().item()
        features["confidence_min"] = confidence.min().item()
        features["confidence_max"] = confidence.max().item()

        if confidence.std().item() > 1e-10:
            features["confidence_stability"] = float(
                confidence.mean().item() / confidence.std().item()
            )
        else:
            features["confidence_stability"] = 100.0

        top2 = torch.topk(probs, k=2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]
        features["confidence_gap_mean"] = gap.mean().item()
        features["confidence_gap_std"] = gap.std().item()

        token_ll = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
        features["neg_mean_loss"] = token_ll.mean().item()
        features["loss_std"] = (-token_ll).std().item()

        return features

    def _compute_hidden_norm_features(self, hidden_states: List[torch.Tensor],
                                      seq_len: int) -> Dict[str, float]:
        features = {}
        layer_norms_mean = []

        for i, hs in enumerate(hidden_states):
            h = hs[0, :seq_len, :].float()
            norms = h.norm(dim=-1)
            n_mean = norms.mean().item()
            layer_norms_mean.append(n_mean)

            if i in [0, self.n_layers // 2, self.n_layers]:
                tag = f"L{i}"
                features[f"hnorm_mean_{tag}"] = n_mean
                features[f"hnorm_std_{tag}"] = norms.std().item()

        arr = np.array(layer_norms_mean)
        features["hnorm_global_mean"] = float(arr.mean())
        features["hnorm_growth_ratio"] = float(arr[-1] / (arr[0] + 1e-10))

        return features

    def _compute_position_features(self, hidden_states: List[torch.Tensor],
                                   seq_len: int) -> Dict[str, float]:
        features = {}
        similarities = []

        for i in [0, self.n_layers // 4, self.n_layers // 2,
                  3 * self.n_layers // 4, self.n_layers]:
            if i >= len(hidden_states):
                continue
            h = hidden_states[i][0, :seq_len, :].float()
            if seq_len < 2:
                continue
            sim = F.cosine_similarity(
                h[0].unsqueeze(0), h[seq_len - 1].unsqueeze(0)
            ).item()
            features[f"first_last_sim_L{i}"] = sim
            similarities.append(sim)

        if similarities:
            features["first_last_sim_mean"] = float(np.mean(similarities))
            features["first_last_sim_std"] = float(np.std(similarities))

        return features

    def _compute_context_evolution(self, hidden_states: List[torch.Tensor],
                                   seq_len: int,
                                   layer_idx: int = -1) -> Dict[str, float]:
        features = {}
        if layer_idx == -1:
            layer_idx = self.n_layers // 2

        h = hidden_states[layer_idx][0, :seq_len, :].float()
        if seq_len < 5:
            return features

        sample_positions = [seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
        evolutions = []
        prev_mean = h[0].unsqueeze(0)

        for pos in sample_positions:
            if pos < 1 or pos >= seq_len:
                continue
            curr_mean = h[:pos + 1].mean(dim=0, keepdim=True)
            delta = (curr_mean - prev_mean).norm().item()
            evolutions.append(delta)
            prev_mean = curr_mean

        if evolutions:
            features["ctx_evolution_mean"] = float(np.mean(evolutions))
            features["ctx_evolution_std"] = float(np.std(evolutions))

        return features

    # ===== GRADIENT FEATURES (NEW) =====

    def _compute_gradient_features(self, text: str) -> Dict[str, float]:
        """Extract gradient-based features with a single backward pass.

        Features:
        - Embedding gradient norm (strongest single gradient signal, AUC 0.6472)
        - Late-layer gradient norms (L24, L28, L29, head — PVC layers from EXP30)
        - Gradient sparsity (Hoyer metric: how concentrated the gradient is)
        - Per-layer gradient-to-state-norm ratio
        """
        features = {}

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                return features

            # Enable gradient for embedding
            self.model.zero_grad()
            embed_layer = self.model.model.embed_tokens
            embed_out = embed_layer(input_ids)
            embed_out.requires_grad_(True)
            embed_out.retain_grad()

            # Manual forward through transformer blocks
            hidden = embed_out
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            # Forward through decoder layers
            for layer in self.model.model.layers:
                layer_out = layer(hidden, position_ids=position_ids)
                hidden = layer_out[0]

            # Final layer norm + lm_head
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

            # Compute loss
            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels)

            # Backward
            loss.backward()

            # --- Embedding gradient ---
            if embed_out.grad is not None:
                grad_embed = embed_out.grad[0].float()  # (T, D)
                grad_norm_embed = grad_embed.norm(dim=-1).mean().item()
                features["grad_norm_embed"] = grad_norm_embed

                # Gradient sparsity (Hoyer metric)
                # H(x) = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
                flat_grad = grad_embed.reshape(-1)
                l1 = flat_grad.abs().sum().item()
                l2 = flat_grad.norm().item()
                n = flat_grad.numel()
                if l2 > 1e-10 and n > 1:
                    hoyer = (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)
                    features["grad_sparsity_embed"] = float(hoyer)

            # --- Late-layer gradient norms ---
            target_layers = {24: "L24", 28: "L28", 29: "L29"}
            for layer_idx, tag in target_layers.items():
                if layer_idx < len(self.model.model.layers):
                    layer_module = self.model.model.layers[layer_idx]
                    layer_grad_norm = 0.0
                    n_params = 0
                    for p in layer_module.parameters():
                        if p.grad is not None:
                            layer_grad_norm += p.grad.float().norm().item() ** 2
                            n_params += 1
                    if n_params > 0:
                        features[f"grad_norm_{tag}"] = np.sqrt(layer_grad_norm)

            # --- Head (lm_head) gradient ---
            head_grad_norm = 0.0
            n_head_p = 0
            for p in self.model.lm_head.parameters():
                if p.grad is not None:
                    head_grad_norm += p.grad.float().norm().item() ** 2
                    n_head_p += 1
            if n_head_p > 0:
                features["grad_norm_head"] = np.sqrt(head_grad_norm)

            # --- Gradient norm of final layer norm ---
            norm_grad_norm = 0.0
            for p in self.model.model.norm.parameters():
                if p.grad is not None:
                    norm_grad_norm += p.grad.float().norm().item() ** 2
            features["grad_norm_final_norm"] = np.sqrt(norm_grad_norm)

            # Clean up
            self.model.zero_grad()
            if embed_out.grad is not None:
                embed_out.grad = None

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO01 GRAD WARN] {type(e).__name__}: {e}")
            self._err_count += 1

        return features

    @torch.no_grad()
    def _extract_memtrace(self, text: str) -> Dict[str, float]:
        """Extract memTrace features (forward-only)."""
        result = {}
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                return result

            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states
            logits = outputs.logits

            result.update(self._compute_transition_features(hidden_states, seq_len))
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))
            result.update(self._compute_hidden_norm_features(hidden_states, seq_len))
            result.update(self._compute_position_features(hidden_states, seq_len))
            result.update(self._compute_context_evolution(hidden_states, seq_len))
            result["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO01 FWD WARN] {type(e).__name__}: {e}")
            self._err_count += 1

        return result

    def extract(self, text: str) -> Dict[str, float]:
        """Extract ALL features: memTrace (forward) + gradient (backward)."""
        # Phase 1: memTrace features (forward-only)
        features = self._extract_memtrace(text)

        # Phase 2: gradient features (backward pass)
        grad_features = self._compute_gradient_features(text)
        features.update(grad_features)

        # Phase 3: Cross-family features (gradient × hidden state)
        if "grad_norm_embed" in features and "hnorm_global_mean" in features:
            hnorm = features["hnorm_global_mean"]
            gnorm = features["grad_norm_embed"]
            if hnorm > 1e-10:
                features["grad_state_ratio_embed"] = gnorm / hnorm
            # Product score (grad × state — orthogonal fusion)
            features["grad_state_product"] = gnorm * hnorm

        if "grad_norm_L29" in features and "hnorm_mean_L15" in features:
            features["grad_late_state_mid_ratio"] = (
                features["grad_norm_L29"] / (features["hnorm_mean_L15"] + 1e-10)
            )

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
        extractor = MemTraceGradExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[COMBO01] Extracting memTrace+Gradient features for {len(df)} samples...")
        print(f"  Phase 1: memTrace (1 forward pass/sample)")
        print(f"  Phase 2: Gradient (1 backward pass/sample)")
        print(f"  Phase 3: Cross-family fusion features")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO01]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[COMBO01] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[COMBO01] Errors: {extractor._err_count}")

        feature_cols = [c for c in feat_df.columns]
        print(f"[COMBO01] Total features extracted: {len(feature_cols)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised signal AUCs ---
        print("\n" + "=" * 70)
        print("   COMBO01: UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)

        unsup_aucs = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best_auc = max(auc, auc_neg)
            direction = "+" if auc >= auc_neg else "-"
            unsup_aucs[col] = (best_auc, direction)

        top_features = sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True)[:20]
        print("\nTop 20 individual features (unsupervised):")
        for col, (auc, direction) in top_features:
            print(f"  {direction}{col:<50} AUC = {auc:.4f}")

        # --- Identify memTrace-only vs gradient-only vs fusion features ---
        grad_cols = [c for c in feature_cols if "grad" in c.lower()]
        memtrace_cols = [c for c in feature_cols if c not in grad_cols and c != "seq_len"]
        fusion_cols = [c for c in feature_cols if "ratio" in c or "product" in c]
        print(f"\n  memTrace features: {len(memtrace_cols)}")
        print(f"  Gradient features: {len(grad_cols)}")
        print(f"  Fusion features:   {len(fusion_cols)}")

        # --- Random Forest Classifier (5-fold CV) ---
        print("\n" + "=" * 70)
        print("   COMBO01: RANDOM FOREST (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy()
        y_all = df.loc[valid_mask, "is_member"].values

        X_all = X_all.fillna(0)
        X_all = X_all.replace([np.inf, -np.inf], 0)

        feature_names = list(X_all.columns)
        X_np = X_all.values.astype(np.float64)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        configs = {
            "ALL (memTrace+Grad)": feature_names,
            "memTrace only": [c for c in feature_names if c in memtrace_cols or c == "seq_len"],
            "Gradient only": [c for c in feature_names if c in grad_cols],
        }

        for config_name, cols in configs.items():
            col_idx = [feature_names.index(c) for c in cols if c in feature_names]
            if len(col_idx) < 2:
                continue
            X_cfg = X_np[:, col_idx]

            n_folds = 5
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)

            fold_aucs = []
            all_scores = np.full(len(X_cfg), np.nan)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_cfg, y_all)):
                X_train, X_test = X_cfg[train_idx], X_cfg[test_idx]
                y_train, y_test = y_all[train_idx], y_all[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    min_samples_split=8, min_samples_leaf=4,
                    max_features="sqrt", class_weight="balanced",
                    random_state=self.args.seed, n_jobs=-1,
                )
                clf.fit(X_train_s, y_train)

                proba = clf.predict_proba(X_test_s)[:, 1]
                auc = roc_auc_score(y_test, proba)
                fold_aucs.append(auc)
                all_scores[test_idx] = proba

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            print(f"\n  {config_name} ({len(col_idx)} features):")
            print(f"    CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
            for fi, a in enumerate(fold_aucs):
                print(f"    Fold {fi+1}: {a:.4f}")

            if config_name == "ALL (memTrace+Grad)":
                df.loc[valid_mask, "combo01_score"] = all_scores

                # Feature importance
                print(f"\n--- Top 15 Feature Importances ({config_name}) ---")
                importances = clf.feature_importances_
                used_names = [cols[i] for i in range(len(cols))]
                imp_pairs = sorted(zip(used_names, importances), key=lambda x: -x[1])
                for rank, (name, imp) in enumerate(imp_pairs[:15]):
                    src = "GRAD" if name in grad_cols else ("FUSE" if name in fusion_cols else "MT")
                    print(f"  {rank+1:2d}. [{src:4s}] {name:<45} imp = {imp:.4f}")

        # --- Per-subset breakdown ---
        print(f"\n{'Subset':<10} | {'Combo01':<10} | {'Loss':<10} | N")
        print("-" * 45)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["combo01_score"])
            r_combo = roc_auc_score(v["is_member"], v["combo01_score"]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            v2 = sub.dropna(subset=["neg_mean_loss"]) if "neg_mean_loss" in sub.columns else pd.DataFrame()
            r_loss = roc_auc_score(v2["is_member"], v2["neg_mean_loss"]) if not v2.empty and len(v2["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r_combo:.4f}     | {r_loss:.4f}     | {len(sub)}")

        # --- Comparison ---
        print(f"\n--- COMPARISON ---")
        v = df.dropna(subset=["combo01_score"])
        if not v.empty and len(v["is_member"].unique()) > 1:
            final_auc = roc_auc_score(v["is_member"], v["combo01_score"])
            print(f"  COMBO01 (memTrace+Grad RF): {final_auc:.4f}")
        print(f"  vs EXP50 memTrace RF:   0.6908 (current best)")
        print(f"  vs EXP43 AttenMIA:      0.6642")
        print(f"  vs EXP41 -grad_z_lang:  0.6539")
        print(f"  vs EXP11 -grad_embed:   0.6472")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO01_{timestamp}.parquet", index=False)
        print(f"\n[COMBO01] Results saved.")


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

    print(f"[COMBO01] memTrace+Gradient: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  1 fwd + 1 bwd pass/sample")
    Experiment(Args).run()
