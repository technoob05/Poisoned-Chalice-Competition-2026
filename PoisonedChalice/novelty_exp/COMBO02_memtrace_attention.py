"""
COMBO02: memTrace + Attention — Hidden State Geometry × Attention Flow Fusion

Rationale:
    EXP50 memTrace (AUC 0.6908) captures hidden state GEOMETRY (norms,
    transitions between layers) but does NOT use attention weights.
    EXP43 AttenMIA (AUC 0.6642) captures attention FLOW patterns
    (concentration, barycenter drift, inter-layer transitions) but
    does NOT use hidden state norms or confidence features.

    These signal families are partially orthogonal:
    - Hidden states measure WHERE information is stored
    - Attention patterns measure HOW information flows between positions
    - A member might have normal hidden norms but distinctive attention
      focusing patterns (or vice versa)

    Architecture:
    - Single forward pass with output_hidden_states=True AND output_attentions=True
    - Requires attn_implementation="eager" for raw attention weights
    - max_length=512 (attention memory: 32 layers × 32 heads × 512² ≈ 1 GB)
    - No perturbation passes (EXP43 perturbation features were inverted/weak)
    - Use ONLY transitional + concentration features from attention (not perturbation)
    - Combine with full memTrace feature set

    Feature count: memTrace (~69) + attention (~20) = ~89 features
    RF classifier with 5-fold stratified CV

Compute: 1 forward pass (eager mode, hidden_states + attentions)
Expected runtime: ~6-10 min on A100 (10% sample)
Expected AUC: 0.71-0.74 (two strongest individual methods fused)
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
    print("  COMBO02: memTrace + Attention Fusion")
    print("  Hidden State Geometry × Attention Flow")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, attn_impl: str = "eager"):
    print(f"[*] Loading model: {model_path} (attn_impl={attn_impl})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded. dtype={dtype}, layers={n_layers}, hidden={hidden_dim}")
    return model, tokenizer


class MemTraceAttnExtractor:
    """Extract memTrace + attention features in a single forward pass.

    Feature families:
    A. memTrace (hidden states):
        1. Layer transition features (L2 surprise + cosine stability)
        2. Prediction confidence features (entropy, confidence variance)
        3. Hidden state norm statistics per layer
        4. Position-based features (first-last similarity)
        5. Context evolution (mean representation changes)
    B. Attention (from EXP43):
        6. KL concentration per layer (attention vs uniform)
        7. Transitional features (barycenter drift, KL between adjacent layers)
    C. Cross-family:
        8. Attention-state correlation (conc × norm, conc × surprise)
    """

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self._n_heads = None

    # ===== MEMTRACE FEATURES =====

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
                                   seq_len: int) -> Dict[str, float]:
        features = {}
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

    # ===== ATTENTION FEATURES (from EXP43, streamlined) =====

    def _compute_attention_concentration(self, attentions: tuple) -> Dict[str, float]:
        """KL(attention || uniform) per layer — how focused attention is."""
        features = {}
        n_layers = len(attentions)
        layer_conc = []

        for layer_idx, attn in enumerate(attentions):
            # attn shape: (1, n_heads, T, T)
            A = attn[0].float()  # (n_heads, T, T)
            H, T, _ = A.shape

            if self._n_heads is None:
                self._n_heads = H
                print(f"[COMBO02] Attention: {n_layers} layers, {H} heads")

            if T < 2:
                layer_conc.append(np.nan)
                continue

            eps = 1e-10
            kl_per_row = (A * torch.log(A + eps)).sum(dim=-1) + np.log(T)  # (H, T)
            skip = max(1, T // 10)
            kl_per_head = kl_per_row[:, skip:].mean(dim=-1)  # (H,)
            layer_conc.append(kl_per_head.mean().item())

        arr = np.array(layer_conc)
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            features["attn_conc_mean"] = float(valid.mean())
            features["attn_conc_std"] = float(valid.std())
            features["attn_conc_max"] = float(valid.max())
            features["attn_conc_late_mean"] = float(valid[-10:].mean()) if len(valid) >= 10 else float(valid.mean())
            if len(valid) >= 20:
                features["attn_conc_early_late_diff"] = float(valid[-10:].mean() - valid[:10].mean())

        return features, arr

    def _compute_attention_transitional(self, attentions: tuple) -> Dict[str, float]:
        """Transitional attention features: barycenter drift, KL between adjacent layers."""
        features = {}
        n_layers = len(attentions)
        bary_drifts = []
        kl_transitions = []

        for l in range(n_layers - 1):
            A_l = attentions[l][0].float()   # (H, T, T)
            A_l1 = attentions[l + 1][0].float()
            H, T, _ = A_l.shape

            if T < 2:
                continue

            # Barycenter drift
            positions = torch.arange(T, dtype=torch.float32, device=A_l.device).view(1, 1, T)
            bary_l = (A_l * positions).sum(dim=-1)    # (H, T)
            bary_l1 = (A_l1 * positions).sum(dim=-1)
            drift = (bary_l1 - bary_l).abs().mean().item()
            bary_drifts.append(drift)

            # KL between adjacent layers
            eps = 1e-10
            kl = (A_l * torch.log((A_l + eps) / (A_l1 + eps))).sum(dim=-1).mean().item()
            kl_transitions.append(kl)

        if bary_drifts:
            arr_bary = np.array(bary_drifts)
            features["attn_bary_drift_mean"] = float(arr_bary.mean())
            features["attn_bary_drift_std"] = float(arr_bary.std())
            features["attn_bary_drift_max"] = float(arr_bary.max())
            # Late-layer drift
            late_start = max(0, len(arr_bary) - 10)
            features["attn_bary_drift_late"] = float(arr_bary[late_start:].mean())

        if kl_transitions:
            arr_kl = np.array(kl_transitions)
            features["attn_kl_trans_mean"] = float(arr_kl.mean())
            features["attn_kl_trans_std"] = float(arr_kl.std())
            features["attn_kl_trans_late"] = float(arr_kl[-10:].mean()) if len(arr_kl) >= 10 else float(arr_kl.mean())

        return features

    # ===== CROSS-FAMILY FEATURES =====

    def _compute_cross_features(self, features: Dict[str, float],
                                layer_conc: np.ndarray,
                                hidden_states: List[torch.Tensor],
                                seq_len: int) -> Dict[str, float]:
        """Cross-family: attention concentration × hidden state norm correlation."""
        cross = {}

        # Compute per-layer hidden norm for correlation
        layer_norms = []
        for i, hs in enumerate(hidden_states[:-1]):  # exclude final embedding
            h = hs[0, :seq_len, :].float()
            layer_norms.append(h.norm(dim=-1).mean().item())
        arr_norms = np.array(layer_norms)

        # Correlation between attention concentration and hidden norm
        n = min(len(layer_conc), len(arr_norms))
        if n > 3:
            conc_valid = layer_conc[:n]
            norms_valid = arr_norms[:n]
            mask = np.isfinite(conc_valid) & np.isfinite(norms_valid)
            if mask.sum() > 3:
                corr = np.corrcoef(conc_valid[mask], norms_valid[mask])[0, 1]
                cross["attn_norm_correlation"] = float(corr) if np.isfinite(corr) else 0.0

        # Product features
        if "attn_conc_mean" in features and "hnorm_global_mean" in features:
            cross["attn_conc_x_hnorm"] = features["attn_conc_mean"] * features["hnorm_global_mean"]
        if "attn_conc_mean" in features and "surprise_global_mean" in features:
            cross["attn_conc_x_surprise"] = features["attn_conc_mean"] * features["surprise_global_mean"]

        return cross

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract ALL features in one forward pass."""
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

            # Single forward pass with BOTH hidden states and attention
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
            )

            hidden_states = outputs.hidden_states
            logits = outputs.logits
            attentions = outputs.attentions

            # A. memTrace features
            result.update(self._compute_transition_features(hidden_states, seq_len))
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))
            result.update(self._compute_hidden_norm_features(hidden_states, seq_len))
            result.update(self._compute_position_features(hidden_states, seq_len))
            result.update(self._compute_context_evolution(hidden_states, seq_len))
            result["seq_len"] = float(seq_len)

            # B. Attention features
            conc_features, layer_conc = self._compute_attention_concentration(attentions)
            result.update(conc_features)
            result.update(self._compute_attention_transitional(attentions))

            # C. Cross-family features
            result.update(self._compute_cross_features(result, layer_conc, hidden_states, seq_len))

            # Free attention memory
            del attentions
            torch.cuda.empty_cache()

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO02 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name, args.attn_impl)

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
        extractor = MemTraceAttnExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[COMBO02] Extracting memTrace+Attention features for {len(df)} samples...")
        print(f"  attn_implementation=eager, output_hidden_states=True+output_attentions=True")
        print(f"  memTrace (~69 features) + Attention (~20 features) + Cross (~3 features)")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO02]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[COMBO02] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[COMBO02] Errors: {extractor._err_count}")

        feature_cols = list(feat_df.columns)
        print(f"[COMBO02] Total features: {len(feature_cols)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised AUCs ---
        print("\n" + "=" * 70)
        print("   COMBO02: UNSUPERVISED SIGNAL AUCs")
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
        print("\nTop 20 features:")
        for col, (auc, direction) in top_features:
            src = "ATTN" if "attn" in col else ("CROSS" if "_x_" in col or "correlation" in col else "MT")
            print(f"  [{src:5s}] {direction}{col:<45} AUC = {auc:.4f}")

        # --- RF 5-fold CV ---
        print("\n" + "=" * 70)
        print("   COMBO02: RANDOM FOREST (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy()
        y_all = df.loc[valid_mask, "is_member"].values

        X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)
        feature_names = list(X_all.columns)
        X_np = X_all.values.astype(np.float64)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        attn_col_names = [c for c in feature_names if "attn" in c or "_x_" in c or "correlation" in c]
        mt_col_names = [c for c in feature_names if c not in attn_col_names]

        configs = {
            "ALL (memTrace+Attn)": feature_names,
            "memTrace only": mt_col_names,
            "Attention only": attn_col_names,
        }

        for config_name, cols in configs.items():
            col_idx = [feature_names.index(c) for c in cols if c in feature_names]
            if len(col_idx) < 2:
                print(f"\n  {config_name}: skipped (too few features)")
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

            if config_name == "ALL (memTrace+Attn)":
                df.loc[valid_mask, "combo02_score"] = all_scores

                print(f"\n--- Top 15 Feature Importances ---")
                importances = clf.feature_importances_
                used_names = [cols[i] for i in range(len(cols))]
                imp_pairs = sorted(zip(used_names, importances), key=lambda x: -x[1])
                for rank, (name, imp) in enumerate(imp_pairs[:15]):
                    src = "ATTN" if name in attn_col_names else "MT"
                    print(f"  {rank+1:2d}. [{src:4s}] {name:<45} imp = {imp:.4f}")

        # --- Per-subset ---
        print(f"\n{'Subset':<10} | {'Combo02':<10} | {'Loss':<10} | N")
        print("-" * 45)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["combo02_score"])
            r_combo = roc_auc_score(v["is_member"], v["combo02_score"]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            v2 = sub.dropna(subset=["neg_mean_loss"]) if "neg_mean_loss" in sub.columns else pd.DataFrame()
            r_loss = roc_auc_score(v2["is_member"], v2["neg_mean_loss"]) if not v2.empty and len(v2["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r_combo:.4f}     | {r_loss:.4f}     | {len(sub)}")

        # --- Comparison ---
        print(f"\n--- COMPARISON ---")
        v = df.dropna(subset=["combo02_score"])
        if not v.empty and len(v["is_member"].unique()) > 1:
            final_auc = roc_auc_score(v["is_member"], v["combo02_score"])
            print(f"  COMBO02 (memTrace+Attn RF):  {final_auc:.4f}")
        print(f"  vs EXP50 memTrace RF:    0.6908")
        print(f"  vs EXP43 AttenMIA:       0.6642")
        print(f"  vs EXP11 -grad_embed:    0.6472")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO02_{timestamp}.parquet", index=False)
        print(f"\n[COMBO02] Results saved.")


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
        attn_impl = "eager"
        seed = 42

    print(f"[COMBO02] memTrace+Attention: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  attn_impl={Args.attn_impl}, 1 fwd pass (hidden+attn)")
    Experiment(Args).run()
