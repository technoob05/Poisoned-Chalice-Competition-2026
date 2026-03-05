"""
COMBO03: memTrace + Logit Lens — Hidden State Geometry × Vocabulary-Space Settling

Rationale:
    EXP50 memTrace (AUC 0.6908) measures hidden state GEOMETRY (norms,
    transitions) but does not look at what the model PREDICTS at
    intermediate layers.

    Logit Lens (Nostalgebraist, 2020) projects hidden states through
    lm_head at each layer to see per-layer vocabulary predictions.
    For members (memorized), intermediate layers should:
    1. "Settle" to the correct prediction EARLIER (lower settling depth)
    2. Show higher AGREEMENT between adjacent layers (stable predictions)
    3. Have a distinctive confidence TRAJECTORY (rapid rise then plateau)

    This is a VOCABULARY-SPACE signal, orthogonal to the GEOMETRY-SPACE
    signals in memTrace. The hidden states are the same, but we extract
    DIFFERENT information from them:
    - memTrace: "how do norms/angles change?"
    - Logit lens: "what does the model predict at each layer?"

    Features:
    1. Settling depth: first layer where top-1 prediction matches final layer
    2. Prediction agreement: fraction of layers that agree with final prediction
    3. Confidence trajectory: how top-1 probability evolves across layers
    4. Entropy trajectory: how prediction entropy evolves across layers
    5. Rank trajectory: rank of correct token across layers

Compute: 1 forward pass (output_hidden_states=True) + lm_head projections
Expected runtime: ~5-8 min on A100 (10% sample, slightly heavier than EXP50)
Expected AUC: 0.70-0.73 (vocabulary-space settling complements geometry)
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
    print("  COMBO03: memTrace + Logit Lens Fusion")
    print("  Hidden State Geometry × Vocabulary-Space Settling")
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


class MemTraceLensExtractor:
    """Extract memTrace features PLUS logit lens features.

    Feature families:
    A. memTrace (hidden state geometry):
        1. Layer transition features (L2 surprise + cosine stability)
        2. Prediction confidence features (entropy, confidence variance)
        3. Hidden state norm statistics per layer
        4. Position-based features (first-last similarity)
        5. Context evolution (mean representation changes)
    B. Logit Lens (vocabulary-space predictions):
        6. Settling depth (when top-1 matches final layer)
        7. Prediction agreement profile (fraction matching final)
        8. Confidence trajectory across layers
        9. Entropy trajectory across layers
        10. Correct-token rank trajectory
    """

    def __init__(self, model, tokenizer, max_length: int = 512,
                 n_lens_layers: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_lens_layers = n_lens_layers  # sample this many layers for logit lens
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

        # Get lm_head and final layer norm for logit lens projections
        self.lm_head = model.lm_head
        self.final_norm = model.model.norm

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
                mid_surprise / (edge_surprise + 1e-10))
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
                confidence.mean().item() / confidence.std().item())
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
                h[0].unsqueeze(0), h[seq_len - 1].unsqueeze(0)).item()
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

    # ===== LOGIT LENS FEATURES (NEW) =====

    def _compute_logit_lens_features(self, hidden_states: List[torch.Tensor],
                                     input_ids: torch.Tensor,
                                     seq_len: int) -> Dict[str, float]:
        """Logit lens: project hidden states through lm_head at sampled layers.

        For each sampled layer, we get per-layer vocabulary predictions and
        extract features about how quickly the model "settles" on its answers.
        """
        features = {}

        # Sample layers to probe (evenly spaced + always include first/last)
        total_layers = len(hidden_states)  # includes embedding (index 0)
        # layer 0 = embedding output, layer i = after transformer block i-1
        # layer total_layers-1 = final layer (same as model output)

        if total_layers < 4:
            return features

        # Select n_lens_layers evenly spaced
        indices = np.linspace(1, total_layers - 1, self.n_lens_layers, dtype=int)
        indices = sorted(set(indices))  # dedup

        shift_labels = input_ids[0, 1:seq_len]  # (T-1,)
        T = shift_labels.shape[0]
        if T < 3:
            return features

        # Get final layer predictions (ground truth for settling)
        final_hs = hidden_states[-1][0, :seq_len, :].float()
        # Apply final layer norm (required for logit lens)
        final_normed = self.final_norm(final_hs.to(self.final_norm.weight.dtype))
        final_logits = self.lm_head(final_normed).float()
        final_preds = final_logits[:-1].argmax(dim=-1)  # (T-1,) final layer top-1

        # Per-layer statistics
        layer_confidences = []
        layer_entropies = []
        layer_agreements = []
        layer_correct_ranks = []
        settling_depths = []  # per-token: first layer matching final prediction

        # Track per-token settling
        token_settled = torch.zeros(T, dtype=torch.bool)
        token_settle_layer = torch.full((T,), float(len(indices)), dtype=torch.float32)

        for depth_idx, layer_idx in enumerate(indices):
            hs = hidden_states[layer_idx][0, :seq_len, :].float()
            # Apply final layer norm + lm_head (logit lens projection)
            normed = self.final_norm(hs.to(self.final_norm.weight.dtype))
            logits_l = self.lm_head(normed).float()

            shift_logits_l = logits_l[:-1]  # (T-1, V)
            probs_l = F.softmax(shift_logits_l, dim=-1)

            # Confidence (max prob)
            conf = probs_l.max(dim=-1).values
            layer_confidences.append(conf.mean().item())

            # Entropy
            log_probs_l = F.log_softmax(shift_logits_l, dim=-1)
            ent = -(probs_l * log_probs_l).sum(dim=-1)
            layer_entropies.append(ent.mean().item())

            # Agreement with final layer
            preds_l = shift_logits_l.argmax(dim=-1)
            agreement = (preds_l == final_preds).float().mean().item()
            layer_agreements.append(agreement)

            # Correct token rank
            correct_probs = probs_l.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Rank = how many tokens have higher probability
            ranks = (probs_l > correct_probs.unsqueeze(-1)).sum(dim=-1).float()
            layer_correct_ranks.append(ranks.mean().item())

            # Settling detection
            newly_settled = (~token_settled) & (preds_l == final_preds)
            token_settle_layer[newly_settled] = float(depth_idx)
            token_settled = token_settled | (preds_l == final_preds)

        # === Logit Lens Features ===

        # 1. Settling depth features
        settle_arr = token_settle_layer.numpy()
        features["lens_settle_mean"] = float(settle_arr.mean())
        features["lens_settle_std"] = float(settle_arr.std())
        features["lens_settle_median"] = float(np.median(settle_arr))
        features["lens_settle_min"] = float(settle_arr.min())
        # Fraction of tokens that settle by mid-point
        mid_idx = len(indices) // 2
        features["lens_early_settle_frac"] = float((settle_arr <= mid_idx).mean())

        # 2. Agreement trajectory
        agree_arr = np.array(layer_agreements)
        features["lens_agreement_mean"] = float(agree_arr.mean())
        features["lens_agreement_std"] = float(agree_arr.std())
        features["lens_agreement_early"] = float(agree_arr[:len(agree_arr)//2].mean())
        features["lens_agreement_late"] = float(agree_arr[len(agree_arr)//2:].mean())
        features["lens_agreement_slope"] = float(
            np.polyfit(np.arange(len(agree_arr)), agree_arr, 1)[0]
        ) if len(agree_arr) > 1 else 0.0

        # 3. Confidence trajectory
        conf_arr = np.array(layer_confidences)
        features["lens_conf_mean"] = float(conf_arr.mean())
        features["lens_conf_std"] = float(conf_arr.std())
        features["lens_conf_early"] = float(conf_arr[:len(conf_arr)//2].mean())
        features["lens_conf_late"] = float(conf_arr[len(conf_arr)//2:].mean())
        features["lens_conf_slope"] = float(
            np.polyfit(np.arange(len(conf_arr)), conf_arr, 1)[0]
        ) if len(conf_arr) > 1 else 0.0
        # Confidence rise = late minus early
        features["lens_conf_rise"] = features["lens_conf_late"] - features["lens_conf_early"]

        # 4. Entropy trajectory
        ent_arr = np.array(layer_entropies)
        features["lens_entropy_mean"] = float(ent_arr.mean())
        features["lens_entropy_std"] = float(ent_arr.std())
        features["lens_entropy_early"] = float(ent_arr[:len(ent_arr)//2].mean())
        features["lens_entropy_late"] = float(ent_arr[len(ent_arr)//2:].mean())
        features["lens_entropy_slope"] = float(
            np.polyfit(np.arange(len(ent_arr)), ent_arr, 1)[0]
        ) if len(ent_arr) > 1 else 0.0

        # 5. Correct-token rank trajectory
        rank_arr = np.array(layer_correct_ranks)
        features["lens_rank_mean"] = float(rank_arr.mean())
        features["lens_rank_std"] = float(rank_arr.std())
        features["lens_rank_early"] = float(rank_arr[:len(rank_arr)//2].mean())
        features["lens_rank_late"] = float(rank_arr[len(rank_arr)//2:].mean())
        features["lens_rank_slope"] = float(
            np.polyfit(np.arange(len(rank_arr)), rank_arr, 1)[0]
        ) if len(rank_arr) > 1 else 0.0

        return features

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract memTrace + logit lens features."""
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

            # A. memTrace features
            result.update(self._compute_transition_features(hidden_states, seq_len))
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))
            result.update(self._compute_hidden_norm_features(hidden_states, seq_len))
            result.update(self._compute_position_features(hidden_states, seq_len))
            result.update(self._compute_context_evolution(hidden_states, seq_len))
            result["seq_len"] = float(seq_len)

            # B. Logit Lens features
            result.update(self._compute_logit_lens_features(hidden_states, input_ids, seq_len))

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO03 WARN] {type(e).__name__}: {e}")
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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = MemTraceLensExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            n_lens_layers=self.args.n_lens_layers,
        )

        print(f"\n[COMBO03] Extracting memTrace+LogitLens features for {len(df)} samples...")
        print(f"  memTrace: transition + confidence + norms + position + evolution")
        print(f"  LogitLens: {self.args.n_lens_layers} layer projections through lm_head")
        print(f"  1 forward pass/sample (output_hidden_states=True)")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO03]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[COMBO03] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[COMBO03] Errors: {extractor._err_count}")

        feature_cols = list(feat_df.columns)
        print(f"[COMBO03] Total features: {len(feature_cols)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised AUCs ---
        print("\n" + "=" * 70)
        print("   COMBO03: UNSUPERVISED SIGNAL AUCs")
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
            src = "LENS" if "lens" in col else "MT"
            print(f"  [{src:4s}] {direction}{col:<50} AUC = {auc:.4f}")

        # --- RF 5-fold CV ---
        print("\n" + "=" * 70)
        print("   COMBO03: RANDOM FOREST (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy()
        y_all = df.loc[valid_mask, "is_member"].values
        X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)
        feature_names = list(X_all.columns)
        X_np = X_all.values.astype(np.float64)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        lens_cols = [c for c in feature_names if "lens" in c]
        mt_cols = [c for c in feature_names if c not in lens_cols]

        configs = {
            "ALL (memTrace+Lens)": feature_names,
            "memTrace only": mt_cols,
            "Logit Lens only": lens_cols,
        }

        for config_name, cols in configs.items():
            col_idx = [feature_names.index(c) for c in cols if c in feature_names]
            if len(col_idx) < 2:
                print(f"\n  {config_name}: skipped (too few features)")
                continue
            X_cfg = X_np[:, col_idx]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
            fold_aucs = []
            all_scores = np.full(len(X_cfg), np.nan)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_cfg, y_all)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_cfg[train_idx])
                X_te = scaler.transform(X_cfg[test_idx])
                y_tr, y_te = y_all[train_idx], y_all[test_idx]

                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    min_samples_split=8, min_samples_leaf=4,
                    max_features="sqrt", class_weight="balanced",
                    random_state=self.args.seed, n_jobs=-1,
                )
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te, proba)
                fold_aucs.append(auc)
                all_scores[test_idx] = proba

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            print(f"\n  {config_name} ({len(col_idx)} features):")
            print(f"    CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
            for fi, a in enumerate(fold_aucs):
                print(f"    Fold {fi+1}: {a:.4f}")

            if config_name == "ALL (memTrace+Lens)":
                df.loc[valid_mask, "combo03_score"] = all_scores
                print(f"\n--- Top 15 Feature Importances ---")
                importances = clf.feature_importances_
                used_names = [cols[i] for i in range(len(cols))]
                imp_pairs = sorted(zip(used_names, importances), key=lambda x: -x[1])
                for rank, (name, imp) in enumerate(imp_pairs[:15]):
                    src = "LENS" if name in lens_cols else "MT"
                    print(f"  {rank+1:2d}. [{src:4s}] {name:<45} imp = {imp:.4f}")

        # --- Per-subset ---
        print(f"\n{'Subset':<10} | {'Combo03':<10} | {'Loss':<10} | N")
        print("-" * 45)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["combo03_score"])
            r1 = roc_auc_score(v["is_member"], v["combo03_score"]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            v2 = sub.dropna(subset=["neg_mean_loss"]) if "neg_mean_loss" in sub.columns else pd.DataFrame()
            r2 = roc_auc_score(v2["is_member"], v2["neg_mean_loss"]) if not v2.empty and len(v2["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r1:.4f}     | {r2:.4f}     | {len(sub)}")

        # --- Comparison ---
        print(f"\n--- COMPARISON ---")
        v = df.dropna(subset=["combo03_score"])
        if not v.empty and len(v["is_member"].unique()) > 1:
            final_auc = roc_auc_score(v["is_member"], v["combo03_score"])
            print(f"  COMBO03 (memTrace+Lens RF): {final_auc:.4f}")
        print(f"  vs EXP50 memTrace RF:   0.6908")
        print(f"  vs EXP43 AttenMIA:      0.6642")
        print(f"  vs EXP55 Histogram RF:  0.6612")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO03_{timestamp}.parquet", index=False)
        print(f"\n[COMBO03] Results saved.")


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
        n_lens_layers = 8
        output_dir = "results"
        seed = 42

    print(f"[COMBO03] memTrace+LogitLens: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  lens_layers={Args.n_lens_layers}")
    Experiment(Args).run()
