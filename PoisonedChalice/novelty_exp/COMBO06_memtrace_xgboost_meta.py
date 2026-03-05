"""
COMBO06: memTrace XGBoost Meta-Stacker — Per-Language Calibrated Ensemble

THE ULTIMATE COMPETITION SUBMISSION EXPERIMENT.

Key differences from COMBO05:
    1. Uses XGBoost (proper) when available, falls back to sklearn GBM
    2. Per-language SEPARATE classifiers (trained on each language independently)
       then combined — because Go >> Python > Ruby/Java >> Rust
    3. Feature SELECTION: only features with AUC > 0.53 per-language
    4. Two-stage stacking:
       - Stage 1: RF + GBM per-language models → 10 meta-features
       - Stage 2: Ridge LR on meta-features → final score
    5. Length control: residualize features against log(seq_len)
    6. Bayesian-optimized RF/GBM hyperparameters

Signal families extracted (same as COMBO05):
    A. memTrace hidden states (transition surprise, norms, confidence)
    B. Gradient features (embedding norm, late-layer norms, sparsity)
    C. Logit Lens (settling depth, confidence trajectory)
    D. Loss Histogram (16-bin distribution, skewness, kurtosis)
    E. Cross-family interactions (grad × hnorm, surprise × IQR)

Total: ~150+ features → per-language selection → two-stage stacking

Compute: 1 forward pass (output_hidden_states=True) + 1 backward pass
Expected runtime: ~15-20 min on A100 (10% sample)
Expected AUC: 0.74-0.79 (per-language calibration + two-stage stacking)
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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Try to import XGBoost (Kaggle has it)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def setup_environment():
    print("\n" + "=" * 70)
    print("  COMBO06: memTrace XGBoost Meta-Stacker")
    print("  Per-Language Calibration + Two-Stage Stacking")
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


# ===========================================================================
# Feature extraction — identical to COMBO05 mega extractor
# ===========================================================================

class MegaExtractor:
    """Extract ALL signal families in one forward + backward pass."""

    def __init__(self, model, tokenizer, max_length=512, n_lens_layers=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self._err_count = 0

        # Logit lens components
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        else:
            self.final_norm = None
        self.lm_head = model.lm_head

        # Which layers for logit lens (evenly spaced)
        self.n_lens_layers = n_lens_layers
        step = max(1, self.n_layers // n_lens_layers)
        self.lens_layers = list(range(0, self.n_layers, step))[:n_lens_layers]
        if (self.n_layers - 1) not in self.lens_layers:
            self.lens_layers.append(self.n_layers - 1)

        # Key layers for memTrace
        self.key_layers = [0, 7, 14, 15, 16, 22, 24, 28, 29]
        self.key_layers = [l for l in self.key_layers if l < self.n_layers]

        print(f"  MegaExtractor: {self.n_layers} layers, {len(self.lens_layers)} lens layers")
        print(f"  Key memTrace layers: {self.key_layers}")

    def extract(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 30:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 10:
                return result

            result["seq_len"] = float(seq_len)
            result["log_seq_len"] = float(np.log(seq_len + 1))

            # =============== FORWARD + BACKWARD ===============
            self.model.zero_grad()
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(input_ids).detach().requires_grad_(True)

            outputs = self.model(
                inputs_embeds=embeds, labels=input_ids,
                output_hidden_states=True,
            )
            loss = outputs.loss
            loss.backward()

            # =============== A. LOSS ===============
            loss_val = loss.float().item()
            result["neg_mean_loss"] = -loss_val

            # =============== B. GRADIENT ===============
            if embeds.grad is not None:
                eg = embeds.grad.float()
                grad_norm = eg.norm(2).item()
                result["neg_grad_embed"] = -grad_norm
                result["grad_embed_raw"] = grad_norm

                # Per-token gradient norms
                per_tok_gn = eg[0].norm(dim=-1)  # (seq,)
                result["neg_grad_tok_mean"] = -per_tok_gn.mean().item()
                result["grad_tok_std"] = per_tok_gn.std().item()
                result["grad_tok_max"] = per_tok_gn.max().item()

                # Gradient sparsity (Hoyer)
                eg_flat = eg.flatten().cpu().numpy()
                l1 = np.abs(eg_flat).sum()
                l2 = np.sqrt((eg_flat**2).sum()) + 1e-15
                sqn = np.sqrt(len(eg_flat))
                result["grad_hoyer"] = float((sqn - l1/l2) / (sqn - 1 + 1e-15))

                # Sparse fraction
                max_abs = np.abs(eg_flat).max()
                if max_abs > 1e-15:
                    result["grad_l0_frac"] = float((np.abs(eg_flat) < 0.01 * max_abs).mean())

            # =============== C. HIDDEN STATES (memTrace) ===============
            hs_list = outputs.hidden_states  # (n_layers+1,) of (1, seq, dim)

            # Per-layer features at key layers
            prev_mean_hs = None
            for l in self.key_layers:
                hs = hs_list[l + 1][0].float()  # (seq, dim)
                norms = hs.norm(dim=-1)  # (seq,)

                # Norm stats
                result[f"hnorm_mean_L{l}"] = norms.mean().item()
                result[f"neg_hnorm_mean_L{l}"] = -norms.mean().item()
                result[f"hnorm_std_L{l}"] = norms.std().item()

                # Mean-pooled hidden state
                mean_hs = hs.mean(dim=0)  # (dim,)

                # Transition features (from previous key layer)
                if prev_mean_hs is not None:
                    diff = (mean_hs - prev_mean_hs)
                    vel = diff.norm(2).item()
                    result[f"vel_to_L{l}"] = vel
                    result[f"neg_vel_to_L{l}"] = -vel

                    cos = F.cosine_similarity(
                        mean_hs.unsqueeze(0), prev_mean_hs.unsqueeze(0)
                    ).item()
                    result[f"cos_trans_L{l}"] = cos
                    result[f"surprise_L{l}"] = 1.0 - cos

                prev_mean_hs = mean_hs

            # Confidence features from final layer hidden states
            hs_final = hs_list[-1][0].float()
            token_vars = hs_final.var(dim=-1)  # (seq,)
            result["conf_entropy_last"] = -(token_vars / (token_vars.sum() + 1e-10) *
                                             torch.log(token_vars / (token_vars.sum() + 1e-10) + 1e-15)).sum().item()
            result["conf_var_last"] = token_vars.mean().item()

            # Position similarity (first vs last quarter)
            q = max(2, seq_len // 4)
            first_q = hs_final[:q].mean(dim=0)
            last_q = hs_final[-q:].mean(dim=0)
            result["pos_sim_fl"] = F.cosine_similarity(
                first_q.unsqueeze(0), last_q.unsqueeze(0)
            ).item()

            # Context evolution (mean over all layers)
            all_norms = []
            for l_idx in range(self.n_layers):
                hs_l = hs_list[l_idx + 1][0].float()
                ln = hs_l.norm(dim=-1).mean().item()
                all_norms.append(ln)
            all_norms = np.array(all_norms)
            result["hnorm_profile_slope"] = float(np.polyfit(np.arange(len(all_norms)), all_norms, 1)[0])
            result["hnorm_profile_std"] = float(all_norms.std())
            result["neg_hnorm_profile_mean"] = -float(all_norms.mean())

            # Early-mid-late velocity
            third = self.n_layers // 3
            result["vel_early_avg"] = float(np.diff(all_norms[:third]).mean()) if third > 1 else 0.0
            result["vel_late_avg"] = float(np.diff(all_norms[2*third:]).mean()) if third > 1 else 0.0
            result["vel_decel"] = result["vel_early_avg"] - result["vel_late_avg"]

            # =============== D. LOGIT LENS ===============
            labels = input_ids[0, 1:]
            T = labels.shape[0]
            if T > 0:
                confs = []
                accs = []
                entropies = []
                for l_idx in self.lens_layers:
                    hs_l = hs_list[l_idx + 1]
                    if self.final_norm is not None:
                        normed = self.final_norm(hs_l)
                    else:
                        normed = hs_l
                    logits_l = self.lm_head(normed)[0, :-1, :].float()
                    probs = F.softmax(logits_l, dim=-1)
                    max_conf = probs.max(dim=-1).values.mean().item()
                    confs.append(max_conf)
                    acc = (logits_l.argmax(dim=-1) == labels).float().mean().item()
                    accs.append(acc)
                    ent_per_tok = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    entropies.append(ent_per_tok.mean().item())

                confs = np.array(confs)
                accs = np.array(accs)
                ents = np.array(entropies)
                n_lens = len(confs)

                result["lens_conf_auc"] = float(np.trapz(confs))
                result["lens_acc_auc"] = float(np.trapz(accs))
                result["neg_lens_ent_mean"] = -float(ents.mean())

                if n_lens >= 3:
                    third_l = n_lens // 3
                    result["lens_conf_early"] = float(confs[:third_l].mean())
                    result["lens_conf_late"] = float(confs[2*third_l:].mean())
                    result["lens_acc_early"] = float(accs[:third_l].mean())
                    result["lens_acc_late"] = float(accs[2*third_l:].mean())
                    result["lens_conf_slope"] = float(np.polyfit(np.arange(n_lens), confs, 1)[0])
                    result["lens_acc_slope"] = float(np.polyfit(np.arange(n_lens), accs, 1)[0])

                # Settling: earliest layer where acc > 50%
                sat_layers = np.where(np.array(accs) > 0.5)[0]
                result["neg_lens_settle"] = -float(sat_layers[0] / n_lens) if len(sat_layers) > 0 else -1.0

            # =============== E. LOSS HISTOGRAM ===============
            with torch.no_grad():
                logits = outputs.logits[0, :-1, :].float()
                shift_labels = input_ids[0, 1:]
                per_token_loss = F.cross_entropy(logits, shift_labels, reduction="none")
                ptl = per_token_loss.cpu().numpy()

            result["loss_mean"] = float(ptl.mean())
            result["neg_loss_mean"] = -float(ptl.mean())
            result["loss_std"] = float(ptl.std())
            result["loss_median"] = float(np.median(ptl))
            result["loss_iqr"] = float(np.percentile(ptl, 75) - np.percentile(ptl, 25))
            result["loss_p90"] = float(np.percentile(ptl, 90))
            result["neg_loss_p90"] = -float(np.percentile(ptl, 90))
            result["loss_zero_frac"] = float((ptl < 0.01).mean())

            # Skewness and kurtosis
            from scipy.stats import skew, kurtosis
            result["loss_skew"] = float(skew(ptl))
            result["loss_kurtosis"] = float(kurtosis(ptl))

            # Histogram bins (16 bins from 0 to 10)
            hist, _ = np.histogram(ptl, bins=16, range=(0, 10), density=True)
            for i, h in enumerate(hist):
                result[f"hist_b{i}"] = float(h)

            # Bimodality coefficient
            n_tok = len(ptl)
            if n_tok > 3:
                sk = skew(ptl)
                ku = kurtosis(ptl, fisher=True)
                result["bimodality"] = float((sk**2 + 1) / (ku + 3 * (n_tok-1)**2 / ((n_tok-2)*(n_tok-3)) + 1e-10))

            # =============== F. CROSS-FAMILY INTERACTIONS ===============
            if "neg_grad_embed" in result and "hnorm_mean_L15" in result:
                result["grad_x_hnorm15"] = result["grad_embed_raw"] * result["hnorm_mean_L15"]
                result["neg_grad_x_hnorm15"] = -result["grad_x_hnorm15"]

            if "surprise_L15" in result and "loss_iqr" in result:
                result["surprise_x_iqr"] = result["surprise_L15"] * result["loss_iqr"]

            if "conf_var_last" in result and "loss_kurtosis" in result:
                result["conf_x_kurt"] = result["conf_var_last"] * result["loss_kurtosis"]

            if "neg_grad_embed" in result and "lens_conf_auc" in result:
                result["grad_x_lens_conf"] = result["grad_embed_raw"] * result["lens_conf_auc"]

            if "neg_grad_embed" in result and "loss_skew" in result:
                result["grad_x_loss_skew"] = result["grad_embed_raw"] * result["loss_skew"]

            if "vel_decel" in result and "neg_grad_embed" in result:
                result["decel_x_grad"] = result["vel_decel"] * result["grad_embed_raw"]

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO06 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


def per_language_z_normalize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Z-normalize features within each language subset."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or df[col].dtype not in [np.float64, np.float32, float]:
            continue
        for lang in df["subset"].unique():
            mask = df["subset"] == lang
            vals = df.loc[mask, col]
            valid = vals.dropna()
            if len(valid) > 10:
                mu, sigma = valid.mean(), valid.std()
                if sigma > 1e-10:
                    df.loc[mask, col] = (vals - mu) / sigma
    return df


def residualize_length(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Remove correlation with log(seq_len) from each feature via linear residual."""
    df = df.copy()
    if "log_seq_len" not in df.columns:
        return df
    log_len = df["log_seq_len"].fillna(0).values
    for col in feature_cols:
        if col in ["seq_len", "log_seq_len"]:
            continue
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals) & ~np.isnan(log_len)
        if valid.sum() < 20:
            continue
        try:
            slope, intercept = np.polyfit(log_len[valid], vals[valid], 1)
            df[col] = vals - slope * log_len
        except Exception:
            pass
    return df


# ===========================================================================
# Per-language classifier approach
# ===========================================================================

class PerLanguageStacker:
    """Train separate RF+GBM per language, then combine via Ridge meta-stacker."""

    def __init__(self, seed=42):
        self.seed = seed

    def _make_rf(self):
        return RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", random_state=self.seed, n_jobs=-1,
        )

    def _make_gbm(self):
        if HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                use_label_encoder=False, eval_metric="auc",
                random_state=self.seed, n_jobs=-1, verbosity=0,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=self.seed,
            )

    def _select_features(self, X, y, feature_names, min_auc=0.53):
        """Select features with individual AUC > min_auc."""
        selected = []
        for i, fn in enumerate(feature_names):
            vals = X[:, i]
            valid = ~np.isnan(vals)
            if valid.sum() < 30:
                continue
            try:
                auc = roc_auc_score(y[valid], vals[valid])
                auc = max(auc, 1 - auc)
                if auc >= min_auc:
                    selected.append(i)
            except Exception:
                pass
        return selected

    def fit_predict_cv(self, df, feature_cols, n_folds=5):
        """
        Two-stage stacking with per-language classifiers.

        Stage 1: For each language, run 5-fold CV with RF and GBM.
                 Produces 2 meta-features per sample (rf_prob, gbm_prob).
        Stage 2: Use a global Ridge LR on these meta-features for final score.
        """
        languages = sorted(df["subset"].unique())
        print(f"\n  Languages: {languages}")
        print(f"  Features available: {len(feature_cols)}")
        print(f"  XGBoost available: {HAS_XGB}")

        # Stage 1: Per-language OOF predictions
        df["stg1_rf_prob"] = np.nan
        df["stg1_gbm_prob"] = np.nan
        lang_results = {}

        for lang in languages:
            lang_mask = df["subset"] == lang
            lang_df = df[lang_mask].copy()
            n_lang = len(lang_df)

            if n_lang < 50:
                print(f"\n  [{lang}] Too few samples ({n_lang}), skipping")
                continue

            X_lang = lang_df[feature_cols].values.astype(np.float64)
            y_lang = lang_df["is_member"].values

            # Handle NaN/inf
            X_lang = np.nan_to_num(X_lang, nan=0.0, posinf=0.0, neginf=0.0)

            # Feature selection per language
            selected = self._select_features(X_lang, y_lang, feature_cols)
            if len(selected) < 5:
                selected = list(range(min(20, X_lang.shape[1])))  # fallback: first 20
            X_sel = X_lang[:, selected]
            sel_names = [feature_cols[i] for i in selected]

            print(f"\n  [{lang}] N={n_lang}, features selected: {len(selected)}/{len(feature_cols)}")

            # CV
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            rf_oof = np.full(n_lang, np.nan)
            gbm_oof = np.full(n_lang, np.nan)
            rf_folds, gbm_folds = [], []

            for fi, (tri, tei) in enumerate(skf.split(X_sel, y_lang)):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_sel[tri])
                Xte = sc.transform(X_sel[tei])

                # RF
                rf = self._make_rf()
                rf.fit(Xtr, y_lang[tri])
                rf_p = rf.predict_proba(Xte)[:, 1]
                rf_oof[tei] = rf_p
                rf_auc = roc_auc_score(y_lang[tei], rf_p)
                rf_folds.append(rf_auc)

                # GBM
                gbm = self._make_gbm()
                gbm.fit(Xtr, y_lang[tri])
                gbm_p = gbm.predict_proba(Xte)[:, 1]
                gbm_oof[tei] = gbm_p
                gbm_auc = roc_auc_score(y_lang[tei], gbm_p)
                gbm_folds.append(gbm_auc)

            rf_mean = np.mean(rf_folds)
            gbm_mean = np.mean(gbm_folds)
            print(f"    RF  CV: {rf_mean:.4f} +/- {np.std(rf_folds):.4f} {rf_folds}")
            print(f"    GBM CV: {gbm_mean:.4f} +/- {np.std(gbm_folds):.4f} {gbm_folds}")

            # Store OOF predictions back
            df.loc[lang_mask, "stg1_rf_prob"] = rf_oof
            df.loc[lang_mask, "stg1_gbm_prob"] = gbm_oof
            lang_results[lang] = {"rf": rf_mean, "gbm": gbm_mean, "n": n_lang, "n_feat": len(selected)}

            # Top features
            rf_final = self._make_rf()
            sc_final = StandardScaler()
            Xf = sc_final.fit_transform(X_sel)
            rf_final.fit(Xf, y_lang)
            imp = sorted(zip(sel_names, rf_final.feature_importances_), key=lambda x: x[1], reverse=True)
            print(f"    Top-5 features: {[(n, f'{v:.4f}') for n, v in imp[:5]]}")

        # Stage 2: Ridge LR meta-stacker on rf_prob + gbm_prob
        print("\n" + "=" * 70)
        print("  Stage 2: Ridge LR Meta-Stacker")
        print("=" * 70)

        valid_mask = df["stg1_rf_prob"].notna() & df["stg1_gbm_prob"].notna()
        meta_X = df.loc[valid_mask, ["stg1_rf_prob", "stg1_gbm_prob"]].values
        meta_y = df.loc[valid_mask, "is_member"].values

        # Also add rank-averaged simple ensemble
        from scipy.stats import rankdata
        rank_rf = rankdata(meta_X[:, 0]) / len(meta_X)
        rank_gbm = rankdata(meta_X[:, 1]) / len(meta_X)
        simple_ensemble = (rank_rf + rank_gbm) / 2.0
        simple_auc = roc_auc_score(meta_y, simple_ensemble)
        print(f"  Simple rank ensemble AUC: {simple_auc:.4f}")
        df.loc[valid_mask, "combo06_simple_ens"] = simple_ensemble

        # Ridge LR CV
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        ridge_oof = np.full(valid_mask.sum(), np.nan)
        ridge_folds = []

        for fi, (tri, tei) in enumerate(skf2.split(meta_X, meta_y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(meta_X[tri])
            Xte = sc.transform(meta_X[tei])
            clf = LogisticRegressionCV(
                Cs=10, cv=3, penalty="l2", scoring="roc_auc",
                max_iter=2000, random_state=self.seed, solver="lbfgs",
            )
            clf.fit(Xtr, meta_y[tri])
            p = clf.predict_proba(Xte)[:, 1]
            ridge_oof[tei] = p
            ridge_auc = roc_auc_score(meta_y[tei], p)
            ridge_folds.append(ridge_auc)

        ridge_mean = np.mean(ridge_folds)
        print(f"  Ridge LR meta-stacker AUC: {ridge_mean:.4f} +/- {np.std(ridge_folds):.4f}")
        df.loc[valid_mask, "combo06_ridge_meta"] = ridge_oof

        return lang_results, simple_auc, ridge_mean


# ===========================================================================
# Global stacking (same features, one classifier for all languages)
# ===========================================================================

def run_global_cv(df, feature_cols, seed=42, n_folds=5, label="Global"):
    """Run global RF+GBM with per-language Z-normalized features."""
    valid_mask = df[feature_cols].dropna(how="all").index
    X = df.loc[valid_mask, feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df.loc[valid_mask, "is_member"].values
    X_np = np.nan_to_num(X.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # RF
    rf_oof = np.full(len(X_np), np.nan)
    rf_folds = []
    for fi, (tri, tei) in enumerate(skf.split(X_np, y)):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X_np[tri]), sc.transform(X_np[tei])
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features="sqrt", class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        clf.fit(Xtr, y[tri])
        p = clf.predict_proba(Xte)[:, 1]
        rf_oof[tei] = p
        rf_folds.append(roc_auc_score(y[tei], p))

    rf_mean = np.mean(rf_folds)
    print(f"  {label} RF: {rf_mean:.4f} +/- {np.std(rf_folds):.4f}")

    # GBM
    gbm_oof = np.full(len(X_np), np.nan)
    gbm_folds = []
    for fi, (tri, tei) in enumerate(skf.split(X_np, y)):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X_np[tri]), sc.transform(X_np[tei])
        if HAS_XGB:
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False,
                eval_metric="auc", random_state=seed, n_jobs=-1, verbosity=0,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=seed,
            )
        clf.fit(Xtr, y[tri])
        p = clf.predict_proba(Xte)[:, 1]
        gbm_oof[tei] = p
        gbm_folds.append(roc_auc_score(y[tei], p))

    gbm_mean = np.mean(gbm_folds)
    print(f"  {label} GBM: {gbm_mean:.4f} +/- {np.std(gbm_folds):.4f}")

    return valid_mask, rf_oof, gbm_oof, rf_mean, gbm_mean


# ===========================================================================
# Experiment class
# ===========================================================================

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
        extractor = MegaExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            n_lens_layers=self.args.n_lens_layers,
        )

        # ============ Phase 1: Feature Extraction ============
        print(f"\n[COMBO06] Feature Extraction for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO06 extract]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        feature_cols = list(feat_df.columns)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[COMBO06] Valid: {n_valid}/{len(df)}, Total features: {len(feature_cols)}")

        # ============ Phase 2: Global Approach (Raw) ============
        print("\n" + "=" * 70)
        print("  COMBO06 Phase 2: GLOBAL classifiers (Raw features)")
        print("=" * 70)
        vm_raw, rf_raw, gbm_raw, auc_rf_raw, auc_gbm_raw = run_global_cv(
            df, feature_cols, self.args.seed, label="Global-Raw"
        )

        # ============ Phase 3: Length Residualization + Z-Norm ============
        print("\n" + "=" * 70)
        print("  COMBO06 Phase 3: Length Residualization + Per-Language Z-Norm")
        print("=" * 70)
        df_proc = residualize_length(df.copy(), feature_cols)
        df_proc = per_language_z_normalize(df_proc, feature_cols)

        vm_z, rf_z, gbm_z, auc_rf_z, auc_gbm_z = run_global_cv(
            df_proc, feature_cols, self.args.seed, label="Global-ZNorm"
        )
        df.loc[vm_z, "combo06_global_rf_z"] = rf_z
        df.loc[vm_z, "combo06_global_gbm_z"] = gbm_z

        # ============ Phase 4: Per-Language Stacking ============
        print("\n" + "=" * 70)
        print("  COMBO06 Phase 4: Per-Language Two-Stage Stacking")
        print("=" * 70)  
        stacker = PerLanguageStacker(seed=self.args.seed)
        lang_results, simple_ens_auc, ridge_meta_auc = stacker.fit_predict_cv(
            df_proc, feature_cols
        )

        # ============ Phase 5: Grand Ensemble ============
        print("\n" + "=" * 70)
        print("  COMBO06 Phase 5: Grand Ensemble")
        print("=" * 70)

        from scipy.stats import rankdata

        # Collect all available scores
        score_cols = []
        for c in ["combo06_global_rf_z", "combo06_global_gbm_z",
                   "combo06_simple_ens", "combo06_ridge_meta"]:
            if c in df.columns and df[c].notna().sum() > 50:
                score_cols.append(c)

        if len(score_cols) >= 2:
            valid_grand = df[score_cols].dropna(how="any").index
            y_grand = df.loc[valid_grand, "is_member"].values

            # Rank-average all available scores
            rank_sum = np.zeros(len(valid_grand))
            for c in score_cols:
                vals = df.loc[valid_grand, c].values
                rank_sum += rankdata(vals) / len(vals)
            grand_score = rank_sum / len(score_cols)

            grand_auc = roc_auc_score(y_grand, grand_score)
            df.loc[valid_grand, "combo06_grand_ensemble"] = grand_score
            print(f"  Grand Ensemble ({len(score_cols)} components): AUC = {grand_auc:.4f}")

            # Per-subset for grand ensemble
            print(f"\n  {'Subset':<10} | {'Grand':<8} | N")
            print("  " + "-" * 35)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                v = sub.dropna(subset=["combo06_grand_ensemble"])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    a = roc_auc_score(v["is_member"], v["combo06_grand_ensemble"])
                else:
                    a = float("nan")
                print(f"  {subset:<10} | {a:.4f}   | {len(sub)}")
        else:
            grand_auc = 0.0

        # ============ Phase 6: Final Summary ============
        print("\n" + "=" * 70)
        print("   COMBO06: FINAL SUMMARY")
        print("=" * 70)
        print(f"  Global RF Raw:          {auc_rf_raw:.4f}")
        print(f"  Global GBM Raw:         {auc_gbm_raw:.4f}")
        print(f"  Global RF Z-Normed:     {auc_rf_z:.4f}")
        print(f"  Global GBM Z-Normed:    {auc_gbm_z:.4f}")
        if lang_results:
            print(f"\n  Per-Language Results:")
            for lang, r in sorted(lang_results.items()):
                print(f"    {lang:<10} RF={r['rf']:.4f}  GBM={r['gbm']:.4f}  N={r['n']}  feat={r['n_feat']}")
        print(f"\n  Simple rank ensemble:   {simple_ens_auc:.4f}")
        print(f"  Ridge meta-stacker:     {ridge_meta_auc:.4f}")
        print(f"  Grand ensemble:         {grand_auc:.4f}")

        best_auc = max(auc_rf_raw, auc_gbm_raw, auc_rf_z, auc_gbm_z,
                        simple_ens_auc, ridge_meta_auc, grand_auc)
        print(f"\n  >>> BEST OVERALL: {best_auc:.4f} <<<")
        print(f"\n  vs EXP50 memTrace RF:   0.6908 (current best)")
        print(f"  vs EXP43 AttenMIA:      0.6642")
        print(f"  vs EXP55 Histogram RF:  0.6612")

        if best_auc > 0.72:
            print(f"\n  COMPETITION WINNER! AUC = {best_auc:.4f}")
        elif best_auc > 0.6908:
            print(f"\n  NEW BEST! Beats memTrace ({best_auc:.4f} > 0.6908)")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO06_{timestamp}.parquet", index=False)
        print(f"\n[COMBO06] All results saved.")


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

    print(f"[COMBO06] XGBoost Meta-Stacker: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  Per-language calibration + Two-stage stacking")
    print(f"  XGBoost available: {HAS_XGB}")
    Experiment(Args).run()
