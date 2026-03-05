"""
COMBO05: memTrace MEGA — Ultimate Multi-Signal Fusion MIA

THE COMPETITION WINNER EXPERIMENT.

Combines ALL proven signal families into one massive feature extraction
pipeline with the strongest possible classifier ensemble:

Signal Families (6 orthogonal sources):
    A. memTrace hidden states (AUC 0.6908 alone):
       - Layer transition surprise/stability
       - Hidden state norms per layer
       - Confidence features (entropy, variance)
       - Position similarity, context evolution
    B. Gradient features (AUC 0.6472 alone):
       - Embedding gradient norm
       - Late-layer gradient norms (L24, L28, L29)
       - Gradient sparsity (Hoyer)
    C. Logit Lens (novel vocabulary-space signals):
       - Settling depth, prediction agreement
       - Confidence/entropy trajectory across layers
       - Correct-token rank trajectory
    D. Loss Histogram (AUC 0.6612 alone):
       - 16-bin distribution + aggregate stats
       - Skewness, kurtosis, bimodality
    E. Per-language Z-normalization (+0.012 proven):
       - Z-normalize numeric features within each language
       - Removes cross-language scale differences
    F. Cross-family interaction features:
       - grad × hidden norm, surprise × IQR
       - confidence variance × kurtosis

Total: ~150+ features → RF + XGBoost comparison
Per-language Z-norm applied before classifier

Compute: 1 forward pass (output_hidden_states=True) + 1 backward pass
Expected runtime: ~12-18 min on A100 (10% sample)
Expected AUC: 0.73-0.78 (6 orthogonal signal families + proven classifier)
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  COMBO05: memTrace MEGA — Ultimate Multi-Signal Fusion")
    print("  6 Signal Families × RF/XGBoost Classifier")
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


class MegaExtractor:
    """Extract features from ALL proven signal families.

    A. memTrace (hidden state geometry) — 69 features
    B. Gradient (loss landscape) — 10 features
    C. Logit Lens (vocabulary predictions) — 25 features
    D. Loss Histogram (token distribution) — 36 features
    E. Cross-family interactions — 8 features
    Total: ~148 features
    """

    N_BINS = 16
    BIN_EDGES = np.linspace(-14, 0, N_BINS + 1)

    def __init__(self, model, tokenizer, max_length: int = 512,
                 n_lens_layers: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_lens_layers = n_lens_layers
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size
        self.lm_head = model.lm_head
        self.final_norm = model.model.norm

    # ===== A. MEMTRACE FEATURES =====

    def _compute_transition_features(self, hidden_states, seq_len):
        features = {}
        n_t = len(hidden_states) - 1
        all_s, all_c = [], []
        for i in range(n_t):
            h0 = hidden_states[i][0, :seq_len, :].float()
            h1 = hidden_states[i+1][0, :seq_len, :].float()
            s = (h1 - h0).norm(dim=-1)
            c = F.cosine_similarity(h0, h1, dim=-1)
            all_s.append(s.mean().item())
            all_c.append(c.mean().item())
            if i in [0, self.n_layers//4, self.n_layers//2, 3*self.n_layers//4, self.n_layers-1]:
                t = f"L{i}"
                features[f"surprise_mean_{t}"] = s.mean().item()
                features[f"surprise_std_{t}"] = s.std().item()
                features[f"surprise_max_{t}"] = s.max().item()
                features[f"stability_mean_{t}"] = c.mean().item()
                features[f"stability_std_{t}"] = c.std().item()
                features[f"stability_min_{t}"] = c.min().item()
        arr_s, arr_c = np.array(all_s), np.array(all_c)
        features["surprise_global_mean"] = float(arr_s.mean())
        features["surprise_global_std"] = float(arr_s.std())
        features["surprise_global_max"] = float(arr_s.max())
        features["surprise_argmax"] = float(arr_s.argmax())
        features["stability_global_mean"] = float(arr_c.mean())
        features["stability_global_std"] = float(arr_c.std())
        features["stability_global_min"] = float(arr_c.min())
        features["stability_argmin"] = float(arr_c.argmin())
        ms, me = self.n_layers//3, 2*self.n_layers//3
        if me > ms:
            features["surprise_mid_edge_ratio"] = float(
                arr_s[ms:me].mean() / (np.concatenate([arr_s[:ms], arr_s[me:]]).mean() + 1e-10))
        return features

    def _compute_confidence_features(self, logits, input_ids, seq_len):
        features = {}
        sl = logits[0, :seq_len-1, :].float()
        labels = input_ids[0, 1:seq_len]
        T = sl.shape[0]
        if T < 3: return features
        probs = F.softmax(sl, dim=-1)
        lp = F.log_softmax(sl, dim=-1)
        ent = -(probs * lp).sum(dim=-1)
        features.update({"entropy_mean": ent.mean().item(), "entropy_std": ent.std().item(),
                         "entropy_min": ent.min().item(), "entropy_max": ent.max().item()})
        conf = probs.max(dim=-1).values
        features.update({"confidence_mean": conf.mean().item(), "confidence_std": conf.std().item(),
                         "confidence_min": conf.min().item(), "confidence_max": conf.max().item()})
        features["confidence_stability"] = float(conf.mean().item() / max(conf.std().item(), 1e-10))
        top2 = torch.topk(probs, k=2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]
        features["confidence_gap_mean"] = gap.mean().item()
        features["confidence_gap_std"] = gap.std().item()
        tll = lp.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        features["neg_mean_loss"] = tll.mean().item()
        features["loss_std"] = (-tll).std().item()
        return features

    def _compute_hidden_norm_features(self, hidden_states, seq_len):
        features = {}
        lnm = []
        for i, hs in enumerate(hidden_states):
            h = hs[0, :seq_len, :].float()
            norms = h.norm(dim=-1)
            nm = norms.mean().item()
            lnm.append(nm)
            if i in [0, self.n_layers//4, self.n_layers//2, 3*self.n_layers//4, self.n_layers]:
                features[f"hnorm_mean_L{i}"] = nm
                features[f"hnorm_std_L{i}"] = norms.std().item()
        arr = np.array(lnm)
        features["hnorm_global_mean"] = float(arr.mean())
        features["hnorm_global_std"] = float(arr.std())
        features["hnorm_growth_ratio"] = float(arr[-1] / (arr[0] + 1e-10))
        # Per-region norms
        thirds = len(arr) // 3
        features["hnorm_early"] = float(arr[:thirds].mean())
        features["hnorm_mid"] = float(arr[thirds:2*thirds].mean())
        features["hnorm_late"] = float(arr[2*thirds:].mean())
        return features

    def _compute_position_features(self, hidden_states, seq_len):
        features = {}
        sims = []
        for i in [0, self.n_layers//4, self.n_layers//2, 3*self.n_layers//4, self.n_layers]:
            if i >= len(hidden_states) or seq_len < 2: continue
            h = hidden_states[i][0, :seq_len, :].float()
            s = F.cosine_similarity(h[0].unsqueeze(0), h[seq_len-1].unsqueeze(0)).item()
            features[f"first_last_sim_L{i}"] = s
            sims.append(s)
        if sims:
            features["first_last_sim_mean"] = float(np.mean(sims))
            features["first_last_sim_std"] = float(np.std(sims))
        return features

    def _compute_context_evolution(self, hidden_states, seq_len):
        features = {}
        li = self.n_layers // 2
        h = hidden_states[li][0, :seq_len, :].float()
        if seq_len < 5: return features
        evos = []
        pm = h[0].unsqueeze(0)
        for p in [seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]:
            if p < 1 or p >= seq_len: continue
            cm = h[:p+1].mean(dim=0, keepdim=True)
            evos.append((cm - pm).norm().item())
            pm = cm
        if evos:
            features["ctx_evolution_mean"] = float(np.mean(evos))
            features["ctx_evolution_std"] = float(np.std(evos))
        return features

    # ===== B. GRADIENT FEATURES =====

    def _compute_gradient_features(self, text: str) -> Dict[str, float]:
        features = {}
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 5: return features

            self.model.zero_grad()
            embed_layer = self.model.model.embed_tokens
            embed_out = embed_layer(input_ids)
            embed_out.requires_grad_(True)
            embed_out.retain_grad()

            hidden = embed_out
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            for layer in self.model.model.layers:
                layer_out = layer(hidden, position_ids=position_ids)
                hidden = layer_out[0]
            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)
            loss = F.cross_entropy(logits[0, :-1, :], input_ids[0, 1:])
            loss.backward()

            if embed_out.grad is not None:
                g = embed_out.grad[0].float()
                features["grad_norm_embed"] = g.norm(dim=-1).mean().item()
                flat = g.reshape(-1)
                l1, l2 = flat.abs().sum().item(), flat.norm().item()
                n = flat.numel()
                if l2 > 1e-10 and n > 1:
                    features["grad_sparsity_embed"] = float((np.sqrt(n) - l1/l2) / (np.sqrt(n) - 1))

            for li, tag in {24: "L24", 28: "L28", 29: "L29"}.items():
                if li < len(self.model.model.layers):
                    gn = sum(p.grad.float().norm().item()**2 for p in self.model.model.layers[li].parameters() if p.grad is not None)
                    features[f"grad_norm_{tag}"] = np.sqrt(gn)

            hgn = sum(p.grad.float().norm().item()**2 for p in self.model.lm_head.parameters() if p.grad is not None)
            features["grad_norm_head"] = np.sqrt(hgn)

            fngn = sum(p.grad.float().norm().item()**2 for p in self.model.model.norm.parameters() if p.grad is not None)
            features["grad_norm_final_norm"] = np.sqrt(fngn)

            self.model.zero_grad()
            if embed_out.grad is not None:
                embed_out.grad = None

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[MEGA GRAD WARN] {type(e).__name__}: {e}")
        return features

    # ===== C. LOGIT LENS FEATURES =====

    def _compute_logit_lens_features(self, hidden_states, input_ids, seq_len):
        features = {}
        total_layers = len(hidden_states)
        if total_layers < 4: return features

        indices = sorted(set(np.linspace(1, total_layers-1, self.n_lens_layers, dtype=int)))
        shift_labels = input_ids[0, 1:seq_len]
        T = shift_labels.shape[0]
        if T < 3: return features

        final_hs = hidden_states[-1][0, :seq_len, :].float()
        final_normed = self.final_norm(final_hs.to(self.final_norm.weight.dtype))
        final_logits = self.lm_head(final_normed).float()
        final_preds = final_logits[:-1].argmax(dim=-1)

        layer_confs, layer_ents, layer_agrees, layer_ranks = [], [], [], []
        token_settled = torch.zeros(T, dtype=torch.bool)
        token_settle_layer = torch.full((T,), float(len(indices)), dtype=torch.float32)

        for di, li in enumerate(indices):
            hs = hidden_states[li][0, :seq_len, :].float()
            normed = self.final_norm(hs.to(self.final_norm.weight.dtype))
            lo = self.lm_head(normed).float()[:-1]
            pr = F.softmax(lo, dim=-1)
            lpr = F.log_softmax(lo, dim=-1)

            layer_confs.append(pr.max(dim=-1).values.mean().item())
            layer_ents.append(-(pr * lpr).sum(dim=-1).mean().item())
            preds = lo.argmax(dim=-1)
            layer_agrees.append((preds == final_preds).float().mean().item())
            cp = pr.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            layer_ranks.append((pr > cp.unsqueeze(-1)).sum(dim=-1).float().mean().item())

            ns = (~token_settled) & (preds == final_preds)
            token_settle_layer[ns] = float(di)
            token_settled = token_settled | (preds == final_preds)

        sa = token_settle_layer.numpy()
        features["lens_settle_mean"] = float(sa.mean())
        features["lens_settle_std"] = float(sa.std())
        features["lens_settle_median"] = float(np.median(sa))
        mid = len(indices) // 2
        features["lens_early_settle_frac"] = float((sa <= mid).mean())

        for name, arr in [("conf", layer_confs), ("entropy", layer_ents),
                          ("agree", layer_agrees), ("rank", layer_ranks)]:
            a = np.array(arr)
            features[f"lens_{name}_mean"] = float(a.mean())
            features[f"lens_{name}_std"] = float(a.std())
            features[f"lens_{name}_early"] = float(a[:len(a)//2].mean())
            features[f"lens_{name}_late"] = float(a[len(a)//2:].mean())
            if len(a) > 1:
                features[f"lens_{name}_slope"] = float(np.polyfit(np.arange(len(a)), a, 1)[0])
            features[f"lens_{name}_rise"] = features[f"lens_{name}_late"] - features[f"lens_{name}_early"]

        return features

    # ===== D. HISTOGRAM FEATURES =====

    def _compute_histogram_features(self, logits, input_ids, seq_len):
        features = {}
        sl = logits[0, :seq_len-1, :].float()
        labels = input_ids[0, 1:seq_len]
        T = sl.shape[0]
        if T < 5: return features

        lp = F.log_softmax(sl, dim=-1)
        tlp = lp.gather(1, labels.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        clipped = np.clip(tlp, self.BIN_EDGES[0], self.BIN_EDGES[-1] - 1e-6)
        hist, _ = np.histogram(clipped, bins=self.BIN_EDGES)
        hn = hist.astype(np.float64) / (hist.sum() + 1e-10)

        for b in range(self.N_BINS):
            features[f"hist_bin_{b}"] = float(hn[b])

        features["hist_agg_mean"] = float(tlp.mean())
        features["hist_agg_std"] = float(tlp.std())
        features["hist_agg_min"] = float(tlp.min())
        features["hist_agg_max"] = float(tlp.max())
        features["hist_agg_median"] = float(np.median(tlp))
        for p in [5, 10, 25, 75, 90, 95]:
            features[f"hist_agg_p{p}"] = float(np.percentile(tlp, p))
        features["hist_iqr"] = features["hist_agg_p75"] - features["hist_agg_p25"]

        if tlp.std() > 1e-10:
            z = (tlp - tlp.mean()) / tlp.std()
            features["hist_skewness"] = float((z**3).mean())
            features["hist_kurtosis"] = float((z**4).mean() - 3.0)
        n = len(tlp)
        if n > 3 and tlp.std() > 1e-10:
            z = (tlp - tlp.mean()) / tlp.std()
            sk, ku = float((z**3).mean()), float((z**4).mean() - 3.0)
            features["hist_bimodality"] = float((sk**2 + 1) / (ku + 3*((n-1)**2)/((n-2)*(n-3)) + 1e-10))

        n20 = max(1, T // 5)
        st = np.sort(tlp)
        features["hist_top_20_mean"] = float(st[-n20:].mean())
        features["hist_bottom_20_mean"] = float(st[:n20].mean())

        return features

    # ===== E. CROSS-FAMILY FEATURES =====

    def _compute_cross_features(self, f: Dict[str, float]) -> Dict[str, float]:
        cross = {}
        if "grad_norm_embed" in f and "hnorm_global_mean" in f:
            cross["grad_state_ratio"] = f["grad_norm_embed"] / (f["hnorm_global_mean"] + 1e-10)
            cross["grad_state_product"] = f["grad_norm_embed"] * f["hnorm_global_mean"]
        if "hnorm_global_mean" in f and "hist_agg_std" in f:
            cross["hnorm_x_loss_std"] = f["hnorm_global_mean"] * f["hist_agg_std"]
        if "confidence_std" in f and "hist_kurtosis" in f:
            cross["confstd_x_kurtosis"] = f["confidence_std"] * abs(f.get("hist_kurtosis", 0))
        if "surprise_global_mean" in f and "hist_iqr" in f:
            cross["surprise_x_iqr"] = f["surprise_global_mean"] * f["hist_iqr"]
        if "lens_settle_mean" in f and "hnorm_global_mean" in f:
            cross["settle_x_hnorm"] = f["lens_settle_mean"] * f["hnorm_global_mean"]
        if "lens_agree_late" in f and "grad_norm_embed" in f:
            cross["agree_x_grad"] = f["lens_agree_late"] * f["grad_norm_embed"]
        if "grad_norm_embed" in f and "hist_agg_max" in f:
            cross["grad_x_lossmax"] = f["grad_norm_embed"] * f["hist_agg_max"]
        return cross

    @torch.no_grad()
    def _extract_forward(self, text: str) -> Tuple[Dict[str, float], bool]:
        """Extract all forward-only features."""
        result = {}
        if not text or len(text) < 20:
            return result, False

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 5: return result, False

            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hs = outputs.hidden_states
            logits = outputs.logits

            result.update(self._compute_transition_features(hs, seq_len))
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))
            result.update(self._compute_hidden_norm_features(hs, seq_len))
            result.update(self._compute_position_features(hs, seq_len))
            result.update(self._compute_context_evolution(hs, seq_len))
            result["seq_len"] = float(seq_len)
            result.update(self._compute_logit_lens_features(hs, input_ids, seq_len))
            result.update(self._compute_histogram_features(logits, input_ids, seq_len))

            return result, True

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[MEGA FWD WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result, False

    def extract(self, text: str) -> Dict[str, float]:
        """Extract ALL features: forward + backward + cross."""
        # Phase 1: Forward-only features
        result, ok = self._extract_forward(text)

        # Phase 2: Gradient features (backward pass)
        if ok:
            grad_feats = self._compute_gradient_features(text)
            result.update(grad_feats)

        # Phase 3: Cross-family features
        result.update(self._compute_cross_features(result))

        return result


def per_language_z_normalize(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Apply per-language Z-normalization to all features (EXP41 insight: +0.012)."""
    df_z = df.copy()
    for col in feature_cols:
        if col == "seq_len":
            continue
        for lang in df["subset"].unique():
            mask = df["subset"] == lang
            vals = df.loc[mask, col].values.astype(np.float64)
            valid = vals[np.isfinite(vals)]
            if len(valid) < 10:
                continue
            mu = valid.mean()
            sigma = valid.std()
            if sigma > 1e-10:
                df_z.loc[mask, col] = (vals - mu) / sigma
    return df_z


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
                if not os.path.exists(path): continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys(): ds = ds["test"]
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

    def _run_rf_cv(self, X_np, y, feature_names, label, n_folds=5):
        """Run RF 5-fold CV and return scores + mean AUC."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)
        fold_aucs = []
        all_scores = np.full(len(X_np), np.nan)

        for fi, (tri, tei) in enumerate(skf.split(X_np, y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_np[tri])
            Xte = sc.transform(X_np[tei])
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=12,
                min_samples_split=6, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced",
                random_state=self.args.seed, n_jobs=-1,
            )
            clf.fit(Xtr, y[tri])
            p = clf.predict_proba(Xte)[:, 1]
            a = roc_auc_score(y[tei], p)
            fold_aucs.append(a)
            all_scores[tei] = p

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  {label} ({X_np.shape[1]} features):")
        print(f"    RF CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        for i, a in enumerate(fold_aucs):
            print(f"    Fold {i+1}: {a:.4f}")

        # Feature importance from last fold
        imp = clf.feature_importances_
        print(f"\n--- Top 20 Feature Importances ({label}) ---")
        imp_pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        for rank, (name, v) in enumerate(imp_pairs[:20]):
            family = "MT"
            if "grad" in name: family = "GRAD"
            elif "lens" in name: family = "LENS"
            elif "hist" in name: family = "HIST"
            elif "_x_" in name or "ratio" in name.split("_")[-1] or "product" in name: family = "CROSS"
            print(f"  {rank+1:2d}. [{family:5s}] {name:<45} imp = {v:.4f}")

        return all_scores, mean_auc, std_auc, fold_aucs

    def _run_xgb_cv(self, X_np, y, feature_names, label, n_folds=5):
        """Run XGBoost-style GBM 5-fold CV."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)
        fold_aucs = []
        all_scores = np.full(len(X_np), np.nan)

        for fi, (tri, tei) in enumerate(skf.split(X_np, y)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_np[tri])
            Xte = sc.transform(X_np[tei])

            # Use sklearn GradientBoosting (no xgboost dependency needed)
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=self.args.seed,
            )
            clf.fit(Xtr, y[tri])
            p = clf.predict_proba(Xte)[:, 1]
            a = roc_auc_score(y[tei], p)
            fold_aucs.append(a)
            all_scores[tei] = p

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  {label} ({X_np.shape[1]} features):")
        print(f"    GBM CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        for i, a in enumerate(fold_aucs):
            print(f"    Fold {i+1}: {a:.4f}")

        return all_scores, mean_auc, std_auc, fold_aucs

    def run(self):
        df = self.load_data()
        extractor = MegaExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            n_lens_layers=self.args.n_lens_layers,
        )

        print(f"\n[COMBO05] MEGA Feature Extraction for {len(df)} samples...")
        print(f"  A. memTrace (hidden states)")
        print(f"  B. Gradient (backward pass)")
        print(f"  C. Logit Lens ({self.args.n_lens_layers} layer projections)")
        print(f"  D. Loss Histogram (16 bins)")
        print(f"  E. Cross-family interactions")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO05]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        feature_cols = list(feat_df.columns)
        print(f"\n[COMBO05] Valid: {n_valid}/{len(df)}, Total features: {len(feature_cols)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised AUCs ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: UNSUPERVISED SIGNAL AUCs (Top 25)")
        print("=" * 70)
        unsup_aucs = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2: continue
            auc = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best_auc = max(auc, auc_neg)
            d = "+" if auc >= auc_neg else "-"
            unsup_aucs[col] = (best_auc, d)
        top = sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True)[:25]
        for col, (auc, d) in top:
            fam = "MT"
            if "grad" in col: fam = "GRAD"
            elif "lens" in col: fam = "LENS"
            elif "hist" in col: fam = "HIST"
            elif "_x_" in col: fam = "CROSS"
            print(f"  [{fam:5s}] {d}{col:<50} AUC = {auc:.4f}")

        # --- Prepare data ---
        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy().fillna(0).replace([np.inf, -np.inf], 0)
        y_all = df.loc[valid_mask, "is_member"].values
        fn = list(X_all.columns)
        X_np = np.nan_to_num(X_all.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # --- RF CV: Raw features ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: RF (Raw Features)")
        print("=" * 70)
        scores_rf_raw, auc_rf_raw, _, _ = self._run_rf_cv(X_np, y_all, fn, "RF Raw")
        df.loc[valid_mask, "mega_rf_raw_score"] = scores_rf_raw

        # --- Per-language Z-normalized features ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: RF (Per-Language Z-Normalized)")
        print("=" * 70)

        df_z = per_language_z_normalize(df.loc[valid_mask].copy(), feature_cols)
        X_z = df_z[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_z_np = np.nan_to_num(X_z.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        scores_rf_z, auc_rf_z, _, _ = self._run_rf_cv(X_z_np, y_all, fn, "RF Z-Normed")
        df.loc[valid_mask, "mega_rf_znorm_score"] = scores_rf_z

        # --- GBM CV ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: GradientBoosting (Raw Features)")
        print("=" * 70)
        scores_gbm, auc_gbm, _, _ = self._run_xgb_cv(X_np, y_all, fn, "GBM Raw")
        df.loc[valid_mask, "mega_gbm_score"] = scores_gbm

        # --- GBM Z-Normed ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: GradientBoosting (Z-Normalized)")
        print("=" * 70)
        scores_gbm_z, auc_gbm_z, _, _ = self._run_xgb_cv(X_z_np, y_all, fn, "GBM Z-Normed")
        df.loc[valid_mask, "mega_gbm_znorm_score"] = scores_gbm_z

        # --- Rank-average ensemble of RF and GBM ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: RF+GBM Rank Ensemble")
        print("=" * 70)
        from scipy.stats import rankdata
        best_rf = scores_rf_z if auc_rf_z > auc_rf_raw else scores_rf_raw
        best_gbm = scores_gbm_z if auc_gbm_z > auc_gbm else scores_gbm
        rank_rf = rankdata(best_rf) / len(best_rf)
        rank_gbm = rankdata(best_gbm) / len(best_gbm)
        ensemble_scores = (rank_rf + rank_gbm) / 2.0
        df.loc[valid_mask, "mega_ensemble_score"] = ensemble_scores
        ens_auc = roc_auc_score(y_all, ensemble_scores)
        print(f"  RF+GBM Rank Ensemble AUC: {ens_auc:.4f}")

        # --- Summary ---
        print("\n" + "=" * 70)
        print("   COMBO05 MEGA: FINAL SUMMARY")
        print("=" * 70)
        print(f"  RF Raw:        {auc_rf_raw:.4f}")
        print(f"  RF Z-Normed:   {auc_rf_z:.4f}")
        print(f"  GBM Raw:       {auc_gbm:.4f}")
        print(f"  GBM Z-Normed:  {auc_gbm_z:.4f}")
        print(f"  RF+GBM Ensem:  {ens_auc:.4f}")
        print(f"\n  vs EXP50 memTrace RF: 0.6908 (current best)")
        print(f"  vs EXP43 AttenMIA:    0.6642")
        print(f"  vs EXP55 Histogram:   0.6612")
        print(f"  vs EXP11 -grad_embed: 0.6472")

        best_score = max(auc_rf_raw, auc_rf_z, auc_gbm, auc_gbm_z, ens_auc)
        if best_score > 0.72:
            print(f"\n  COMPETITION WINNER! AUC = {best_score:.4f}")
        elif best_score > 0.6908:
            print(f"\n  NEW BEST! Beats memTrace ({best_score:.4f} > 0.6908)")
        else:
            print(f"\n  Best = {best_score:.4f}")

        # --- Per-subset ---
        best_col = "mega_ensemble_score"
        print(f"\n{'Subset':<10} | {'Ensemble':<10} | N")
        print("-" * 30)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=[best_col])
            r = roc_auc_score(v["is_member"], v[best_col]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r:.4f}     | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO05_{timestamp}.parquet", index=False)
        print(f"\n[COMBO05] MEGA results saved.")


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

    print(f"[COMBO05] MEGA Fusion: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  6 signal families × RF+GBM classifier")
    Experiment(Args).run()
