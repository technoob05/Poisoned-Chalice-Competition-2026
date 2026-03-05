"""
COMBO04: memTrace + Loss Histogram — Hidden State Geometry × Token Loss Distribution

Rationale:
    EXP50 memTrace (AUC 0.6908) uses neg_mean_loss as ONE feature but
    doesn't capture the DISTRIBUTION of per-token losses.
    EXP55 Histogram RF (AUC 0.6612) captures loss distribution shape
    via 16-bin histograms, showing +0.100 over scalar mean_loss.

    The two approaches are partially orthogonal:
    - memTrace: hidden state norms, transitions, confidence variance
    - Histogram: token-level loss distribution shape (skew, tails, bimodality)

    A member that's well-memorized has:
    - Low hidden state norms at middle layers (memTrace)
    - More tokens in the "easy" (low-loss) bins (histogram)
    - Distinctive tail behavior in loss distribution
    - Higher bimodality (some tokens very easy, some moderately hard)

    Features: memTrace (~69) + histogram (16 bins + 20 aggregate) = ~105 features

Compute: 1 forward pass (output_hidden_states=True, logits for histogram)
Expected runtime: ~4-6 min on A100 (10% sample)
Expected AUC: 0.71-0.74
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
    print("  COMBO04: memTrace + Loss Histogram Fusion")
    print("  Hidden State Geometry × Token Loss Distribution")
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


class MemTraceHistExtractor:
    """Extract memTrace + loss histogram features.

    Feature families:
    A. memTrace: transition, confidence, norms, position, evolution
    B. Histogram: 16-bin normalized loss distribution
    C. Aggregate loss stats: percentiles, skew, kurtosis, bimodality
    D. Cross-family: histogram features × hidden state features
    """

    N_BINS = 16
    BIN_EDGES = np.linspace(-14, 0, N_BINS + 1)  # log-prob range [-14, 0]

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    # ===== MEMTRACE FEATURES =====
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
            nm = h.norm(dim=-1).mean().item()
            lnm.append(nm)
            if i in [0, self.n_layers//2, self.n_layers]:
                features[f"hnorm_mean_L{i}"] = nm
                features[f"hnorm_std_L{i}"] = h.norm(dim=-1).std().item()
        arr = np.array(lnm)
        features["hnorm_global_mean"] = float(arr.mean())
        features["hnorm_growth_ratio"] = float(arr[-1] / (arr[0] + 1e-10))
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

    # ===== HISTOGRAM FEATURES (from EXP55 adapted) =====

    def _compute_histogram_features(self, logits: torch.Tensor,
                                    input_ids: torch.Tensor,
                                    seq_len: int) -> Dict[str, float]:
        """Extract loss distribution histogram and aggregate statistics."""
        features = {}

        shift_logits = logits[0, :seq_len - 1, :].float()
        shift_labels = input_ids[0, 1:seq_len]
        T = shift_logits.shape[0]
        if T < 5:
            return features

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_lp_np = token_lp.cpu().numpy()

        # Clip to bin range
        clipped = np.clip(token_lp_np, self.BIN_EDGES[0], self.BIN_EDGES[-1] - 1e-6)

        # 16-bin histogram (normalized to sum=1)
        hist, _ = np.histogram(clipped, bins=self.BIN_EDGES)
        hist_norm = hist.astype(np.float64) / (hist.sum() + 1e-10)

        for b in range(self.N_BINS):
            features[f"hist_bin_{b}"] = float(hist_norm[b])

        # Aggregate statistics
        features["hist_agg_mean"] = float(token_lp_np.mean())
        features["hist_agg_std"] = float(token_lp_np.std())
        features["hist_agg_min"] = float(token_lp_np.min())
        features["hist_agg_max"] = float(token_lp_np.max())
        features["hist_agg_median"] = float(np.median(token_lp_np))
        features["hist_agg_p5"] = float(np.percentile(token_lp_np, 5))
        features["hist_agg_p10"] = float(np.percentile(token_lp_np, 10))
        features["hist_agg_p25"] = float(np.percentile(token_lp_np, 25))
        features["hist_agg_p75"] = float(np.percentile(token_lp_np, 75))
        features["hist_agg_p90"] = float(np.percentile(token_lp_np, 90))
        features["hist_agg_p95"] = float(np.percentile(token_lp_np, 95))

        # Skewness and kurtosis
        if token_lp_np.std() > 1e-10:
            z = (token_lp_np - token_lp_np.mean()) / token_lp_np.std()
            features["hist_skewness"] = float(z.mean() ** 3 if len(z) > 0 else 0)
            features["hist_kurtosis"] = float((z ** 4).mean() - 3.0)
        else:
            features["hist_skewness"] = 0.0
            features["hist_kurtosis"] = 0.0

        # Bimodality coefficient
        n = len(token_lp_np)
        if n > 3 and token_lp_np.std() > 1e-10:
            z = (token_lp_np - token_lp_np.mean()) / token_lp_np.std()
            skew = float((z ** 3).mean())
            kurt = float((z ** 4).mean() - 3.0)
            bc = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)) + 1e-10)
            features["hist_bimodality"] = float(bc)

        # Top/bottom token means
        sorted_lp = np.sort(token_lp_np)
        n20 = max(1, T // 5)
        features["hist_top_20_mean"] = float(sorted_lp[-n20:].mean())
        features["hist_bottom_20_mean"] = float(sorted_lp[:n20].mean())
        features["hist_iqr"] = features["hist_agg_p75"] - features["hist_agg_p25"]

        return features

    # ===== CROSS-FAMILY FEATURES =====

    def _compute_cross_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Cross-family features combining hidden state and histogram signals."""
        cross = {}

        # Hidden norm × loss distribution shape
        if "hnorm_global_mean" in features and "hist_agg_std" in features:
            cross["hnorm_x_loss_std"] = features["hnorm_global_mean"] * features["hist_agg_std"]
        if "hnorm_global_mean" in features and "hist_agg_max" in features:
            cross["hnorm_x_loss_max"] = features["hnorm_global_mean"] * features["hist_agg_max"]

        # Confidence variance × loss kurtosis (both capture distributional shape)
        if "confidence_std" in features and "hist_kurtosis" in features:
            cross["conf_std_x_kurtosis"] = features["confidence_std"] * abs(features["hist_kurtosis"])

        # Surprise × IQR (transition sharpness × loss spread)
        if "surprise_global_mean" in features and "hist_iqr" in features:
            cross["surprise_x_iqr"] = features["surprise_global_mean"] * features["hist_iqr"]

        return cross

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
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

            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            logits = outputs.logits

            # A. memTrace features
            result.update(self._compute_transition_features(hidden_states, seq_len))
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))
            result.update(self._compute_hidden_norm_features(hidden_states, seq_len))
            result.update(self._compute_position_features(hidden_states, seq_len))
            result.update(self._compute_context_evolution(hidden_states, seq_len))
            result["seq_len"] = float(seq_len)

            # B. Histogram features
            result.update(self._compute_histogram_features(logits, input_ids, seq_len))

            # C. Cross-family features
            result.update(self._compute_cross_features(result))

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[COMBO04 WARN] {type(e).__name__}: {e}")
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

    def run(self):
        df = self.load_data()
        extractor = MemTraceHistExtractor(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[COMBO04] Extracting memTrace+Histogram features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[COMBO04]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[COMBO04] Valid: {n_valid}/{len(df)}")

        feature_cols = list(feat_df.columns)
        print(f"[COMBO04] Total features: {len(feature_cols)}")
        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised AUCs ---
        print("\n" + "=" * 70)
        print("   COMBO04: UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)
        unsup_aucs = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2: continue
            auc = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best_auc = max(auc, auc_neg)
            direction = "+" if auc >= auc_neg else "-"
            unsup_aucs[col] = (best_auc, direction)

        top = sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True)[:20]
        for col, (auc, d) in top:
            src = "HIST" if "hist" in col else ("CROSS" if "_x_" in col else "MT")
            print(f"  [{src:5s}] {d}{col:<50} AUC = {auc:.4f}")

        # --- RF 5-fold CV ---
        print("\n" + "=" * 70)
        print("   COMBO04: RANDOM FOREST (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy().fillna(0).replace([np.inf, -np.inf], 0)
        y_all = df.loc[valid_mask, "is_member"].values
        feature_names = list(X_all.columns)
        X_np = np.nan_to_num(X_all.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        hist_cols = [c for c in feature_names if "hist" in c]
        cross_cols = [c for c in feature_names if "_x_" in c]
        mt_cols = [c for c in feature_names if c not in hist_cols and c not in cross_cols]

        configs = {"ALL (memTrace+Hist)": feature_names, "memTrace only": mt_cols, "Histogram only": hist_cols}

        for cn, cols in configs.items():
            ci = [feature_names.index(c) for c in cols if c in feature_names]
            if len(ci) < 2:
                print(f"\n  {cn}: skipped"); continue
            X_cfg = X_np[:, ci]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
            fa = []
            scores = np.full(len(X_cfg), np.nan)
            for fi, (tri, tei) in enumerate(skf.split(X_cfg, y_all)):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_cfg[tri]); Xte = sc.transform(X_cfg[tei])
                clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=8,
                    min_samples_leaf=4, max_features="sqrt", class_weight="balanced",
                    random_state=self.args.seed, n_jobs=-1)
                clf.fit(Xtr, y_all[tri])
                p = clf.predict_proba(Xte)[:, 1]
                a = roc_auc_score(y_all[tei], p)
                fa.append(a); scores[tei] = p
            print(f"\n  {cn} ({len(ci)} features): CV Mean AUC: {np.mean(fa):.4f} +/- {np.std(fa):.4f}")
            for i, a in enumerate(fa): print(f"    Fold {i+1}: {a:.4f}")

            if cn == "ALL (memTrace+Hist)":
                df.loc[valid_mask, "combo04_score"] = scores
                print(f"\n--- Top 15 Feature Importances ---")
                imp = clf.feature_importances_
                ip = sorted(zip([cols[i] for i in range(len(cols))], imp), key=lambda x: -x[1])
                for r, (n, v) in enumerate(ip[:15]):
                    src = "HIST" if n in hist_cols else ("CROSS" if n in cross_cols else "MT")
                    print(f"  {r+1:2d}. [{src:5s}] {n:<45} imp = {v:.4f}")

        print(f"\n{'Subset':<10} | {'Combo04':<10} | N")
        print("-" * 30)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["combo04_score"])
            r = roc_auc_score(v["is_member"], v["combo04_score"]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r:.4f}     | {len(sub)}")

        print(f"\n--- COMPARISON ---")
        v = df.dropna(subset=["combo04_score"])
        if not v.empty and len(v["is_member"].unique()) > 1:
            print(f"  COMBO04 (memTrace+Hist RF): {roc_auc_score(v['is_member'], v['combo04_score']):.4f}")
        print(f"  vs EXP50 memTrace RF:   0.6908")
        print(f"  vs EXP55 Histogram RF:  0.6612")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"COMBO04_{timestamp}.parquet", index=False)
        print(f"\n[COMBO04] Results saved.")


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
    print(f"[COMBO04] memTrace+Histogram: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    Experiment(Args).run()
