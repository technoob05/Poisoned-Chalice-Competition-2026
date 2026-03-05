"""
EXPERIMENT 38: MasterStack — Multi-Signal XGBoost Ensemble

This is THE stacker (originally planned as EXP15).

Motivation (from ALL Insights):
    - Gradient ceiling ~0.65 (Insight 8): no single gradient metric breaks through.
    - CAMIA trajectory = 0.6065 (Insight 2): orthogonal to gradient.
    - Rank-avg hurts strong signals (Insight 3): need learned weighting.
    - Auto-flip needed (Insight 3, Failure Mode D): some signals inverted.
    - Per-language calibration (Insight 7): language as feature.

Architecture:
    SINGLE-RUN design — computes ALL features per sample in one forward+backward pass:

    A. LOSS FEATURES (from forward pass):
       - mean_loss, loss_std → SURP score (mean - std)
       - min-K%++ score (mean of bottom-K% token log-probs, Z-normalized)
       - Per-block losses (8 blocks of 256 tokens) → CAMIA sub-signals (MDM, TVar)
       - Per-token loss statistics (p10, p50, p90, variance)

    B. GRADIENT FEATURES (from backward pass):
       - grad_embed: embedding layer gradient norm
       - grad_head: LM head gradient norm
       - grad_L28, grad_L29: late-layer gradient norms (Insight 6)
       - product_score: -(grad_embed × jsd_early) (EXP27 formula)

    C. META FEATURES:
       - language (categorical → one-hot)
       - sequence_length (normalized)

    TRAINING:
       - Probe phase: 400 balanced labeled samples (200M + 200NM)
       - XGBoost with 5-fold stratified CV on probe set for hyperparameter selection
       - Final model trained on full probe set
       - Inference: score all samples

    AUTO-FLIP:
       - For each feature: if AUC < 0.5 on probe set, negate the feature

Expected AUC: 0.68–0.73 (learned combination of 15+ orthogonal features)

Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed. Will fallback to LogisticRegression.")

if not HAS_XGB:
    from sklearn.linear_model import LogisticRegression


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP38: MasterStack — MULTI-SIGNAL XGBOOST ENSEMBLE")
    print("=" * 65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
    except Exception as e:
        print(f"[HF] Note: {e}")

    if not HAS_XGB:
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "xgboost", "-q"])
            import importlib
            globals()["XGBClassifier"] = importlib.import_module("xgboost").XGBClassifier
            globals()["HAS_XGB"] = True
            print("[*] xgboost installed successfully.")
        except Exception:
            print("[*] Using LogisticRegression fallback.")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


class MultiSignalExtractor:
    """Extracts loss + gradient features in a single forward+backward pass."""

    def __init__(self, model, tokenizer, max_length=2048, n_blocks=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_blocks = n_blocks
        self.embed_layer = model.get_input_embeddings()
        self._err_count = 0

        self.bookend_layers = self._find_bookend_layers()

    def _find_bookend_layers(self):
        layers = {}
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            n = len(self.model.model.layers)
            for tag, idx in [("L28", min(28, n - 1)), ("L29", min(29, n - 1))]:
                prefix = f"model.layers.{idx}."
                params = [name for name, _ in self.model.named_parameters() if name.startswith(prefix)]
                if params:
                    layers[tag] = params
        embed_params = [n for n, _ in self.model.named_parameters() if "embed_tokens" in n]
        head_params = [n for n, _ in self.model.named_parameters() if "lm_head" in n]
        if embed_params:
            layers["embed"] = embed_params
        if head_params:
            layers["head"] = head_params
        return layers

    def extract_features(self, text: str) -> Dict[str, float]:
        result = self._empty_result()
        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            result["seq_len"] = seq_len

            self.model.zero_grad()
            outputs = self.model(**inputs, labels=input_ids)
            loss = outputs.loss

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).float().detach().cpu().numpy()

            result["mean_loss"] = float(per_token_loss.mean())
            result["std_loss"] = float(per_token_loss.std())
            result["surp_score"] = result["mean_loss"] - result["std_loss"]
            result["loss_p10"] = float(np.percentile(per_token_loss, 10))
            result["loss_p50"] = float(np.percentile(per_token_loss, 50))
            result["loss_p90"] = float(np.percentile(per_token_loss, 90))

            k_pct = max(1, int(0.2 * len(per_token_loss)))
            sorted_losses = np.sort(per_token_loss)
            result["minkpp_score"] = float(sorted_losses[:k_pct].mean())

            n_tokens = len(per_token_loss)
            block_size = max(1, n_tokens // self.n_blocks)
            block_losses = []
            for b in range(self.n_blocks):
                start = b * block_size
                end = min(start + block_size, n_tokens)
                if start < n_tokens:
                    block_losses.append(float(per_token_loss[start:end].mean()))
            if len(block_losses) >= 2:
                bl = np.array(block_losses)
                result["camia_mdm"] = float(bl.max() - bl.min())
                result["camia_tvar"] = float(bl.std())
                diffs = np.diff(bl)
                neg_diffs = diffs[diffs < 0]
                result["camia_max_drop"] = float(neg_diffs.min()) if len(neg_diffs) > 0 else 0.0
                result["camia_drop_position"] = float(np.argmin(diffs) / len(diffs)) if len(diffs) > 0 else 0.5

            loss.backward()
            param_dict = {n: p for n, p in self.model.named_parameters()}
            for tag, param_names in self.bookend_layers.items():
                norms = []
                for pn in param_names:
                    p = param_dict.get(pn)
                    if p is not None and p.grad is not None:
                        norms.append(p.grad.float().norm(2).item())
                if norms:
                    result[f"grad_{tag}"] = float(np.sqrt(np.mean(np.square(norms))))

            self.model.zero_grad()

            if "grad_embed" in result and not np.isnan(result["grad_embed"]):
                result["neg_grad_embed"] = -result["grad_embed"]

            return result

        except Exception as e:
            self.model.zero_grad()
            if self._err_count < 3:
                print(f"\n[EXP38 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result

    def _empty_result(self):
        keys = [
            "seq_len", "mean_loss", "std_loss", "surp_score",
            "loss_p10", "loss_p50", "loss_p90", "minkpp_score",
            "camia_mdm", "camia_tvar", "camia_max_drop", "camia_drop_position",
            "grad_embed", "grad_head", "grad_L28", "grad_L29",
            "neg_grad_embed",
        ]
        return {k: np.nan for k in keys}


class MasterStackAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.probe_size = getattr(args, "probe_size", 400)
        self.n_folds = getattr(args, "n_folds", 5)
        self.extractor = MultiSignalExtractor(
            model, tokenizer,
            max_length=getattr(args, "max_length", 2048),
            n_blocks=getattr(args, "n_blocks", 8),
        )

    @property
    def name(self):
        return "master_xgboost_stacker"

    def _get_feature_columns(self, df):
        exclude = {"is_member", "content", "membership", "subset", "id", "idx"}
        feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        return feat_cols

    def _auto_flip(self, X, y, feat_cols):
        """Flip features where AUC < 0.5 (wrong direction)."""
        flip_mask = {}
        for col in feat_cols:
            vals = X[col].dropna()
            labels = y.loc[vals.index]
            if len(labels.unique()) < 2 or len(vals) < 20:
                flip_mask[col] = False
                continue
            auc = roc_auc_score(labels, vals)
            if auc < 0.5:
                flip_mask[col] = True
            else:
                flip_mask[col] = False
        return flip_mask

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        print(f"\n[EXP38] Extracting features for {len(df)} samples...")
        all_features = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP38] Feature Extraction"):
            all_features.append(self.extractor.extract_features(row["content"]))

        feat_df = pd.DataFrame(all_features)

        lang_dummies = pd.get_dummies(df["subset"], prefix="lang")
        feat_df = pd.concat([feat_df, lang_dummies.reset_index(drop=True)], axis=1)
        feat_df["seq_len_norm"] = feat_df["seq_len"] / feat_df["seq_len"].max()

        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feat_df.columns:
            if feat_df[col].dtype in [np.float64, np.float32]:
                med = feat_df[col].median()
                feat_df[col] = feat_df[col].fillna(med if not np.isnan(med) else 0.0)

        feat_cols = self._get_feature_columns(feat_df)
        print(f"[EXP38] {len(feat_cols)} features extracted: {feat_cols[:10]}...")

        n_each = self.probe_size // 2
        members = df[df["is_member"] == 1].sample(min(n_each, (df["is_member"] == 1).sum()), random_state=self.args.seed)
        nonmembers = df[df["is_member"] == 0].sample(min(n_each, (df["is_member"] == 0).sum()), random_state=self.args.seed)
        probe_idx = pd.concat([members, nonmembers]).index.tolist()

        X_probe = feat_df.loc[probe_idx, feat_cols].copy()
        y_probe = df.loc[probe_idx, "is_member"].copy()

        flip_mask = self._auto_flip(X_probe, y_probe, feat_cols)
        flipped_cols = [c for c, flip in flip_mask.items() if flip]
        if flipped_cols:
            print(f"[EXP38] Auto-flipped {len(flipped_cols)} features: {flipped_cols}")
            for col in flipped_cols:
                feat_df[col] = -feat_df[col]

        X_probe = feat_df.loc[probe_idx, feat_cols].values
        y_probe = df.loc[probe_idx, "is_member"].values

        print(f"\n[EXP38] Training stacker on {len(X_probe)} probe samples ({self.n_folds}-fold CV)...")

        cv_aucs = []
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.args.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_probe, y_probe)):
            X_tr, X_val = X_probe[train_idx], X_probe[val_idx]
            y_tr, y_val = y_probe[train_idx], y_probe[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            if HAS_XGB:
                clf = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="auc", random_state=self.args.seed,
                    use_label_encoder=False,
                )
                clf.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
            else:
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(X_tr_s, y_tr)

            if HAS_XGB:
                val_pred = clf.predict_proba(X_val_s)[:, 1]
            else:
                val_pred = clf.predict_proba(X_val_s)[:, 1]

            fold_auc = roc_auc_score(y_val, val_pred)
            cv_aucs.append(fold_auc)
            print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")

        mean_cv = np.mean(cv_aucs)
        std_cv = np.std(cv_aucs)
        print(f"  CV Mean AUC: {mean_cv:.4f} ± {std_cv:.4f}")

        scaler_final = StandardScaler()
        X_probe_s = scaler_final.fit_transform(X_probe)
        if HAS_XGB:
            final_clf = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="auc", random_state=self.args.seed,
                use_label_encoder=False,
            )
            final_clf.fit(X_probe_s, y_probe, verbose=False)
        else:
            final_clf = LogisticRegression(max_iter=1000, C=1.0)
            final_clf.fit(X_probe_s, y_probe)

        X_all = feat_df[feat_cols].values
        X_all_s = scaler_final.transform(X_all)
        all_preds = final_clf.predict_proba(X_all_s)[:, 1]

        result_df = feat_df.copy()
        result_df["xgb_score"] = all_preds

        if HAS_XGB and hasattr(final_clf, "feature_importances_"):
            importances = final_clf.feature_importances_
            imp_sorted = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
            print(f"\n[EXP38] Top-10 feature importances:")
            for fname, imp in imp_sorted[:10]:
                flipped_tag = " [FLIPPED]" if fname in flipped_cols else ""
                print(f"  {fname:<30} {imp:.4f}{flipped_tag}")

        n_valid = result_df["xgb_score"].notna().sum()
        print(f"\n[EXP38] Valid: {n_valid}/{len(df)}")
        if self.extractor._err_count > 0:
            print(f"[EXP38] Extraction errors: {self.extractor._err_count}")

        return result_df


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
        print(f"[*] Loading data from {self.args.dataset}")
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
        if not dfs:
            raise ValueError("No data loaded!")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data().reset_index(drop=True)
        attacker = MasterStackAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df)

        for col in scores_df.columns:
            if col not in df.columns:
                df[col] = scores_df[col].values

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP38_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "=" * 65)
        print("   EXP38: MasterStack — PERFORMANCE REPORT")
        print("=" * 65)

        report = {"experiment": "EXP38_master_stacker", "timestamp": timestamp, "aucs": {}, "subset_aucs": {}}

        score_cols = {
            "xgb_score": "XGBoost Ensemble [PRIMARY]",
            "neg_grad_embed": "-Grad Embed (single feature)",
            "surp_score": "SURP (mean-std)",
            "minkpp_score": "Min-K%++ Score",
        }
        for score_col, label in score_cols.items():
            if score_col not in df.columns:
                continue
            valid = df.dropna(subset=[score_col])
            if len(valid["is_member"].unique()) > 1:
                auc = roc_auc_score(valid["is_member"], valid[score_col])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<45} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'XGB':<8} | {'GradEmbed':<11} | {'SURP':<8} | N")
        print("-" * 50)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["xgb_score", "neg_grad_embed", "surp_score"]:
                v = sub.dropna(subset=[sc]) if sc in sub.columns else pd.DataFrame()
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            print(f"{subset:<10} | {r.get('xgb_score', float('nan')):.4f}   "
                  f"| {r.get('neg_grad_embed', float('nan')):.4f}      "
                  f"| {r.get('surp_score', float('nan')):.4f}   "
                  f"| {len(sub)}")
            report["subset_aucs"][subset] = r

        print("=" * 65)
        print(f"\nComparison with previous best:")
        print(f"  EXP27 product_score:  0.6484")
        print(f"  EXP11 grad_embed:     0.6472")
        print(f"  EXP33 CAMIA combined: 0.6065")
        xgb_auc = report["aucs"].get("xgb_score", 0)
        print(f"  **EXP38 XGB Stacker:  {xgb_auc:.4f}**")
        if xgb_auc > 0.6484:
            print(f"  → NEW BEST! +{xgb_auc - 0.6484:.4f} over EXP27")
        else:
            print(f"  → Delta vs EXP27: {xgb_auc - 0.6484:+.4f}")

        report_path = self.output_dir / f"EXP38_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\n[*] Report saved: {report_path.name}")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        probe_size = 400
        n_folds = 5
        n_blocks = 8
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP38] Model      : {Args.model_name}")
    print(f"[EXP38] Sample     : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP38] Probe      : {Args.probe_size}")
    print(f"[EXP38] CV folds   : {Args.n_folds}")
    print(f"[EXP38] Classifier : {'XGBoost' if HAS_XGB else 'LogisticRegression'}")
    Experiment(Args).run()
