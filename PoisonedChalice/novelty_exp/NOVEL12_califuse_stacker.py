"""
NOVEL EXPERIMENT 12: CaliFuse-MIA — Multi-Signal Calibrated Fusion Stacker

NOVELTY: Final mega-stacker combining the BEST signals from ALL experiments
    with per-language Z-normalization, auto-direction detection, and
    Ridge logistic regression with nested cross-validation.

Core Idea:
    After 55+ experiments, we know which individual signals work:
    - Hidden state norms (mid-layer): ~0.69 (EXP50)
    - Attention concentration: ~0.66 (EXP43)
    - Gradient norm (embed): ~0.65 (EXP11/41)
    - Logit lens features: ~0.60-0.67 (NOVEL01/08/09)
    - Representation velocity: ~0.60-0.66 (NOVEL04)
    - Gradient sparsity: ~0.58-0.65 (NOVEL10)
    - Residual decomposition: ~0.58-0.65 (NOVEL11)

    CaliFuse extracts the TOP features from EACH signal family in a SINGLE
    forward + backward pass, then stacks them using Ridge LR with CV.

    Key innovations:
    1. Per-language Z-normalization (EXP41: +0.012 boost)
    2. Auto-direction detection (no manual +/- assignment)
    3. Length control (seq_len as feature + normalization)
    4. Feature selection: only use features with AUC > 0.55
    5. Ridge regularization to prevent overfitting on sparse signals

    This is the COMPETITION-READY ensemble that combines everything.

Compute: 1 forward pass (hidden states + attention) + 1 backward (gradient).
    All features extracted in a single pass.
    Ridge LR stacking with 5-fold stratified CV.
Expected runtime: ~15-20 min on A100.
Expected AUC: 0.72-0.76 (optimistic based on orthogonal signal fusion).
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL12: CaliFuse-MIA — Calibrated Multi-Signal Fusion")
    print("  Novelty: Final competition stacker with all signal families")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str, need_attention: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype, "device_map": "auto"}
    if need_attention:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    print(f"  Loaded. dtype={dtype}, attn=eager, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class AllSignalScorer:
    """Extract features from ALL signal families in a single pass."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        else:
            self.final_norm = None
        self.lm_head = model.lm_head

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

            # ==================== FORWARD + BACKWARD ====================
            self.model.zero_grad()
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(input_ids).detach().requires_grad_(True)

            outputs = self.model(
                inputs_embeds=embeds, labels=input_ids,
                output_hidden_states=True, output_attentions=True,
            )
            loss = outputs.loss
            loss.backward()

            # ==================== SIGNAL FAMILY 1: LOSS ====================
            result["neg_mean_loss"] = -loss.float().item()

            # ==================== SIGNAL FAMILY 2: GRADIENT ====================
            if embeds.grad is not None:
                eg = embeds.grad.float()
                grad_norm = eg.norm(2).item()
                result["neg_grad_norm"] = -grad_norm

                # Gradient sparsity
                eg_flat = eg.flatten().cpu().numpy()
                max_abs = np.abs(eg_flat).max()
                if max_abs > 1e-15:
                    result["grad_l0_proxy"] = float((np.abs(eg_flat) < 0.01 * max_abs).mean())

            # ==================== SIGNAL FAMILY 3: HIDDEN STATES ====================
            hs_list = outputs.hidden_states
            mid_layer = self.n_layers // 2

            # Hidden state norms at key layers
            for name, idx in [("early", self.n_layers // 4),
                               ("mid", mid_layer),
                               ("late", 3 * self.n_layers // 4)]:
                hs = hs_list[idx + 1][0].float()  # (seq, dim)
                norms = hs.norm(dim=-1)  # (seq,)
                result[f"neg_hs_norm_{name}"] = -norms.mean().item()
                result[f"hs_norm_std_{name}"] = norms.std().item()

            # Token-level hidden state heterogeneity at mid-layer
            hs_mid = hs_list[mid_layer + 1][0].float()
            result[f"hs_token_var_mid"] = hs_mid.var(dim=0).mean().item()

            # Representation velocity at key transitions
            for name, (l_from, l_to) in [("early_mid", (self.n_layers // 4, mid_layer)),
                                           ("mid_late", (mid_layer, 3 * self.n_layers // 4))]:
                h_from = hs_list[l_from + 1][0].float().mean(dim=0)
                h_to = hs_list[l_to + 1][0].float().mean(dim=0)
                vel = (h_to - h_from).norm(2).item()
                result[f"neg_vel_{name}"] = -vel

            # ==================== SIGNAL FAMILY 4: ATTENTION ====================
            attentions = outputs.attentions
            kl_vals = []
            for layer_idx, attn in enumerate(attentions):
                attn_mean = attn[0].float().mean(dim=0)  # (seq, seq)
                T_attn = attn_mean.shape[0]
                uniform = torch.ones_like(attn_mean) / T_attn
                kl = F.kl_div(uniform.log(), attn_mean, reduction='batchmean').item()
                kl_vals.append(kl)

            kl_arr = np.array(kl_vals)
            result["attn_conc_mean"] = float(kl_arr.mean())
            result["attn_conc_late"] = float(kl_arr[-5:].mean()) if len(kl_arr) >= 5 else float(kl_arr.mean())

            # Attention entropy
            for layer_idx in [mid_layer, self.n_layers - 1]:
                if layer_idx < len(attentions):
                    attn_mean = attentions[layer_idx][0].float().mean(dim=0)
                    attn_clamp = attn_mean.clamp(min=1e-10)
                    entropy = -(attn_clamp * attn_clamp.log()).sum(dim=-1).mean().item()
                    name = "mid" if layer_idx == mid_layer else "last"
                    result[f"neg_attn_entropy_{name}"] = -entropy

            # ==================== SIGNAL FAMILY 5: LOGIT LENS ====================
            # Quick logit lens at 3 key layers: early, mid, late
            labels = input_ids[0, 1:]
            T = labels.shape[0]

            for name, idx in [("early", self.n_layers // 4),
                               ("mid", mid_layer),
                               ("late", 3 * self.n_layers // 4)]:
                hs = hs_list[idx + 1]
                if self.final_norm is not None:
                    normed = self.final_norm(hs)
                else:
                    normed = hs
                logits_l = self.lm_head(normed)
                logits_shifted = logits_l[0, :-1, :].float()
                probs = F.softmax(logits_shifted, dim=-1)
                max_conf = probs.max(dim=-1).values.mean().item()
                result[f"lens_conf_{name}"] = max_conf

                # Top-1 accuracy
                top1 = logits_shifted.argmax(dim=-1)
                acc = (top1 == labels).float().mean().item()
                result[f"lens_acc_{name}"] = acc

            # ==================== SIGNAL FAMILY 6: PRODUCTS ====================
            # Orthogonal signal products (key fusion)
            if "neg_grad_norm" in result and "attn_conc_mean" in result:
                result["product_grad_conc"] = result["neg_grad_norm"] * result["attn_conc_mean"]
            if "neg_grad_norm" in result and "neg_hs_norm_mid" in result:
                result["product_grad_hs"] = result["neg_grad_norm"] * result["neg_hs_norm_mid"]

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL12 WARN] {type(e).__name__}: {e}")
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
        scorer = AllSignalScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL12] Extracting ALL signals for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL12]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Phase 1: Individual AUCs ---
        print("\n" + "=" * 70)
        print("   NOVEL12: Individual Signal AUCs")
        print("=" * 70)

        score_cols = [c for c in feat_df.columns if c not in ["seq_len", "log_seq_len"]]
        aucs = {}
        for col in sorted(score_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= auc_neg else "-"
            aucs[col] = (best, d)
            print(f"  {d}{col:<40} AUC = {best:.4f}")

        # --- Phase 2: Per-language Z-normalization ---
        print("\n[NOVEL12] Phase 2: Per-language Z-normalization...")

        feat_cols_for_stack = [c for c, (auc, _) in aucs.items() if auc >= 0.55]
        print(f"  Features with AUC >= 0.55: {len(feat_cols_for_stack)}")

        z_feat_cols = []
        for col in feat_cols_for_stack:
            z_col = f"{col}_z"
            df[z_col] = np.nan
            for lang in df["subset"].unique():
                mask = df["subset"] == lang
                vals = df.loc[mask, col].dropna()
                if len(vals) > 10:
                    mu, sigma = vals.mean(), vals.std()
                    if sigma > 1e-10:
                        df.loc[mask, z_col] = (df.loc[mask, col] - mu) / sigma
            # Auto-direction: flip sign if AUC direction is negative
            if aucs[col][1] == "-":
                df[z_col] = -df[z_col]
            z_feat_cols.append(z_col)

        # --- Phase 3: Ridge LR Stacking with CV ---
        print(f"\n[NOVEL12] Phase 3: Ridge LR stacking ({len(z_feat_cols)} Z-features)...")

        # Prepare feature matrix
        X = df[z_feat_cols].values
        y = df["is_member"].values

        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        print(f"  Valid samples: {valid_mask.sum()}/{len(df)}")

        if valid_mask.sum() < 100:
            print("  [WARN] Too few valid samples for stacking!")
            return

        # Nested CV: outer 5-fold for evaluation, inner LogisticRegressionCV for regularization
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.args.seed)
        oof_probs = np.zeros(len(X_valid))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_valid, y_valid)):
            X_train, X_val = X_valid[train_idx], X_valid[val_idx]
            y_train, y_val = y_valid[train_idx], y_valid[val_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Replace NaN after scaling (shouldn't happen but safety)
            X_train_s = np.nan_to_num(X_train_s, 0)
            X_val_s = np.nan_to_num(X_val_s, 0)

            clf = LogisticRegressionCV(
                Cs=10, cv=3, penalty="l2", scoring="roc_auc",
                max_iter=2000, random_state=self.args.seed,
                solver="lbfgs",
            )
            clf.fit(X_train_s, y_train)
            oof_probs[val_idx] = clf.predict_proba(X_val_s)[:, 1]

            fold_auc = roc_auc_score(y_val, oof_probs[val_idx])
            print(f"  Fold {fold+1}: AUC = {fold_auc:.4f} (C={clf.C_[0]:.4f})")

        overall_auc = roc_auc_score(y_valid, oof_probs)
        print(f"\n  CaliFuse CV AUC = {overall_auc:.4f}")

        # Store OOF predictions
        df.loc[valid_mask, "califuse_score"] = oof_probs

        # --- Phase 4: Feature Importance ---
        print(f"\n[NOVEL12] Phase 4: Feature importance (final model)...")
        scaler_final = StandardScaler()
        X_s = scaler_final.fit_transform(np.nan_to_num(X_valid, 0))
        clf_final = LogisticRegressionCV(
            Cs=10, cv=5, penalty="l2", scoring="roc_auc",
            max_iter=2000, random_state=self.args.seed, solver="lbfgs",
        )
        clf_final.fit(X_s, y_valid)

        coefs = clf_final.coef_[0]
        importance = list(zip(z_feat_cols, coefs))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top features by |coefficient|:")
        for feat, coef in importance[:15]:
            base_feat = feat.replace("_z", "")
            base_auc = aucs.get(base_feat, (0, "?"))[0]
            print(f"    {coef:+.4f}  {feat:<35} (raw AUC={base_auc:.4f})")

        # --- Final Report ---
        print("\n" + "=" * 70)
        print("   NOVEL12: CaliFuse-MIA FINAL RESULTS")
        print("=" * 70)
        print(f"\n  CaliFuse 5-fold CV AUC = {overall_auc:.4f}")
        print(f"  Features used: {len(z_feat_cols)}")
        print(f"  Best C: {clf_final.C_[0]:.4f}")
        print(f"\n  vs EXP50 memTrace RF CV: 0.6908")
        print(f"  vs EXP43 AttenMIA:       0.6642")
        print(f"  vs EXP41 -grad_z_lang:   0.6539")

        # Per-subset AUC
        calif_eval = df.dropna(subset=["califuse_score"])
        print(f"\n{'Subset':<10} | CaliFuse AUC | N")
        print("-" * 40)
        for subset in sorted(calif_eval["subset"].unique()):
            sub = calif_eval[calif_eval["subset"] == subset]
            if len(sub) > 10 and len(sub["is_member"].unique()) > 1:
                auc = roc_auc_score(sub["is_member"], sub["califuse_score"])
            else:
                auc = float("nan")
            print(f"  {subset:<10} | {auc:.4f}       | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL12_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL12] Results saved.")


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

    print(f"[NOVEL12] CaliFuse: Multi-Signal Calibrated Fusion")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
