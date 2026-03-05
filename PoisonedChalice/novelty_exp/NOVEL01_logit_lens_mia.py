"""
NOVEL EXPERIMENT 01: LensMIA — Logit Lens Convergence for Membership Inference

NOVELTY: First application of the "logit lens" interpretability technique
    (nostalgebraist 2020; Geva et al. NeurIPS 2022) to membership inference.
    No prior MIA work uses layer-wise vocabulary projections.

Core Idea:
    The logit lens projects intermediate hidden states through the final
    LayerNorm + LM head to obtain vocabulary-space predictions at EVERY layer.
    For memorized (member) code, the model should "know" the correct next token
    at EARLIER layers — the prediction SETTLES sooner because the model has
    already encoded this specific sequence during training.

    For non-member code, the model needs more layers of computation to refine
    its prediction → correct token appears in top-K at LATER layers.

    Signal: "settling depth" = earliest layer where the correct next token
    first appears in top-K predictions. Lower depth = more likely member.

    This is FUNDAMENTALLY DIFFERENT from:
    - EXP50 memTrace (hidden state NORMS — scalar magnitude, not predictions)
    - EXP43 AttenMIA (attention PATTERNS — routing, not vocabulary predictions)
    - EXP11 gradient norm (loss-surface geometry, not layer-wise predictions)
    - EXP26 ESR/JSD (distribution divergence, not specific token prediction)

    The logit lens directly measures WHAT the model predicts at each layer,
    not just how its representations look statistically.

Builds on Insights:
    - Insight 22: mid-layer (L15) representations are most informative
    - Insight 8: gradient ceiling at ~0.65 → need new signal families
    - Insight 16: attention broke gradient ceiling → logit lens may push further

Features extracted (per sample):
    1. settling_depth_top1: earliest layer where correct token is top-1 (mean across tokens)
    2. settling_depth_top5: earliest layer where correct token is in top-5
    3. settled_frac_mid: fraction of tokens settled by mid-layer (L15)
    4. settled_frac_early: fraction of tokens settled by early layer (L7)
    5. settling_variance: std of settling depths across token positions
    6. confidence_at_settle: mean confidence (softmax prob) when token first settles
    7. never_settled_frac: fraction of tokens that NEVER reach correct top-1
    8. agreement_ratio: fraction of consecutive layers that agree on top-1 prediction
    9. prediction_stability: mean run-length of consistent top-1 predictions

Compute: 1 forward pass per sample (output_hidden_states=True), no backward pass.
    ~31 logit lens projections per sample (one per layer).
    Forward-only, 10% sample.
Expected runtime: ~8-12 min on A100
Expected AUC: 0.60-0.68
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

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL01: LensMIA — Logit Lens Convergence for MIA")
    print("  Novelty: First use of logit lens for membership inference")
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
    print(f"  Loaded. dtype={dtype}, layers={n_layers}")
    return model, tokenizer


class LogitLensScorer:
    """Extract logit lens features: settling depth, agreement, confidence trajectory."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 top_k: int = 5, layer_stride: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k = top_k
        self.layer_stride = layer_stride
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        # Get final layer norm and lm_head for logit lens projection
        # StarCoder2: model.model.norm + model.lm_head
        # GPT-2 style: model.transformer.ln_f + model.lm_head
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            self.final_norm = model.transformer.ln_f
        else:
            self.final_norm = None
            print("  [WARN] Could not find final layer norm, using identity")

        self.lm_head = model.lm_head
        self.layer_indices = list(range(0, self.n_layers, self.layer_stride))
        print(f"  Logit lens: {len(self.layer_indices)} layers, top_k={top_k}")

    @torch.no_grad()
    def _logit_lens_at_layer(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state through final norm + lm_head to get logits."""
        if self.final_norm is not None:
            normed = self.final_norm(hidden_state)
        else:
            normed = hidden_state
        return self.lm_head(normed)

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

            # Forward pass with all hidden states
            outputs = self.model(
                input_ids=input_ids, output_hidden_states=True, labels=input_ids,
            )
            result["neg_mean_loss"] = -outputs.loss.float().item()

            # Labels: next token at each position
            labels = input_ids[0, 1:]  # shape (T-1,)
            T = labels.shape[0]

            # For each layer, project through logit lens and check predictions
            n_check_layers = len(self.layer_indices)
            # settled_at[t] = earliest layer index where correct token is in top-K
            settled_top1 = np.full(T, n_check_layers, dtype=np.float32)  # default: never settled
            settled_top5 = np.full(T, n_check_layers, dtype=np.float32)
            conf_at_settle = np.zeros(T, dtype=np.float32)

            # Track per-layer predictions for agreement computation
            layer_predictions = []  # list of (T,) arrays of top-1 token ids
            layer_confidences = []  # list of (T,) arrays of max softmax prob

            for li, layer_idx in enumerate(self.layer_indices):
                # hidden_states[0] = embedding, hidden_states[i+1] = layer i
                hs = outputs.hidden_states[layer_idx + 1]
                logits_l = self._logit_lens_at_layer(hs)

                # Shift: predict next token, so use positions [0, T-1] to predict [1, T]
                logits_shifted = logits_l[0, :-1, :].float()  # (T, vocab)

                # Top-K predictions
                topk_vals, topk_ids = torch.topk(logits_shifted, self.top_k, dim=-1)
                top1_ids = topk_ids[:, 0].cpu().numpy()  # (T,)

                # Max confidence (softmax of top-1)
                probs = F.softmax(logits_shifted, dim=-1)
                max_conf = probs.max(dim=-1).values.cpu().numpy()  # (T,)

                # Check if correct token is in top-1
                correct_top1 = (topk_ids[:, 0].cpu() == labels.cpu()).numpy()
                # Check if correct token is in top-K
                correct_topk = (topk_ids.cpu() == labels.unsqueeze(-1).cpu()).any(dim=-1).numpy()

                # Update settling depths
                for t in range(T):
                    if correct_top1[t] and settled_top1[t] == n_check_layers:
                        settled_top1[t] = li
                        conf_at_settle[t] = max_conf[t]
                    if correct_topk[t] and settled_top5[t] == n_check_layers:
                        settled_top5[t] = li

                layer_predictions.append(top1_ids)
                layer_confidences.append(max_conf)

            # --- Feature extraction ---

            # 1. Settling depth (normalized by number of layers)
            norm_factor = float(n_check_layers)
            result["settling_depth_top1"] = float(settled_top1.mean() / norm_factor)
            result["settling_depth_top5"] = float(settled_top5.mean() / norm_factor)
            result["settling_depth_top1_median"] = float(np.median(settled_top1) / norm_factor)

            # 2. Fraction settled by mid-layer and early layer
            mid_layer_idx = n_check_layers // 2
            early_layer_idx = n_check_layers // 4
            result["settled_frac_mid_top1"] = float((settled_top1 <= mid_layer_idx).mean())
            result["settled_frac_mid_top5"] = float((settled_top5 <= mid_layer_idx).mean())
            result["settled_frac_early_top1"] = float((settled_top1 <= early_layer_idx).mean())

            # 3. Never-settled fraction
            result["never_settled_frac"] = float((settled_top1 == n_check_layers).mean())

            # 4. Settling depth variance
            settled_mask = settled_top1 < n_check_layers
            if settled_mask.sum() > 2:
                result["settling_variance"] = float(settled_top1[settled_mask].std() / norm_factor)
            else:
                result["settling_variance"] = 1.0

            # 5. Confidence at settling point
            settled_conf = conf_at_settle[settled_mask]
            if len(settled_conf) > 0:
                result["conf_at_settle_mean"] = float(settled_conf.mean())
            else:
                result["conf_at_settle_mean"] = 0.0

            # 6. Cross-layer prediction agreement
            if len(layer_predictions) >= 2:
                agreements = []
                for i in range(len(layer_predictions) - 1):
                    agree = (layer_predictions[i] == layer_predictions[i + 1]).mean()
                    agreements.append(agree)
                result["agreement_ratio"] = float(np.mean(agreements))
                result["agreement_early"] = float(np.mean(agreements[:len(agreements)//2]))
                result["agreement_late"] = float(np.mean(agreements[len(agreements)//2:]))
            else:
                result["agreement_ratio"] = 0.0

            # 7. Prediction stability: mean run-length of consistent top-1
            if len(layer_predictions) >= 2:
                stabilities = []
                for t in range(T):
                    preds_at_t = [lp[t] for lp in layer_predictions]
                    runs = []
                    current_run = 1
                    for i in range(1, len(preds_at_t)):
                        if preds_at_t[i] == preds_at_t[i - 1]:
                            current_run += 1
                        else:
                            runs.append(current_run)
                            current_run = 1
                    runs.append(current_run)
                    stabilities.append(max(runs) / len(preds_at_t))
                result["prediction_stability"] = float(np.mean(stabilities))
            else:
                result["prediction_stability"] = 0.0

            # 8. Confidence trajectory features
            conf_curve = np.array([np.mean(lc) for lc in layer_confidences])
            if len(conf_curve) >= 4:
                # Slope of confidence rise
                x = np.arange(len(conf_curve), dtype=np.float32)
                slope = np.polyfit(x, conf_curve, 1)[0]
                result["conf_slope"] = float(slope)
                # Saturation: layer where confidence first exceeds 90% of final
                final_conf = conf_curve[-1]
                if final_conf > 0.01:
                    threshold = 0.9 * final_conf
                    sat_layers = np.where(conf_curve >= threshold)[0]
                    result["conf_saturation_depth"] = float(sat_layers[0] / norm_factor) if len(sat_layers) > 0 else 1.0
                else:
                    result["conf_saturation_depth"] = 1.0
                # Monotonicity: fraction of consecutive increases
                diffs = np.diff(conf_curve)
                result["conf_monotonicity"] = float((diffs > 0).mean())

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL01 WARN] {type(e).__name__}: {e}")
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
        scorer = LogitLensScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            top_k=self.args.top_k,
            layer_stride=self.args.layer_stride,
        )

        print(f"\n[NOVEL01] Extracting logit lens features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL01]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[NOVEL01] Valid: {n_valid}/{len(df)}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL01: LensMIA — Logit Lens RESULTS")
        print("=" * 70)

        score_cols = [c for c in feat_df.columns if c not in ["seq_len"]]
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
            tag = " <-- PRIMARY" if col == "settling_depth_top1" else ""
            print(f"  {d}{col:<35} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")
            print(f"  vs EXP50 memTrace RF:  0.6908 (current best)")
            print(f"  vs EXP43 AttenMIA:     0.6642")
            print(f"  vs EXP41 -grad_z_lang: 0.6539")

            # Per-subset
            best_col = best_sig[0]
            best_dir = best_sig[1][1]
            print(f"\n{'Subset':<10} | {best_col:<25} | N")
            print("-" * 50)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                v = sub.dropna(subset=[best_col])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    vals = v[best_col] if best_dir == "+" else -v[best_col]
                    auc = roc_auc_score(v["is_member"], vals)
                else:
                    auc = float("nan")
                print(f"  {subset:<10} | {auc:.4f}                 | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL01_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL01] Results saved.")


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
        top_k = 5
        layer_stride = 1  # check every layer (set to 2 for speed)

    print(f"[NOVEL01] LensMIA — Logit Lens Convergence")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    print(f"  top_k={Args.top_k}, stride={Args.layer_stride}")
    Experiment(Args).run()
