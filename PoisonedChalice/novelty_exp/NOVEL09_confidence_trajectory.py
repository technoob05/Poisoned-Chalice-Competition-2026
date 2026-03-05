"""
NOVEL EXPERIMENT 09: ConfTraj-MIA — Confidence Trajectory via Logit Lens

NOVELTY: First MIA method that characterizes the SHAPE of the confidence
    curve (max softmax probability) across transformer layers.
    Maps the "confidence trajectory" as a fingerprint for memorization.

Core Idea:
    Using the logit lens at each layer, compute the maximum softmax probability
    (i.e., how confident the model would be if we stopped at that layer).
    The shape of this confidence curve across layers is a fingerprint:

    MEMBERS (memorized):
    - Confidence rises STEEPLY in early layers (quick recognition)
    - Reaches HIGH plateau by mid-layer (already matched to memory)
    - Curve shape: steep_rise → flat_plateau
    - High AUC under confidence curve

    NON-MEMBERS:
    - Confidence rises SLOWLY and GRADUALLY
    - May not reach as high a plateau
    - Curve shape: gradual_rise → late_saturation
    - Lower AUC under confidence curve

    Features extracted from the confidence trajectory:
    1. conf_auc: area under the confidence curve
    2. conf_slope_early: slope of confidence in first third of layers
    3. conf_half_life: layer where confidence reaches 50% of final
    4. conf_plateau_height: max sustained confidence level
    5. conf_curvature: second derivative (concavity) of trajectory
    6. conf_ratio_early_late: ratio of early to late confidence

Builds on Insights:
    - NOVEL01 logit lens concept validated
    - EXP50: mid-layer is informative → confidence should plateau there
    - EXP26 ESR: early settling captures partial signal

Compute: 1 forward pass, logit lens softmax at each layer.
Expected runtime: ~10-14 min on A100.
Expected AUC: 0.58-0.66
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
    print("  NOVEL09: ConfTraj-MIA — Confidence Trajectory")
    print("  Novelty: Shape of per-layer confidence curve for MIA")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class ConfTrajScorer:
    """Extract confidence trajectory features via logit lens."""

    def __init__(self, model, tokenizer, max_length: int = 512, layer_stride: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self.layer_stride = layer_stride
        self._err_count = 0

        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        else:
            self.final_norm = None
        self.lm_head = model.lm_head
        self.layer_indices = list(range(0, self.n_layers, self.layer_stride))

    @torch.no_grad()
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

            outputs = self.model(
                input_ids=input_ids, output_hidden_states=True, labels=input_ids,
            )
            result["neg_mean_loss"] = -outputs.loss.float().item()

            labels = input_ids[0, 1:]  # (T,)
            T = labels.shape[0]

            # Collect per-layer confidence metrics
            max_conf_curve = []        # max softmax prob (avg over tokens)
            correct_conf_curve = []    # softmax prob of CORRECT token (avg)
            top1_accuracy_curve = []   # fraction of correct top-1 predictions

            for layer_idx in self.layer_indices:
                hs = outputs.hidden_states[layer_idx + 1]
                if self.final_norm is not None:
                    normed = self.final_norm(hs)
                else:
                    normed = hs
                logits_l = self.lm_head(normed)
                logits_shifted = logits_l[0, :-1, :].float()  # (T, vocab)

                probs = F.softmax(logits_shifted, dim=-1)

                # Max confidence
                max_conf = probs.max(dim=-1).values.mean().item()
                max_conf_curve.append(max_conf)

                # Confidence of correct token
                correct_probs = probs[torch.arange(T, device=probs.device), labels].mean().item()
                correct_conf_curve.append(correct_probs)

                # Top-1 accuracy
                top1 = logits_shifted.argmax(dim=-1)
                acc = (top1 == labels).float().mean().item()
                top1_accuracy_curve.append(acc)

            mc = np.array(max_conf_curve)
            cc = np.array(correct_conf_curve)
            ac = np.array(top1_accuracy_curve)
            n = len(mc)

            # --- CONFIDENCE TRAJECTORY FEATURES ---

            # 1. Area under curves
            result["conf_max_auc"] = float(np.trapz(mc))
            result["conf_correct_auc"] = float(np.trapz(cc))
            result["acc_auc"] = float(np.trapz(ac))

            # 2. Early/mid/late confidence
            third = max(1, n // 3)
            result["conf_max_early"] = float(mc[:third].mean())
            result["conf_max_mid"] = float(mc[third:2*third].mean())
            result["conf_max_late"] = float(mc[2*third:].mean())
            result["conf_correct_early"] = float(cc[:third].mean())
            result["conf_correct_late"] = float(cc[2*third:].mean())

            # 3. Early-to-late ratio
            late_mean = mc[2*third:].mean()
            result["conf_early_late_ratio"] = float(mc[:third].mean() / (late_mean + 1e-10))

            # 4. Half-life: layer where confidence first exceeds 50% of final
            final_conf = mc[-1]
            if final_conf > 0.01:
                half_conf = 0.5 * final_conf
                half_layers = np.where(mc >= half_conf)[0]
                result["neg_conf_half_life"] = -float(half_layers[0] / n) if len(half_layers) > 0 else -1.0
            else:
                result["neg_conf_half_life"] = -1.0

            # 5. 90%-life (saturation depth)
            if final_conf > 0.01:
                sat_thresh = 0.9 * final_conf
                sat_layers = np.where(mc >= sat_thresh)[0]
                result["neg_conf_saturation"] = -float(sat_layers[0] / n) if len(sat_layers) > 0 else -1.0
            else:
                result["neg_conf_saturation"] = -1.0

            # 6. Slope of confidence rise
            x = np.arange(n, dtype=np.float32)
            if n >= 3:
                slope = np.polyfit(x, mc, 1)[0]
                result["conf_slope"] = float(slope)

                # Early slope (first third)
                x_early = np.arange(third, dtype=np.float32)
                slope_early = np.polyfit(x_early, mc[:third], 1)[0] if third >= 2 else 0.0
                result["conf_slope_early"] = float(slope_early)

            # 7. Curvature (second derivative) — concave = rapid early rise then plateau
            if n >= 5:
                second_deriv = np.diff(np.diff(mc))
                result["conf_curvature_mean"] = float(second_deriv.mean())
                result["conf_curvature_early"] = float(second_deriv[:len(second_deriv)//2].mean())

            # 8. Monotonicity (fraction of increasing steps)
            diffs = np.diff(mc)
            result["conf_monotonicity"] = float((diffs > 0).mean())

            # 9. Plateau height (max sustained level in last quarter)
            result["conf_plateau"] = float(mc[-max(1, n//4):].mean())

            # 10. Correct-token confidence features
            result["correct_conf_mean"] = float(cc.mean())
            if n >= 3:
                slope_cc = np.polyfit(x, cc, 1)[0]
                result["correct_conf_slope"] = float(slope_cc)

            # 11. Accuracy trajectory
            result["acc_early"] = float(ac[:third].mean())
            result["acc_late"] = float(ac[2*third:].mean())
            result["acc_early_late_ratio"] = float(ac[:third].mean() / (ac[2*third:].mean() + 1e-10))

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL09 WARN] {type(e).__name__}: {e}")
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
        scorer = ConfTrajScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            layer_stride=self.args.layer_stride,
        )

        print(f"\n[NOVEL09] Extracting confidence trajectory for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL09]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL09: ConfTraj-MIA RESULTS")
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
            print(f"  {d}{col:<35} AUC = {best:.4f}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL09_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL09] Results saved.")


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
        layer_stride = 2  # every other layer for speed

    print(f"[NOVEL09] ConfTraj: Confidence Trajectory")
    print(f"  model: {Args.model_name}, stride={Args.layer_stride}")
    Experiment(Args).run()
