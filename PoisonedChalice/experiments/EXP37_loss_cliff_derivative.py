"""
EXPERIMENT 37: CAMIA-D — Loss Cliff Detection via Trajectory Derivatives

Novelty & motivation (from Insights 2, 3):
    EXP33 CAMIA showed that loss TRAJECTORY captures a different signal
    than loss magnitude (0.6065 combined, +0.11 over sub-signals).
    But CAMIA used fixed 256-token blocks and simple statistics (MDM, TVar, AUCG).

    Key observation from Insight 2: model "progressively recognizes" member files
    → loss drops sharply at semantic boundaries (import→class→method→usage).
    These "recognition cliffs" should be detectable as peaks in the FIRST DERIVATIVE
    of the smoothed loss trajectory.

    THIS EXPERIMENT:
    1. Computes per-token loss trajectory (single forward pass)
    2. Smooths with rolling window to remove token-level noise
    3. Computes first derivative (rate of change) and second derivative (curvature)
    4. Detects "loss cliffs" — locations where loss drops sharply
    5. Extracts features: cliff_count, max_cliff_magnitude, cliff_positions,
       trajectory_curvature, early_vs_late_slope

    Members: sharp, deep cliffs in the first half of the file (model recognizes
             memorized patterns early). Late portion has very low, flat loss.
    Non-members: gradual, smooth loss decline. No dramatic cliffs.

    FAST: forward-only, no backward pass needed. ~same speed as EXP36.

Expected AUC: 0.60–0.67 (extends CAMIA from 0.6065 with finer-grained features)

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
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP37: CAMIA-D — LOSS CLIFF DETECTION (Trajectory Derivatives)")
    print("=" * 65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
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
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


class LossCliffAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.smooth_window = getattr(args, "smooth_window", 32)
        self.cliff_threshold = getattr(args, "cliff_threshold", 0.3)
        self._err_count = 0

    @property
    def name(self):
        return "loss_cliff_derivative"

    def _get_per_token_losses(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True,
        ).to(self.model.device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 10:
            return None

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).float().cpu().numpy()
        return per_token

    def compute_cliff_features(self, text: str) -> Dict[str, float]:
        result = {
            "mean_loss": np.nan, "loss_first_half": np.nan, "loss_second_half": np.nan,
            "slope_overall": np.nan, "slope_first_half": np.nan, "slope_second_half": np.nan,
            "max_cliff_mag": np.nan, "cliff_count": np.nan,
            "cliff_position_mean": np.nan, "cliff_position_first": np.nan,
            "d1_mean": np.nan, "d1_std": np.nan, "d1_min": np.nan,
            "d2_mean": np.nan, "d2_std": np.nan,
            "early_late_ratio": np.nan,
            "trajectory_auc": np.nan,
            "combined_cliff_score": np.nan,
        }
        if not text or len(text) < 30:
            return result

        try:
            losses = self._get_per_token_losses(text)
            if losses is None or len(losses) < self.smooth_window * 2:
                return result

            n = len(losses)
            result["mean_loss"] = float(losses.mean())

            mid = n // 2
            result["loss_first_half"] = float(losses[:mid].mean())
            result["loss_second_half"] = float(losses[mid:].mean())

            if losses[mid:].mean() > 1e-6:
                result["early_late_ratio"] = float(losses[:mid].mean() / losses[mid:].mean())

            smoothed = uniform_filter1d(losses.astype(np.float64), size=self.smooth_window)

            x = np.arange(len(smoothed), dtype=np.float64)
            if len(x) > 1:
                slope, _ = np.polyfit(x / len(x), smoothed, 1)
                result["slope_overall"] = float(slope)

                slope_1h, _ = np.polyfit(x[:mid] / len(x), smoothed[:mid], 1)
                slope_2h, _ = np.polyfit(x[mid:] / len(x), smoothed[mid:], 1)
                result["slope_first_half"] = float(slope_1h)
                result["slope_second_half"] = float(slope_2h)

            d1 = np.diff(smoothed)
            result["d1_mean"] = float(d1.mean())
            result["d1_std"] = float(d1.std())
            result["d1_min"] = float(d1.min())

            if len(d1) > 1:
                d2 = np.diff(d1)
                result["d2_mean"] = float(d2.mean())
                result["d2_std"] = float(d2.std())

            cliff_mask = d1 < -self.cliff_threshold
            cliff_count = cliff_mask.sum()
            result["cliff_count"] = int(cliff_count)

            if cliff_count > 0:
                cliff_positions = np.where(cliff_mask)[0]
                cliff_magnitudes = -d1[cliff_mask]

                result["max_cliff_mag"] = float(cliff_magnitudes.max())
                result["cliff_position_mean"] = float(cliff_positions.mean() / len(d1))
                result["cliff_position_first"] = float(cliff_positions[0] / len(d1))
            else:
                result["max_cliff_mag"] = 0.0
                result["cliff_position_mean"] = 0.5
                result["cliff_position_first"] = 1.0

            norm_traj = smoothed / (smoothed.max() + 1e-8)
            result["trajectory_auc"] = float(np.trapz(norm_traj) / len(norm_traj))

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP37 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result

    def compute_scores(self, texts: List[str]) -> pd.DataFrame:
        print(f"\n[EXP37] Processing {len(texts)} samples (forward-only)")
        rows = []
        for text in tqdm(texts, desc="[EXP37] Loss Cliff Detection"):
            rows.append(self.compute_cliff_features(text))

        df = pd.DataFrame(rows)

        df["score_neg_loss"] = -df["mean_loss"]
        df["score_max_cliff"] = df["max_cliff_mag"]
        df["score_cliff_count"] = df["cliff_count"]
        df["score_early_cliff"] = -df["cliff_position_first"]
        df["score_slope"] = -df["slope_overall"]
        df["score_d1_min"] = -df["d1_min"]

        rank_sources = [
            "score_neg_loss", "score_max_cliff", "score_early_cliff",
            "score_slope", "score_d1_min",
        ]
        valid_cols = [c for c in rank_sources if c in df.columns]
        if valid_cols:
            rank_sum = np.zeros(len(df))
            for col in valid_cols:
                vals = df[col].fillna(df[col].min()).values
                rank_sum += rankdata(vals, method="average") / len(vals)
            df["combined_rank"] = rank_sum / len(valid_cols)

        n_valid = df["mean_loss"].notna().sum()
        print(f"[EXP37] Valid: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP37] Errors: {self._err_count}")
        return df


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
        df = self.load_data()
        attacker = LossCliffAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df["content"].tolist())
        df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP37_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "=" * 65)
        print("   EXP37: CAMIA-D — PERFORMANCE REPORT")
        print("=" * 65)

        report = {"experiment": "EXP37_camia_d", "timestamp": timestamp, "aucs": {}, "subset_aucs": {}}

        for score_col, label in [
            ("combined_rank", "Combined Rank [PRIMARY]"),
            ("score_neg_loss", "-Mean Loss"),
            ("score_max_cliff", "Max Cliff Magnitude"),
            ("score_cliff_count", "Cliff Count"),
            ("score_early_cliff", "Early Cliff Position"),
            ("score_slope", "-Overall Slope"),
            ("score_d1_min", "-d1_min (sharpest drop)"),
            ("early_late_ratio", "Early/Late Loss Ratio"),
        ]:
            if score_col not in df.columns:
                continue
            valid = df.dropna(subset=[score_col])
            if len(valid["is_member"].unique()) > 1:
                auc = roc_auc_score(valid["is_member"], valid[score_col])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<40} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'Combined':<10} | {'MaxCliff':<10} | {'Slope':<8} | {'Cliffs/seq'}")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["combined_rank", "score_max_cliff", "score_slope"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            avg_cliffs = sub["cliff_count"].mean()
            print(f"{subset:<10} | {r.get('combined_rank', float('nan')):.4f}     "
                  f"| {r.get('score_max_cliff', float('nan')):.4f}     "
                  f"| {r.get('score_slope', float('nan')):.4f}   "
                  f"| {avg_cliffs:.1f}")
            report["subset_aucs"][subset] = r

        print(f"\nCliff statistics (Member vs Non-member):")
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for feat in ["max_cliff_mag", "cliff_count", "cliff_position_first", "slope_overall"]:
            if feat in df.columns:
                print(f"  {feat:<25} M: {m[feat].mean():.4f}  NM: {nm[feat].mean():.4f}")

        print("=" * 65)
        report_path = self.output_dir / f"EXP37_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[*] Report saved: {report_path.name}")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        output_dir = "results"
        max_length = 2048
        smooth_window = 32
        cliff_threshold = 0.3
        seed = 42

    print(f"[EXP37] Model     : {Args.model_name}")
    print(f"[EXP37] Sample    : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP37] Smooth    : {Args.smooth_window} tokens")
    print(f"[EXP37] Cliff thr : {Args.cliff_threshold}")
    Experiment(Args).run()
