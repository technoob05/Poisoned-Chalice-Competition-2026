"""
EXPERIMENT 36: CodeMIF — Token-Level Memorization Fingerprint for Code

Novelty & motivation (from Insights 1, 2, 7):
    All previous experiments operate at SEQUENCE level (one score per file).
    But code has rich TOKEN-LEVEL structure: function definitions, imports,
    class declarations, type annotations are structural "anchors" that
    the model memorizes differently from variable names or logic.

    Key hypothesis:
    - Members: model has memorized the STRUCTURE of the file. It predicts
      anchor tokens (def, class, import, func, fn, struct, package) with
      very HIGH confidence (low per-token loss) because it has seen these
      exact patterns during training.
    - Non-members: model is less certain about anchor tokens because the
      specific code structure is unfamiliar.

    Signal: The GAP between anchor-token loss and body-token loss is larger
    for members (very low anchor loss, moderate body loss) than non-members
    (both moderate loss).

Architecture:
    1. Single forward pass → per-token log-probabilities (NO backward needed!)
    2. Identify anchor tokens via string matching on decoded tokens
    3. Compute token-level statistics:
       - mean_loss_anchor: average loss on anchor tokens
       - mean_loss_body: average loss on non-anchor tokens
       - anchor_body_ratio: anchor_loss / body_loss (low = memorized structure)
       - loss_variance: overall loss spread across tokens
       - pct_confident: % of tokens with loss < threshold
       - min_loss_window: minimum loss in any 32-token window
    4. Combined score via rank-averaging

    This is FAST (forward-only, ~2× faster than gradient-based methods)
    and captures a fundamentally different signal family (token-level confidence
    vs. gradient magnitude).

Expected AUC: 0.58–0.65 (orthogonal to gradient signals → valuable for EXP15 stacking)

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

warnings.filterwarnings("ignore")

ANCHOR_KEYWORDS = {
    "python": ["def", "class", "import", "from", "return", "self", "async", "await", "yield"],
    "go": ["func", "package", "import", "type", "struct", "interface", "return", "defer", "go"],
    "java": ["class", "public", "private", "import", "return", "void", "static", "interface", "extends"],
    "ruby": ["def", "class", "module", "require", "end", "return", "attr", "include"],
    "rust": ["fn", "pub", "use", "struct", "impl", "trait", "let", "mut", "return", "mod"],
}

ALL_ANCHORS = set()
for kws in ANCHOR_KEYWORDS.values():
    ALL_ANCHORS.update(kws)


def setup_environment():
    print("\n" + "=" * 65)
    print("  EXP36: CodeMIF — TOKEN-LEVEL MEMORIZATION FINGERPRINT")
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


class CodeMIFAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.confident_threshold = getattr(args, "confident_threshold", 0.5)
        self.window_size = getattr(args, "window_size", 32)
        self._err_count = 0

    @property
    def name(self):
        return "code_memorization_fingerprint"

    def _get_per_token_losses(self, text: str):
        """Forward pass → per-token cross-entropy losses."""
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True,
        ).to(self.model.device)
        input_ids = inputs["input_ids"]  # (1, seq)
        if input_ids.shape[1] < 4:
            return None, None

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, seq, vocab)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).float().cpu().numpy()

        token_ids = input_ids[0, 1:].cpu().tolist()
        return per_token_loss, token_ids

    def _identify_anchor_mask(self, token_ids: List[int], language: str) -> np.ndarray:
        """Returns boolean mask: True for anchor tokens."""
        lang_anchors = ANCHOR_KEYWORDS.get(language.lower(), ALL_ANCHORS)
        mask = np.zeros(len(token_ids), dtype=bool)

        for i, tid in enumerate(token_ids):
            decoded = self.tokenizer.decode([tid]).strip().lower()
            if decoded in lang_anchors:
                mask[i] = True

        return mask

    def compute_token_features(self, text: str, language: str) -> Dict[str, float]:
        result = {
            "mean_loss": np.nan, "std_loss": np.nan,
            "mean_loss_anchor": np.nan, "mean_loss_body": np.nan,
            "anchor_body_ratio": np.nan, "anchor_body_gap": np.nan,
            "pct_confident": np.nan, "min_loss_window": np.nan,
            "loss_p10": np.nan, "loss_p90": np.nan,
            "n_anchors": 0, "seq_len": 0,
            "surp_score": np.nan,
        }
        if not text or len(text) < 20:
            return result

        try:
            per_token_loss, token_ids = self._get_per_token_losses(text)
            if per_token_loss is None:
                return result

            n_tokens = len(per_token_loss)
            result["seq_len"] = n_tokens
            result["mean_loss"] = float(per_token_loss.mean())
            result["std_loss"] = float(per_token_loss.std())
            result["loss_p10"] = float(np.percentile(per_token_loss, 10))
            result["loss_p90"] = float(np.percentile(per_token_loss, 90))

            result["surp_score"] = float(per_token_loss.mean() - per_token_loss.std())

            result["pct_confident"] = float((per_token_loss < self.confident_threshold).mean())

            if n_tokens >= self.window_size:
                windows = np.lib.stride_tricks.sliding_window_view(per_token_loss, self.window_size)
                result["min_loss_window"] = float(windows.mean(axis=1).min())

            anchor_mask = self._identify_anchor_mask(token_ids, language)
            n_anchors = anchor_mask.sum()
            result["n_anchors"] = int(n_anchors)

            if n_anchors >= 3:
                anchor_losses = per_token_loss[anchor_mask]
                body_losses = per_token_loss[~anchor_mask]

                result["mean_loss_anchor"] = float(anchor_losses.mean())
                if len(body_losses) > 0:
                    result["mean_loss_body"] = float(body_losses.mean())
                    body_mean = body_losses.mean()
                    if body_mean > 1e-6:
                        result["anchor_body_ratio"] = float(anchor_losses.mean() / body_mean)
                    result["anchor_body_gap"] = float(body_losses.mean() - anchor_losses.mean())

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP36 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[EXP36] Processing {len(df)} samples (forward-only, fast)")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP36] CodeMIF"):
            lang = row.get("subset", "python")
            rows.append(self.compute_token_features(row["content"], lang))

        scores_df = pd.DataFrame(rows)

        scores_df["score_neg_loss"] = -scores_df["mean_loss"]
        scores_df["score_surp"] = scores_df["surp_score"]
        scores_df["score_confident"] = scores_df["pct_confident"]
        scores_df["score_anchor_gap"] = scores_df["anchor_body_gap"]
        scores_df["score_neg_anchor_ratio"] = -scores_df["anchor_body_ratio"]

        rank_sources = ["score_neg_loss", "score_surp", "score_confident", "score_anchor_gap"]
        valid_cols = [c for c in rank_sources if c in scores_df.columns]
        if valid_cols:
            rank_sum = np.zeros(len(scores_df))
            for col in valid_cols:
                vals = scores_df[col].fillna(scores_df[col].min()).values
                rank_sum += rankdata(vals, method="average") / len(vals)
            scores_df["combined_rank"] = rank_sum / len(valid_cols)

        n_valid = scores_df["mean_loss"].notna().sum()
        print(f"[EXP36] Valid: {n_valid}/{len(df)} ({100*n_valid/max(1,len(df)):.1f}%)")
        if self._err_count > 0:
            print(f"[EXP36] Errors: {self._err_count}")
        return scores_df


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
        attacker = CodeMIFAttack(self.args, self.model, self.tokenizer)
        scores_df = attacker.compute_scores(df)
        df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP36_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "=" * 65)
        print("   EXP36: CodeMIF — PERFORMANCE REPORT")
        print("=" * 65)

        report = {"experiment": "EXP36_code_mif", "timestamp": timestamp, "aucs": {}, "subset_aucs": {}}

        for score_col, label in [
            ("combined_rank", "Combined Rank [PRIMARY]"),
            ("score_neg_loss", "-Mean Loss"),
            ("score_surp", "SURP (mean-std)"),
            ("score_confident", "% Confident Tokens"),
            ("score_anchor_gap", "Anchor-Body Gap"),
            ("score_neg_anchor_ratio", "-Anchor/Body Ratio"),
            ("min_loss_window", "Min-Loss Window"),
        ]:
            if score_col not in df.columns:
                continue
            valid = df.dropna(subset=[score_col])
            if len(valid["is_member"].unique()) > 1:
                auc = roc_auc_score(valid["is_member"], valid[score_col])
                report["aucs"][score_col] = float(auc)
                tag = " ← PRIMARY" if "PRIMARY" in label else ""
                print(f"  {label:<40} AUC = {auc:.4f}{tag}")

        print(f"\n{'Subset':<10} | {'Combined':<10} | {'SURP':<8} | {'AnchorGap':<10} | {'Anchors/seq'}")
        print("-" * 55)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["combined_rank", "score_surp", "score_anchor_gap"]:
                v = sub.dropna(subset=[sc])
                r[sc] = roc_auc_score(v["is_member"], v[sc]) if not v.empty and len(v["is_member"].unique()) > 1 else float("nan")
            avg_anchors = sub["n_anchors"].mean()
            print(f"{subset:<10} | {r.get('combined_rank', float('nan')):.4f}     "
                  f"| {r.get('score_surp', float('nan')):.4f}   "
                  f"| {r.get('score_anchor_gap', float('nan')):.4f}     "
                  f"| {avg_anchors:.1f}")
            report["subset_aucs"][subset] = r

        print(f"\nAnchor token statistics:")
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            m = sub[sub["is_member"] == 1]
            nm = sub[sub["is_member"] == 0]
            m_gap = m["anchor_body_gap"].mean()
            nm_gap = nm["anchor_body_gap"].mean()
            print(f"  {subset:<10} anchor_body_gap — M: {m_gap:.4f}  NM: {nm_gap:.4f}")

        print("=" * 65)
        report_path = self.output_dir / f"EXP36_report_{timestamp}.json"
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
        confident_threshold = 0.5
        window_size = 32
        seed = 42

    print(f"[EXP36] Model  : {Args.model_name}")
    print(f"[EXP36] Sample : {Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
