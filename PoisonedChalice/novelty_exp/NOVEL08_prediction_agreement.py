"""
NOVEL EXPERIMENT 08: PredAgree-MIA — Token Prediction Depth & Agreement

NOVELTY: First MIA method that tracks how early and how consistently the
    model "commits" to its final token prediction across transformer layers.
    Uses logit lens projections to measure cross-layer AGREEMENT stability.

Core Idea:
    At each transformer layer, we project hidden states through the final
    LN + LM head (the "logit lens") to see what the model would predict
    if we stopped processing at that layer.

    For MEMBERS (memorized sequences):
    - The model commits to the correct prediction EARLY (few layers)
    - Once committed, the prediction is STABLE through remaining layers
    - High cross-layer agreement, long "run lengths" of consistent predictions
    - The prediction RARELY flip-flops between layers

    For NON-MEMBERS:
    - The model takes LONGER to settle on a prediction
    - Predictions may CHANGE between layers (flip-flop)
    - Lower agreement ratio, shorter run lengths
    - More "exploration" before "exploitation"

    This is distinct from NOVEL01 (which measures settling DEPTH) by focusing
    specifically on STABILITY and FLIP-FLOP patterns across prediction layers.

    Features:
    1. flip_flop_rate: how often top-1 prediction changes between layers
    2. first_commit_depth: layer where prediction first matches final prediction
    3. commitment_ratio: fraction of layers agreeing with final prediction
    4. longest_stable_run: longest consecutive sequence of same prediction
    5. early_commitment: fraction of first-half layers committed to final pred

Builds on Insights:
    - Insight 22: mid-layer representations are most informative
    - EXP50: hidden states encode membership signal (0.6908)
    - NOVEL01 hypothesis: logit lens reveals memorization timing

Compute: 1 forward pass, 31 logit lens projections.
Expected runtime: ~10-14 min on A100.
Expected AUC: 0.60-0.67
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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL08: PredAgree-MIA — Prediction Agreement Depth")
    print("  Novelty: Cross-layer prediction stability via logit lens")
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


class PredAgreeScorer:
    """Track prediction agreement/flip-flop across transformer layers."""

    def __init__(self, model, tokenizer, max_length: int = 512, layer_stride: int = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self.layer_stride = layer_stride
        self._err_count = 0

        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self.final_norm = model.model.norm
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            self.final_norm = model.transformer.ln_f
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

            # Get final-layer predictions as reference
            final_logits = outputs.logits[0, :-1, :].float()  # (T, vocab)
            T = final_logits.shape[0]
            final_preds = final_logits.argmax(dim=-1).cpu().numpy()  # (T,)

            # For each layer, get logit lens predictions
            layer_preds_list = []  # list of (T,) arrays

            for layer_idx in self.layer_indices:
                hs = outputs.hidden_states[layer_idx + 1]
                if self.final_norm is not None:
                    normed = self.final_norm(hs)
                else:
                    normed = hs
                logits_l = self.lm_head(normed)
                preds_l = logits_l[0, :-1, :].float().argmax(dim=-1).cpu().numpy()
                layer_preds_list.append(preds_l)

            n_layers_checked = len(layer_preds_list)
            if n_layers_checked < 3:
                return result

            # --- Feature extraction ---

            # 1. Flip-flop rate: how often top-1 changes between consecutive layers
            flips = []
            for i in range(1, n_layers_checked):
                flip = (layer_preds_list[i] != layer_preds_list[i-1]).mean()
                flips.append(flip)
            result["neg_flip_rate_mean"] = -float(np.mean(flips))  # lower flip = member
            result["neg_flip_rate_late"] = -float(np.mean(flips[len(flips)//2:]))

            # 2. Agreement with final prediction at each layer
            agreements = []
            for i, preds in enumerate(layer_preds_list):
                agree = (preds == final_preds).mean()
                agreements.append(agree)
            agree_arr = np.array(agreements)

            result["final_agree_mean"] = float(agree_arr.mean())
            result["final_agree_early"] = float(agree_arr[:n_layers_checked//3].mean())
            result["final_agree_mid"] = float(agree_arr[n_layers_checked//3:2*n_layers_checked//3].mean())
            result["final_agree_late"] = float(agree_arr[2*n_layers_checked//3:].mean())

            # 3. First commit depth: earliest layer where >50% of tokens match final
            commit_layers = np.where(agree_arr >= 0.5)[0]
            if len(commit_layers) > 0:
                result["neg_first_commit_depth"] = -float(commit_layers[0] / n_layers_checked)
            else:
                result["neg_first_commit_depth"] = -1.0

            # 4. Strong commit: earliest layer where >80% match final
            strong_commit = np.where(agree_arr >= 0.8)[0]
            if len(strong_commit) > 0:
                result["neg_strong_commit_depth"] = -float(strong_commit[0] / n_layers_checked)
            else:
                result["neg_strong_commit_depth"] = -1.0

            # 5. Per-token stability analysis
            stabilities = []  # per-token: longest consecutive run of same prediction
            for t in range(T):
                token_preds = [lp[t] for lp in layer_preds_list]
                max_run = 1
                current_run = 1
                for i in range(1, len(token_preds)):
                    if token_preds[i] == token_preds[i-1]:
                        current_run += 1
                        max_run = max(max_run, current_run)
                    else:
                        current_run = 1
                stabilities.append(max_run / n_layers_checked)

            stab_arr = np.array(stabilities)
            result["stability_mean"] = float(stab_arr.mean())
            result["stability_median"] = float(np.median(stab_arr))
            result["stability_min"] = float(stab_arr.min())

            # 6. Early commitment ratio: fraction of tokens committed by mid-layer
            mid_idx = n_layers_checked // 2
            early_committed = 0
            for t in range(T):
                preds_after_mid = [layer_preds_list[i][t] for i in range(mid_idx, n_layers_checked)]
                if len(set(preds_after_mid)) == 1:  # same prediction from mid to end
                    early_committed += 1
            result["early_commit_ratio"] = float(early_committed / T)

            # 7. Agreement slope (does agreement increase? how fast?)
            if len(agree_arr) >= 3:
                x = np.arange(len(agree_arr), dtype=np.float32)
                slope = np.polyfit(x, agree_arr, 1)[0]
                result["agree_slope"] = float(slope)

            # 8. Commitment AUC (area under agreement curve)
            result["agree_auc"] = float(np.trapz(agree_arr))

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL08 WARN] {type(e).__name__}: {e}")
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
        scorer = PredAgreeScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            layer_stride=self.args.layer_stride,
        )

        print(f"\n[NOVEL08] Extracting prediction agreement features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL08]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL08: PredAgree-MIA RESULTS")
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
            tag = " <-- PRIMARY" if "early_commit" in col else ""
            print(f"  {d}{col:<35} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")
            print(f"  vs EXP50 memTrace RF:  0.6908")

            m, nm = df[df["is_member"]==1], df[df["is_member"]==0]
            for col in ["neg_flip_rate_mean", "stability_mean", "early_commit_ratio", "final_agree_early"]:
                if col in df.columns:
                    mv, nmv = m[col].dropna(), nm[col].dropna()
                    if len(mv)>0 and len(nmv)>0:
                        print(f"  {col:<30} M={mv.mean():.4f}  NM={nmv.mean():.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL08_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL08] Results saved.")


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
        layer_stride = 1

    print(f"[NOVEL08] PredAgree: Prediction Agreement Depth")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
