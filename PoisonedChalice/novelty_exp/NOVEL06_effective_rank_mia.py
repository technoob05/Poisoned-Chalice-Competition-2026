"""
NOVEL EXPERIMENT 06: RankDef-MIA — Effective Rank of Hidden State Matrices

NOVELTY: First use of matrix rank (via singular value decomposition) of
    hidden state matrices for MIA. No prior MIA work measures the
    geometric "dimensionality" of the representation at each layer.

Core Idea:
    At each transformer layer, the hidden state forms a matrix of shape
    (seq_len × hidden_dim). The EFFECTIVE RANK of this matrix measures
    how many dimensions are actually being used to encode the input.

    Effective rank = exp(entropy of normalized singular values)
    (Roy & Vetterli, 2007)

    For MEMBERS (memorized content):
    - The model has a SPECIALIZED encoding for this exact sequence
    - Hidden states occupy a LOWER-dimensional subspace (fewer effective dims)
    - Information is COMPRESSED into learned attractors
    - Lower effective rank

    For NON-MEMBERS:
    - The model must use a more GENERAL encoding strategy
    - Hidden states spread across MORE dimensions (higher effective rank)
    - Representation is less structured/compressed
    - Higher effective rank

    This connects to the flat-minima hypothesis: memorized samples sit
    in well-structured low-dimensional valleys of the loss landscape,
    which is reflected in the low-rank structure of their hidden states.

Builds on Insights:
    - Insight 22: mid-layer hidden states are most discriminative
    - EXP50: hidden state statistics encode strong membership signal (0.6908)
    - Flat minima: members have more structured/constrained representations

Compute: 1 forward pass + SVD on hidden state matrix per layer.
    SVD on (seq_len × hidden_dim) is cheap when seq_len << hidden_dim.
Expected runtime: ~8-12 min on A100.
Expected AUC: 0.55-0.63
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
    print("  NOVEL06: RankDef-MIA — Effective Rank of Hidden States")
    print("  Novelty: SVD-based dimensionality analysis for MIA")
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


def effective_rank(matrix: np.ndarray) -> float:
    """Compute effective rank via entropy of normalized singular values."""
    try:
        sv = np.linalg.svd(matrix, compute_uv=False)
        sv = sv[sv > 1e-10]
        if len(sv) == 0:
            return 0.0
        sv_norm = sv / sv.sum()
        entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-15))
        return np.exp(entropy)
    except Exception:
        return 0.0


def stable_rank(matrix: np.ndarray) -> float:
    """Compute stable rank = ||A||_F^2 / ||A||_2^2."""
    try:
        sv = np.linalg.svd(matrix, compute_uv=False)
        if sv[0] < 1e-10:
            return 0.0
        return float(np.sum(sv**2) / (sv[0]**2))
    except Exception:
        return 0.0


class RankDefScorer:
    """Extract effective rank features from hidden state matrices."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 check_layers: str = "all"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        if check_layers == "all":
            self.check_layers = list(range(self.n_layers))
        elif check_layers == "key":
            q = self.n_layers // 4
            self.check_layers = [0, q, 2*q, 3*q, self.n_layers-1]
        elif check_layers == "quartile":
            step = max(1, self.n_layers // 8)
            self.check_layers = list(range(0, self.n_layers, step))
        else:
            self.check_layers = list(range(self.n_layers))

        print(f"  Checking {len(self.check_layers)} layers for rank: {self.check_layers[:5]}...")

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

            eff_ranks = []
            stab_ranks = []
            sv_entropies = []
            top_sv_fracs = []
            sv_decays = []

            for layer_idx in self.check_layers:
                hs = outputs.hidden_states[layer_idx + 1]
                mat = hs[0].float().cpu().numpy()  # (seq_len, dim)

                # Effective rank
                er = effective_rank(mat)
                eff_ranks.append(er)

                # Stable rank
                sr = stable_rank(mat)
                stab_ranks.append(sr)

                # SVD analysis
                try:
                    sv = np.linalg.svd(mat, compute_uv=False)
                    sv = sv[sv > 1e-10]
                    if len(sv) > 0:
                        sv_norm = sv / sv.sum()
                        entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-15))
                        sv_entropies.append(entropy)

                        # Top SV fraction (how much variance is captured by top-1)
                        top_sv_fracs.append(sv[0]**2 / np.sum(sv**2))

                        # SV decay rate (slope of log SV)
                        if len(sv) >= 3:
                            x = np.arange(min(20, len(sv)), dtype=np.float32)
                            log_sv = np.log(sv[:len(x)])
                            slope = np.polyfit(x, log_sv, 1)[0]
                            sv_decays.append(slope)
                except Exception:
                    pass

            er_arr = np.array(eff_ranks)
            sr_arr = np.array(stab_ranks)

            # --- Features ---

            # Effective rank at key layers
            n_check = len(self.check_layers)
            if n_check >= 4:
                result["eff_rank_early"] = float(er_arr[:n_check//4].mean())
                result["eff_rank_mid"] = float(er_arr[n_check//4:3*n_check//4].mean())
                result["eff_rank_late"] = float(er_arr[3*n_check//4:].mean())

            result["eff_rank_mean"] = float(er_arr.mean())
            result["eff_rank_std"] = float(er_arr.std())
            result["eff_rank_min"] = float(er_arr.min())
            result["eff_rank_max"] = float(er_arr.max())

            # Effective rank at specific percentile layers
            if len(er_arr) >= 5:
                mid_idx = len(er_arr) // 2
                result["eff_rank_at_mid"] = float(er_arr[mid_idx])
                result["eff_rank_at_last"] = float(er_arr[-1])

            # Stable rank
            result["stable_rank_mean"] = float(sr_arr.mean())
            result["stable_rank_mid"] = float(sr_arr[len(sr_arr)//2]) if len(sr_arr) > 0 else 0.0

            # Rank decay across layers (does rank decrease = compression?)
            if len(er_arr) >= 3:
                x = np.arange(len(er_arr), dtype=np.float32)
                slope = np.polyfit(x, er_arr, 1)[0]
                result["rank_decay_rate"] = float(slope)

            # SV analysis
            if sv_entropies:
                result["sv_entropy_mean"] = float(np.mean(sv_entropies))
            if top_sv_fracs:
                result["top_sv_frac_mean"] = float(np.mean(top_sv_fracs))
                result["neg_top_sv_frac_mean"] = -result["top_sv_frac_mean"]
            if sv_decays:
                result["sv_decay_mean"] = float(np.mean(sv_decays))

            # Normalized by seq_len (to control for length confound)
            result["eff_rank_per_token"] = float(er_arr.mean() / seq_len)

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL06 WARN] {type(e).__name__}: {e}")
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
        scorer = RankDefScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            check_layers=self.args.check_layers,
        )

        print(f"\n[NOVEL06] Extracting rank features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL06]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL06: RankDef-MIA RESULTS")
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
            print(f"  vs EXP50 memTrace RF:  0.6908")

            # M vs NM
            m, nm = df[df["is_member"]==1], df[df["is_member"]==0]
            for col in ["eff_rank_mean", "eff_rank_mid", "stable_rank_mean", "top_sv_frac_mean"]:
                if col in df.columns:
                    mv, nmv = m[col].dropna(), nm[col].dropna()
                    if len(mv)>0 and len(nmv)>0:
                        print(f"  {col:<30} M={mv.mean():.4f}  NM={nmv.mean():.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL06_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL06] Results saved.")


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
        check_layers = "quartile"  # "all", "key", "quartile"

    print(f"[NOVEL06] RankDef-MIA: Effective Rank Analysis")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
