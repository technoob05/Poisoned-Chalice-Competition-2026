"""
NOVEL EXPERIMENT 04: RepVel-MIA — Layer-wise Representation Velocity Profile

NOVELTY: First use of inter-layer "velocity" (rate of change of hidden states
    between consecutive transformer layers) as a membership inference signal.
    No prior MIA work characterizes the layer-to-layer evolution SPEED of
    representations.

Core Idea:
    As input passes through transformer layers, the hidden state evolves.
    The SPEED of this evolution (||H_l - H_{l-1}|| / ||H_{l-1}||) at each
    layer transition creates a "velocity profile."

    For MEMBERS (memorized sequences):
    - Early layers: FAST velocity (model quickly activates memorized patterns)
    - Late layers: SLOW velocity (representation has SETTLED into the
      memorized attractor — consistent with EXP50's finding that members
      have lower hidden state norms at mid-layer)
    - Profile shape: rapid DECELERATION

    For NON-MEMBERS:
    - More UNIFORM velocity across layers (model consistently works to
      map unfamiliar input to plausible continuations at every layer)
    - Late layers may show ACCELERATION (model still refining predictions)
    - Profile shape: flat or slightly increasing

    This is FUNDAMENTALLY DIFFERENT from:
    - EXP50 memTrace: measures hidden state MAGNITUDE (norm), not velocity
    - EXP43 AttenMIA: measures attention patterns, not representation dynamics
    - EXP26 ESR: measures output distribution divergence across layers
    - NOVEL01 Logit Lens: measures vocabulary-space predictions, not representation speed

Features extracted:
    1. velocity_profile: ||H_l - H_{l-1}|| for each consecutive layer pair
    2. velocity_decay_rate: slope of log(velocity) across layers
    3. early_late_velocity_ratio: mean(vel_early) / mean(vel_late)
    4. velocity_peak_layer: layer with maximum velocity
    5. velocity_variance: std of velocity profile
    6. deceleration_score: fraction of consecutive layers where vel decreases
    7. terminal_velocity: velocity at last layer transition
    8. velocity_auc: area under the velocity curve (total representation work)

Compute: 1 forward pass per sample (output_hidden_states=True), no backward.
Expected runtime: ~5-8 min on A100.
Expected AUC: 0.60-0.66
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
    print("  NOVEL04: RepVel-MIA — Representation Velocity Profile")
    print("  Novelty: Inter-layer evolution speed for membership inference")
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


class VelocityScorer:
    """Extract representation velocity features across transformer layers."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

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

            # hidden_states: tuple of (1, seq_len, dim) for embed + 30 layers
            hs_list = outputs.hidden_states  # len = n_layers + 1

            # Compute per-layer velocity: ||H_l - H_{l-1}|| (mean-pooled)
            velocities = []
            relative_velocities = []
            cosine_velocities = []

            for l in range(1, len(hs_list)):
                h_prev = hs_list[l - 1][0].float()  # (seq, dim)
                h_curr = hs_list[l][0].float()       # (seq, dim)

                # Mean-pooled versions
                mp_prev = h_prev.mean(dim=0)  # (dim,)
                mp_curr = h_curr.mean(dim=0)

                # Absolute velocity (L2 distance)
                vel = (mp_curr - mp_prev).norm(2).item()
                velocities.append(vel)

                # Relative velocity (normalized by prev norm)
                prev_norm = mp_prev.norm(2).item()
                rel_vel = vel / (prev_norm + 1e-10)
                relative_velocities.append(rel_vel)

                # Cosine velocity (1 - cosine similarity = directional change)
                cos_sim = torch.nn.functional.cosine_similarity(
                    mp_prev.unsqueeze(0), mp_curr.unsqueeze(0)
                ).item()
                cosine_velocities.append(1.0 - cos_sim)

            vel = np.array(velocities)
            rel_vel = np.array(relative_velocities)
            cos_vel = np.array(cosine_velocities)
            n = len(vel)

            # --- Feature extraction ---

            # 1. Basic statistics
            result["vel_mean"] = float(vel.mean())
            result["vel_std"] = float(vel.std())
            result["vel_max"] = float(vel.max())
            result["rel_vel_mean"] = float(rel_vel.mean())
            result["cos_vel_mean"] = float(cos_vel.mean())

            # 2. Early/mid/late velocity
            third = n // 3
            result["vel_early"] = float(vel[:third].mean())
            result["vel_mid"] = float(vel[third:2*third].mean())
            result["vel_late"] = float(vel[2*third:].mean())

            # 3. Early-late ratio (deceleration indicator)
            result["vel_early_late_ratio"] = float(
                vel[:third].mean() / (vel[2*third:].mean() + 1e-10)
            )
            result["vel_mid_late_ratio"] = float(
                vel[third:2*third].mean() / (vel[2*third:].mean() + 1e-10)
            )

            # 4. Velocity decay rate (slope of log velocity)
            log_vel = np.log(vel + 1e-10)
            x = np.arange(n, dtype=np.float32)
            if n >= 3:
                slope = np.polyfit(x, log_vel, 1)[0]
                result["vel_decay_rate"] = float(slope)
            else:
                result["vel_decay_rate"] = 0.0

            # 5. Peak velocity layer
            result["vel_peak_layer"] = float(np.argmax(vel) / n)

            # 6. Deceleration score (fraction of transitions where vel decreases)
            if n >= 2:
                diffs = np.diff(vel)
                result["deceleration_frac"] = float((diffs < 0).mean())
            else:
                result["deceleration_frac"] = 0.5

            # 7. Terminal velocity (last layer transition)
            result["vel_terminal"] = float(vel[-1])
            result["vel_terminal_relative"] = float(vel[-1] / (vel.mean() + 1e-10))

            # 8. Velocity AUC (total representation work)
            result["vel_auc"] = float(np.trapz(vel))

            # 9. Cosine velocity features
            result["cos_vel_early_late_ratio"] = float(
                cos_vel[:third].mean() / (cos_vel[2*third:].mean() + 1e-10)
            )
            result["cos_vel_max"] = float(cos_vel.max())
            result["cos_vel_late"] = float(cos_vel[2*third:].mean())

            # 10. Velocity profile entropy (shape diversity)
            vel_norm = vel / (vel.sum() + 1e-10)
            vel_entropy = -np.sum(vel_norm * np.log(vel_norm + 1e-15))
            result["vel_entropy"] = float(vel_entropy)

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL04 WARN] {type(e).__name__}: {e}")
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
        scorer = VelocityScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL04] Extracting velocity features for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL04]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[NOVEL04] Valid: {n_valid}/{len(df)}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL04: RepVel-MIA — Representation Velocity RESULTS")
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
            print(f"  vs EXP41 -grad_z_lang: 0.6539")

            # Distribution stats
            print("\n--- Member vs Non-Member Velocity Stats ---")
            m = df[df["is_member"] == 1]
            nm = df[df["is_member"] == 0]
            for col in ["vel_mean", "vel_early", "vel_late", "vel_early_late_ratio",
                         "vel_decay_rate", "deceleration_frac", "cos_vel_mean"]:
                if col not in df.columns:
                    continue
                mv = m[col].dropna()
                nmv = nm[col].dropna()
                if len(mv) > 0 and len(nmv) > 0:
                    print(f"  {col:<30} M={mv.mean():.4f}  NM={nmv.mean():.4f}  Δ={mv.mean()-nmv.mean():.4f}")

            # Per-subset
            best_col, (_, best_dir) = best_sig
            print(f"\n{'Subset':<10} | {best_col:<30} | N")
            print("-" * 55)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                v = sub.dropna(subset=[best_col])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    vals = v[best_col] if best_dir == "+" else -v[best_col]
                    auc = roc_auc_score(v["is_member"], vals)
                else:
                    auc = float("nan")
                print(f"  {subset:<10} | {auc:.4f}                     | {len(sub)}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL04_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL04] Results saved.")


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

    print(f"[NOVEL04] RepVel-MIA: Representation Velocity")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
