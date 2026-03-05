"""
NOVEL EXPERIMENT 11: ResDecomp-MIA — Residual Stream Decomposition

NOVELTY: First MIA method that decomposes the transformer's residual stream
    update at each block into its ATTENTION vs MLP components and uses their
    relative contributions as a membership signal.
    No prior MIA work separates attention and MLP contributions.

Core Idea:
    In a transformer layer, the hidden state update is:
        H_{l+1} = H_l + Attn_l(H_l) + MLP_l(H_l + Attn_l(H_l))

    The update comes from two sources:
    - Attention output: information routing / token mixing
    - MLP output: token-level knowledge retrieval / computation

    For MEMBERS, the balance between attention and MLP contributions may
    differ from non-members because:
    - Memorized content may rely more on MLP (knowledge stored in FF weights)
    - Non-member content may require more attention (novel pattern matching)

    We use forward hooks to capture:
    1. Attention sublayer output norm: ||Attn_l||
    2. MLP sublayer output norm: ||MLP_l||
    3. Attention/MLP ratio: ||Attn_l|| / ||MLP_l||
    4. Component-wise cosine: cos(Attn_l, MLP_l) — are they aligned?
    5. Component dominance: which component contributes more at each layer

    These create a "decomposition profile" that may fingerprint memorization.

Builds on Insights:
    - EXP50: hidden state NORMS are powerful (0.6908) → sublayer norms may add info
    - EXP34: blockwise features worked (block_sign_profile AUC 0.6408)
    - Insight 22: mid-layer features are most discriminative

Compute: 1 forward pass with hooks on attention and MLP sublayers.
Expected runtime: ~8-12 min on A100.
Expected AUC: 0.58-0.65
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
    print("  NOVEL11: ResDecomp-MIA — Residual Stream Decomposition")
    print("  Novelty: Attention vs MLP contribution balance for MIA")
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
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded. dtype={dtype}, layers={n_layers}")
    return model, tokenizer


class ResDecompScorer:
    """Decompose residual stream into attention and MLP contributions."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        # Detect architecture to find attention and MLP sublayers
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect transformer architecture to locate sublayers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layers = self.model.model.layers
            # StarCoder2 architecture
            layer0 = self.layers[0]
            if hasattr(layer0, 'self_attn'):
                self.attn_attr = 'self_attn'
            elif hasattr(layer0, 'attn'):
                self.attn_attr = 'attn'
            else:
                self.attn_attr = None

            if hasattr(layer0, 'mlp'):
                self.mlp_attr = 'mlp'
            else:
                self.mlp_attr = None
            print(f"  Architecture: attn={self.attn_attr}, mlp={self.mlp_attr}")
        else:
            self.layers = None
            self.attn_attr = None
            self.mlp_attr = None
            print("  [WARN] Could not detect architecture")

    def _register_hooks(self):
        """Register hooks on attention and MLP sublayers."""
        self._attn_outputs = {}
        self._mlp_outputs = {}
        self._hooks = []

        if self.layers is None:
            return

        for idx in range(self.n_layers):
            layer = self.layers[idx]

            if self.attn_attr:
                attn = getattr(layer, self.attn_attr)
                def attn_hook(module, input, output, layer_idx=idx):
                    if isinstance(output, tuple):
                        self._attn_outputs[layer_idx] = output[0].detach()
                    else:
                        self._attn_outputs[layer_idx] = output.detach()
                h = attn.register_forward_hook(attn_hook)
                self._hooks.append(h)

            if self.mlp_attr:
                mlp = getattr(layer, self.mlp_attr)
                def mlp_hook(module, input, output, layer_idx=idx):
                    if isinstance(output, tuple):
                        self._mlp_outputs[layer_idx] = output[0].detach()
                    else:
                        self._mlp_outputs[layer_idx] = output.detach()
                h = mlp.register_forward_hook(mlp_hook)
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attn_outputs = {}
        self._mlp_outputs = {}

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

            self._register_hooks()
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            result["neg_mean_loss"] = -outputs.loss.float().item()

            if not self._attn_outputs or not self._mlp_outputs:
                self._remove_hooks()
                return result

            attn_norms = []
            mlp_norms = []
            attn_mlp_ratios = []
            attn_mlp_cosines = []

            for idx in range(self.n_layers):
                if idx not in self._attn_outputs or idx not in self._mlp_outputs:
                    continue

                attn_out = self._attn_outputs[idx][0].float()  # (seq, dim)
                mlp_out = self._mlp_outputs[idx][0].float()

                # Mean-pooled norms
                attn_norm = attn_out.mean(dim=0).norm(2).item()
                mlp_norm = mlp_out.mean(dim=0).norm(2).item()
                attn_norms.append(attn_norm)
                mlp_norms.append(mlp_norm)

                # Ratio
                ratio = attn_norm / (mlp_norm + 1e-10)
                attn_mlp_ratios.append(ratio)

                # Cosine similarity between attn and mlp outputs
                a_flat = attn_out.flatten()
                m_flat = mlp_out.flatten()
                cos = F.cosine_similarity(a_flat.unsqueeze(0), m_flat.unsqueeze(0)).item()
                attn_mlp_cosines.append(cos)

            self._remove_hooks()

            if not attn_norms:
                return result

            an = np.array(attn_norms)
            mn = np.array(mlp_norms)
            ratios = np.array(attn_mlp_ratios)
            cosines = np.array(attn_mlp_cosines)
            n = len(an)

            # --- DECOMPOSITION FEATURES ---

            # Attention norms
            result["neg_attn_norm_mean"] = -float(an.mean())
            result["neg_attn_norm_mid"] = -float(an[n//2]) if n > 0 else 0.0

            # MLP norms
            result["neg_mlp_norm_mean"] = -float(mn.mean())
            result["neg_mlp_norm_mid"] = -float(mn[n//2]) if n > 0 else 0.0

            # Attention/MLP ratio
            result["attn_mlp_ratio_mean"] = float(ratios.mean())
            result["attn_mlp_ratio_std"] = float(ratios.std())
            if n >= 3:
                third = n // 3
                result["attn_mlp_ratio_early"] = float(ratios[:third].mean())
                result["attn_mlp_ratio_mid"] = float(ratios[third:2*third].mean())
                result["attn_mlp_ratio_late"] = float(ratios[2*third:].mean())

            # Which component dominates more (per-layer count)
            result["attn_dominant_frac"] = float((ratios > 1.0).mean())

            # Cosine between attn and mlp
            result["attn_mlp_cos_mean"] = float(cosines.mean())
            result["neg_attn_mlp_cos_mean"] = -float(cosines.mean())
            if n >= 3:
                result["attn_mlp_cos_late"] = float(cosines[2*n//3:].mean())

            # Ratio trajectory (slope)
            if n >= 3:
                x = np.arange(n, dtype=np.float32)
                slope = np.polyfit(x, ratios, 1)[0]
                result["ratio_slope"] = float(slope)

            # Total contribution balance (attn vs mlp across all layers)
            total_attn = an.sum()
            total_mlp = mn.sum()
            result["total_attn_frac"] = float(total_attn / (total_attn + total_mlp + 1e-10))

            # Combined MLP + attn norm (total update magnitude)
            total_update = an + mn
            result["neg_total_update_mean"] = -float(total_update.mean())

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            self._remove_hooks()
            if self._err_count < 3:
                print(f"\n[NOVEL11 WARN] {type(e).__name__}: {e}")
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
        scorer = ResDecompScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL11] Extracting residual decomposition for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL11]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL11: ResDecomp-MIA RESULTS")
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
            tag = " <-- DECOMP" if "ratio" in col or "frac" in col else ""
            print(f"  {d}{col:<40} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL11_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL11] Results saved.")


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

    print(f"[NOVEL11] ResDecomp: Residual Stream Decomposition")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
