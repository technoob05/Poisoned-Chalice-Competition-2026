"""
NOVEL EXPERIMENT 10: GradSparse-MIA — Gradient Sparsity Profile

NOVELTY: First MIA method using gradient SPARSITY (fraction of near-zero
    gradients and L1/L2 ratio) as a membership inference signal.
    Existing gradient MIA methods measure magnitude (norm), not distribution
    shape (sparsity).

Core Idea:
    At a flat minimum (where members reside after training), the gradient
    is not just SMALL but also SPARSE: most parameter directions require
    NO update, and only a few outlier directions have nonzero gradient.

    For NON-MEMBERS, the gradient is DENSER: many parameters need adjusting
    because the model hasn't optimized for this specific input.

    Sparsity metrics:
    1. L0-proxy: fraction of gradient elements < threshold (near-zero)
    2. L1/L2 ratio: measures spread (sparse → low L1/L2 ratio)
    3. Gini coefficient: measures inequality in gradient magnitudes
    4. Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1)

    We measure these at EACH layer to create a "sparsity profile."

    This is FUNDAMENTALLY DIFFERENT from:
    - EXP11/13: gradient MAGNITUDE (L2 norm) per layer → how big
    - EXP21: gradient in sliding windows → temporal variation
    - EXP23: gradient DIRECTION variance → angular spread
    - NOVEL07: gradient-hidden ANGLE → geometric relationship

    Sparsity measures gradient DISTRIBUTION SHAPE, not magnitude or direction.

Builds on Insights:
    - Flat minima: members have lower gradient (Insight 11)
    - Gradient ceiling: need new gradient FEATURES, not just magnitude
    - Layer profile shape matters more than single-layer values

Compute: 1 forward + 1 backward pass. Gradient analysis at selected layers.
Expected runtime: ~10-14 min on A100.
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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  NOVEL10: GradSparse-MIA — Gradient Sparsity Profile")
    print("  Novelty: Gradient distribution shape (sparsity) for MIA")
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


def hoyer_sparsity(x: np.ndarray) -> float:
    """Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1). Range [0,1]."""
    n = len(x)
    if n < 2:
        return 0.0
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x**2).sum())
    if l2 < 1e-15:
        return 1.0  # all zeros = maximally sparse
    ratio = l1 / l2
    sqrt_n = np.sqrt(n)
    return float((sqrt_n - ratio) / (sqrt_n - 1 + 1e-15))


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient of absolute values. Range [0,1]."""
    x = np.abs(x)
    if len(x) < 2 or x.sum() < 1e-15:
        return 0.0
    sorted_x = np.sort(x)
    n = len(sorted_x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_x) / (n * sorted_x.sum())) - (n + 1) / n)


class GradSparseScorer:
    """Extract gradient sparsity features at each layer."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        # Select layers to analyze
        step = max(1, self.n_layers // 8)
        self.check_layers = list(range(0, self.n_layers, step))
        if (self.n_layers - 1) not in self.check_layers:
            self.check_layers.append(self.n_layers - 1)
        print(f"  Checking {len(self.check_layers)} layers for sparsity")

    def _register_hooks(self):
        """Register hooks to capture hidden states for gradient computation."""
        self._hidden_states = {}
        self._hooks = []

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            return

        for idx in self.check_layers:
            if idx >= len(layers):
                continue
            layer = layers[idx]

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    hs.retain_grad()
                    self._hidden_states[layer_idx] = hs
                return hook_fn

            h = layer.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._hidden_states = {}

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

            # Also get embedding gradient
            self.model.zero_grad()
            embed_layer = self.model.get_input_embeddings()
            embeds = embed_layer(input_ids).detach().requires_grad_(True)

            self._register_hooks()
            outputs = self.model(inputs_embeds=embeds, labels=input_ids)
            loss = outputs.loss
            result["neg_mean_loss"] = -loss.float().item()
            loss.backward()

            # Embedding gradient sparsity
            if embeds.grad is not None:
                eg = embeds.grad.float().flatten().cpu().numpy()
                result["embed_grad_norm"] = float(np.sqrt((eg**2).sum()))
                result["neg_embed_grad_norm"] = -result["embed_grad_norm"]
                result["embed_hoyer"] = hoyer_sparsity(eg)
                result["embed_gini"] = gini_coefficient(eg)

                # L0-proxy: fraction of elements < 1% of max
                max_abs = np.abs(eg).max()
                if max_abs > 1e-15:
                    result["embed_l0_proxy"] = float((np.abs(eg) < 0.01 * max_abs).mean())
                else:
                    result["embed_l0_proxy"] = 1.0

                # L1/L2 ratio
                l1 = np.abs(eg).sum()
                l2 = np.sqrt((eg**2).sum())
                result["embed_l1l2_ratio"] = float(l1 / (l2 * np.sqrt(len(eg)) + 1e-10))

            # Per-layer gradient sparsity
            hoyer_values = []
            gini_values = []
            l0_values = []
            l1l2_values = []
            grad_norms = []

            for idx in self.check_layers:
                if idx not in self._hidden_states:
                    continue
                hs = self._hidden_states[idx]
                if hs.grad is None:
                    continue

                g = hs.grad[0].float().flatten().cpu().numpy()
                if len(g) == 0:
                    continue

                grad_norms.append(float(np.sqrt((g**2).sum())))
                hoyer_values.append(hoyer_sparsity(g))
                gini_values.append(gini_coefficient(g))

                max_abs = np.abs(g).max()
                if max_abs > 1e-15:
                    l0_values.append(float((np.abs(g) < 0.01 * max_abs).mean()))
                else:
                    l0_values.append(1.0)

                l1 = np.abs(g).sum()
                l2 = np.sqrt((g**2).sum())
                l1l2_values.append(float(l1 / (l2 * np.sqrt(len(g)) + 1e-10)))

            self._remove_hooks()

            if not hoyer_values:
                return result

            h_arr = np.array(hoyer_values)
            g_arr = np.array(gini_values)
            l0_arr = np.array(l0_values)
            n = len(h_arr)

            # --- SPARSITY FEATURES ---

            # Hoyer sparsity
            result["hoyer_mean"] = float(h_arr.mean())
            result["hoyer_std"] = float(h_arr.std())
            if n >= 3:
                third = n // 3
                result["hoyer_early"] = float(h_arr[:third].mean())
                result["hoyer_mid"] = float(h_arr[third:2*third].mean())
                result["hoyer_late"] = float(h_arr[2*third:].mean())
                result["hoyer_early_late_ratio"] = float(h_arr[:third].mean() / (h_arr[2*third:].mean() + 1e-10))

            # Gini
            result["gini_mean"] = float(g_arr.mean())

            # L0-proxy
            result["l0_mean"] = float(l0_arr.mean())

            # L1/L2 ratio
            if l1l2_values:
                result["neg_l1l2_mean"] = -float(np.mean(l1l2_values))

            # Gradient norm (baseline)
            if grad_norms:
                result["neg_layer_grad_norm_mean"] = -float(np.mean(grad_norms))

            # Sparsity profile slope (does sparsity increase across layers?)
            if n >= 3:
                x = np.arange(n, dtype=np.float32)
                slope = np.polyfit(x, h_arr, 1)[0]
                result["hoyer_slope"] = float(slope)

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            self._remove_hooks()
            if self._err_count < 3:
                print(f"\n[NOVEL10 WARN] {type(e).__name__}: {e}")
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
        scorer = GradSparseScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL10] Extracting gradient sparsity for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL10]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL10: GradSparse-MIA RESULTS")
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
            tag = " <-- SPARSITY" if "hoyer" in col or "gini" in col else ""
            print(f"  {d}{col:<35} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL10_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL10] Results saved.")


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

    print(f"[NOVEL10] GradSparse: Gradient Sparsity Profile")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
