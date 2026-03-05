"""
NOVEL EXPERIMENT 07: GradOrth-MIA — Gradient-Hidden State Orthogonality

NOVELTY: First MIA method measuring the ANGULAR RELATIONSHIP between gradient
    vectors and hidden state vectors at each transformer layer.
    No prior MIA work examines whether gradients are aligned or orthogonal
    to the representation at the same layer.

Core Idea:
    At each layer, we have two vectors of the same dimensionality:
    1. Hidden state H_l (the representation)
    2. Gradient ∂L/∂H_l (the update direction)

    For MEMBERS (at flat minimum):
    - The model has ALREADY learned this sample
    - The gradient is SMALL (confirmed: EXP11 grad_norm, M/NM ≈ 0.63)
    - The gradient is more ORTHOGONAL to the hidden state direction
      (no systematic push toward or away from the representation manifold)
    - cos(grad, hidden) ≈ 0 (random orientation of small gradient)

    For NON-MEMBERS:
    - The model needs to UPDATE its representation
    - The gradient pushes the representation toward the learned manifold
    - Gradient is more ALIGNED with the hidden state vector
    - |cos(grad, hidden)| > 0 (systematic push direction)

    Signal: |cos(∂L/∂H_l, H_l)| — absolute cosine similarity.
    Members should have LOWER alignment (more orthogonal).

Builds on Insights:
    - Insight 11: gradient carries directional information beyond magnitude
    - Insight 8: gradient magnitude ceiling ~0.65, need new gradient features
    - Flat minima: small, randomly-oriented gradients for members
    - EXP23: gradient direction variance is informative

Compute: 1 forward + 1 backward pass per sample.
    gradient_checkpointing for memory efficiency.
Expected runtime: ~12-15 min on A100.
Expected AUC: 0.56-0.64
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    print("  NOVEL07: GradOrth-MIA — Gradient-Hidden Orthogonality")
    print("  Novelty: Angular relationship between gradient and representation")
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
    print(f"  Loaded. dtype={dtype}, layers={model.config.num_hidden_layers}")
    return model, tokenizer


class GradOrthScorer:
    """Measure gradient-hidden state orthogonality at each layer."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 layer_indices: List[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_layers = model.config.num_hidden_layers
        self._err_count = 0

        if layer_indices is None:
            step = max(1, self.n_layers // 8)
            self.layer_indices = list(range(0, self.n_layers, step))
        else:
            self.layer_indices = layer_indices
        print(f"  Check layers: {self.layer_indices}")

    def _register_hooks(self):
        """Register hooks to capture hidden states with grad retention."""
        self._hidden_states = {}
        self._hooks = []

        # Access model layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            return

        for idx in self.layer_indices:
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

            # Register hooks to capture hidden states
            self._register_hooks()
            self.model.zero_grad()

            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            result["neg_mean_loss"] = -loss.float().item()

            # Backward to compute gradients of hidden states
            loss.backward()

            # Compute orthogonality at each layer
            ortho_values = []
            abs_ortho_values = []
            grad_norms = []
            hs_norms = []

            for idx in self.layer_indices:
                if idx not in self._hidden_states:
                    continue
                hs = self._hidden_states[idx]
                if hs.grad is None:
                    continue

                # Mean-pool over sequence dimension
                hs_vec = hs[0].float().mean(dim=0)  # (dim,)
                grad_vec = hs.grad[0].float().mean(dim=0)  # (dim,)

                # Cosine similarity between gradient and hidden state
                cos_sim = F.cosine_similarity(
                    hs_vec.unsqueeze(0), grad_vec.unsqueeze(0)
                ).item()
                ortho_values.append(cos_sim)
                abs_ortho_values.append(abs(cos_sim))

                # Individual norms
                grad_norms.append(grad_vec.norm(2).item())
                hs_norms.append(hs_vec.norm(2).item())

            self._remove_hooks()

            if not ortho_values:
                return result

            ortho = np.array(ortho_values)
            abs_ortho = np.array(abs_ortho_values)
            gn = np.array(grad_norms)
            hn = np.array(hs_norms)
            n = len(ortho)

            # --- Features ---

            # 1. Overall orthogonality
            result["neg_abs_ortho_mean"] = -float(abs_ortho.mean())  # lower abs = more orthogonal = member
            result["ortho_mean"] = float(ortho.mean())  # signed cosine sim
            result["ortho_std"] = float(ortho.std())

            # 2. Layer-wise orthogonality
            if n >= 4:
                q = n // 4
                result["neg_abs_ortho_early"] = -float(abs_ortho[:q].mean())
                result["neg_abs_ortho_mid"] = -float(abs_ortho[q:3*q].mean())
                result["neg_abs_ortho_late"] = -float(abs_ortho[3*q:].mean())

            # 3. Orthogonality trajectory
            if n >= 3:
                x = np.arange(n, dtype=np.float32)
                slope = np.polyfit(x, abs_ortho, 1)[0]
                result["ortho_slope"] = float(slope)  # negative slope = increasing orthogonality

            # 4. Combined with gradient magnitude
            # grad_magnitude × alignment = "effective pressure"
            pressure = gn * abs_ortho  # how much aligned force the gradient exerts
            result["neg_effective_pressure"] = -float(pressure.mean())

            # 5. Gradient-to-hidden norm ratio (flat minima signature)
            ratio = gn / (hn + 1e-10)
            result["neg_gh_ratio_mean"] = -float(ratio.mean())

            # 6. Orthogonality at specifically informative layers
            if n >= 2:
                mid_idx = n // 2
                result["neg_abs_ortho_at_mid"] = -float(abs_ortho[mid_idx])
                result["neg_abs_ortho_at_last"] = -float(abs_ortho[-1])

            # 7. Pure gradient features (bonus)
            result["neg_grad_norm_layers_mean"] = -float(gn.mean())
            result["neg_hs_norm_layers_mean"] = -float(hn.mean())

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            self._remove_hooks()
            if self._err_count < 3:
                print(f"\n[NOVEL07 WARN] {type(e).__name__}: {e}")
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
        scorer = GradOrthScorer(
            self.model, self.tokenizer, max_length=self.args.max_length,
        )

        print(f"\n[NOVEL07] Extracting gradient-hidden orthogonality for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL07]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL07: GradOrth-MIA RESULTS")
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
            tag = " <-- PRIMARY" if col == "neg_abs_ortho_mean" else ""
            print(f"  {d}{col:<35} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL07_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL07] Results saved.")


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

    print(f"[NOVEL07] GradOrth: Gradient-Hidden Orthogonality")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
