"""
EXPERIMENT 48: ICIMIA-Adapted — Input Corruption-Inspired MIA via
                Representation Robustness

Paper: "Image Corruption-Inspired Membership Inference Attacks against
        Large Vision-Language Models"
       Wu, Lin, Zhang, Wang, Zhang, Zhang, Wang
       Penn State University (arXiv:2506.12340v3, Feb 2026)

Core insight (adapted from vision to code):
    The original paper observes that VLLMs produce MORE ROBUST image embeddings
    for member images under corruption (blur, compression) than non-members.
    The cosine similarity between original and corrupted image embeddings is
    higher for members (AUC 0.881 on LLaVA-1.5-7B with Gaussian Blur).

    We adapt this to code LLMs:
    - Instead of image corruption → use TEXT corruption (token masking,
      token dropping, span noise)
    - Instead of image embeddings → use HIDDEN STATE embeddings from
      intermediate/last transformer layers
    - Core hypothesis: StarCoder2-3b produces more robust internal
      representations for training code (members) under text corruption
      because the model "recalls" memorized content even when input is degraded

    This is FUNDAMENTALLY DIFFERENT from our failed perturbation experiments
    (EXP23/24/28/31/32 — Insight 9) which measured LOSS or GRADIENT change.
    Here we measure REPRESENTATION SIMILARITY (cosine sim of hidden states),
    a different signal that captures how the model's internal encoding changes,
    not just its prediction confidence.

Corruption methods for code:
    1. Token masking: replace random K% of tokens with random vocab tokens
    2. Token dropping: remove random K% of tokens from sequence
    3. Span corruption: mask contiguous spans of L tokens at random positions

Algorithm (per sample):
    1. Forward pass on original code → extract hidden states H(x) at layer(s)
    2. Corrupt code → x'
    3. Forward pass on corrupted code → extract hidden states H(x')
    4. score = cosine_similarity(mean_pool(H(x)), mean_pool(H(x')))
    5. Higher similarity → more likely member

Paper results (on VLLMs):
    - White-box (Gaussian Blur, kernel=5): AUC 0.881 on LLaVA-1.5-7B
    - Black-box (text output similarity): AUC 0.652
    - Outperforms all logit-based baselines (best baseline AUC 0.743)

Adaptation considerations:
    - Original paper: FINE-TUNED VLLMs where vision encoder memorizes images
    - Our setup: PRE-TRAINED code LLM where memorization is weaker
    - Previous perturbation experiments (5/5 failed, Insight 9) suggest
      perturbation-based signals are weak on code LLMs
    - BUT: embedding similarity is a different signal class than loss/gradient
    - The paper's white-box approach (embedding comparison) worked much better
      than their black-box (text output comparison): 0.881 vs 0.652

Compute: 2-4 forward passes per sample (original + 1-3 corruption methods)
    Forward-only (no backward pass), max_length=512
Expected runtime: ~10-15 min on A100 (10% sample)
Expected AUC: 0.50-0.60 (perturbation signals have been weak on code LLMs,
    but embedding similarity is a new signal class worth testing;
    if AUC > 0.55 this is a genuinely new orthogonal signal for stacking)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    print("  EXP48: ICIMIA-Adapted — Corruption Robustness MIA")
    print("  Paper: Wu et al. (arXiv:2506.12340v3, Feb 2026)")
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


class TextCorruptor:
    """Text corruption methods adapted from ICIMIA's image corruption."""

    def __init__(self, tokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = np.random.RandomState(seed)
        self.vocab_size = tokenizer.vocab_size

    def token_mask(self, input_ids: torch.Tensor, rate: float = 0.10) -> torch.Tensor:
        """Replace random tokens with random vocab tokens (analogous to Gaussian blur)."""
        ids = input_ids.clone()
        seq_len = ids.shape[1]
        n_mask = max(1, int(seq_len * rate))
        positions = self.rng.choice(seq_len, size=n_mask, replace=False)
        replacements = self.rng.randint(0, self.vocab_size, size=n_mask)
        for pos, rep in zip(positions, replacements):
            ids[0, pos] = rep
        return ids

    def token_drop(self, input_ids: torch.Tensor, rate: float = 0.10) -> torch.Tensor:
        """Remove random tokens from sequence (analogous to JPEG compression)."""
        ids = input_ids.squeeze(0).tolist()
        seq_len = len(ids)
        n_drop = max(1, int(seq_len * rate))
        drop_positions = set(self.rng.choice(seq_len, size=n_drop, replace=False))
        kept = [t for i, t in enumerate(ids) if i not in drop_positions]
        if len(kept) < 5:
            kept = ids[:5]
        return torch.tensor([kept], dtype=input_ids.dtype, device=input_ids.device)

    def span_noise(self, input_ids: torch.Tensor, n_spans: int = 3,
                   span_len: int = 5) -> torch.Tensor:
        """Mask contiguous spans with random tokens (analogous to Motion blur)."""
        ids = input_ids.clone()
        seq_len = ids.shape[1]
        for _ in range(n_spans):
            if seq_len <= span_len:
                break
            start = self.rng.randint(0, seq_len - span_len)
            replacements = self.rng.randint(0, self.vocab_size, size=span_len)
            for j in range(span_len):
                ids[0, start + j] = replacements[j]
        return ids


class ICIMIAScorer:
    """ICIMIA-adapted scorer: measures representation robustness under corruption."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 corruption_rates: List[float] = None,
                 extract_layers: List[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0
        self.n_layers = model.config.num_hidden_layers

        self.corruption_rates = corruption_rates or [0.05, 0.10, 0.20]
        self.corruptor = TextCorruptor(tokenizer)

        # Layers to extract embeddings from
        # "last" = final layer, "mid" = middle layer, "early" = 1/4 layer
        self.layer_indices = {
            "early": max(0, self.n_layers // 4),
            "mid": self.n_layers // 2,
            "late": self.n_layers - 2,
            "last": self.n_layers - 1,
        }
        if extract_layers:
            self.layer_indices = {k: v for k, v in self.layer_indices.items()
                                  if k in extract_layers}

        print(f"  Corruption rates: {self.corruption_rates}")
        print(f"  Extract layers: {self.layer_indices}")

    @torch.no_grad()
    def _get_hidden_states(self, input_ids: torch.Tensor) -> Dict[str, np.ndarray]:
        """Forward pass returning mean-pooled hidden states at selected layers."""
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        result = {}
        for name, layer_idx in self.layer_indices.items():
            # hidden_states[0] = embedding, hidden_states[i+1] = layer i output
            hs = outputs.hidden_states[layer_idx + 1]  # (1, seq_len, hidden_dim)
            pooled = hs.float().mean(dim=1).squeeze(0).detach().cpu().numpy()  # (hidden_dim,)
            result[name] = pooled
        return result

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def extract(self, text: str) -> Dict[str, float]:
        """Extract corruption-robustness features for a single sample."""
        feature_keys = []
        for layer_name in self.layer_indices:
            for rate in self.corruption_rates:
                r_tag = f"{int(rate*100)}"
                feature_keys.append(f"sim_mask_{r_tag}_{layer_name}")
                feature_keys.append(f"sim_drop_{r_tag}_{layer_name}")
            feature_keys.append(f"sim_span_{layer_name}")

        feature_keys += [
            "sim_mask_avg", "sim_drop_avg", "sim_span_avg",
            "sim_combined_avg",
            "neg_mean_loss",
        ]

        result = {k: np.nan for k in feature_keys}

        if not text or len(text) < 30:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 10:
                return result

            # Original forward pass → hidden states + loss
            orig_out = self.model(input_ids=input_ids, labels=input_ids,
                                  output_hidden_states=True)
            result["neg_mean_loss"] = -orig_out.loss.float().item()

            orig_hs = {}
            for name, layer_idx in self.layer_indices.items():
                hs = orig_out.hidden_states[layer_idx + 1]
                orig_hs[name] = hs.float().mean(dim=1).squeeze(0).detach().cpu().numpy()

            all_sims = []

            # Token masking at various rates
            for rate in self.corruption_rates:
                r_tag = f"{int(rate*100)}"
                corrupted_ids = self.corruptor.token_mask(input_ids, rate=rate)
                corr_hs = self._get_hidden_states(corrupted_ids)
                for layer_name in self.layer_indices:
                    sim = self._cosine_sim(orig_hs[layer_name], corr_hs[layer_name])
                    result[f"sim_mask_{r_tag}_{layer_name}"] = sim
                    all_sims.append(("mask", sim))

            # Token dropping at various rates
            for rate in self.corruption_rates:
                r_tag = f"{int(rate*100)}"
                corrupted_ids = self.corruptor.token_drop(input_ids, rate=rate)
                corr_hs = self._get_hidden_states(corrupted_ids)
                for layer_name in self.layer_indices:
                    # For token drop, seq lengths differ → still compare mean-pooled
                    sim = self._cosine_sim(orig_hs[layer_name], corr_hs[layer_name])
                    result[f"sim_drop_{r_tag}_{layer_name}"] = sim
                    all_sims.append(("drop", sim))

            # Span corruption
            corrupted_ids = self.corruptor.span_noise(input_ids, n_spans=3, span_len=5)
            corr_hs = self._get_hidden_states(corrupted_ids)
            for layer_name in self.layer_indices:
                sim = self._cosine_sim(orig_hs[layer_name], corr_hs[layer_name])
                result[f"sim_span_{layer_name}"] = sim
                all_sims.append(("span", sim))

            # Aggregated scores
            mask_sims = [s for t, s in all_sims if t == "mask"]
            drop_sims = [s for t, s in all_sims if t == "drop"]
            span_sims = [s for t, s in all_sims if t == "span"]

            if mask_sims:
                result["sim_mask_avg"] = float(np.mean(mask_sims))
            if drop_sims:
                result["sim_drop_avg"] = float(np.mean(drop_sims))
            if span_sims:
                result["sim_span_avg"] = float(np.mean(span_sims))
            if all_sims:
                result["sim_combined_avg"] = float(np.mean([s for _, s in all_sims]))

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP48 WARN] {type(e).__name__}: {e}")
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
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()

        scorer = ICIMIAScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            corruption_rates=self.args.corruption_rates,
            extract_layers=self.args.extract_layers,
        )

        print(f"\n[EXP48] Extracting corruption-robustness features for {len(df)} samples...")
        print(f"  Corruption methods: token_mask, token_drop, span_noise")
        print(f"  Forward passes per sample: ~{1 + len(self.args.corruption_rates)*2 + 1}")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP48]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        n_valid = feat_df["sim_combined_avg"].notna().sum()
        print(f"\n[EXP48] Valid: {n_valid}/{len(df)}")
        if scorer._err_count > 0:
            print(f"[EXP48] Errors: {scorer._err_count}")

        # --- Report ---
        print("\n" + "=" * 70)
        print("   EXP48: ICIMIA-Adapted — Corruption Robustness REPORT")
        print("=" * 70)

        # Collect all score columns
        sim_cols = [c for c in df.columns if c.startswith("sim_")]
        sim_cols.append("neg_mean_loss")

        print("\n--- All Signal AUCs ---")
        aucs = {}
        for col in sorted(sim_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc = roc_auc_score(v["is_member"], v[col])
            aucs[col] = auc
            tag = ""
            if col == "sim_combined_avg":
                tag = " <-- PRIMARY"
            elif col == "neg_mean_loss":
                tag = " (baseline)"
            print(f"  {col:<40} AUC = {auc:.4f}{tag}")

        if aucs:
            best = max(aucs, key=aucs.get)
            print(f"\n  Best signal: {best} = {aucs[best]:.4f}")

        print(f"\n  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP11 -grad_embed:    0.6472")

        # Key comparison: does embedding similarity beat loss/gradient?
        combined_auc = aucs.get("sim_combined_avg", 0)
        loss_auc = aucs.get("neg_mean_loss", 0)
        print(f"\n  Corruption robustness vs loss: {combined_auc:.4f} vs {loss_auc:.4f}")
        if combined_auc > loss_auc + 0.005:
            print(f"  -> REPRESENTATION ROBUSTNESS adds signal beyond loss (+{combined_auc-loss_auc:.4f})")
        elif combined_auc < loss_auc - 0.005:
            print(f"  -> Representation robustness WEAKER than loss ({combined_auc-loss_auc:.4f})")
        else:
            print(f"  -> Representation robustness ~= loss (within noise)")

        # Distribution statistics
        print("\n--- Distribution Statistics (Members vs Non-Members) ---")
        m = df[df["is_member"] == 1]
        nm = df[df["is_member"] == 0]
        for col in ["sim_combined_avg", "sim_mask_avg", "sim_drop_avg", "sim_span_avg"]:
            if col not in df.columns:
                continue
            m_val = m[col].dropna()
            nm_val = nm[col].dropna()
            if len(m_val) > 0 and len(nm_val) > 0:
                print(
                    f"  {col:<25} M={m_val.mean():.6f}+-{m_val.std():.6f}  "
                    f"NM={nm_val.mean():.6f}+-{nm_val.std():.6f}  "
                    f"delta={m_val.mean()-nm_val.mean():.6f}"
                )

        # Per-layer analysis
        print("\n--- Per-Layer Best AUC ---")
        for layer_name in scorer.layer_indices:
            layer_aucs = {k: v for k, v in aucs.items() if layer_name in k}
            if layer_aucs:
                best_k = max(layer_aucs, key=layer_aucs.get)
                print(f"  {layer_name:<8} best: {best_k} = {layer_aucs[best_k]:.4f}")

        # Per-corruption-method analysis
        print("\n--- Per-Corruption-Method Best AUC ---")
        for method in ["mask", "drop", "span"]:
            method_aucs = {k: v for k, v in aucs.items()
                           if method in k and "avg" not in k}
            if method_aucs:
                best_k = max(method_aucs, key=method_aucs.get)
                print(f"  {method:<8} best: {best_k} = {method_aucs[best_k]:.4f}")

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'Combined':<10} | {'Mask_avg':<10} | {'Drop_avg':<10} | {'Loss':<8} | N")
        print("-" * 70)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            r = {}
            for sc in ["sim_combined_avg", "sim_mask_avg", "sim_drop_avg", "neg_mean_loss"]:
                if sc not in sub.columns:
                    r[sc] = float("nan")
                    continue
                v = sub.dropna(subset=[sc])
                if not v.empty and len(v["is_member"].unique()) > 1:
                    r[sc] = roc_auc_score(v["is_member"], v[sc])
                else:
                    r[sc] = float("nan")
            print(
                f"{subset:<10} | {r.get('sim_combined_avg', float('nan')):.4f}     "
                f"| {r.get('sim_mask_avg', float('nan')):.4f}     "
                f"| {r.get('sim_drop_avg', float('nan')):.4f}     "
                f"| {r.get('neg_mean_loss', float('nan')):.4f}   "
                f"| {len(sub)}"
            )

        # Verdict
        print("\n--- VERDICT ---")
        if combined_auc > 0.55:
            print(f"  PROMISING: sim_combined_avg={combined_auc:.4f} > 0.55")
            print(f"  → Representation robustness IS a viable signal for code LLM MIA")
            print(f"  → Add to EXP15 stacker as orthogonal feature to gradient")
        elif combined_auc > 0.52:
            print(f"  WEAK: sim_combined_avg={combined_auc:.4f} (barely above random)")
            print(f"  → Representation robustness adds marginal signal, not useful standalone")
        else:
            print(f"  FAILED: sim_combined_avg={combined_auc:.4f} (near random)")
            print(f"  → Confirms Insight 9: perturbation-based signals fail on code LLMs")
            print(f"  → Even embedding-level similarity doesn't help")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP48_{timestamp}.parquet", index=False)
        print(f"\n[EXP48] Results saved.")


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
        corruption_rates = [0.05, 0.10, 0.20]
        extract_layers = ["mid", "last"]  # focus on mid+last for efficiency

    n_fwd = 1 + len(Args.corruption_rates) * 2 + 1  # orig + mask×rates + drop×rates + span
    print(f"[EXP48] ICIMIA-Adapted: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  corruption_rates={Args.corruption_rates}, layers={Args.extract_layers}")
    print(f"  ~{n_fwd} forward passes per sample")
    Experiment(Args).run()
