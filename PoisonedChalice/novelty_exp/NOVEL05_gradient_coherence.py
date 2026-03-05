"""
NOVEL EXPERIMENT 05: GradCoh-MIA — Cross-Subsequence Gradient Coherence

NOVELTY: First MIA method that measures gradient COHERENCE between different
    parts of the same input. No prior MIA work decomposes the gradient by
    input subsequence.

Core Idea:
    During pre-training, member code files were processed as COMPLETE units.
    The model learned a unified representation where ALL parts of the file
    contribute to a coherent optimization direction.

    If we split the code into two halves and compute gradients separately:
    - MEMBERS: gradients from each half point in SIMILAR directions
      (cosine sim ≈ high) because the model learned this file as a coherent whole
    - NON-MEMBERS: gradients from each half point in DIFFERENT directions
      (cosine sim ≈ low) because the model has no unified representation

    This tests the "holistic memorization" hypothesis: does the model
    memorize code files as integrated units, or just local patterns?

    Signal: cosine_similarity(grad_first_half, grad_second_half)
    Members should have HIGHER coherence.

    This is FUNDAMENTALLY DIFFERENT from:
    - EXP05 SIA-TTS: split by token type, not by position
    - EXP21 GradWindow: sliding window gradient MAGNITUDE, not direction coherence
    - EXP23 GradDirection: variance of full-sequence gradient, not cross-subsequence

Builds on Insights:
    - Insight 11: gradient carries information beyond just magnitude
    - Insight 12: embedding gradient is strongest gradient signal
    - Flat minima hypothesis: directionality matters, not just norm

Compute: 2 backward passes per sample (one per half).
Expected runtime: ~15-20 min on A100.
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
    print("  NOVEL05: GradCoh-MIA — Cross-Subsequence Gradient Coherence")
    print("  Novelty: Gradient direction coherence between code halves")
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


class GradCoherenceScorer:
    """Compute gradient coherence between code subsequences."""

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0

    def _get_embedding_gradient(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get flattened gradient of loss w.r.t. embedding layer parameters."""
        self.model.zero_grad()
        embed_layer = self.model.get_input_embeddings()
        embeds = embed_layer(input_ids).detach().requires_grad_(True)
        outputs = self.model(inputs_embeds=embeds, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        grad = embeds.grad.float().flatten()
        return grad, loss.float().item()

    def _get_param_gradient(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get gradient w.r.t. model parameters (not embedding input)."""
        self.model.zero_grad()
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        # Collect gradient from embedding weight matrix
        embed_param = self.model.get_input_embeddings().weight
        if embed_param.grad is not None:
            return embed_param.grad.float().flatten(), loss.float().item()
        return None, loss.float().item()

    def extract(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 60:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 20:
                return result

            # Full sequence gradient and loss
            grad_full, loss_full = self._get_embedding_gradient(input_ids)
            result["neg_mean_loss"] = -loss_full
            result["neg_grad_norm_full"] = -grad_full.norm(2).item()

            # Split into halves
            mid = seq_len // 2
            ids_first = input_ids[:, :mid]
            ids_second = input_ids[:, mid:]

            if ids_first.shape[1] < 10 or ids_second.shape[1] < 10:
                return result

            # Gradient for each half
            grad_first, loss_first = self._get_embedding_gradient(ids_first)
            grad_second, loss_second = self._get_embedding_gradient(ids_second)

            # Trim to same length for cosine similarity
            min_len = min(grad_first.shape[0], grad_second.shape[0])
            g1 = grad_first[:min_len]
            g2 = grad_second[:min_len]

            # --- CORE: Gradient coherence (cosine similarity) ---
            cos_sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            result["grad_coherence"] = cos_sim

            # L2 distance between gradient directions (normalized)
            g1_dir = g1 / (g1.norm(2) + 1e-10)
            g2_dir = g2 / (g2.norm(2) + 1e-10)
            result["grad_dir_distance"] = -(g1_dir - g2_dir).norm(2).item()

            # Magnitude coherence: ratio of half gradients to full
            norm_first = grad_first.norm(2).item()
            norm_second = grad_second.norm(2).item()
            norm_ratio = min(norm_first, norm_second) / (max(norm_first, norm_second) + 1e-10)
            result["grad_mag_ratio"] = norm_ratio

            # Loss coherence: difference between halves
            result["loss_coherence"] = -abs(loss_first - loss_second)

            # Combined gradient coherence (cosine × magnitude ratio)
            result["combined_coherence"] = cos_sim * norm_ratio

            # Also try thirds (split into 3 parts)
            third = seq_len // 3
            if third >= 10:
                ids_t1 = input_ids[:, :third]
                ids_t2 = input_ids[:, third:2*third]
                ids_t3 = input_ids[:, 2*third:]

                if ids_t1.shape[1] >= 10 and ids_t2.shape[1] >= 10 and ids_t3.shape[1] >= 10:
                    gt1, _ = self._get_embedding_gradient(ids_t1)
                    gt2, _ = self._get_embedding_gradient(ids_t2)
                    gt3, _ = self._get_embedding_gradient(ids_t3)

                    min3 = min(gt1.shape[0], gt2.shape[0], gt3.shape[0])
                    gt1, gt2, gt3 = gt1[:min3], gt2[:min3], gt3[:min3]

                    cos_12 = F.cosine_similarity(gt1.unsqueeze(0), gt2.unsqueeze(0)).item()
                    cos_13 = F.cosine_similarity(gt1.unsqueeze(0), gt3.unsqueeze(0)).item()
                    cos_23 = F.cosine_similarity(gt2.unsqueeze(0), gt3.unsqueeze(0)).item()

                    result["grad_coh_thirds_mean"] = (cos_12 + cos_13 + cos_23) / 3
                    result["grad_coh_thirds_min"] = min(cos_12, cos_13, cos_23)
                    result["grad_coh_adjacent"] = (cos_12 + cos_23) / 2
                    result["grad_coh_distant"] = cos_13

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[NOVEL05 WARN] {type(e).__name__}: {e}")
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
        scorer = GradCoherenceScorer(self.model, self.tokenizer, max_length=self.args.max_length)

        print(f"\n[NOVEL05] Extracting gradient coherence for {len(df)} samples...")
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[NOVEL05]"):
            rows.append(scorer.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Report ---
        print("\n" + "=" * 70)
        print("   NOVEL05: GradCoh-MIA RESULTS")
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
            tag = " <-- PRIMARY" if col == "grad_coherence" else ""
            print(f"  {d}{col:<35} AUC = {best:.4f}{tag}")

        if aucs:
            best_sig = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_sig[1][1]}{best_sig[0]} = {best_sig[1][0]:.4f}")
            print(f"  vs EXP50 memTrace RF:  0.6908")

            # M vs NM stats
            m = df[df["is_member"] == 1]
            nm = df[df["is_member"] == 0]
            for col in ["grad_coherence", "combined_coherence", "grad_coh_thirds_mean"]:
                if col not in df.columns:
                    continue
                mv, nmv = m[col].dropna(), nm[col].dropna()
                if len(mv) > 0 and len(nmv) > 0:
                    print(f"  {col:<30} M={mv.mean():.4f}  NM={nmv.mean():.4f}")

        print("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"NOVEL05_{timestamp}.parquet", index=False)
        print(f"\n[NOVEL05] Results saved.")


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

    print(f"[NOVEL05] GradCoh-MIA: Cross-Subsequence Gradient Coherence")
    print(f"  model: {Args.model_name}, sample={Args.sample_fraction*100:.0f}%")
    Experiment(Args).run()
