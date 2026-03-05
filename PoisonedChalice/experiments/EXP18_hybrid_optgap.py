"""
EXPERIMENT 18: Hybrid Optimization Gap Ensemble
Method: Combine ICP-MIA-SP score with Gradient Norm and SURP.
Goal: Fuse optimization-gap proxy + white-box signal for stronger AUC.
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

# ============================================================================
# Kaggle & Environment Setup
# ============================================================================

def setup_environment():
    print("--- Environment Setup Starting ---")
    try:
        import transformers
        import datasets
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub")

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face.")
    except Exception as e:
        print(f"Login Note: {e}")
    print("--- Environment Setup Complete ---")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model, tokenizer

# ============================================================================
# Hybrid Attack Components
# ============================================================================

class ICPMIAAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.mask_ratios = args.mask_ratios
        self.probes_per_ratio = args.probes_per_ratio
        self.mask_token = args.mask_token
        self.separator = args.separator

    @property
    def name(self):
        return "icp_sp"

    def _random_mask(self, text: str, ratio: float) -> str:
        tokens = text.split()
        if len(tokens) < 4:
            return text
        num_to_mask = max(1, int(len(tokens) * ratio))
        indices = random.sample(range(len(tokens)), k=min(num_to_mask, len(tokens)))
        for idx in indices:
            tokens[idx] = self.mask_token
        return " ".join(tokens)

    def _build_input_ids(self, prefix: str, target: str):
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids:
            return None, 0, []

        total_len = len(prefix_ids) + len(target_ids)
        if total_len > self.max_length:
            if len(target_ids) >= self.max_length:
                target_ids = target_ids[-self.max_length:]
                prefix_ids = []
            else:
                keep_prefix = self.max_length - len(target_ids)
                prefix_ids = prefix_ids[-keep_prefix:]

        input_ids = prefix_ids + target_ids
        prefix_len = len(prefix_ids)
        return input_ids, prefix_len, target_ids

    def _log_likelihood(self, text: str, prefix: str = "") -> float:
        input_ids, prefix_len, target_ids = self._build_input_ids(prefix, text)
        if not target_ids:
            return np.nan

        input_tensor = torch.tensor([input_ids], device=self.model.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            log_probs = log_softmax(outputs.logits, dim=-1)

        total_ll = 0.0
        for t_idx, token_id in enumerate(target_ids):
            pos = prefix_len + t_idx
            if pos == 0:
                continue
            total_ll += log_probs[0, pos - 1, token_id].item()
        return total_ll

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="ICP-MIA-SP"):
            if not text:
                scores.append(np.nan)
                continue

            base_ll = self._log_likelihood(text)
            if np.isnan(base_ll):
                scores.append(np.nan)
                continue

            min_icp = None
            for ratio in self.mask_ratios:
                for _ in range(self.probes_per_ratio):
                    masked = self._random_mask(text, ratio)
                    probe_prefix = masked + self.separator
                    probe_ll = self._log_likelihood(text, prefix=probe_prefix)
                    if np.isnan(probe_ll):
                        continue
                    icp_score = base_ll - probe_ll
                    if (min_icp is None) or (icp_score < min_icp):
                        min_icp = icp_score

            if min_icp is None:
                scores.append(np.nan)
            else:
                scores.append(min_icp)
        return scores

class GradientNormAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "grad_norm"

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="Gradient Norm"):
            if not text:
                scores.append(np.nan)
                continue
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)
                self.model.zero_grad()
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()

                embed_layer = self.model.get_input_embeddings()
                if embed_layer.weight.grad is None:
                    scores.append(np.nan)
                else:
                    norm = embed_layer.weight.grad.norm(2).item()
                    scores.append(-norm)
            except Exception:
                scores.append(np.nan)
            finally:
                self.model.zero_grad()
        return scores

class SURPAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    @property
    def name(self):
        return "surp"

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="SURP"):
            if not text:
                scores.append(np.nan)
                continue
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    log_probs = log_softmax(logits, dim=-1)

                target_probs = []
                for i in range(inputs["input_ids"].shape[1] - 1):
                    token_id = inputs["input_ids"][0, i + 1]
                    target_probs.append(log_probs[0, i, token_id].item())

                if not target_probs:
                    scores.append(np.nan)
                    continue

                mean_lp = float(np.mean(target_probs))
                std_lp = float(np.std(target_probs))
                scores.append(mean_lp - std_lp)
            except Exception:
                scores.append(np.nan)
        return scores

class MinKPlusPlusAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.k = args.mink_pp_k
        self.use_sliding_window = args.mink_pp_sliding_window
        self.window_size = args.mink_pp_window_size

    @property
    def name(self):
        return "mink_pp"

    def _token_z_scores_truncation(self, text: str) -> np.ndarray:
        if not text:
            return np.array([])

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            log_probs = log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            mu = (probs * log_probs).sum(dim=-1)
            var = (probs * (log_probs ** 2)).sum(dim=-1) - (mu ** 2)
            sigma = torch.sqrt(torch.clamp(var, min=1e-12))

            target_z_scores = []
            for i in range(inputs["input_ids"].shape[1] - 1):
                token_id = inputs["input_ids"][0, i + 1]
                token_lp = log_probs[0, i, token_id]
                z_score = (token_lp - mu[0, i]) / sigma[0, i]
                target_z_scores.append(z_score.item())

        return np.array(target_z_scores)

    def _token_z_scores_sliding(self, text: str) -> np.ndarray:
        if not text:
            return np.array([])

        encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encodings.input_ids[0]
        all_z_scores = []

        for i in range(0, len(input_ids), self.window_size):
            chunk_ids = input_ids[i: i + self.window_size]
            if len(chunk_ids) < 2:
                continue

            chunk_tensor = chunk_ids.unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(chunk_tensor)
                logits = outputs.logits
                log_probs = log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                mu = (probs * log_probs).sum(dim=-1)
                var = (probs * (log_probs ** 2)).sum(dim=-1) - (mu ** 2)
                sigma = torch.sqrt(torch.clamp(var, min=1e-12))

                for j in range(chunk_ids.shape[0] - 1):
                    token_id = chunk_ids[j + 1]
                    token_lp = log_probs[0, j, token_id]
                    z_score = (token_lp - mu[0, j]) / sigma[0, j]
                    all_z_scores.append(z_score.item())

        return np.array(all_z_scores)

    def compute_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for text in tqdm(texts, desc="Min-K%++"):
            try:
                if self.use_sliding_window:
                    z_scores = self._token_z_scores_sliding(text)
                else:
                    z_scores = self._token_z_scores_truncation(text)
                if len(z_scores) == 0:
                    scores.append(np.nan)
                    continue

                sorted_z = np.sort(z_scores)
                k_len = max(1, int(len(sorted_z) * self.k))
                min_k_z = sorted_z[:k_len]
                scores.append(float(np.mean(min_k_z)))
            except Exception:
                scores.append(np.nan)
        return scores

# ============================================================================
# Experiment Runner
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir) / f"EXP18_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self):
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"Loading data from {self.args.dataset}...")
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
            raise ValueError("No data found!")

        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)

        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"Sampled {len(df)} rows.")
        return df

    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        return series.rank(pct=True)

    def run(self):
        df = self.load_data()

        icp = ICPMIAAttack(self.args, self.model, self.tokenizer)
        grad = GradientNormAttack(self.args, self.model, self.tokenizer)
        surp = SURPAttack(self.args, self.model, self.tokenizer)
        mink_pp = MinKPlusPlusAttack(self.args, self.model, self.tokenizer)

        df[f"{icp.name}_score"] = icp.compute_scores(df["content"].tolist())
        df[f"{grad.name}_score"] = grad.compute_scores(df["content"].tolist())
        df[f"{surp.name}_score"] = surp.compute_scores(df["content"].tolist())
        df[f"{mink_pp.name}_score"] = mink_pp.compute_scores(df["content"].tolist())

        score_cols = [
            f"{icp.name}_score",
            f"{grad.name}_score",
            f"{surp.name}_score",
            f"{mink_pp.name}_score",
        ]
        for col in score_cols:
            df[col] = df[col].fillna(df[col].mean())

        df["icp_rank"] = self._rank_normalize(df[f"{icp.name}_score"])
        df["grad_rank"] = self._rank_normalize(df[f"{grad.name}_score"])
        df["surp_rank"] = self._rank_normalize(df[f"{surp.name}_score"])
        df["mink_pp_rank"] = self._rank_normalize(df[f"{mink_pp.name}_score"])

        weights = {
            "icp_rank": 2.0,
            "mink_pp_rank": 1.6,
            "grad_rank": 1.5,
            "surp_rank": 1.0,
        }

        total_w = sum(weights.values())
        df["hybrid_score"] = 0.0
        for key, w in weights.items():
            df["hybrid_score"] += df[key] * w
        df["hybrid_score"] /= total_w

        fname = "results.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {self.output_dir / fname}")

        try:
            for col in score_cols + ["hybrid_score"]:
                auc = roc_auc_score(df["is_member"], df[col].fillna(0))
                print(f"AUC ({col}): {auc:.4f}")
        except Exception as e:
            print(f"AUC Error: {e}")

if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.1
        output_dir = "results"
        max_length = 2048
        seed = 42
        mask_ratios = [0.2, 0.4, 0.6]
        probes_per_ratio = 3
        mask_token = "/*MASK*/"
        separator = "\n\n"
        mink_pp_k = 0.2
        mink_pp_sliding_window = True
        mink_pp_window_size = 256

    print(f"[EXP18] Model: {Args.model_name}")
    Experiment(Args).run()
