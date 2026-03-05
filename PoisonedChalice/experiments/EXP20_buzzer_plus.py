"""
EXPERIMENT 20: BUZZER+ (Multi-Signal + ICP + GradNorm)
Method: BUZZER-style calibrated signals plus ICP-MIA-SP and gradient norm.
Goal: Stronger CMI by combining calibrated logits + white-box signals.
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================================
# Kaggle & Environment Setup
# ============================================================================

def setup_environment():
    print("--- Environment Setup Starting ---")
    try:
        import transformers
        import datasets
        import sklearn
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

def load_model(model_path: str):
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
# BUZZER+ Attack
# ============================================================================

class BuzzerPlusAttack:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.k = args.mink_pp_k
        self.use_sliding_window = args.use_sliding_window
        self.window_size = args.window_size
        self.calibrate = args.use_calibration
        self.calibration_ratio = args.calibration_drop_ratio
        self.mask_ratios = args.mask_ratios
        self.probes_per_ratio = args.probes_per_ratio
        self.mask_token = args.mask_token
        self.separator = args.separator

    @property
    def name(self):
        return "buzzer_plus"

    def _perturb_text(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) < 4:
            return text
        drop_count = max(1, int(len(tokens) * self.calibration_ratio))
        drop_idx = set(random.sample(range(len(tokens)), k=min(drop_count, len(tokens))))
        kept = [tok for i, tok in enumerate(tokens) if i not in drop_idx]
        return " ".join(kept) if kept else text

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

    def _collect_token_stats(self, input_ids: torch.Tensor) -> Dict[str, List[float]]:
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            log_probs = log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

        mu = (probs * log_probs).sum(dim=-1)
        var = (probs * (log_probs ** 2)).sum(dim=-1) - (mu ** 2)
        sigma = torch.sqrt(torch.clamp(var, min=1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)

        token_log_probs = []
        token_z_scores = []
        token_entropy = []

        for i in range(input_ids.shape[1] - 1):
            token_id = input_ids[0, i + 1]
            token_lp = log_probs[0, i, token_id]
            z_score = (token_lp - mu[0, i]) / sigma[0, i]
            token_log_probs.append(token_lp.item())
            token_z_scores.append(z_score.item())
            token_entropy.append(entropy[0, i].item())

        return {
            "log_probs": token_log_probs,
            "z_scores": token_z_scores,
            "entropy": token_entropy,
        }

    def _compute_signals(self, text: str) -> Dict[str, float]:
        if not text:
            return {}

        if self.use_sliding_window:
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids[0]
            log_probs = []
            z_scores = []
            entropies = []

            for i in range(0, len(input_ids), self.window_size):
                chunk_ids = input_ids[i: i + self.window_size]
                if len(chunk_ids) < 2:
                    continue
                chunk_tensor = chunk_ids.unsqueeze(0).to(self.model.device)
                stats = self._collect_token_stats(chunk_tensor)
                log_probs.extend(stats["log_probs"])
                z_scores.extend(stats["z_scores"])
                entropies.extend(stats["entropy"])
        else:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)
            stats = self._collect_token_stats(inputs["input_ids"])
            log_probs = stats["log_probs"]
            z_scores = stats["z_scores"]
            entropies = stats["entropy"]

        if not log_probs:
            return {}

        mean_lp = float(np.mean(log_probs))
        std_lp = float(np.std(log_probs))
        surp = mean_lp - std_lp
        neg_entropy = -float(np.mean(entropies)) if entropies else np.nan

        sorted_z = np.sort(np.array(z_scores))
        k_len = max(1, int(len(sorted_z) * self.k))
        mink_pp = float(np.mean(sorted_z[:k_len]))

        return {
            "mean_lp": mean_lp,
            "surp": surp,
            "mink_pp": mink_pp,
            "neg_entropy": neg_entropy,
        }

    def _calibrate_signals(self, base: Dict[str, float], cal: Dict[str, float]) -> Dict[str, float]:
        calibrated = {}
        for key, value in base.items():
            cal_value = cal.get(key, np.nan)
            if np.isnan(value) or np.isnan(cal_value):
                calibrated[f"{key}_cal"] = np.nan
                calibrated[f"{key}_ratio"] = np.nan
            else:
                calibrated[f"{key}_cal"] = cal_value
                calibrated[f"{key}_ratio"] = value / (cal_value + 1e-8)
        return calibrated

    def _compute_icp_score(self, text: str) -> float:
        base_ll = self._log_likelihood(text)
        if np.isnan(base_ll):
            return np.nan

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

        return np.nan if min_icp is None else min_icp

    def _compute_grad_norm(self, text: str) -> float:
        if not text:
            return np.nan
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
                return np.nan
            return -embed_layer.weight.grad.norm(2).item()
        except Exception:
            return np.nan
        finally:
            self.model.zero_grad()

    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        rows = []
        for text in tqdm(texts, desc="BUZZER+ Signals"):
            base = self._compute_signals(text)
            if not base:
                rows.append({})
                continue

            row = dict(base)
            if self.calibrate:
                perturbed = self._perturb_text(text)
                cal = self._compute_signals(perturbed)
                row.update(self._calibrate_signals(base, cal))

            row["icp_score"] = self._compute_icp_score(text)
            row["grad_norm"] = self._compute_grad_norm(text)

            rows.append(row)

        return pd.DataFrame(rows)

    def train_and_score(self, feature_df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        feature_df = feature_df.copy()
        # Replace ±inf with NaN first, then fill NaN with column median
        # (median is more robust than mean when there are extreme outliers)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_df.columns:
            col_median = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(col_median if not np.isnan(col_median) else 0.0)
        # Clip any remaining extreme values to the 1st–99th percentile per column
        for col in feature_df.columns:
            lo, hi = feature_df[col].quantile(0.01), feature_df[col].quantile(0.99)
            feature_df[col] = feature_df[col].clip(lo, hi)

        features = feature_df.values
        if self.args.train_fraction <= 0:
            return feature_df.mean(axis=1).values

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=max(0.1, 1.0 - self.args.train_fraction),
            random_state=self.args.seed
        )
        train_idx, _ = next(splitter.split(features, labels))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_all = scaler.transform(features)

        clf = LogisticRegression(max_iter=200, class_weight="balanced")
        clf.fit(X_train, labels[train_idx])
        return clf.predict_proba(X_all)[:, 1]

# ============================================================================
# Experiment Runner
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir) / f"EXP20_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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

    def run(self):
        df = self.load_data()
        attacker = BuzzerPlusAttack(self.args, self.model, self.tokenizer)
        feature_df = attacker.extract_features(df["content"].tolist())
        df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

        scores = attacker.train_and_score(feature_df, df["is_member"].values)
        df["buzzer_plus_score"] = scores

        fname = "results.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"Saved to {self.output_dir / fname}")

        try:
            auc = roc_auc_score(df["is_member"], df["buzzer_plus_score"].fillna(0))
            print(f"AUC (buzzer_plus_score): {auc:.4f}")
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
        mink_pp_k = 0.2
        use_sliding_window = True
        window_size = 256
        use_calibration = True
        calibration_drop_ratio = 0.1
        train_fraction = 0.2
        mask_ratios = [0.2, 0.4, 0.6]
        probes_per_ratio = 2
        mask_token = "/*MASK*/"
        separator = "\n\n"

    print(f"[EXP20] Model: {Args.model_name}")
    Experiment(Args).run()
