"""
EXPERIMENT 30: Privacy-Vulnerable Layer Gradient (Data-Driven Selective Gradient)

Paper inspiration:
    "Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights"
    (ICLR 2026 Oral) — Privacy leakage / memorization concentrates in a small subset
    of model weights, not spread uniformly.

Problem with EXP11 / EXP22:
    - EXP11: Gradient norm over the full embedding layer (noisy signal from all parameters).
    - EXP22: Hand-picked "strategic" layers (embedding, early, mid, late, head).
      Works well (AUC 0.6337) but layer selection is heuristic, not data-driven.

Innovation (this experiment):
    1. PROBE PHASE  — run a small balanced labeled set (~200 samples) through the model.
       For each sample: backward pass → record per-component gradient norms.
       Components: embedding layer + each transformer block (36×) + LM head = 38 total.
       Compute "Vulnerability Score" = mean_grad_member / (mean_grad_nonmember + ε).
       Select the top-K most discriminative components as the
       "Privacy-Vulnerable Component Mask" (PVC Mask).

    2. INFERENCE PHASE — for all remaining samples, run backward pass as usual,
       but compute the MIA score using gradient norms ONLY at PVC mask components.
       This eliminates the noise from irrelevant layers and sharpens the signal.

Expected benefit:
    By restricting to the ~5 most discriminative layers instead of averaging
    over all 38, we expect a meaningful AUC boost over EXP22 (0.6337).

Architecture notes:
    The probe accesses three component types per transformer layer:
        • self_attn   : Q/K/V/O projection weights combined (via named_parameters)
        • mlp         : feed-forward network weights combined
        • layernorms  : input_layernorm + post_attention_layernorm
    This makes the approach compatible with any standard CausalLM.

Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment():
    print("\n" + "="*65)
    print("  EXP30: PRIVACY-VULNERABLE LAYER GRADIENT (Data-Driven)")
    print("="*65)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Authenticated.")
    except Exception as e:
        print(f"[HF] Note: {e}")


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"[*] Model loaded. dtype={dtype}  device={model.device}")
    return model, tokenizer


# ============================================================================
# Component Gradient Extractor
# ============================================================================

class ComponentGradientExtractor:
    """
    Defines and extracts gradient norms for named model components.

    Components (for a model with N transformer layers):
        "embed"         : input embedding layer
        "layer_{i}_attn": self-attention block of layer i
        "layer_{i}_mlp" : feed-forward block of layer i
        "layer_{i}_norm": layer-normalisation weights of layer i
        "head"          : LM head (output projection)
    Total components: 1 + 3N + 1 = 3N + 2
    """

    def __init__(self, model):
        self.model = model
        self.components = self._build_component_map()
        print(f"[EXP30] Defined {len(self.components)} gradient components.")

    def _build_component_map(self) -> Dict[str, List[str]]:
        """
        Returns {component_name: [param_name_prefix, ...]}.
        We'll use startswith matching on param.name() to group parameters.
        """
        components = {}

        # Embedding
        components["embed"] = []
        for name, _ in self.model.named_parameters():
            if "embed_tokens" in name or (
                "wte" in name and "weight" in name
            ):
                components["embed"].append(name)

        # Transformer layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            num_layers = len(self.model.model.layers)
            for i in range(num_layers):
                prefix = f"model.layers.{i}."
                attn_names, mlp_names, norm_names = [], [], []
                for name, _ in self.model.named_parameters():
                    if not name.startswith(prefix):
                        continue
                    local = name[len(prefix):]
                    if local.startswith("self_attn") or local.startswith("attn"):
                        attn_names.append(name)
                    elif local.startswith("mlp"):
                        mlp_names.append(name)
                    elif "norm" in local or "ln" in local:
                        norm_names.append(name)
                if attn_names:
                    components[f"layer_{i}_attn"] = attn_names
                if mlp_names:
                    components[f"layer_{i}_mlp"] = mlp_names
                if norm_names:
                    components[f"layer_{i}_norm"] = norm_names

        # LM head
        components["head"] = []
        for name, _ in self.model.named_parameters():
            if "lm_head" in name:
                components["head"].append(name)

        return {k: v for k, v in components.items() if v}

    def compute_component_grad_norms(self, text: str, tokenizer, max_length: int) -> Dict[str, float]:
        """
        Forward + backward on text.
        Returns {component_name: rms_grad_norm} for all components.
        Empty dict on failure.
        """
        if not text or len(text) < 20:
            return {}
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.model.device)

            self.model.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()

            # Build name → param lookup once
            param_by_name = {n: p for n, p in self.model.named_parameters()}

            norms: Dict[str, float] = {}
            for comp_name, param_names in self.components.items():
                grad_vals = []
                for pname in param_names:
                    p = param_by_name.get(pname)
                    if p is not None and p.grad is not None:
                        grad_vals.append(p.grad.norm(2).item())
                if grad_vals:
                    norms[comp_name] = float(np.sqrt(np.mean(np.square(grad_vals))))
                else:
                    norms[comp_name] = np.nan

            self.model.zero_grad()
            return norms

        except Exception:
            self.model.zero_grad()
            return {}


# ============================================================================
# Privacy-Vulnerable Layer Gradient Attack
# ============================================================================

class PrivacyVulnerableGradientAttack:
    """
    Two-phase MIA attacker using data-driven gradient component selection.
    """

    def __init__(self, args, model, tokenizer):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = getattr(args, "max_length", 2048)
        self.probe_size = getattr(args, "probe_size", 200)     # balanced probe samples
        self.top_k_components = getattr(args, "top_k_components", 5)
        self.extractor = ComponentGradientExtractor(model)
        self.pvc_mask: List[str] = []  # Populated after probe

    @property
    def name(self) -> str:
        return "privacy_vulnerable_gradient"

    def _run_probe(self, df: pd.DataFrame) -> List[str]:
        """
        Run the probe phase on a small balanced labeled sample.
        Returns the list of top-K most discriminative component names.
        """
        n_each = self.probe_size // 2
        members = df[df["is_member"] == 1].sample(
            min(n_each, (df["is_member"] == 1).sum()), random_state=self.args.seed
        )
        nonmembers = df[df["is_member"] == 0].sample(
            min(n_each, (df["is_member"] == 0).sum()), random_state=self.args.seed
        )
        probe_df = pd.concat([members, nonmembers], ignore_index=True)

        print(
            f"\n[EXP30] PROBE PHASE: {len(probe_df)} samples "
            f"({len(members)} members + {len(nonmembers)} non-members)"
        )

        member_norms: Dict[str, List[float]] = {c: [] for c in self.extractor.components}
        nonmember_norms: Dict[str, List[float]] = {c: [] for c in self.extractor.components}

        for _, row in tqdm(probe_df.iterrows(), total=len(probe_df), desc="[PROBE] Gradient scan"):
            norms = self.extractor.compute_component_grad_norms(
                row["content"], self.tokenizer, self.max_length
            )
            target = member_norms if row["is_member"] == 1 else nonmember_norms
            for comp, val in norms.items():
                if not np.isnan(val):
                    target[comp].append(val)

        # Compute vulnerability scores
        vuln_scores: Dict[str, float] = {}
        for comp in self.extractor.components:
            m_vals = member_norms[comp]
            nm_vals = nonmember_norms[comp]
            if not m_vals or not nm_vals:
                vuln_scores[comp] = 0.0
                continue
            # Ratio: how much higher is the member gradient vs non-member?
            # (member gradient lower due to flat minima → ratio < 1 → vulnerability inverted)
            m_mean = np.mean(m_vals)
            nm_mean = np.mean(nm_vals)
            # |member_mean - nonmember_mean| / (member_mean + nonmember_mean + eps)
            # Normalized absolute difference — independent of direction
            vuln_scores[comp] = abs(m_mean - nm_mean) / (m_mean + nm_mean + 1e-9)

        # Rank by vulnerability
        sorted_comps = sorted(vuln_scores, key=vuln_scores.get, reverse=True)
        selected = sorted_comps[: self.top_k_components]

        print(f"\n[EXP30] Privacy-Vulnerable Components (top {self.top_k_components}):")
        for rank, comp in enumerate(selected, 1):
            m_mean = np.mean(member_norms[comp]) if member_norms[comp] else float("nan")
            nm_mean = np.mean(nonmember_norms[comp]) if nonmember_norms[comp] else float("nan")
            print(
                f"  #{rank:2d}  {comp:<30s}  vuln={vuln_scores[comp]:.4f}  "
                f"member={m_mean:.4f}  nonmember={nm_mean:.4f}"
            )

        return selected

    def compute_pvc_grad_score(self, text: str) -> Dict[str, float]:
        """
        Compute gradient norm restricted to the PVC mask.
        Score = -rms(pvc_grad_norms) → lower = more likely member = higher score.
        """
        if not self.pvc_mask:
            return {"pvc_score": np.nan, "pvc_grad_norm": np.nan}

        all_norms = self.extractor.compute_component_grad_norms(
            text, self.tokenizer, self.max_length
        )
        if not all_norms:
            return {"pvc_score": np.nan, "pvc_grad_norm": np.nan}

        pvc_vals = [all_norms[c] for c in self.pvc_mask if c in all_norms and not np.isnan(all_norms[c])]
        if not pvc_vals:
            return {"pvc_score": np.nan, "pvc_grad_norm": np.nan}

        pvc_rms = float(np.sqrt(np.mean(np.square(pvc_vals))))
        return {
            "pvc_grad_norm": pvc_rms,
            "pvc_score": -pvc_rms,       # lower norm = member = higher score
            **{f"pvc_{c}": all_norms.get(c, np.nan) for c in self.pvc_mask},
        }

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- Phase 1: Probe ----
        self.pvc_mask = self._run_probe(df)

        # ---- Phase 2: Inference ----
        print(f"\n[EXP30] INFERENCE PHASE: {len(df)} samples with PVC mask…")
        rows = []
        for text in tqdm(df["content"].tolist(), desc="[EXP30] PVC Gradient"):
            rows.append(self.compute_pvc_grad_score(text))

        return pd.DataFrame(rows)


# ============================================================================
# Experiment Runner
# ============================================================================

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
        print(f"[*] Loading data from {self.args.dataset}…")
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
            raise ValueError("No data loaded!")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
            print(f"[*] Sampled {len(df)} rows.")
        return df

    def run(self):
        df = self.load_data()
        attacker = PrivacyVulnerableGradientAttack(self.args, self.model, self.tokenizer)

        # Pass full df to compute_scores (probe uses the labels inside)
        scores_df = attacker.compute_scores(df)

        df = pd.concat(
            [df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"EXP30_{self.args.model_name.replace('/', '_')}_{timestamp}.parquet"
        df.to_parquet(self.output_dir / fname, index=False)
        print(f"\n[*] Results saved: {fname}")

        print("\n" + "="*65)
        print("   EXP30: PRIVACY-VULNERABLE GRADIENT — PERFORMANCE REPORT")
        print("="*65)

        report = {
            "experiment": "EXP30_privacy_vulnerable_gradient",
            "model": self.args.model_name,
            "timestamp": timestamp,
            "pvc_mask": attacker.pvc_mask,
            "top_k_components": self.args.top_k_components,
            "probe_size": self.args.probe_size,
            "aucs": {},
            "subset_aucs": {},
        }

        if "pvc_score" in df.columns:
            valid = df.dropna(subset=["pvc_score"])
            if len(valid["is_member"].unique()) > 1:
                overall_auc = roc_auc_score(valid["is_member"], valid["pvc_score"])
                report["aucs"]["pvc_score"] = float(overall_auc)
                print(f"OVERALL AUC (PVC Gradient Score): {overall_auc:.4f}")
                print(f"Compare — EXP11 Baseline (full embed grad): 0.6472")
                print(f"Compare — EXP22 Heuristic layers:           0.6337")

                print(f"\n{'Subset':<10} | {'AUC':<8} | {'N':<6} | "
                      f"{'PVC Norm (M)':<15} | {'PVC Norm (NM)'}")
                print("-" * 60)
                for subset in sorted(df["subset"].unique()):
                    sub = df[df["subset"] == subset].dropna(subset=["pvc_score"])
                    if len(sub["is_member"].unique()) > 1:
                        auc = roc_auc_score(sub["is_member"], sub["pvc_score"])
                        m_norm = sub[sub["is_member"] == 1]["pvc_grad_norm"].mean()
                        nm_norm = sub[sub["is_member"] == 0]["pvc_grad_norm"].mean()
                        print(
                            f"{subset:<10} | {auc:.4f}   | {len(sub):<6} | "
                            f"{m_norm:<15.4f} | {nm_norm:.4f}"
                        )
                        report["subset_aucs"][subset] = {"auc": float(auc)}

        print("="*65)
        print("\n[EXP30] Privacy-Vulnerable Components selected:")
        for c in attacker.pvc_mask:
            print(f"  → {c}")

        report_path = self.output_dir / f"EXP30_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\n[*] Report saved: {report_path.name}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        # Probe phase: 200 samples (fast, ~3 min on A100)
        # Inference phase: full sample_fraction of dataset
        sample_fraction = 0.05     # ~12 500 inference samples + 200 probe
        probe_size = 200           # balanced (100 members + 100 non-members)
        top_k_components = 5       # Select 5 most vulnerable components (out of ~110)
        output_dir = "results"
        max_length = 2048
        seed = 42

    print(f"[EXP30] Model           : {Args.model_name}")
    print(f"[EXP30] Sample          : {Args.sample_fraction*100:.0f}%")
    print(f"[EXP30] Probe size      : {Args.probe_size}")
    print(f"[EXP30] Top-K components: {Args.top_k_components}")
    Experiment(Args).run()
