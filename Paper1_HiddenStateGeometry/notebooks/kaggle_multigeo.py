#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║  MultiGeo-MIA — Kaggle Notebook                              ║
║  Clone repo → Smoke test → Full multi-model evaluation        ║
║  GPU: H100 80 GB                                              ║
╚═══════════════════════════════════════════════════════════════╝

Kaggle Dataset: minh2duy/poisoned-chalice-dataset
  /kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/
    ├── poisoned_chalice_dataset/
    ├── kaggle_wikimia/
    ├── kaggle_mimir/
    ├── kaggle_bookmia/
    ├── kaggle_xsum_mia/
    └── kaggle_agnews_mia/
"""

# ═══════════════════════════════════════════
# Cell 1: Setup & Clone Repo
# ═══════════════════════════════════════════

import os
import subprocess
import sys

# Clone the repo
REPO_URL = "https://github.com/technoob05/Poisoned-Chalice-Competition-2026.git"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"✓ Cloned to {REPO_DIR}")
else:
    print("Updating repository...")
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], capture_output=True)
    print(f"✓ Repo updated at {REPO_DIR}")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub", "pyarrow"],
               capture_output=True)

# HuggingFace auth — try HF_TOKEN first, then posioned
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    user_secrets = UserSecretsClient()
    token = None
    for secret_name in ["HF_TOKEN", "posioned"]:
        try:
            token = user_secrets.get_secret(secret_name)
            if token:
                print(f"✓ Found secret: {secret_name}")
                break
        except Exception:
            continue
    if token:
        login(token=token, add_to_git_credential=True)
        print("✓ HuggingFace authenticated")
    else:
        print("⚠ No HF token found in secrets")
except Exception as e:
    print(f"○ Kaggle secrets unavailable: {e}")

# Add experiment dir to path
EXP_DIR = os.path.join(REPO_DIR, "Paper1_HiddenStateGeometry", "experiments")
sys.path.insert(0, EXP_DIR)

print(f"✓ Added {EXP_DIR} to sys.path")

# ═══════════════════════════════════════════
# Cell 2: Verify GPU & Data Paths
# ═══════════════════════════════════════════

import torch

print("=" * 50)
print("  ENVIRONMENT CHECK")
print("=" * 50)

# GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name} ({vram:.0f} GB)")
else:
    print("  ⚠ No GPU!")

# Dataset paths
KAGGLE_ROOT = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
datasets = {
    "Poisoned Chalice": os.path.join(KAGGLE_ROOT, "poisoned_chalice_dataset"),
    "WikiMIA": os.path.join(KAGGLE_ROOT, "kaggle_wikimia"),
    "MIMIR": os.path.join(KAGGLE_ROOT, "kaggle_mimir"),
    "BookMIA": os.path.join(KAGGLE_ROOT, "kaggle_bookmia"),
    "XSum-MIA": os.path.join(KAGGLE_ROOT, "kaggle_xsum_mia"),
    "AGNews-MIA": os.path.join(KAGGLE_ROOT, "kaggle_agnews_mia"),
}

for name, path in datasets.items():
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists} {name}: {path}")

# ═══════════════════════════════════════════
# Cell 3: Import & Smoke Test
# ═══════════════════════════════════════════

print("\n" + "=" * 50)
print("  SMOKE TEST")
print("=" * 50)

from multigeo import (
    Config, MultiGeoExperiment,
    load_model, free_model,
    MultiGeoExtractor,
    WIKIMIA_MODELS, MIMIR_MODELS, BOOKMIA_MODELS,
)

print(f"  ✓ Imports OK")
print(f"  Models — WikiMIA: {len(WIKIMIA_MODELS)}, MIMIR: {len(MIMIR_MODELS)}, BookMIA: {len(BOOKMIA_MODELS)}")

# Test smallest model
print("\n  Testing pythia-160m-deduped...")
model, tokenizer, n_layers = load_model("EleutherAI/pythia-160m-deduped", "bfloat16")
cfg_test = Config()
extractor = MultiGeoExtractor(model, tokenizer, n_layers, cfg_test)

features = extractor.extract("def hello():\n    print('Hello, world!')\n")
print(f"  ✓ Extraction OK: {len(features)} features")
for k in ["signal_magnitude", "signal_dimensionality", "signal_dynamics", "signal_routing", "loss"]:
    print(f"    {k}: {features[k]:.4f}")

free_model(model, tokenizer, extractor, model_name="EleutherAI/pythia-160m-deduped")
print("\n  ✓ SMOKE TEST PASSED — ready for full evaluation")

# ═══════════════════════════════════════════
# Cell 4: Full Evaluation — All Benchmarks
# ═══════════════════════════════════════════

import gc

cfg = Config()
cfg.output_dir = "/kaggle/working/results_multigeo"
cfg.multi_model = True  # Run ALL models
# cfg.split defaults to "test" — only test set for Poisoned Chalice

os.makedirs(cfg.output_dir, exist_ok=True)

print("=" * 60)
print("  MultiGeo-MIA — FULL MULTI-MODEL EVALUATION")
print(f"  Output: {cfg.output_dir}")
print(f"  Multi-model: {cfg.multi_model}")
print("=" * 60)

exp = MultiGeoExperiment(cfg)

# ── 1. WikiMIA ──
print("\n\n" + "█" * 60)
print("  [1/4] WIKIMIA — 19 models × 4 lengths")
print("█" * 60)
wikimia_results = exp.run_wikimia()

gc.collect()
torch.cuda.empty_cache()

# ── 2. MIMIR ──
print("\n\n" + "█" * 60)
print("  [2/4] MIMIR — 10 models × 7 domains")
print("█" * 60)
mimir_results = exp.run_mimir()

gc.collect()
torch.cuda.empty_cache()

# ── 3. BookMIA ──
print("\n\n" + "█" * 60)
print("  [3/4] BOOKMIA — 7 models")
print("█" * 60)
bookmia_results = exp.run_bookmia()

gc.collect()
torch.cuda.empty_cache()

# ── 4. Poisoned Chalice (Competition — TEST only) ──
print("\n\n" + "█" * 60)
print("  [4/4] POISONED CHALICE (test split)")
print("█" * 60)
pc_results = exp.run_poisoned_chalice()

# ═══════════════════════════════════════════
# Cell 5: Summary & Export
# ═══════════════════════════════════════════

import pandas as pd

print("\n" + "═" * 60)
print("  FINAL SUMMARY — MultiGeo-MIA")
print("═" * 60)

summary_rows = []
all_results = {}
if pc_results and "results" in pc_results:
    all_results["PoisonedChalice"] = pc_results
for d in [wikimia_results, mimir_results, bookmia_results]:
    if isinstance(d, dict):
        all_results.update(d)

for key, res in all_results.items():
    if isinstance(res, dict) and "results" in res and len(res["results"]) > 0:
        best = res["results"].iloc[0]
        print(f"  {key:45s}  {best['score']:25s}  AUC={best['auc']:.4f}")
        summary_rows.append({
            "benchmark_model": key,
            "best_signal": best["score"],
            "auroc": float(best["auc"]),
        })

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    path = os.path.join(cfg.output_dir, "multimodel_summary.csv")
    df_summary.to_csv(path, index=False)
    print(f"\n  Summary → {path}")
    print(f"\n{df_summary.to_string(index=False)}")

# List all output files
print("\n  Output files:")
for f in sorted(os.listdir(cfg.output_dir)):
    size = os.path.getsize(os.path.join(cfg.output_dir, f)) / 1024
    print(f"    {f} ({size:.0f} KB)")

print("\n  ✓ ALL DONE!")
