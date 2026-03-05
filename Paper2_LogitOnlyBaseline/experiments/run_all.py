#!/usr/bin/env python3
"""
ESP-Cal — Full Multi-Model Benchmark Evaluation
================================================
Runs all 4 benchmarks × all models (H100 80 GB).

Usage:
    python run_all.py                       # all benchmarks, all models
    python run_all.py --benchmark wikimia   # single benchmark
    python run_all.py --no-multi-model      # one model per benchmark (fast)
"""

import os
import sys
import argparse
import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub", "pyarrow"],
               capture_output=True)

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    token = UserSecretsClient().get_secret("posioned")
    login(token=token, add_to_git_credential=True)
    print("✓ HuggingFace authenticated (Kaggle)")
except Exception:
    print("○ Not on Kaggle or no HF secret — using local auth")

import gc
import pandas as pd
import torch
from core import Config, ESPCalExperiment


def main():
    parser = argparse.ArgumentParser(description="ESP-Cal Full Evaluation")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "poisoned_chalice", "wikimia", "mimir", "bookmia"])
    parser.add_argument("--no-multi-model", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    cfg = Config()
    cfg.split = args.split
    if args.no_multi_model:
        cfg.multi_model = False

    if os.path.exists("/kaggle/working"):
        cfg.output_dir = "/kaggle/working/results_espcal"
    elif args.output_dir:
        cfg.output_dir = args.output_dir

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("  ESP-Cal — Full Benchmark Evaluation")
    print(f"  Multi-model: {cfg.multi_model}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Split: {cfg.split}")
    print("=" * 60)

    exp = ESPCalExperiment(cfg)
    results = {}

    benchmarks = {
        "poisoned_chalice": exp.run_poisoned_chalice,
        "wikimia": exp.run_wikimia,
        "mimir": exp.run_mimir,
        "bookmia": exp.run_bookmia,
    }

    targets = benchmarks if args.benchmark == "all" else {args.benchmark: benchmarks[args.benchmark]}

    for name, run_fn in targets.items():
        try:
            results[name] = run_fn()
        except Exception as e:
            print(f"\n  ✗ {name} failed: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final summary ──
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY — ESP-Cal")
    print("═" * 60)

    summary_rows = []
    for bname, bres in results.items():
        if isinstance(bres, dict):
            for key, res in bres.items() if not isinstance(bres.get("results"), pd.DataFrame) else [(bname, bres)]:
                if isinstance(res, dict) and "results" in res and len(res["results"]) > 0:
                    best = res["results"].iloc[0]
                    print(f"  {key:45s}  {best['score']:25s}  AUC={best['auc']:.4f}")
                    summary_rows.append({
                        "benchmark_model": key,
                        "best_signal": best["score"],
                        "auroc": best["auc"],
                    })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        path = os.path.join(cfg.output_dir, "multimodel_summary.csv")
        df_summary.to_csv(path, index=False)
        print(f"\n  Summary CSV → {path}")

    print("\n  ✓ All done!")


if __name__ == "__main__":
    main()
