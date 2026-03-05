#!/usr/bin/env python3
"""
exp_00_Baseline — Standard Logit-Only Baselines
=================================================
Methods evaluated (no ESP, pure loss-based):
  • Loss          : -mean(token_loss)
  • Min-K%        : mean of bottom-20% token log-probs  (Shi et al., 2024)
  • Min-K%++      : z-score normalised Min-K%           (Zhang et al., ICLR 2025)
  • SURP          : -(mean_loss - std_loss)
  • Zlib          : -mean_loss / zlib_len

Quick-test (default): 10 % of each language × split.
Full run            : pass --full (or --frac 1.0).

Usage
-----
  python run.py                # quick test, 10 %
  python run.py --full         # 100 %
  python run.py --frac 0.3     # 30 %
  python run.py --split train  # use train split
"""

import os, sys, argparse, gc, json, time
from datetime import datetime

# ── resolve repo root so 'core' package is importable ──────────────────────
_EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
_EXPS_DIR = os.path.dirname(_EXP_DIR)
sys.path.insert(0, _EXPS_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from core import (Config, load_model, free_model, ESPExtractor,
                  load_poisoned_chalice, evaluate_scores, evaluate_per_subset)

# ── Experiment name (used for output filenames) ─────────────────────────────
EXP_NAME = "exp_00_Baseline"

# ── Baseline signal columns produced by ESPExtractor ───────────────────────
BASELINE_COLS = [
    "neg_mean_loss",   # Loss
    "minkprob_20",     # Min-K%  (Shi et al., 2024)
    "minkpp_20",       # Min-K%++ (Zhang et al., ICLR 2025)
    "minkpp_10",
    "minkpp_50",
    "surp",            # SURP
    "zlib_ratio",      # Zlib
    "neg_mean_rank",   # Rank
    "signal_minkpp",   # = minkpp_20 (pre-computed alias)
    "signal_mink",     # = minkprob_20
    "signal_loss",     # = neg_mean_loss
    "signal_zlib",     # = zlib_ratio
]

# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline MIA methods (no ESP)")
    p.add_argument("--full",  action="store_true",  help="Run on 100 %% of data")
    p.add_argument("--frac",  type=float, default=0.10,
                   help="Data fraction for quick test (default: 0.10)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model", default="bigcode/starcoder2-3b")
    p.add_argument("--out",   default=None, help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    frac = 1.0 if args.full else args.frac

    cfg = Config()
    cfg.model_name      = args.model
    cfg.split           = args.split
    cfg.sample_fraction = frac
    cfg.multi_model     = False

    out_dir = args.out or os.path.join(_EXPS_DIR, "..", "results", EXP_NAME)
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg.output_dir = out_dir

    mode_tag = "FULL" if frac >= 1.0 else f"QUICK-TEST {frac:.0%}"
    print("=" * 60)
    print(f"  {EXP_NAME}  [{mode_tag}]")
    print(f"  Model:  {cfg.model_name}")
    print(f"  Split:  {cfg.split}  |  Frac: {frac:.0%}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/3] Loading data …")
    df = load_poisoned_chalice(cfg)
    print(f"  {len(df)} samples  ({df.is_member.sum()} members)")

    # ── Load model & extract features ─────────────────────────────────────
    print("\n[2/3] Loading model & extracting features …")
    model, tok = load_model(cfg.model_name, cfg.torch_dtype)
    extractor = ESPExtractor(model, tok, cfg)

    t0 = time.time()
    feats = []
    for i, row in df.iterrows():
        if i > 0 and i % 200 == 0:
            rate = i / (time.time() - t0)
            print(f"  [{i}/{len(df)}]  {rate:.1f} it/s  ETA {(len(df)-i)/rate:.0f}s")
        feats.append(extractor.extract(row["text"]))

    feat_df = pd.DataFrame(feats)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    print(f"  ✓ Extraction done in {time.time()-t0:.0f}s")

    del model, tok, extractor
    free_model(model_name=cfg.model_name)

    # ── Evaluate ────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating …")
    eval_cols = [c for c in BASELINE_COLS if c in df.columns]
    results = evaluate_scores(df, eval_cols)

    print("\n" + "─" * 50)
    print(f"  RESULTS — {EXP_NAME}  (all languages combined)")
    print("─" * 50)
    for _, r in results.iterrows():
        print(f"  {r['score']:25s}  AUC={r['auc']:.4f}  ({r['polarity']})")

    # Per-language breakdown
    best_col = results.iloc[0]["score"] if len(results) > 0 else "signal_minkpp"
    print(f"\n  Per-language ({best_col}):")
    per_sub = evaluate_per_subset(df, best_col)
    for _, r in per_sub.iterrows():
        print(f"    {r['subset']:10s}  AUC={r['auc']:.4f}  (n={r['n']})")

    # ── Save ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(out_dir, f"{EXP_NAME}_{ts}.parquet"), index=False)
    with open(os.path.join(out_dir, f"{EXP_NAME}_{ts}.json"), "w") as f:
        json.dump({
            "exp": EXP_NAME, "frac": frac, "split": cfg.split,
            "model": cfg.model_name, "timestamp": ts,
            "results": results.to_dict("records"),
            "per_subset": per_sub.to_dict("records"),
        }, f, indent=2)
    print(f"\n  ✓ Saved → {out_dir}/")

    if frac < 1.0:
        print(f"\n  [ Quick-test passed — re-run with --full for production results ]")

    print("\n" + "═" * 60)
    print(f"  DONE — {EXP_NAME}")
    print("═" * 60)


if __name__ == "__main__":
    main()
