#!/usr/bin/env python3
"""
exp_04_Combined — All Features + Calibration + Best Ensemble
=============================================================
Combines every signal from ESPExtractor (ESP, surprise trajectory,
loss, Min-K%++) under unified 3-scale calibration, then ranks all
columns by AUROC to find the best individual signal and the best
feature ensemble.

Key questions answered:
  1. What is the single best logit-only feature for Poisoned Chalice?
  2. Does combining ESP + SurpriseTraj beat either alone?
  3. Does the ensemble beat Min-K%++ (ICLR 2025 SOTA)?

Quick-test (default): 10 % of data.
Full run            : --full

Usage
-----
  python run.py               # quick test
  python run.py --full        # production run
"""

import os, sys, argparse, json, time
from datetime import datetime

_EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
_EXPS_DIR = os.path.dirname(_EXP_DIR)
sys.path.insert(0, _EXPS_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from core import (Config, load_model, free_model, ESPExtractor,
                  MultiScaleCalibrator,
                  load_poisoned_chalice, evaluate_scores, evaluate_per_subset)

EXP_NAME = "exp_04_Combined"

# All individual signals (pre-calibration)
ALL_SIGNAL_COLS = [
    # ESP family
    "signal_esp", "signal_h_drop", "z_esp_slope",
    "h_drop", "h_curvature",
    # Surprise trajectory family
    "surprise_drop", "surprise_accel", "neg_surprise_vol",
    "neg_loss_q_range", "max_loss_drop", "settle_frac",
    "signal_surprise_drop",
    # Loss family
    "signal_loss", "neg_mean_loss", "loss_slope",
    # Min-K% family
    "signal_mink", "minkprob_20",
    # Min-K%++ family
    "signal_minkpp", "minkpp_20", "minkpp_10", "minkpp_50",
    # SURP, Zlib
    "surp", "signal_zlib", "zlib_ratio",
]

# Ensemble groups (for combined score experiments)
ENSEMBLE_GROUPS = {
    "esp_traj":        ["signal_esp", "signal_surprise_drop"],
    "esp_minkpp":      ["signal_esp", "signal_minkpp"],
    "traj_minkpp":     ["signal_surprise_drop", "signal_minkpp"],
    "esp_traj_minkpp": ["signal_esp", "signal_surprise_drop", "signal_minkpp"],
    "all_main":        ["signal_esp", "signal_surprise_drop", "signal_minkpp",
                        "signal_loss", "signal_zlib"],
}


def ensemble_score(df: pd.DataFrame, cols: list) -> pd.Series:
    """Simple mean ensemble (all cols already z-normalised)."""
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return pd.Series(np.nan, index=df.index)
    return df[valid].mean(axis=1)


def parse_args():
    p = argparse.ArgumentParser(description="Combined: all features + ensembles")
    p.add_argument("--full",  action="store_true")
    p.add_argument("--frac",  type=float, default=0.10)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model", default="bigcode/starcoder2-3b")
    p.add_argument("--out",   default=None)
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
    print("=" * 60)

    # ── 1. Load ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    df = load_poisoned_chalice(cfg)
    print(f"  {len(df)} samples  ({df.is_member.sum()} members)")

    # ── 2. Extract ───────────────────────────────────────────────────────
    print("\n[2/4] Extracting features …")
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
    print(f"  ✓ {time.time()-t0:.0f}s")

    del model, tok, extractor
    free_model(model_name=cfg.model_name)

    # ── 3. Calibrate all signals ─────────────────────────────────────────
    print("\n[3/4] 3-scale calibration …")
    calib_cols = [c for c in ALL_SIGNAL_COLS if c in df.columns]
    calibrator = MultiScaleCalibrator(cfg)
    df = calibrator.calibrate(df, calib_cols)
    print(f"  ✓ Calibrated {len(calib_cols)} columns")

    # Build ensemble scores (after calibration, all on same z-scale)
    for name, members in ENSEMBLE_GROUPS.items():
        df[f"ens_{name}"] = ensemble_score(df, members)
    print(f"  ✓ Built {len(ENSEMBLE_GROUPS)} ensemble scores")

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    print("\n[4/4] Evaluating …")
    all_eval_cols = [c for c in df.columns
                     if c not in ["text","is_member","subset","seq_len","n_tokens"]
                     and not c.endswith("_raw")]
    results_all = evaluate_scores(df, all_eval_cols)

    # Print top 25
    print("\n" + "─" * 55)
    print(f"  TOP RESULTS — {EXP_NAME} (all features + ensembles)")
    print("─" * 55)
    for _, r in results_all.head(25).iterrows():
        tag = "ENS" if r["score"].startswith("ens_") else "   "
        m   = "★" if r["score"] in ("signal_esp", "signal_minkpp") or r["score"].startswith("ens_") else " "
        print(f"  {m} [{tag}] {r['score']:35s}  AUC={r['auc']:.4f}")

    # Best single feature
    best_col = results_all.iloc[0]["score"] if len(results_all) > 0 else "signal_esp"
    per_sub = evaluate_per_subset(df, best_col)
    print(f"\n  Per-language  best={best_col}:")
    for _, r in per_sub.iterrows():
        print(f"    {r['subset']:10s}  AUC={r['auc']:.4f}  (n={r['n']})")

    # Ensemble summary table
    ens_cols = [f"ens_{k}" for k in ENSEMBLE_GROUPS]
    results_ens = evaluate_scores(df, [c for c in ens_cols if c in df.columns])
    print("\n  ENSEMBLE SCORES:")
    for _, r in results_ens.iterrows():
        print(f"    {r['score']:35s}  AUC={r['auc']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Drop text to keep parquet small
    df_save = df.drop(columns=["text"], errors="ignore")
    df_save.to_parquet(os.path.join(out_dir, f"{EXP_NAME}_{ts}.parquet"), index=False)
    with open(os.path.join(out_dir, f"{EXP_NAME}_{ts}.json"), "w") as f:
        json.dump({
            "exp": EXP_NAME, "frac": frac, "split": cfg.split,
            "model": cfg.model_name, "timestamp": ts,
            "results_top25": results_all.head(25).to_dict("records"),
            "results_ensembles": results_ens.to_dict("records"),
            "per_subset_best": per_sub.to_dict("records"),
            "best_feature": best_col,
        }, f, indent=2)
    print(f"\n  ✓ Saved → {out_dir}/")

    if frac < 1.0:
        print(f"\n  [ Quick-test passed — re-run with --full for production results ]")
    print("\n" + "═" * 60)
    print(f"  DONE — {EXP_NAME}")
    print("═" * 60)


if __name__ == "__main__":
    main()
