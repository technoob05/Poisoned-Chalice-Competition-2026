#!/usr/bin/env python3
"""
exp_03_SurpriseTraj — Surprise Trajectory Features
====================================================
Novel technique: instead of entropy slope, characterise HOW per-token
loss (surprise) evolves over the sequence.

Features
--------
  surprise_drop      : first-half mean loss − second-half mean loss
                       Members: model "recognises" text → loss drops
  surprise_accel     : quadratic curvature of loss trajectory
                       Members: concave loss curve (fast initial drop)
  neg_surprise_vol   : −std(Δloss)  — stability of consecutive changes
                       Members: more predictable ↔ lower volatility
  neg_loss_q_range   : −(q90 − q10)  — spread of token losses
                       Members: more uniform ↔ smaller range
  max_loss_drop      : largest single-token loss decrease
                       Members: stronger "recognition" bursts
  settle_frac        : fraction of second-half tokens below median loss
                       Members: loss "settles" low after recognition

Calibration: 3-scale (same as ESPCal) is applied to all features.

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
from core import (Config, load_model, free_model, ESPExtractor,
                  MultiScaleCalibrator,
                  load_poisoned_chalice, evaluate_scores, evaluate_per_subset)

EXP_NAME = "exp_03_SurpriseTraj"

SURPRISE_COLS = [
    "surprise_drop",
    "surprise_accel",
    "neg_surprise_vol",
    "neg_loss_q_range",
    "max_loss_drop",
    "settle_frac",
    "signal_surprise_drop",   # alias = surprise_drop
    # Include loss baseline for comparison in same run
    "signal_loss",
    "neg_mean_loss",
]


def parse_args():
    p = argparse.ArgumentParser(description="Surprise Trajectory features")
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

    print("\n[1/4] Loading data …")
    df = load_poisoned_chalice(cfg)
    print(f"  {len(df)} samples  ({df.is_member.sum()} members)")

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

    print("\n[3/4] Calibrating …")
    calib_cols = [c for c in SURPRISE_COLS if c in df.columns]
    calibrator = MultiScaleCalibrator(cfg)
    df = calibrator.calibrate(df, calib_cols)
    print(f"  ✓ Calibrated {len(calib_cols)} columns")

    print("\n[4/4] Evaluating …")
    eval_cols = [c for c in SURPRISE_COLS if c in df.columns]
    results = evaluate_scores(df, eval_cols)

    print("\n" + "─" * 50)
    print(f"  RESULTS — {EXP_NAME}")
    print("─" * 50)
    for _, r in results.iterrows():
        m = "★" if "surprise" in r["score"] or "settle" in r["score"] else " "
        print(f"  {m} {r['score']:28s}  AUC={r['auc']:.4f}  ({r['polarity']})")

    best_col = results.iloc[0]["score"] if len(results) > 0 else "surprise_drop"
    per_sub = evaluate_per_subset(df, best_col)
    print(f"\n  Per-language ({best_col}):")
    for _, r in per_sub.iterrows():
        print(f"    {r['subset']:10s}  AUC={r['auc']:.4f}  (n={r['n']})")

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
