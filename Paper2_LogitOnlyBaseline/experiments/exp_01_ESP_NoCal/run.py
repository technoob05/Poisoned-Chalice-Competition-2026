#!/usr/bin/env python3
"""
exp_01_ESP_NoCal — Raw Entropy Slope (no calibration)
======================================================
Validates the core ESP signal BEFORE applying any calibration.
Shows what the raw entropy slope gives us, and documents the
baseline that Scale-2/3 calibration builds on top of.

Ablation breakdown:
  (a) Raw esp_slope          — no normalisation at all
  (b) z_esp_slope            — Scale 1 (token z-norm, done inside extractor)
  (c) signal_esp (uncalib)   — just negated slope, no position/domain calib

Quick-test (default): 10 % of data.
Full run            : --full

Usage
-----
  python run.py               # quick test
  python run.py --full        # production run
"""

import os, sys, argparse, gc, json, time
from datetime import datetime

_EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
_EXPS_DIR = os.path.dirname(_EXP_DIR)
sys.path.insert(0, _EXPS_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from core import (Config, load_model, free_model, ESPExtractor,
                  load_poisoned_chalice, evaluate_scores, evaluate_per_subset)

EXP_NAME = "exp_01_ESP_NoCal"

ESP_NOCAL_COLS = [
    "esp_slope",       # raw slope (lower → more negative → member)
    "z_esp_slope",     # token z-norm (Scale 1 only)
    "signal_esp",      # negated raw slope (higher → more likely member)
    "h_drop",          # first-half minus second-half entropy
    "signal_h_drop",   # = h_drop
    "h_curvature",     # quadratic curvature of entropy trajectory
    "h_mean",
    "h_std",
]


def parse_args():
    p = argparse.ArgumentParser(description="Raw ESP slope — no calibration")
    p.add_argument("--full",  action="store_true")
    p.add_argument("--frac",  type=float, default=0.10)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model", default="bigcode/starcoder2-3b")
    p.add_argument("--out",   default=None)
    return p.parse_args()


def _ablation(df: pd.DataFrame):
    """Print per-column ablation table."""
    print("\n  ABLATION — Entropy Slope Variants (no calibration)")
    print("  " + "─" * 48)
    rows = []
    for col, desc in [
        ("esp_slope",    "(a) Raw ESP slope"),
        ("z_esp_slope",  "(b) + Scale 1 (token z-norm)"),
        ("signal_esp",   "(c) Negated slope (= -esp_slope)"),
        ("h_drop",       "(d) Entropy drop (first vs second half)"),
        ("h_curvature",  "(e) Quadratic curvature"),
    ]:
        if col not in df.columns:
            continue
        v = df[col].notna() & df["is_member"].notna()
        if v.sum() < 10:
            continue
        auc = roc_auc_score(df.loc[v, "is_member"], df.loc[v, col])
        best = max(auc, 1 - auc)
        rows.append({"col": col, "desc": desc, "auc": best})
        star = "★" if best == max(r["auc"] for r in rows) else " "
        print(f"  {star} {desc:40s}  AUC={best:.4f}")
    return pd.DataFrame(rows)


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

    print("\n[1/3] Loading data …")
    df = load_poisoned_chalice(cfg)
    print(f"  {len(df)} samples  ({df.is_member.sum()} members)")

    print("\n[2/3] Loading model & extracting features …")
    model, tok = load_model(cfg.model_name, cfg.torch_dtype)
    extractor = ESPExtractor(model, tok, cfg)

    t0 = time.time()
    feats = [extractor.extract(r["text"]) for _, r in df.iterrows()]
    feat_df = pd.DataFrame(feats)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    print(f"  ✓ {time.time()-t0:.0f}s")

    del model, tok, extractor
    free_model(model_name=cfg.model_name)

    print("\n[3/3] Evaluating …")
    eval_cols = [c for c in ESP_NOCAL_COLS if c in df.columns]
    results = evaluate_scores(df, eval_cols)

    print("\n" + "─" * 50)
    print(f"  RESULTS — {EXP_NAME}  (no calibration)")
    print("─" * 50)
    for _, r in results.iterrows():
        m = "★" if "esp" in r["score"].lower() else " "
        print(f"  {m} {r['score']:25s}  AUC={r['auc']:.4f}  ({r['polarity']})")

    # Per-language
    best_col = results.iloc[0]["score"] if len(results) > 0 else "signal_esp"
    per_sub = evaluate_per_subset(df, best_col)
    print(f"\n  Per-language ({best_col}):")
    for _, r in per_sub.iterrows():
        print(f"    {r['subset']:10s}  AUC={r['auc']:.4f}  (n={r['n']})")

    abl = _ablation(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(out_dir, f"{EXP_NAME}_{ts}.parquet"), index=False)
    with open(os.path.join(out_dir, f"{EXP_NAME}_{ts}.json"), "w") as f:
        json.dump({
            "exp": EXP_NAME, "frac": frac, "split": cfg.split,
            "model": cfg.model_name, "timestamp": ts,
            "results": results.to_dict("records"),
            "per_subset": per_sub.to_dict("records"),
            "ablation": abl.to_dict("records"),
        }, f, indent=2)
    print(f"\n  ✓ Saved → {out_dir}/")

    if frac < 1.0:
        print(f"\n  [ Quick-test passed — re-run with --full for production results ]")
    print("\n" + "═" * 60)
    print(f"  DONE — {EXP_NAME}")
    print("═" * 60)


if __name__ == "__main__":
    main()
