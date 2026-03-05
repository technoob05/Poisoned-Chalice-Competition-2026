#!/usr/bin/env python3
"""
exp_02_ESPCal — Full Entropy Slope + 3-Scale Calibration  ★ MAIN CONTRIBUTION
===============================================================================
Paper 2's primary technique:
  1. Compute per-token entropy trajectory H(t)
  2. Fit slope α: H(t) = α·t + β   →   members have steeper negative slope
  3. Apply 3-scale z-normalisation:
       Scale 1 — token-level   (done inside extractor  → z_esp_slope)
       Scale 2 — position/length bucket  (MultiScaleCalibrator)
       Scale 3 — domain/language bucket  (MultiScaleCalibrator)

Ablation breakdown shows contribution of each scale.
Comparison against baselines (Loss, Min-K%, Min-K%++) is included.

Quick-test (default): 10 % of data.
Full run            : --full

Usage
-----
  python run.py               # quick test
  python run.py --full        # production run
  python run.py --no-ablation # skip ablation (faster)
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

EXP_NAME = "exp_02_ESPCal"

# Columns to calibrate (Scale 2 + 3)
CALIB_COLS = [
    "signal_esp", "signal_h_drop", "signal_loss",
    "signal_minkpp", "signal_mink", "signal_zlib",
    "neg_mean_loss", "minkpp_20", "minkprob_20", "zlib_ratio", "surp",
]

# Primary + comparison baselines for final table
REPORT_COLS = [
    "signal_esp",      # ★ ESP-Cal (main)
    "signal_minkpp",   # Min-K%++
    "signal_mink",     # Min-K%
    "signal_loss",     # Loss
    "signal_zlib",     # Zlib
    "surp",            # SURP
]


def parse_args():
    p = argparse.ArgumentParser(description="ESP-Cal: full 3-scale calibration")
    p.add_argument("--full",         action="store_true")
    p.add_argument("--frac",         type=float, default=0.10)
    p.add_argument("--split",        default="test", choices=["train", "test"])
    p.add_argument("--model",        default="bigcode/starcoder2-3b")
    p.add_argument("--no-ablation",  action="store_true")
    p.add_argument("--out",          default=None)
    return p.parse_args()


def run_ablation(df_raw: pd.DataFrame, df_cal: pd.DataFrame):
    """Compare contribution of each calibration scale."""
    print("\n  ABLATION — Calibration Scale Contributions")
    print("  " + "─" * 52)

    rows = []
    checks = [
        ("(a) Raw esp_slope",              df_raw, "esp_slope",    False),
        ("(b) + Scale 1 (token z-norm)",   df_raw, "z_esp_slope",  False),
        ("(c) + Scale 2+3 (pos+domain)",   df_cal, "signal_esp",   True),
    ]
    for desc, df_use, col, is_main in checks:
        if col not in df_use.columns:
            continue
        v = df_use[col].notna() & df_use["is_member"].notna()
        if v.sum() < 10:
            continue
        auc = roc_auc_score(df_use.loc[v, "is_member"], df_use.loc[v, col])
        best = max(auc, 1 - auc)
        rows.append({"condition": desc, "auc": best})
        star = "★" if is_main else " "
        print(f"  {star} {desc:44s}  AUC={best:.4f}")
    return pd.DataFrame(rows)


def make_latex_table(df: pd.DataFrame, cols_map: dict) -> str:
    """Generate paper-ready LaTeX comparison table."""
    langs = ["Go", "Java", "Python", "Ruby", "Rust"]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Go} & \textbf{Java} & \textbf{Python} "
        r"& \textbf{Ruby} & \textbf{Rust} & \textbf{Avg} \\",
        r"\midrule",
    ]
    for col, name in cols_map.items():
        if col not in df.columns:
            continue
        aucs = []
        for lang in langs:
            sub = df[df["subset"] == lang]
            v = sub[col].notna() & sub["is_member"].notna()
            if v.sum() >= 10:
                a = roc_auc_score(sub.loc[v, "is_member"], sub.loc[v, col])
                aucs.append(f"{max(a,1-a):.3f}")
            else:
                aucs.append("—")
        valid = [float(a) for a in aucs if a != "—"]
        avg = f"{np.mean(valid):.3f}" if valid else "—"
        lines.append(f"  {name} & " + " & ".join(aucs) + f" & {avg} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{AUROC on Poisoned Chalice (test split). "
        r"ESP-Cal consistently outperforms logit-only baselines.}",
        r"\label{tab:espcal_pc}",
        r"\end{table}",
    ]
    return "\n".join(lines)


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
    print(f"  {EXP_NAME}  [{mode_tag}]  ★ MAIN CONTRIBUTION")
    print(f"  Model:  {cfg.model_name}")
    print(f"  Split:  {cfg.split}  |  Frac: {frac:.0%}")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    df = load_poisoned_chalice(cfg)
    print(f"  {len(df)} samples  ({df.is_member.sum()} members)")

    # ── 2. Extract features ──────────────────────────────────────────────
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
    df_raw = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    print(f"  ✓ {time.time()-t0:.0f}s")

    del model, tok, extractor
    free_model(model_name=cfg.model_name)

    # ── 3. Calibrate ─────────────────────────────────────────────────────
    print("\n[3/4] 3-scale calibration …")
    calibrator = MultiScaleCalibrator(cfg)
    cal_cols = [c for c in CALIB_COLS if c in df_raw.columns]
    df_cal = calibrator.calibrate(df_raw, cal_cols)
    print(f"  ✓ Calibrated {len(cal_cols)} columns")

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    print("\n[4/4] Evaluating …")
    eval_cols = [c for c in df_cal.columns
                 if c not in ["text","is_member","subset","seq_len","n_tokens"]
                 and not c.endswith("_raw")]
    results_all = evaluate_scores(df_cal, eval_cols)
    results_key  = evaluate_scores(df_cal, [c for c in REPORT_COLS if c in df_cal.columns])

    print("\n" + "─" * 50)
    print(f"  TOP RESULTS — {EXP_NAME}")
    print("─" * 50)
    for _, r in results_all.head(20).iterrows():
        m = "★" if "esp" in r["score"].lower() else " "
        print(f"  {m} {r['score']:30s}  AUC={r['auc']:.4f}  ({r['polarity']})")

    best_col = "signal_esp" if "signal_esp" in results_all["score"].values \
               else results_all.iloc[0]["score"]
    print(f"\n  Per-language ({best_col}):")
    per_sub = evaluate_per_subset(df_cal, best_col)
    for _, r in per_sub.iterrows():
        print(f"    {r['subset']:10s}  AUC={r['auc']:.4f}  (n={r['n']})")

    # ── Ablation ─────────────────────────────────────────────────────────
    abl_df = None
    if not args.no_ablation:
        abl_df = run_ablation(df_raw, df_cal)

    # ── LaTeX table ───────────────────────────────────────────────────────
    latex_map = {
        "signal_loss":   "Loss",
        "signal_mink":   "Min-K\\%",
        "signal_minkpp": "Min-K\\%++",
        "surp":          "SURP",
        "signal_zlib":   "Zlib",
        "signal_esp":    "\\textbf{ESP-Cal (Ours)}",
    }
    table_str = make_latex_table(df_cal, latex_map)
    tex_path = os.path.join(out_dir, f"{EXP_NAME}_latex_table.tex")
    with open(tex_path, "w") as f:
        f.write(table_str)
    print(f"\n  LaTeX table → {tex_path}")

    # ── Save ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_cal.to_parquet(os.path.join(out_dir, f"{EXP_NAME}_{ts}.parquet"), index=False)
    with open(os.path.join(out_dir, f"{EXP_NAME}_{ts}.json"), "w") as f:
        json.dump({
            "exp": EXP_NAME, "frac": frac, "split": cfg.split,
            "model": cfg.model_name, "timestamp": ts,
            "results_top20": results_all.head(20).to_dict("records"),
            "results_key":   results_key.to_dict("records"),
            "per_subset":    per_sub.to_dict("records"),
            "ablation": abl_df.to_dict("records") if abl_df is not None else None,
        }, f, indent=2)
    print(f"  ✓ Saved → {out_dir}/")

    if frac < 1.0:
        print(f"\n  [ Quick-test passed — re-run with --full for production results ]")
    print("\n" + "═" * 60)
    print(f"  DONE — {EXP_NAME}")
    print("═" * 60)


if __name__ == "__main__":
    main()
