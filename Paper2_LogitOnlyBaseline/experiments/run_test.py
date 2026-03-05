#!/usr/bin/env python3
"""
run_test.py — Quick-Test Orchestrator (10 % data, all experiments)
===================================================================
Runs every exp_XX_*/run.py with --frac 0.10 so you can validate each
technique quickly before committing to a full run.

Workflow
--------
  1. python run_test.py              → run ALL experiments at 10 %
  2. Inspect results/ — if AUC > 0.55 for at least one exp, proceed
  3. python run_test.py --full       → promote ALL to 100 %
  4. Or cherry-pick:  cd exp_02_ESPCal && python run.py --full

Usage
-----
  python run_test.py                 # all exps, 10%
  python run_test.py --exps 02 04    # only exp_02 and exp_04
  python run_test.py --full          # all exps, 100%
  python run_test.py --frac 0.3      # all exps, 30%
"""

import os
import sys
import subprocess
import time
import argparse

_EXPS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── All registered experiments (order matters: baseline first) ─────────────
EXPERIMENTS = [
    "exp_00_Baseline",
    "exp_01_ESP_NoCal",
    "exp_02_ESPCal",
    "exp_03_SurpriseTraj",
    "exp_04_Combined",
]


def parse_args():
    p = argparse.ArgumentParser(description="Quick-test all experiments")
    p.add_argument("--full",  action="store_true",
                   help="Run at 100%% (production mode)")
    p.add_argument("--frac",  type=float, default=0.10,
                   help="Data fraction (default 0.10 = 10%%)")
    p.add_argument("--exps",  nargs="*", default=None,
                   help="Filter by experiment number(s), e.g. --exps 02 04")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--model", default=None,
                   help="Override model for all experiments")
    return p.parse_args()


def run_experiment(exp_name: str, frac: float, split: str, model: str = None) -> bool:
    run_py = os.path.join(_EXPS_DIR, exp_name, "run.py")
    if not os.path.exists(run_py):
        print(f"  ✗ {exp_name}: run.py not found — skipping")
        return False

    cmd = [sys.executable, run_py, "--frac", str(frac), "--split", split]
    if frac >= 1.0:
        cmd = [sys.executable, run_py, "--full", "--split", split]
    if model:
        cmd += ["--model", model]

    print(f"\n{'▶'*3} {exp_name}  (frac={frac:.0%}  split={split})")
    print("  cmd: " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.join(_EXPS_DIR, exp_name))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✓ OK" if ok else f"✗ FAILED (code {result.returncode})"
    print(f"  [{status}]  {elapsed:.0f}s")
    return ok


def check_imports():
    """Verify core/ package is importable."""
    try:
        sys.path.insert(0, _EXPS_DIR)
        from core import Config, ESPExtractor, WIKIMIA_MODELS, MIMIR_MODELS
        print(f"  ✓ core package OK")
        print(f"    WikiMIA models registered: {len(WIKIMIA_MODELS)}")
        print(f"    MIMIR  models registered:  {len(MIMIR_MODELS)}")
        return True
    except Exception as e:
        print(f"  ✗ core import failed: {e}")
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {name}  ({vram:.0f} GB VRAM)")
        else:
            print("  ⚠ No CUDA GPU — will use CPU (much slower)")
    except Exception:
        print("  ⚠ Could not check GPU")


def main():
    args = parse_args()
    frac = 1.0 if args.full else args.frac

    # Filter experiments if --exps given
    exps = EXPERIMENTS
    if args.exps:
        exps = [e for e in EXPERIMENTS if any(e.startswith(f"exp_{n}") for n in args.exps)]
        if not exps:
            print(f"  ✗ No experiments matched: {args.exps}")
            sys.exit(1)

    mode_tag = "FULL (100%)" if frac >= 1.0 else f"QUICK-TEST ({frac:.0%})"
    print("=" * 60)
    print(f"  Paper 2 — Experiment Orchestrator  [{mode_tag}]")
    print(f"  Experiments : {len(exps)}")
    print(f"  Split       : {args.split}")
    if args.model:
        print(f"  Model override: {args.model}")
    print("=" * 60)

    # Pre-flight: check imports + GPU
    print("\n[pre-flight]")
    check_imports()
    check_gpu()

    t_total = time.time()
    passed, failed = [], []

    for exp in exps:
        ok = run_experiment(exp, frac, args.split, args.model)
        (passed if ok else failed).append(exp)

    elapsed = time.time() - t_total
    print("\n" + "═" * 60)
    print(f"  SUMMARY  ({elapsed:.0f}s total)")
    print("═" * 60)
    for e in passed:
        print(f"  ✓  {e}")
    for e in failed:
        print(f"  ✗  {e}  ← FAILED")

    if failed:
        print(f"\n  {len(failed)} experiment(s) failed.")
        sys.exit(1)
    else:
        if frac < 1.0:
            print(f"\n  All quick-tests passed!")
            print(f"  Next step: python run_test.py --full")
            print(f"  Or cherry-pick: cd exp_02_ESPCal && python run.py --full")
        else:
            print(f"\n  All production runs complete.")
            print(f"  Results in: Paper2_LogitOnlyBaseline/results/")


if __name__ == "__main__":
    main()

