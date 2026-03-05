#!/usr/bin/env python3
"""
package_for_kaggle.py — Package core/ + experiments for Kaggle upload
======================================================================
Creates two artefacts:

  1. kaggle_upload/espcal_core.zip
       → Upload this as a Kaggle Dataset (once; re-upload when core/ changes)
       Dataset slug: paper2-espcal-core

  2. kaggle_upload/kaggle_exp_XX.py  (one per experiment)
       → Paste as a Kaggle notebook cell, or use as a Kaggle Script notebook

Usage
-----
  python package_for_kaggle.py          # build all
  python package_for_kaggle.py --upload # build + auto-upload via kaggle CLI
"""

import os, sys, shutil, zipfile, argparse, subprocess, textwrap

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, "kaggle_upload")

CORE_DIR    = os.path.join(_SCRIPT_DIR, "core")
EXP_DIRS = {
    "exp_00_Baseline":    "00",
    "exp_01_ESP_NoCal":   "01",
    "exp_02_ESPCal":      "02",
    "exp_03_SurpriseTraj":"03",
    "exp_04_Combined":    "04",
}

# ── Kaggle Dataset metadata template ─────────────────────────────────────────
DATASET_META = """{
  "title": "paper2-espcal-core",
  "id": "YOUR_USERNAME/paper2-espcal-core",
  "licenses": [{"name": "CC0-1.0"}]
}
"""

# ── Kaggle notebook cell preamble (common to all experiments) ─────────────────
def _preamble(exp_name: str, frac: float = 0.10) -> str:
    return textwrap.dedent(f"""\
    #!/usr/bin/env python3
    # =============================================================
    # Kaggle Script — Paper 2: {exp_name}
    # =============================================================
    # Requirements:
    #   Dataset attached: YOUR_USERNAME/paper2-espcal-core
    #   (contains the core/ package)
    # GPU: T4 x2 or P100
    # =============================================================

    import os, sys, subprocess

    # ── Install dependencies ──────────────────────────────────────
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.40", "accelerate", "datasets",
                    "scikit-learn", "scipy", "huggingface_hub>=0.23"],
                   capture_output=True)

    # ── HuggingFace auth ─────────────────────────────────────────
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        for name in ["HF_TOKEN", "posioned"]:
            try:
                tok = UserSecretsClient().get_secret(name)
                if tok:
                    login(token=tok, add_to_git_credential=True)
                    print(f"✓ HF auth via secret: {{name}}")
                    break
            except Exception:
                pass
    except Exception:
        pass

    # ── Add core package to path ──────────────────────────────────
    # 'paper2-espcal-core' dataset is attached as input
    _CORE_ROOTS = [
        "/kaggle/input/paper2-espcal-core",
        "/kaggle/input/paper2espcalcore",          # Kaggle strips hyphens
    ]
    for _r in _CORE_ROOTS:
        if os.path.isdir(os.path.join(_r, "core")):
            sys.path.insert(0, _r)
            print(f"✓ core package found at: {{_r}}")
            break
    else:
        raise RuntimeError("core/ not found in attached dataset. "
                           "Attach 'paper2-espcal-core' dataset.")

    from core import (Config, load_model, free_model, ESPExtractor,
                      MultiScaleCalibrator,
                      load_poisoned_chalice, evaluate_scores, evaluate_per_subset)
    print("✓ core imports OK")

    # ── Configuration ─────────────────────────────────────────────
    FRAC  = {frac}         # set to 1.0 for production run
    SPLIT = "test"
    MODEL = "bigcode/starcoder2-3b"
    OUT   = "/kaggle/working/results/{exp_name}"
    os.makedirs(OUT, exist_ok=True)

    cfg = Config()
    cfg.model_name      = MODEL
    cfg.split           = SPLIT
    cfg.sample_fraction = FRAC
    cfg.output_dir      = OUT
    cfg.multi_model     = False

    """)


def _exp_body(exp_key: str) -> str:
    """Read the local run.py and extract the core logic (after imports)."""
    run_py = os.path.join(_SCRIPT_DIR, exp_key, "run.py")
    if not os.path.exists(run_py):
        return f"# {exp_key}/run.py not found\n"
    with open(run_py, "r", encoding="utf-8") as f:
        src = f.read()
    # Strip the argparse / path-setup boilerplate at the top;
    # keep everything from the first non-import function definition onward.
    lines = src.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class ") or line.startswith("EXP_NAME"):
            start = i
            break
    body_lines = lines[start:]
    # Replace argparse main() call pattern with direct invocation
    body = "\n".join(body_lines)
    body = body.replace(
        'if __name__ == "__main__":\n    main()',
        '# ── Kaggle: run directly ──\nmain()\n'
    )
    # Remove argparse dependency from main()
    body = body.replace("args = parse_args()\n    frac = 1.0 if args.full else args.frac\n",
                        "frac = FRAC\n    ")
    body = body.replace("args.split",  "SPLIT")
    body = body.replace("args.model",  "MODEL")
    body = body.replace("args.out or ", "")
    body = body.replace("args.no_ablation", "False")
    return body


# ── Build zip of core/ ────────────────────────────────────────────────────────
def build_core_zip():
    zip_path = os.path.join(_OUT_DIR, "espcal_core.zip")
    print(f"  Building {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(CORE_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if fname.endswith(".pyc"):
                    continue
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, os.path.dirname(CORE_DIR))
                zf.write(full, arcname)
    print(f"  ✓ {zip_path}  ({os.path.getsize(zip_path)//1024} KB)")
    return zip_path


# ── Write dataset-metadata.json ──────────────────────────────────────────────
def write_meta():
    meta_path = os.path.join(_OUT_DIR, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        f.write(DATASET_META)
    print(f"  ✓ {meta_path}  ← edit 'id' to set your Kaggle username")


# ── Build per-experiment Kaggle scripts ───────────────────────────────────────
def build_kaggle_scripts(frac: float = 0.10):
    for exp_key, num in EXP_DIRS.items():
        out_file = os.path.join(_OUT_DIR, f"kaggle_{exp_key}.py")
        preamble = _preamble(exp_key, frac)
        body     = _exp_body(exp_key)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(preamble + body)
        print(f"  ✓ {out_file}")


# ── Optional: upload via Kaggle CLI ──────────────────────────────────────────
def upload_dataset():
    """Requires `kaggle` CLI and ~/.kaggle/kaggle.json credentials."""
    print("\n  Uploading dataset via Kaggle CLI ...")

    # Unzip core into a staging folder (Kaggle dataset = folder, not zip)
    stage_dir = os.path.join(_OUT_DIR, "_stage_core")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    shutil.copytree(CORE_DIR, os.path.join(stage_dir, "core"))
    shutil.copy(os.path.join(_OUT_DIR, "dataset-metadata.json"), stage_dir)

    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", stage_dir, "--dir-mode", "zip"],
        capture_output=False
    )
    if result.returncode == 0:
        print("  ✓ Dataset uploaded!")
        print("  Update existing: kaggle datasets version -p <stage_dir> -m 'Update core'")
    else:
        print("  ✗ Upload failed — check Kaggle CLI setup.")
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true", help="Upload via Kaggle CLI after building")
    p.add_argument("--frac",   type=float, default=0.10, help="Default sample fraction in generated scripts")
    args = p.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)

    print("\n[1/3] Packaging core/ → zip")
    build_core_zip()

    print("\n[2/3] Writing dataset metadata")
    write_meta()

    print(f"\n[3/3] Generating Kaggle experiment scripts (frac={args.frac:.0%})")
    build_kaggle_scripts(args.frac)

    print("\n" + "═" * 60)
    print("  DONE — Files in:", _OUT_DIR)
    print("═" * 60)
    print("""
  Next steps
  ──────────
  1. Edit kaggle_upload/dataset-metadata.json
       Set "id" to "YOUR_USERNAME/paper2-espcal-core"

  2a. Upload manually:
        kaggle.com → Datasets → New Dataset
        Upload the kaggle_upload/ folder (or espcal_core.zip)
        Slug: paper2-espcal-core

  2b. Upload via CLI  (needs ~/.kaggle/kaggle.json):
        python package_for_kaggle.py --upload

  3. In each Kaggle notebook:
        + Data → Search "paper2-espcal-core" → Add
        Paste kaggle_upload/kaggle_exp_02_ESPCal.py as a Script notebook
        OR copy cells into an existing notebook

  4. Run quick-test first (FRAC=0.10 by default in generated scripts)
     Then set FRAC=1.0 and re-run for production.
""")

    if args.upload:
        upload_dataset()


if __name__ == "__main__":
    main()
