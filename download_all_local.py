"""
download_all_local.py — Download WikiMIA + MIMIR locally, then zip everything for Kaggle.

Usage:
    python download_all_local.py

Output:
    ./benchmark_data/WikiMIA/          — WikiMIA (4 lengths × original + paraphrased)
    ./benchmark_data/MIMIR/            — MIMIR (7 domains)
    ./benchmark_data/PoisonedChalice/  — symlink/copy of existing data
    ./kaggle_benchmarks.zip            — ready to upload to Kaggle
"""

import os
import sys
import json
import shutil
from pathlib import Path

# ═══════════════════════════════════
# 1. Download WikiMIA
# ═══════════════════════════════════

def download_wikimia(base_dir: str):
    from datasets import load_dataset

    save_dir = os.path.join(base_dir, "WikiMIA")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  DOWNLOADING WikiMIA (Shi et al., ICLR 2024)")
    print("  Source: huggingface.co/datasets/swj0419/WikiMIA")
    print("=" * 60)

    # Original splits
    for length in [32, 64, 128, 256]:
        out_path = os.path.join(save_dir, f"length_{length}")
        if os.path.exists(out_path):
            print(f"  [SKIP] length_{length} already exists")
            continue

        print(f"\n  Downloading WikiMIA length={length}...")
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
            ds.save_to_disk(out_path)
            n_mem = sum(1 for r in ds if r["label"] == 1)
            print(f"  ✓ {len(ds)} samples ({n_mem} members) → {out_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Paraphrased splits (important for robustness evaluation)
    for length in [32, 64, 128, 256]:
        out_path = os.path.join(save_dir, f"length_{length}_paraphrased")
        if os.path.exists(out_path):
            print(f"  [SKIP] length_{length}_paraphrased already exists")
            continue

        print(f"\n  Downloading WikiMIA length={length} (paraphrased)...")
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}_paraphrased")
            ds.save_to_disk(out_path)
            print(f"  ✓ {len(ds)} samples → {out_path}")
        except Exception as e:
            print(f"  ✗ Paraphrased not available: {e}")

    print(f"\n  WikiMIA done → {save_dir}")


# ═══════════════════════════════════
# 2. Download MIMIR
# ═══════════════════════════════════

def download_mimir(base_dir: str):
    from datasets import load_dataset

    save_dir = os.path.join(base_dir, "MIMIR")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  DOWNLOADING MIMIR (Duan et al., COLM 2024)")
    print("  Source: huggingface.co/datasets/iamgroot42/mimir")
    print("=" * 60)

    domains = [
        "wikipedia", "github", "pile_cc", "pubmed_central",
        "arxiv", "dm_mathematics", "hackernews"
    ]

    for domain in domains:
        out_path = os.path.join(save_dir, domain)
        if os.path.exists(out_path):
            print(f"  [SKIP] {domain} already exists")
            continue

        print(f"\n  Downloading MIMIR/{domain}...")
        try:
            ds = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)
            # Save each split separately
            os.makedirs(out_path, exist_ok=True)
            for split_name in ds.keys():
                split_path = os.path.join(out_path, split_name)
                ds[split_name].save_to_disk(split_path)
                print(f"    {split_name}: {len(ds[split_name])} samples → {split_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Try loading full dataset without config
            try:
                print(f"  Trying alternative loading for {domain}...")
                ds = load_dataset("iamgroot42/mimir", domain,
                                  trust_remote_code=True,
                                  revision="main")
                os.makedirs(out_path, exist_ok=True)
                for split_name in ds.keys():
                    split_path = os.path.join(out_path, split_name)
                    ds[split_name].save_to_disk(split_path)
                    print(f"    {split_name}: {len(ds[split_name])} samples")
            except Exception as e2:
                print(f"  ✗ Fallback also failed: {e2}")

    print(f"\n  MIMIR done → {save_dir}")


# ═══════════════════════════════════
# 3. Copy Poisoned Chalice
# ═══════════════════════════════════

def copy_poisoned_chalice(base_dir: str, source_dir: str):
    save_dir = os.path.join(base_dir, "PoisonedChalice")

    print("\n" + "=" * 60)
    print("  COPYING Poisoned Chalice (local)")
    print("=" * 60)

    if os.path.exists(save_dir):
        print(f"  [SKIP] Already exists at {save_dir}")
        return

    if not os.path.exists(source_dir):
        print(f"  ✗ Source not found: {source_dir}")
        print("  Downloading from HuggingFace instead...")
        from datasets import load_dataset
        os.makedirs(save_dir, exist_ok=True)
        for lang in ["Go", "Java", "Python", "Ruby", "Rust"]:
            print(f"    Downloading {lang}...")
            try:
                ds = load_dataset("AISE-TUDelft/Poisoned-Chalice", lang)
                for split_name in ds.keys():
                    out = os.path.join(save_dir, lang, split_name)
                    ds[split_name].save_to_disk(out)
                    print(f"      {split_name}: {len(ds[split_name])} → {out}")
            except Exception as e:
                print(f"      Error: {e}")
        return

    print(f"  Copying {source_dir} → {save_dir}")
    shutil.copytree(source_dir, save_dir)
    # Verify
    for lang in ["Go", "Java", "Python", "Ruby", "Rust"]:
        lang_path = os.path.join(save_dir, lang)
        if os.path.exists(lang_path):
            print(f"  ✓ {lang}")
        else:
            print(f"  ✗ {lang} missing!")

    print(f"  Poisoned Chalice done → {save_dir}")


# ═══════════════════════════════════
# 4. Create Kaggle-ready ZIP
# ═══════════════════════════════════

def create_kaggle_zip(base_dir: str, zip_name: str = "kaggle_benchmarks"):
    print("\n" + "=" * 60)
    print("  CREATING KAGGLE ZIP")
    print("=" * 60)

    # Show what we have
    total_size = 0
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print(f"  Total data size: {total_size / (1024**3):.2f} GB")

    # Create zip
    zip_path = os.path.join(os.path.dirname(base_dir), zip_name)
    print(f"  Zipping → {zip_path}.zip ...")
    shutil.make_archive(zip_path, "zip", base_dir)

    zip_file = zip_path + ".zip"
    zip_size = os.path.getsize(zip_file) / (1024**3)
    print(f"  ✓ Created {zip_file} ({zip_size:.2f} GB)")
    print(f"\n  Upload this to Kaggle as a dataset!")
    return zip_file


# ═══════════════════════════════════
# 5. Summary
# ═══════════════════════════════════

def print_summary(base_dir: str):
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(item_path) for f in files
            )
            n_files = sum(len(files) for _, _, files in os.walk(item_path))
            print(f"  {item:25s}  {n_files:5d} files  {size/(1024**2):8.1f} MB")


# ═══════════════════════════════════
# MAIN
# ═══════════════════════════════════

if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
    PC_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poisoned_chalice_dataset")

    os.makedirs(BASE, exist_ok=True)

    print("╔" + "═" * 58 + "╗")
    print("║  BENCHMARK DOWNLOADER — Local + Kaggle ZIP               ║")
    print("║  WikiMIA + MIMIR + Poisoned Chalice                      ║")
    print("╚" + "═" * 58 + "╝")

    # 1. WikiMIA
    download_wikimia(BASE)

    # 2. MIMIR
    download_mimir(BASE)

    # 3. Poisoned Chalice
    copy_poisoned_chalice(BASE, PC_SOURCE)

    # 4. Summary
    print_summary(BASE)

    # 5. ZIP for Kaggle
    create_kaggle_zip(BASE, "kaggle_benchmarks")

    print("\n  ALL DONE!")
