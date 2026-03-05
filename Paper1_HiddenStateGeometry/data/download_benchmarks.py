"""
download_benchmarks.py — Download ALL benchmark datasets for Paper 1 & Paper 2

Benchmarks:
1. WikiMIA (Shi et al., 2024)  — HuggingFace: swj0419/WikiMIA
2. MIMIR (Duan et al., 2024)   — HuggingFace: iamgroot42/mimir  
3. Poisoned Chalice (2026)     — HuggingFace: AISE-TUDelft/Poisoned-Chalice
4. BookMIA (Shi et al., 2024)  — HuggingFace: swj0419/BookMIA

Usage:
    pip install datasets huggingface_hub
    python download_benchmarks.py --output_dir ./data
    
For Kaggle: upload the downloaded data as a Kaggle dataset,
    or use HuggingFace directly in the notebook.
"""

import argparse
import os
import json
from pathlib import Path


def download_wikimia(output_dir: str):
    """
    WikiMIA: Wikipedia temporal split benchmark.
    - Member: Wikipedia events before model training cutoff
    - Non-member: Wikipedia events after cutoff
    - Splits by length: 32, 64, 128, 256 tokens
    - Settings: original, paraphrased
    
    Source: https://huggingface.co/datasets/swj0419/WikiMIA
    Paper: Shi et al., "Detecting Pretraining Data from Large Language Models", ICLR 2024
    """
    from datasets import load_dataset
    
    save_dir = os.path.join(output_dir, "WikiMIA")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Downloading WikiMIA benchmark")
    print("=" * 60)
    
    # WikiMIA has multiple configs for different lengths
    for length in [32, 64, 128, 256]:
        print(f"\n  [WikiMIA] Length = {length}")
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
            ds.save_to_disk(os.path.join(save_dir, f"length_{length}"))
            print(f"    ✓ Saved {len(ds)} samples → {save_dir}/length_{length}")
            # Show sample
            if len(ds) > 0:
                sample = ds[0]
                print(f"    Columns: {list(sample.keys())}")
                label_col = "label" if "label" in sample else list(sample.keys())[0]
                print(f"    Sample label: {sample.get('label', 'N/A')}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Try alternative config names
            try:
                ds = load_dataset("swj0419/WikiMIA", f"WikiMIA_length{length}")
                if hasattr(ds, "keys"):
                    for split_name in ds.keys():
                        ds[split_name].save_to_disk(
                            os.path.join(save_dir, f"length_{length}_{split_name}")
                        )
                        print(f"    ✓ Saved split '{split_name}' ({len(ds[split_name])} samples)")
            except Exception as e2:
                print(f"    ✗ Fallback also failed: {e2}")
    
    # Also try paraphrased versions
    for length in [32, 64, 128, 256]:
        print(f"\n  [WikiMIA] Paraphrased, Length = {length}")
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}_paraphrased")
            ds.save_to_disk(os.path.join(save_dir, f"length_{length}_paraphrased"))
            print(f"    ✓ Saved {len(ds)} samples")
        except Exception as e:
            print(f"    ✗ Not available or error: {e}")
    
    print(f"\n  WikiMIA download complete → {save_dir}")


def download_mimir(output_dir: str):
    """
    MIMIR: Membership Inference on Models Trained on The Pile.
    - Member: Pile train split samples
    - Non-member: Pile test split samples
    - 7 domains: Wikipedia, Github, Pile-CC, PubMed Central, ArXiv, DM Mathematics, HackerNews
    
    Source: https://huggingface.co/datasets/iamgroot42/mimir
    Paper: Duan et al., "Do Membership Inference Attacks Work on LLMs?", COLM 2024
    
    Note: MIMIR also provides pre-computed scores for some methods.
    We download the raw text data for running our own methods.
    """
    from datasets import load_dataset
    
    save_dir = os.path.join(output_dir, "MIMIR")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Downloading MIMIR benchmark")
    print("=" * 60)
    
    # MIMIR domains (subsets of The Pile)
    domains = [
        "wikipedia", "github", "pile_cc", "pubmed_central",
        "arxiv", "dm_mathematics", "hackernews"
    ]
    
    for domain in domains:
        print(f"\n  [MIMIR] Domain: {domain}")
        try:
            ds = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)
            if hasattr(ds, "keys"):
                for split_name in ds.keys():
                    save_path = os.path.join(save_dir, f"{domain}_{split_name}")
                    ds[split_name].save_to_disk(save_path)
                    print(f"    ✓ Split '{split_name}': {len(ds[split_name])} samples → {save_path}")
            else:
                save_path = os.path.join(save_dir, domain)
                ds.save_to_disk(save_path)
                print(f"    ✓ {len(ds)} samples → {save_path}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Try without subset name
            try:
                ds = load_dataset("iamgroot42/mimir", trust_remote_code=True)
                print(f"    Fallback: loaded full dataset with keys: {list(ds.keys()) if hasattr(ds, 'keys') else 'single'}")
            except Exception as e2:
                print(f"    ✗ Fallback also failed: {e2}")
    
    print(f"\n  MIMIR download complete → {save_dir}")


def download_poisoned_chalice(output_dir: str):
    """
    Poisoned Chalice: Code MIA benchmark on StarCoder2-3b.
    - 5 languages: Go, Java, Python, Ruby, Rust
    - 100K samples total (20K per language, 10K member + 10K non-member)
    
    Source: https://huggingface.co/datasets/AISE-TUDelft/Poisoned-Chalice
    """
    from datasets import load_dataset
    
    save_dir = os.path.join(output_dir, "PoisonedChalice")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Downloading Poisoned Chalice benchmark")
    print("=" * 60)
    
    languages = ["Go", "Java", "Python", "Ruby", "Rust"]
    
    for lang in languages:
        print(f"\n  [PC] Language: {lang}")
        try:
            ds = load_dataset("AISE-TUDelft/Poisoned-Chalice", lang)
            if hasattr(ds, "keys"):
                for split_name in ds.keys():
                    save_path = os.path.join(save_dir, f"{lang}_{split_name}")
                    ds[split_name].save_to_disk(save_path)
                    print(f"    ✓ Split '{split_name}': {len(ds[split_name])} samples → {save_path}")
            else:
                save_path = os.path.join(save_dir, lang)
                ds.save_to_disk(save_path)
                print(f"    ✓ {len(ds)} samples → {save_path}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n  Poisoned Chalice download complete → {save_dir}")


def download_bookmia(output_dir: str):
    """
    BookMIA: Copyright book membership inference benchmark.
    - Member: text excerpts from books published before model training cutoff
    - Non-member: text excerpts from books published in 2023 
    - Text length: 512 tokens
    - 9,870 balanced samples (4,935 member + 4,935 non-member)
    
    Source: https://huggingface.co/datasets/swj0419/BookMIA
    Paper: Shi et al., "Detecting Pretraining Data from Large Language Models", ICLR 2024
    
    Note: SoK paper (Meeus et al., SaTML 2025) shows BoW AUC ~0.94 on BookMIA,
    indicating some distribution shift. Use alongside MIMIR for fair comparison.
    """
    from datasets import load_dataset
    
    save_dir = os.path.join(output_dir, "BookMIA")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Downloading BookMIA benchmark")
    print("=" * 60)
    
    try:
        ds = load_dataset("swj0419/BookMIA", split="train")
        ds.save_to_disk(save_dir)
        n_member = sum(1 for row in ds if row["label"] == 1)
        n_nonmember = len(ds) - n_member
        print(f"    ✓ Saved {len(ds)} samples ({n_member} members, {n_nonmember} non-members)")
        print(f"    Columns: {ds.column_names}")
        print(f"    Text field: 'snippet' (512 tokens)")
        print(f"    Label: 0=non-member (2023 books), 1=member (older books)")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print(f"\n  BookMIA download complete → {save_dir}")


def download_all(output_dir: str):
    """Download all benchmarks."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("╔" + "═" * 58 + "╗")
    print("║  BENCHMARK DATA DOWNLOADER                               ║")
    print("║  Paper 1: MultiGeo-MIA (Hidden-State Geometry)           ║")
    print("║  Paper 2: ESP-Cal (Logit-Only Baseline)                  ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Datasets:                                                ║")
    print("║    1. WikiMIA  (Shi et al., ICLR 2024)                   ║")
    print("║    2. MIMIR    (Duan et al., COLM 2024)                  ║")
    print("║    3. Poisoned Chalice (AISE-TUDelft, 2026)              ║")
    print("║    4. BookMIA  (Shi et al., ICLR 2024)                   ║")
    print("╚" + "═" * 58 + "╝")
    
    download_wikimia(output_dir)
    download_mimir(output_dir)
    download_poisoned_chalice(output_dir)
    download_bookmia(output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            for d in sorted(dirs):
                print(f"{indent}  {d}/")
    print("\nDone! Upload this folder to Kaggle as a dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MIA benchmark datasets")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save downloaded datasets")
    parser.add_argument("--only", type=str, default="all",
                        choices=["all", "wikimia", "mimir", "poisoned_chalice", "bookmia"],
                        help="Download only a specific benchmark")
    args = parser.parse_args()
    
    if args.only == "all":
        download_all(args.output_dir)
    elif args.only == "wikimia":
        download_wikimia(args.output_dir)
    elif args.only == "mimir":
        download_mimir(args.output_dir)
    elif args.only == "poisoned_chalice":
        download_poisoned_chalice(args.output_dir)
    elif args.only == "bookmia":
        download_bookmia(args.output_dir)
