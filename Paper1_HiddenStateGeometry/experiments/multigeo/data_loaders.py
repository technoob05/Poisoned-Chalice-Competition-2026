"""Data loaders for all benchmarks."""
import os
import numpy as np
import pandas as pd
from typing import Dict

from .config import Config


KAGGLE_ROOT = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"


def load_poisoned_chalice(cfg: Config) -> pd.DataFrame:
    from datasets import load_dataset, load_from_disk

    print("\n  Loading Poisoned Chalice dataset...")
    all_rows = []
    kaggle_path = os.path.join(KAGGLE_ROOT, "poisoned_chalice_dataset")

    for lang in cfg.languages:
        try:
            if os.path.exists(os.path.join(kaggle_path, lang)):
                ds = load_from_disk(os.path.join(kaggle_path, lang, cfg.split))
            else:
                ds = load_dataset(cfg.dataset_name, lang, split=cfg.split)

            for row in ds:
                text = row.get("content") or ""
                if not text or not text.strip():
                    continue
                all_rows.append({
                    "text": text,
                    "is_member": 1 if row["membership"] == "member" else 0,
                    "subset": lang,
                })
            print(f"    {lang}: {len(ds)} samples (kept {sum(1 for r in all_rows if r['subset']==lang)})")
        except Exception as e:
            print(f"    {lang}: ERROR — {e}")

    df = pd.DataFrame(all_rows)
    if cfg.sample_fraction < 1.0:
        df = df.groupby(["subset", "is_member"]).apply(
            lambda x: x.sample(frac=cfg.sample_fraction, random_state=cfg.seed)
        ).reset_index(drop=True)
        print(f"  Sampled {cfg.sample_fraction:.0%} → {len(df)} rows")
    print(f"  Total: {len(df)} samples, {df['is_member'].sum()} members")
    return df


def load_wikimia(cfg: Config) -> Dict[str, pd.DataFrame]:
    from datasets import load_dataset, load_from_disk

    print("\n  Loading WikiMIA dataset...")
    data_by_length = {}
    kaggle_path = os.path.join(KAGGLE_ROOT, "kaggle_wikimia")

    for length in cfg.wikimia_lengths:
        try:
            local_split = os.path.join(kaggle_path, f"WikiMIA_length{length}")
            if os.path.exists(local_split):
                ds = load_from_disk(local_split)
            else:
                ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")

            rows = []
            for row in ds:
                rows.append({
                    "text": row["input"],
                    "is_member": int(row["label"]),
                    "subset": f"len{length}",
                })
            df = pd.DataFrame(rows)
            data_by_length[f"len{length}"] = df
            mem = df["is_member"].sum()
            print(f"    Length {length}: {len(df)} ({mem}M/{len(df)-mem}NM)")
        except Exception as e:
            print(f"    Length {length}: ERROR — {e}")
    return data_by_length


def load_mimir(cfg: Config) -> Dict[str, pd.DataFrame]:
    from datasets import load_dataset

    print("\n  Loading MIMIR dataset...")
    data_by_domain = {}
    kaggle_path = os.path.join(KAGGLE_ROOT, "kaggle_mimir")

    for domain in cfg.mimir_domains:
        try:
            local_dir = os.path.join(kaggle_path, domain)
            if os.path.exists(local_dir):
                # Load from pre-downloaded JSONL
                rows = []
                for fname, label in [("member.jsonl", 1), ("nonmember.jsonl", 0)]:
                    fpath = os.path.join(local_dir, fname)
                    if os.path.exists(fpath):
                        import json
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                text = line.strip()
                                if text:
                                    rows.append({"text": text, "is_member": label, "subset": domain})
            else:
                ds = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)
                rows = []
                for split_name, label in [("member", 1), ("nonmember", 0)]:
                    if split_name in ds:
                        for row in ds[split_name]:
                            text = row.get("text", row.get("input", ""))
                            rows.append({"text": text, "is_member": label, "subset": domain})

            if rows:
                df = pd.DataFrame(rows)
                data_by_domain[domain] = df
                mem = df["is_member"].sum()
                print(f"    {domain}: {len(df)} ({mem}M/{len(df)-mem}NM)")
        except Exception as e:
            print(f"    {domain}: ERROR — {e}")
    return data_by_domain


def load_bookmia(cfg: Config) -> pd.DataFrame:
    from datasets import load_dataset, load_from_disk

    print("\n  Loading BookMIA dataset...")
    kaggle_path = os.path.join(KAGGLE_ROOT, "kaggle_bookmia")

    try:
        if os.path.exists(kaggle_path):
            ds = load_from_disk(kaggle_path)
        else:
            ds = load_dataset("swj0419/BookMIA", split="train")

        rows = []
        for row in ds:
            rows.append({
                "text": row["snippet"],
                "is_member": int(row["label"]),
                "subset": f"book_{row.get('book_id', 0)}",
            })
        df = pd.DataFrame(rows)
        mem = df["is_member"].sum()
        print(f"    BookMIA: {len(df)} ({mem}M/{len(df)-mem}NM)")
        return df
    except Exception as e:
        print(f"    BookMIA: ERROR — {e}")
        return pd.DataFrame()
