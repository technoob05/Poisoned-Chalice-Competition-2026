"""Evaluation utilities: AUROC, rank-averaging, z-normalization."""
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import roc_auc_score


def rank_average(df: pd.DataFrame, columns: List[str], name: str = "rank_avg") -> pd.Series:
    """Compute rank-average of multiple score columns (unsupervised fusion)."""
    ranks = pd.DataFrame()
    for col in columns:
        valid = df[col].notna()
        r = pd.Series(np.nan, index=df.index)
        r[valid] = df.loc[valid, col].rank(pct=True)
        ranks[col] = r
    return ranks.mean(axis=1)


def per_language_znorm(df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """Z-normalize score columns per-language subset."""
    df = df.copy()
    for col in score_columns:
        df[f"{col}_raw"] = df[col]
        grouped = df.groupby("subset")[col]
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, 1)
        df[col] = (df[col] - means) / stds
    return df


def evaluate_scores(df: pd.DataFrame, score_columns: List[str],
                    label_col: str = "is_member") -> pd.DataFrame:
    """Compute AUROC for each score column."""
    results = []
    for col in score_columns:
        valid = df[col].notna() & df[label_col].notna()
        if valid.sum() < 10:
            continue
        y_true = df.loc[valid, label_col].values
        y_score = df.loc[valid, col].values
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_score)
        best_auc = max(auc, 1 - auc)
        polarity = "+" if auc >= 0.5 else "-"
        results.append({
            "score": col, "auc": best_auc, "auc_raw": auc,
            "polarity": polarity, "n_samples": int(valid.sum()),
        })
    return pd.DataFrame(results).sort_values("auc", ascending=False)


def evaluate_per_subset(df: pd.DataFrame, score_col: str,
                        label_col: str = "is_member") -> pd.DataFrame:
    """Compute per-subset AUROC."""
    results = []
    for subset, group in df.groupby("subset"):
        valid = group[score_col].notna() & group[label_col].notna()
        if valid.sum() < 10:
            continue
        y_true = group.loc[valid, label_col].values
        y_score = group.loc[valid, score_col].values
        if len(np.unique(y_true)) < 2:
            continue
        auc = roc_auc_score(y_true, y_score)
        best_auc = max(auc, 1 - auc)
        results.append({"subset": subset, "auc": best_auc, "n": int(valid.sum())})
    return pd.DataFrame(results)
