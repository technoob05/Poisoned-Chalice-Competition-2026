"""Evaluation utilities: AUROC, per-subset metrics."""
import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import roc_auc_score


def evaluate_scores(df: pd.DataFrame, score_columns: List[str],
                    label_col: str = "is_member") -> pd.DataFrame:
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
