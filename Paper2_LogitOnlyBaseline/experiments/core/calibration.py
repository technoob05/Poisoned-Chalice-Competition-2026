"""Multi-Scale Calibration (3-scale z-normalization)."""
import pandas as pd
from typing import List

from .config import Config


class MultiScaleCalibrator:
    """
    3-scale hierarchical z-normalization for ESP scores.

    Scale 1 (Token):    z-normalize per-token entropy (done in extractor)
    Scale 2 (Position): z-normalize by sequence-length bucket
    Scale 3 (Domain):   z-normalize per language/domain subset
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def calibrate(self, df: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in score_columns:
            df[f"{col}_raw"] = df[col]

            if self.cfg.enable_scale2_position and "n_tokens" in df.columns:
                df["_len_bucket"] = pd.qcut(
                    df["n_tokens"].fillna(0),
                    q=min(self.cfg.position_buckets, df["n_tokens"].nunique()),
                    duplicates="drop", labels=False,
                )
                grouped = df.groupby("_len_bucket")[col]
                means = grouped.transform("mean")
                stds = grouped.transform("std").replace(0, 1)
                df[col] = (df[col] - means) / stds
                df.drop(columns=["_len_bucket"], inplace=True)

            if self.cfg.enable_scale3_domain and "subset" in df.columns:
                grouped = df.groupby("subset")[col]
                means = grouped.transform("mean")
                stds = grouped.transform("std").replace(0, 1)
                df[col] = (df[col] - means) / stds
        return df
