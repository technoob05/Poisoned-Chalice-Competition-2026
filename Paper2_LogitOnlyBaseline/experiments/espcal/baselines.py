"""Baseline MIA methods for comparison."""
import zlib
import numpy as np


class BaselineComparison:
    """Compute baseline MIA scores for comparison in the paper."""

    @staticmethod
    def minkprob(log_probs: np.ndarray, k_pct: float = 0.2) -> float:
        k = max(1, int(len(log_probs) * k_pct))
        return np.sort(log_probs)[:k].mean()

    @staticmethod
    def minkprob_plus_plus(log_probs: np.ndarray, mu: np.ndarray,
                            sigma: np.ndarray, k_pct: float = 0.2) -> float:
        z_scores = (log_probs - mu) / np.maximum(sigma, 1e-10)
        k = max(1, int(len(z_scores) * k_pct))
        return np.sort(z_scores)[:k].mean()

    @staticmethod
    def loss_attack(token_losses: np.ndarray) -> float:
        return -token_losses.mean()

    @staticmethod
    def surp(token_losses: np.ndarray) -> float:
        return -(token_losses.mean() - token_losses.std())

    @staticmethod
    def zlib_ratio(text: str, mean_loss: float) -> float:
        zlib_len = len(zlib.compress(text.encode("utf-8")))
        return mean_loss / max(zlib_len, 1)
