"""ESP feature extractor (logit-only, grey-box)."""
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

from .config import Config


class ESPExtractor:
    """
    Extract Entropy Slope Profile (ESP) features from logits.

    For each token position t, compute:
        H(t) = -Σ_v p(v|x_{<t}) log p(v|x_{<t})
    Then fit: H(t) = α·t + β
    ESP score = α (slope of entropy trajectory)
    Members: steeper negative slope → negate for signal
    """

    def __init__(self, model, tokenizer, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_length, padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        seq_len = input_ids.shape[1]

        if seq_len < self.cfg.min_tokens:
            return self._empty_features()

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]

        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Per-token entropy
        entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
        entropy_np = entropy.cpu().numpy()

        # Per-token loss
        token_lp = log_probs.squeeze(0).gather(1, shift_labels.squeeze(0).unsqueeze(1)).squeeze(1)
        token_loss = -token_lp.cpu().numpy()

        n_tokens = len(entropy_np)
        positions = np.arange(n_tokens)

        # 1. Entropy Slope (ESP) — original
        esp_slope, esp_intercept = np.polyfit(positions, entropy_np, 1)

        # 2. Entropy trajectory features
        h_mean = entropy_np.mean()
        h_std = entropy_np.std()
        mid = n_tokens // 2
        h_first = entropy_np[:mid].mean() if mid > 0 else h_mean
        h_second = entropy_np[mid:].mean() if mid < n_tokens else h_mean
        h_drop = h_first - h_second
        h_curvature = np.polyfit(positions, entropy_np, 2)[0] if n_tokens >= 6 else 0.0

        # 3. Token-level z-normalized entropy slope
        z_entropy_slope = np.polyfit(positions, (entropy_np - h_mean) / max(h_std, 1e-10), 1)[0] if h_std > 0 else 0.0

        # 4. Loss features
        mean_loss = token_loss.mean()
        loss_std = token_loss.std()
        loss_slope = np.polyfit(positions, token_loss, 1)[0]

        # ═══════════════════════════════════════════════════════
        # 5. ★ Surprise Trajectory Features (token-level loss shape)
        #    These capture HOW the model's per-token surprise evolves,
        #    which is a direct memorization signal.
        # ═══════════════════════════════════════════════════════

        # 5a. Surprise drop: first-half loss minus second-half loss
        #     Members: model "recognizes" text → loss drops more → positive
        loss_first = token_loss[:mid].mean() if mid > 0 else mean_loss
        loss_second = token_loss[mid:].mean() if mid < n_tokens else mean_loss
        surprise_drop = loss_first - loss_second

        # 5b. Surprise acceleration: curvature of loss trajectory
        #     Members: concave loss curve (fast initial drop) → negative
        surprise_accel = np.polyfit(positions, token_loss, 2)[0] if n_tokens >= 6 else 0.0

        # 5c. Surprise volatility: stability of consecutive loss changes
        #     Members: more predictable → lower volatility
        loss_diffs = np.diff(token_loss)
        surprise_volatility = loss_diffs.std() if len(loss_diffs) > 1 else 0.0

        # 5d. Loss quantile range: spread of token losses
        #     Members: more uniform (less extreme) → smaller range
        if n_tokens > 2:
            loss_q_range = float(np.quantile(token_loss, 0.9) - np.quantile(token_loss, 0.1))
        else:
            loss_q_range = 0.0

        # 5e. Max surprise step: largest single-token loss decrease
        #     Members: stronger "recognition" moments → larger drops
        max_loss_drop = float(-loss_diffs.min()) if len(loss_diffs) > 0 else 0.0

        # 5f. Settle fraction: proportion of second-half tokens below median
        #     Members: loss "settles" low after model recognizes the text
        median_loss = np.median(token_loss)
        settle_frac = float((token_loss[mid:] < median_loss).mean()) if mid < n_tokens else 0.5

        # 6. MinK% Prob
        k = max(1, int(n_tokens * 0.2))
        sorted_lp = np.sort(token_lp.cpu().numpy())
        minkprob = sorted_lp[:k].mean()

        # 7. Rank features
        ranks = (shift_logits.squeeze(0).argsort(dim=-1, descending=True)
                 .argsort(dim=-1).gather(1, shift_labels.squeeze(0).unsqueeze(1))
                 .squeeze(1).float().cpu().numpy())
        mean_rank = ranks.mean()
        median_rank = np.median(ranks)

        # 8. Composite
        surp = mean_loss - loss_std
        zlib_len = len(zlib.compress(text.encode("utf-8")))
        zlib_ratio = mean_loss / (zlib_len / n_tokens) if zlib_len > 0 else 0.0

        features = {
            # Entropy features
            "esp_slope": esp_slope, "esp_intercept": esp_intercept,
            "z_esp_slope": z_entropy_slope,
            "h_mean": h_mean, "h_std": h_std, "h_drop": h_drop, "h_curvature": h_curvature,
            # Loss features
            "neg_mean_loss": -mean_loss, "loss_slope": loss_slope, "loss_std": loss_std,
            # ★ Surprise trajectory features
            "surprise_drop": surprise_drop,
            "surprise_accel": surprise_accel,
            "neg_surprise_vol": -surprise_volatility,
            "neg_loss_q_range": -loss_q_range,
            "max_loss_drop": max_loss_drop,
            "settle_frac": settle_frac,
            # Baselines
            "minkprob_20": minkprob, "surp": -surp,
            "neg_mean_rank": -mean_rank, "neg_median_rank": -median_rank,
            "zlib_ratio": zlib_ratio,
            # Meta
            "seq_len": seq_len, "n_tokens": n_tokens,
            # Primary signals (higher → more likely member)
            "signal_esp": -esp_slope,
            "signal_h_drop": h_drop,
            "signal_loss": -mean_loss,
            "signal_surprise_drop": surprise_drop,
        }

        del outputs, logits, probs, log_probs, entropy
        torch.cuda.empty_cache()
        return features

    def _empty_features(self) -> Dict[str, float]:
        keys = ["esp_slope", "esp_intercept", "z_esp_slope",
                "h_mean", "h_std", "h_drop", "h_curvature",
                "neg_mean_loss", "loss_slope", "loss_std",
                "surprise_drop", "surprise_accel", "neg_surprise_vol",
                "neg_loss_q_range", "max_loss_drop", "settle_frac",
                "minkprob_20", "surp", "neg_mean_rank", "neg_median_rank",
                "zlib_ratio", "seq_len", "n_tokens",
                "signal_esp", "signal_h_drop", "signal_loss",
                "signal_surprise_drop"]
        return {k: np.nan for k in keys}
