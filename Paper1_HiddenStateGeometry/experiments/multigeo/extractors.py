"""MultiGeo 4-axis feature extractor (single forward pass)."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List

from .config import Config


class MultiGeoExtractor:
    """
    Extract 4-axis geometric features from a single forward pass.

    Axis 1 — Magnitude:       ‖h_mid‖₂ (mid-layer hidden-state L2 norm)
    Axis 2 — Dimensionality:  effective rank of H matrix (SVD)
    Axis 3 — Dynamics:        mean cosine distance between consecutive layers
    Axis 4 — Routing:         mean attention entropy across heads and layers
    """

    def __init__(self, model, tokenizer, n_layers: int, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.cfg = cfg
        self.device = next(model.parameters()).device

        mid = n_layers // 2
        self.mid_layer_idx = mid
        self.cascade_layer_indices = self._select_cascade_layers()
        self.attn_layer_indices = self._select_attn_layers()

    def _select_cascade_layers(self) -> List[int]:
        n = self.n_layers
        k = min(self.cfg.cascade_pairs + 1, n)
        return [int(i * (n - 1) / (k - 1)) for i in range(k)]

    def _select_attn_layers(self) -> List[int]:
        n = self.n_layers
        return [0, n // 4, n // 2, 3 * n // 4, n - 1]

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.cfg.max_length, padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        seq_len = input_ids.shape[1]

        if seq_len < 4:
            return self._empty_features()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.device),
            output_hidden_states=True,
            output_attentions=True,
        )

        features = {}
        features.update(self._compute_magnitude(outputs.hidden_states))
        features.update(self._compute_dimensionality(outputs.hidden_states))
        features.update(self._compute_dynamics(outputs.hidden_states))
        features.update(self._compute_routing(outputs.attentions))
        features["loss"] = self._compute_loss(outputs.logits, input_ids)
        features["seq_len"] = seq_len

        del outputs, input_ids
        torch.cuda.empty_cache()
        return features

    def _compute_magnitude(self, hidden_states) -> Dict[str, float]:
        mid_hs = hidden_states[self.mid_layer_idx + 1].float()
        norms = torch.norm(mid_hs[0], dim=-1)
        mid_norm = norms.mean().item()
        mid_norm_std = norms.std().item() if norms.numel() > 1 else 0.0

        last_hs = hidden_states[-1].float()
        last_norm = torch.norm(last_hs[0], dim=-1).mean().item()

        all_norms = []
        for li in range(0, self.n_layers + 1, max(1, self.n_layers // 8)):
            hs = hidden_states[li].float()
            all_norms.append(torch.norm(hs[0], dim=-1).mean().item())
        norm_slope = np.polyfit(range(len(all_norms)), all_norms, 1)[0] if len(all_norms) > 1 else 0.0

        return {
            "mag_mid_norm": mid_norm,
            "mag_mid_norm_std": mid_norm_std,
            "mag_last_norm": last_norm,
            "mag_norm_slope": norm_slope,
            "signal_magnitude": -mid_norm,
        }

    def _compute_dimensionality(self, hidden_states) -> Dict[str, float]:
        H = hidden_states[self.mid_layer_idx + 1].float().squeeze(0)
        seq_len, d = H.shape
        try:
            k = min(self.cfg.svd_top_k, seq_len, d)
            U, S, Vh = torch.linalg.svd(H, full_matrices=False)
            S = S[:k]
            S_norm = S / S.sum()
            S_norm = S_norm.clamp(min=1e-10)
            sv_entropy = -(S_norm * torch.log(S_norm)).sum().item()
            eff_rank = np.exp(sv_entropy)
            top5_ratio = S[:min(5, k)].sum().item() / S.sum().item()
            spectral_gap = (S[0] / S[1]).item() if k > 1 and S[1] > 0 else 0.0
        except Exception:
            sv_entropy, eff_rank, top5_ratio, spectral_gap = 0.0, 0.0, 1.0, 0.0

        return {
            "dim_eff_rank": eff_rank,
            "dim_sv_entropy": sv_entropy,
            "dim_top5_ratio": top5_ratio,
            "dim_spectral_gap": spectral_gap,
            "signal_dimensionality": -eff_rank,
        }

    def _compute_dynamics(self, hidden_states) -> Dict[str, float]:
        drifts = []
        for i in range(len(self.cascade_layer_indices) - 1):
            l1 = self.cascade_layer_indices[i]
            l2 = self.cascade_layer_indices[i + 1]
            h1 = hidden_states[l1 + 1].float().squeeze(0).mean(dim=0)
            h2 = hidden_states[l2 + 1].float().squeeze(0).mean(dim=0)
            cos_sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            drifts.append(1.0 - cos_sim)

        mean_drift = np.mean(drifts) if drifts else 0.0
        max_drift = np.max(drifts) if drifts else 0.0
        drift_std = np.std(drifts) if len(drifts) > 1 else 0.0

        if len(drifts) >= 4:
            early_drift = np.mean(drifts[:len(drifts) // 2])
            late_drift = np.mean(drifts[len(drifts) // 2:])
            drift_asymmetry = early_drift - late_drift
        else:
            drift_asymmetry = 0.0

        return {
            "dyn_mean_drift": mean_drift,
            "dyn_max_drift": max_drift,
            "dyn_drift_std": drift_std,
            "dyn_drift_asymmetry": drift_asymmetry,
            "signal_dynamics": -mean_drift,
        }

    def _compute_routing(self, attentions) -> Dict[str, float]:
        entropies, concentrations = [], []
        for li in self.attn_layer_indices:
            if li >= len(attentions):
                continue
            attn = attentions[li].float().squeeze(0)
            n_heads, seq_q, seq_k = attn.shape
            attn_clamped = attn.clamp(min=1e-10)
            H = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)
            mean_entropy = H.mean().item()
            entropies.append(mean_entropy)
            max_ent = np.log(seq_k) if seq_k > 0 else 1.0
            concentrations.append(max_ent - mean_entropy)

        mean_attn_entropy = np.mean(entropies) if entropies else 0.0
        mean_concentration = np.mean(concentrations) if concentrations else 0.0

        if len(entropies) >= 3:
            early_ent = np.mean(entropies[:len(entropies) // 2])
            late_ent = np.mean(entropies[len(entropies) // 2:])
            entropy_trajectory = late_ent - early_ent
        else:
            entropy_trajectory = 0.0

        return {
            "rout_mean_entropy": mean_attn_entropy,
            "rout_mean_concentration": mean_concentration,
            "rout_entropy_trajectory": entropy_trajectory,
            "signal_routing": -mean_attn_entropy,
        }

    def _compute_loss(self, logits, input_ids) -> float:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="mean"
        )
        return -loss.item()

    def _empty_features(self) -> Dict[str, float]:
        return {
            "mag_mid_norm": np.nan, "mag_mid_norm_std": np.nan,
            "mag_last_norm": np.nan, "mag_norm_slope": np.nan,
            "signal_magnitude": np.nan,
            "dim_eff_rank": np.nan, "dim_sv_entropy": np.nan,
            "dim_top5_ratio": np.nan, "dim_spectral_gap": np.nan,
            "signal_dimensionality": np.nan,
            "dyn_mean_drift": np.nan, "dyn_max_drift": np.nan,
            "dyn_drift_std": np.nan, "dyn_drift_asymmetry": np.nan,
            "signal_dynamics": np.nan,
            "rout_mean_entropy": np.nan, "rout_mean_concentration": np.nan,
            "rout_entropy_trajectory": np.nan, "signal_routing": np.nan,
            "loss": np.nan, "seq_len": 0,
        }
