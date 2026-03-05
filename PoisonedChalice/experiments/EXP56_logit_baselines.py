"""
EXPERIMENT 56: Next-Generation Logit-Only Baselines — Min-K%++ Replacements
============================================================================

Goal: Find a SINGLE logit-only unsupervised score that REPLACES Min-K%++ as
the standard MIA baseline. Same requirements, stronger signal, simpler math.

Context (from our tracker):
  - Min-K%++ (EXP02): 0.5770    ← current "standard" baseline
  - Mean loss (EXP01): 0.5807   ← trivially beats Min-K%++
  - SURP (EXP16):     0.5884   ← mean - std, 2 statistics
  - CDD entropy slope: 0.6292   ← strongest logit-only known

WHY Min-K%++ is a BAD baseline:
  1. Z-score normalization assumes Gaussian p(·|x<t) — WRONG for code LLMs
     (code has multimodal token distributions: keywords vs identifiers)
  2. Bottom-K% selection discards 80% of tokens — wasteful, because
     memorization is HOLISTIC (Insight 13: members memorize EVERYTHING)
  3. Ignores positional/trajectory information (each token treated i.i.d.)
  4. More complex than simpler methods that outperform it

CANDIDATES (all novel or under-explored for code MIA):

  1. EWL  — Entropy-Weighted Loss
           score = Σ H(t)·log p(x_t) / Σ H(t)
           Upweight uncertain positions where M/NM diverge most.

  2. MLR  — Mean Log-Rank
           score = -mean(log₂(rank(x_t) + 1))
           Ordinal measure, immune to logit scale/calibration.

  3. QCL  — Quantile-Calibrated Loss
           score = mean(Φ_t(log p(x_t)))  where Φ_t = empirical CDF at pos t
           Full CDF calibration vs Min-K%++'s Gaussian Z-score approximation.

  4. DCG  — Discounted Cumulative Gain (IR-inspired)
           score = mean(1/log₂(rank+2))
           Rewards correct predictions proportionally to confidence.

  5. GAL  — Gated Loss (entropy-thresholded)
           score = mean(log p(x_t) WHERE H(t) > median(H))
           Only grade the model on tokens where it's genuinely uncertain.

  6. ESP  — Entropy Slope (CDD-derived, included as comparison)
           score = -corr(t, H(t)) × std(H) / std(t)  [= regression slope]
           Members' entropy decreases faster across the sequence.

  7. MAX  — Max-Rank Score
           score = -max(rank(x_t)) / V  (normalized worst prediction)
           Members' worst token is still well-ranked; NM have outlier failures.

  8. FCS  — Fractional Confidence Score
           score = mean(p(x_t) > 1/k) for k in {10,100,1000}
           Fraction of tokens where model beats uniform over top-k.

  9. KLD  — KL from Uniform (mean model confidence)
           score = mean(log V + log p(x_t))  = mean(log(V · p(x_t)))
           How much better than random is the model at each position.

 10. NES  — Negative Entropy-Surprisal product
           score = -mean(H(t) × s(t))  where s(t) = -log p(x_t)
           When entropy AND surprisal are both high → model is lost → NM.
           Members: either low H (certain) or low s (correct) → lower product.

Baselines for comparison:
  - `loss`: -mean(log p)
  - `minkpp_k20`: bottom-20% Z-scored log-probs (Min-K%++ paper)
  - `surp`: mean(lp) - std(lp)

All from 1 forward pass. All unsupervised. Zero overhead beyond Min-K%++.
Usage: Copy-paste into a Kaggle cell.
"""

import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP56: Next-Gen Logit-Only Baselines — Min-K%++ Replacements")
    print("  10 candidates × 1 forward pass × zero overhead")
    print("=" * 70)
    try:
        import transformers, datasets
    except ImportError:
        os.system("pip install -q transformers datasets accelerate scikit-learn pandas numpy huggingface_hub")
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
        print("[HF] Logged in.")
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"\n[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"    dtype={model.dtype}, device={model.device}")
    return model, tokenizer


# ============================================================================
# Core: Extract ALL logit-only scores in a single forward pass
# ============================================================================

class LogitBaselineExtractor:
    """
    Extracts 10 novel logit-only MIA scores + 3 baselines from 1 forward pass.
    Every score: scalar, unsupervised, same compute as Min-K%++.
    """

    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = model.config.vocab_size
        self._err = 0

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Return dict of {score_name: value} for one sample."""
        f = {}
        if not text or len(text) < 20:
            return f

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            if seq_len < 8:
                return f

            # ── Single forward pass ──────────────────────────────────
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, :-1, :].float()   # (T, V)
            labels = input_ids[0, 1:]                      # (T,)
            T = logits.shape[0]
            V = logits.shape[1]
            if T < 5:
                return f

            # ── Shared computations ──────────────────────────────────
            probs = F.softmax(logits, dim=-1)              # (T, V)
            log_probs = F.log_softmax(logits, dim=-1)      # (T, V)

            # Per-token log-prob of the correct next token
            token_lp = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)                                   # (T,)
            tlp = token_lp.cpu().numpy()                    # numpy
            surprisals = -tlp                               # higher = worse

            # Per-position entropy H(t)
            H = -(probs * log_probs).sum(dim=-1)           # (T,)
            H_np = H.cpu().numpy()

            # Per-position probability of correct token
            token_p = probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)                                   # (T,)

            # ── Ranks: position of correct token in sorted logits ────
            # rank = 0 means correct token has highest logit
            # Efficient: count how many logits are strictly higher
            correct_logit = logits.gather(
                dim=-1, index=labels.unsqueeze(-1)
            )                                               # (T, 1)
            ranks = (logits > correct_logit).sum(dim=-1)   # (T,) int
            ranks_np = ranks.cpu().float().numpy()

            # ── Min-K%++ Z-scores (baseline) ─────────────────────────
            mu = (probs * log_probs).sum(dim=-1)
            sigma_sq = (probs * log_probs.pow(2)).sum(dim=-1) - mu.pow(2)
            sigma = sigma_sq.clamp(min=1e-10).sqrt()
            z = ((token_lp - mu) / sigma).cpu().numpy()

            # ==========================================================
            # BASELINES
            # ==========================================================

            # B1: Loss (negative mean log-prob)
            f["loss"] = float(np.mean(tlp))

            # B2: Min-K%++ k=20%
            n20 = max(1, int(T * 0.2))
            f["minkpp_k20"] = float(np.mean(np.sort(z)[:n20]))

            # B3: SURP (mean - std)
            f["surp"] = float(np.mean(tlp) - np.std(tlp))

            # ==========================================================
            # CANDIDATE 1: EWL — Entropy-Weighted Loss
            # ==========================================================
            # Upweight tokens where model is uncertain → where M/NM differ
            H_sum = float(H_np.sum())
            if H_sum > 1e-8:
                f["ewl"] = float(np.sum(H_np * tlp) / H_sum)

            # ==========================================================
            # CANDIDATE 2: MLR — Mean Log-Rank
            # ==========================================================
            # Ordinal metric immune to logit calibration
            f["mlr"] = float(-np.mean(np.log2(ranks_np + 1)))

            # ==========================================================
            # CANDIDATE 3: QCL — Quantile-Calibrated Loss
            # ==========================================================
            # At each position, compute the CDF percentile of the correct
            # token's log-prob within that position's full distribution.
            # Instead of Z-score (Gaussian assumption), use the empirical
            # fraction of vocab tokens with LOWER log-prob than correct token.
            # quantile = (V - rank) / V  ∈ [0, 1]
            quantiles = (V - ranks_np) / V
            f["qcl"] = float(np.mean(quantiles))

            # Also: bottom-20% of quantiles (analogous to Min-K% but CDF-based)
            n20 = max(1, int(T * 0.2))
            f["qcl_k20"] = float(np.mean(np.sort(quantiles)[:n20]))

            # ==========================================================
            # CANDIDATE 4: DCG — Discounted Cumulative Gain
            # ==========================================================
            # IR-inspired: 1/log₂(rank+2) gives high value for rank 0,
            # decays proportionally. Mean DCG = how well-ranked on average.
            dcg = 1.0 / np.log2(ranks_np + 2)
            f["dcg"] = float(np.mean(dcg))

            # ==========================================================
            # CANDIDATE 5: GAL — Gated Loss (entropy-thresholded)
            # ==========================================================
            # Only grade on uncertain tokens (H > median) — ignore easy tokens
            # where both M and NM perform well
            H_med = float(np.median(H_np))
            mask_uncertain = H_np > H_med
            if mask_uncertain.sum() > 3:
                f["gal_hi"] = float(np.mean(tlp[mask_uncertain]))
            # Also: only confident tokens (H < median)
            mask_confident = ~mask_uncertain
            if mask_confident.sum() > 3:
                f["gal_lo"] = float(np.mean(tlp[mask_confident]))

            # ==========================================================
            # CANDIDATE 6: ESP — Entropy Slope
            # ==========================================================
            # Regression slope of entropy vs position
            # Members: entropy decreases faster → more negative slope
            positions = np.arange(T, dtype=np.float64)
            pm, hm = positions.mean(), H_np.mean()
            cov_ph = np.sum((positions - pm) * (H_np - hm))
            var_p = np.sum((positions - pm) ** 2)
            if var_p > 1e-10:
                slope = cov_ph / var_p
                f["esp"] = float(-slope)  # negate: faster decrease = higher score

            # Pearson correlation version (scale-invariant)
            h_std = np.std(H_np)
            if h_std > 1e-8 and var_p > 1e-10:
                pearson = cov_ph / (np.sqrt(var_p) * h_std * T)
                f["esp_corr"] = float(-pearson)

            # ==========================================================
            # CANDIDATE 7: MAX — Normalized Max-Rank
            # ==========================================================
            # Members' WORST prediction still well-ranked; NM have outliers
            f["max_rank"] = float(-(ranks_np.max()) / V)

            # Also: 95th percentile of rank (robust to single outlier)
            f["p95_rank"] = float(-(np.percentile(ranks_np, 95)) / V)

            # ==========================================================
            # CANDIDATE 8: FCS — Fractional Confidence Score
            # ==========================================================
            # Fraction of tokens where p(correct) > threshold
            token_p_np = token_p.cpu().numpy()
            for thresh_name, thresh in [("fc01", 0.1), ("fc001", 0.01), ("fc0001", 0.001)]:
                f[f"fcs_{thresh_name}"] = float(np.mean(token_p_np > thresh))

            # Top-1 accuracy (special case: p(correct) = max(p))
            predicted = logits.argmax(dim=-1).cpu()
            f["fcs_top1"] = float((predicted == labels.cpu()).float().mean().item())

            # ==========================================================
            # CANDIDATE 9: KLD — KL from Uniform
            # ==========================================================
            # log(V) + log p(x_t) = log(V · p(x_t))
            # How much better than random the model does at each position
            log_v = np.log(V)
            f["kld"] = float(np.mean(tlp + log_v))

            # Bottom-K version
            kld_scores = tlp + log_v
            n20 = max(1, int(T * 0.2))
            f["kld_k20"] = float(np.mean(np.sort(kld_scores)[:n20]))

            # ==========================================================
            # CANDIDATE 10: NES — Negative Entropy-Surprisal Product
            # ==========================================================
            # H(t) × s(t): both high when model is clueless (NM)
            # Members: either low entropy (certain) or low surprisal (correct)
            product = H_np * surprisals
            f["nes"] = float(-np.mean(product))

            # Also: just the worst (max product) positions
            n20 = max(1, int(T * 0.2))
            f["nes_worst20"] = float(-np.mean(np.sort(product)[-n20:]))

            # ==========================================================
            # BONUS: Combined / derived signals
            # ==========================================================

            # SURP-rank: rank-based SURP analog
            log_ranks = np.log2(ranks_np + 1)
            f["surp_rank"] = float(-(np.mean(log_ranks) + np.std(log_ranks)))

            # Entropy-weighted rank (EWR): EWL but for ranks
            if H_sum > 1e-8:
                f["ewr"] = float(-np.sum(H_np * np.log2(ranks_np + 1)) / H_sum)

            # Loss slope (trajectory): members improve more across sequence
            cov_pl = np.sum((positions - pm) * (tlp - np.mean(tlp)))
            if var_p > 1e-10:
                f["loss_slope"] = float(cov_pl / var_p)

            # Per-language calibration marker (the language ID is NOT used here;
            # the caller applies z-norm per language after extraction)
            f["seq_len"] = float(seq_len)

        except Exception as e:
            if self._err < 3:
                print(f"\n[WARN] {type(e).__name__}: {e}")
            self._err += 1

        return f


# ============================================================================
# Experiment Runner
# ============================================================================

class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self) -> pd.DataFrame:
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        print(f"\n[*] Loading dataset: {self.args.dataset} (local={is_local})")
        for subset in subsets:
            try:
                if is_local:
                    path = os.path.join(self.args.dataset, subset)
                    if not os.path.exists(path):
                        continue
                    ds = load_from_disk(path)
                    if hasattr(ds, "keys") and "test" in ds.keys():
                        ds = ds["test"]
                else:
                    ds = load_dataset(self.args.dataset, subset, split="test")
                sub_df = ds.to_pandas()
                sub_df["subset"] = subset
                dfs.append(sub_df)
                print(f"    {subset}: {len(sub_df)} samples")
            except Exception as e:
                print(f"    [WARN] {subset}: {e}")
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)

        if self.args.sample_fraction < 1.0:
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
        print(f"[*] Total: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        print(f"    Members: {df['is_member'].sum()} | Non-members: {(~df['is_member'].astype(bool)).sum()}")
        return df

    def per_language_znorm(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Per-language Z-normalization (proven +0.012 from EXP41)."""
        print("\n[LangCal] Applying per-language Z-normalization...")
        new_cols = []
        for col in cols:
            if col not in df.columns or col == "seq_len":
                continue
            zcol = f"z_{col}"
            df[zcol] = np.nan
            for subset in df["subset"].unique():
                mask = df["subset"] == subset
                vals = df.loc[mask, col].dropna()
                if len(vals) < 10:
                    continue
                mu, sig = vals.mean(), vals.std()
                if sig > 1e-8:
                    df.loc[mask, zcol] = (df.loc[mask, col] - mu) / sig
            new_cols.append(zcol)
        print(f"    Created {len(new_cols)} z-normed columns")
        return df, new_cols

    def run(self):
        import time
        df = self.load_data()
        ext = LogitBaselineExtractor(
            self.model, self.tokenizer, self.args.max_length
        )

        # ── Extract features ─────────────────────────────────────────
        print(f"\n[EXP56] Extracting logit-only scores (10 candidates + 3 baselines)...")
        t0 = time.time()
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP56]"):
            rows.append(ext.extract(row["content"]))

        feat_df = pd.DataFrame(rows)
        score_cols = [c for c in feat_df.columns if c != "seq_len"]
        for col in feat_df.columns:
            df[col] = feat_df[col].values

        elapsed = time.time() - t0
        valid = df[score_cols[0]].notna().sum() if score_cols else 0
        print(f"\n[EXP56] Extraction done: {elapsed/60:.1f} min, {valid}/{len(df)} valid")

        # ── Per-language z-normalization ──────────────────────────────
        key_signals = ["loss", "minkpp_k20", "surp", "ewl", "mlr", "qcl",
                        "dcg", "esp", "nes", "kld"]
        df, z_cols = self.per_language_znorm(df, key_signals)
        all_score_cols = score_cols + z_cols

        # ── Evaluate ALL scores ──────────────────────────────────────
        print("\n" + "=" * 70)
        print("   EXP56: LOGIT-ONLY BASELINE LEADERBOARD")
        print("=" * 70)

        results = {}
        for col in sorted(all_score_cols):
            if col not in df.columns:
                continue
            v = df.dropna(subset=[col])
            if len(v) < 50 or v["is_member"].nunique() < 2:
                continue
            auc_pos = roc_auc_score(v["is_member"], v[col])
            auc_neg = 1.0 - auc_pos
            best = max(auc_pos, auc_neg)
            d = "+" if auc_pos >= 0.5 else "-"
            results[col] = (best, d, auc_pos)

        # ── Leaderboard ──────────────────────────────────────────────
        ranked = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

        # Split baselines vs candidates
        baseline_names = {"loss", "minkpp_k20", "surp", "z_loss", "z_minkpp_k20", "z_surp"}

        print(f"\n  {'Rank':<5} {'Signal':<25} {'AUC':>8}  {'Dir':>3}  Category")
        print("  " + "─" * 60)
        for i, (col, (auc, d, _)) in enumerate(ranked):
            cat = "BASELINE" if col in baseline_names else "★ NEW"
            marker = " ◀ Min-K%++ baseline" if col == "minkpp_k20" else ""
            marker = " ◀ Mean loss" if col == "loss" else marker
            marker = " ◀ SURP" if col == "surp" else marker
            print(f"  {i+1:<5} {d}{col:<24} {auc:>8.4f}  {d:>3}  {cat}{marker}")

        # ── How many candidates BEAT Min-K%++? ───────────────────────
        minkpp_auc = results.get("minkpp_k20", (0.5, "+", 0.5))[0]
        loss_auc = results.get("loss", (0.5, "+", 0.5))[0]
        surp_auc = results.get("surp", (0.5, "+", 0.5))[0]

        beats_minkpp = [(c, v) for c, v in ranked
                        if c not in baseline_names and v[0] > minkpp_auc]
        beats_loss = [(c, v) for c, v in ranked
                      if c not in baseline_names and v[0] > loss_auc]
        beats_surp = [(c, v) for c, v in ranked
                      if c not in baseline_names and v[0] > surp_auc]

        print(f"\n  ═══ COMPARISON WITH ESTABLISHED BASELINES ═══")
        print(f"  Min-K%++ (k=20%):  {minkpp_auc:.4f}")
        print(f"  Mean loss:         {loss_auc:.4f}")
        print(f"  SURP (mean-std):   {surp_auc:.4f}")
        print(f"  ──────────────────────────────────")
        print(f"  Candidates beating Min-K%++:      {len(beats_minkpp)}")
        print(f"  Candidates beating mean loss:     {len(beats_loss)}")
        print(f"  Candidates beating SURP:          {len(beats_surp)}")

        # ── Best overall → per-subset breakdown ──────────────────────
        if ranked:
            best_col = ranked[0][0]
            best_auc = ranked[0][1][0]
            best_d = ranked[0][1][1]
            print(f"\n  ═══ BEST SIGNAL: {best_d}{best_col} = {best_auc:.4f} ═══")
            print(f"  Per-subset breakdown:")
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset].dropna(subset=[best_col])
                if len(sub) > 10 and sub["is_member"].nunique() > 1:
                    vals = sub[best_col] if best_d == "+" else -sub[best_col]
                    auc = roc_auc_score(sub["is_member"], vals)
                    print(f"    {subset:<10} AUC = {auc:.4f}")

            # Per-subset for top-3
            print(f"\n  ═══ TOP-3 PER-SUBSET COMPARISON ═══")
            top3 = ranked[:3]
            header = f"  {'Subset':<10}" + "".join(f" {c[1][1]}{c[0]:<20}" for c in top3)
            print(header)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                parts = [f"  {subset:<10}"]
                for col, (_, d, _) in top3:
                    s = sub.dropna(subset=[col])
                    if len(s) > 10 and s["is_member"].nunique() > 1:
                        vals = s[col] if d == "+" else -s[col]
                        auc = roc_auc_score(s["is_member"], vals)
                        parts.append(f" {auc:<20.4f}")
                    else:
                        parts.append(f" {'N/A':<20}")
                print("".join(parts))

        # ── Family champion comparison ───────────────────────────────
        families = {
            "Baselines": ["loss", "minkpp_k20", "surp"],
            "Entropy-Weighted": ["ewl", "z_ewl"],
            "Rank-Based": ["mlr", "qcl", "qcl_k20", "dcg", "z_mlr", "z_qcl", "z_dcg"],
            "Gated": ["gal_hi", "gal_lo"],
            "Trajectory": ["esp", "esp_corr", "loss_slope", "z_esp"],
            "Worst-Case": ["max_rank", "p95_rank"],
            "Threshold": ["fcs_fc01", "fcs_fc001", "fcs_fc0001", "fcs_top1"],
            "KL-Based": ["kld", "kld_k20", "z_kld"],
            "Product": ["nes", "nes_worst20", "z_nes"],
            "Hybrid": ["surp_rank", "ewr"],
        }

        print(f"\n  ═══ FAMILY CHAMPIONS ═══")
        for fam, cols in families.items():
            fam_results = [(c, results[c]) for c in cols if c in results]
            if fam_results:
                best_in_fam = max(fam_results, key=lambda x: x[1][0])
                c, (a, d, _) = best_in_fam
                delta = a - minkpp_auc
                marker = f"+{delta:.4f} vs MinK++" if delta > 0 else f"{delta:.4f} vs MinK++"
                print(f"    {fam:<20} → {d}{c:<22} = {a:.4f}  ({marker})")

        # ── Final verdict ────────────────────────────────────────────
        print(f"\n  ═══ VERDICT ═══")
        if ranked:
            bc, (ba, bd, _) = ranked[0]
            delta_mk = ba - minkpp_auc
            delta_loss = ba - loss_auc
            delta_surp = ba - surp_auc
            print(f"  Best logit-only score: {bd}{bc} = {ba:.4f}")
            print(f"    vs Min-K%++:  {'+' if delta_mk > 0 else ''}{delta_mk:.4f}")
            print(f"    vs Mean loss: {'+' if delta_loss > 0 else ''}{delta_loss:.4f}")
            print(f"    vs SURP:      {'+' if delta_surp > 0 else ''}{delta_surp:.4f}")
            if delta_mk > 0.01:
                print(f"  ✅ RECOMMENDED: Replace Min-K%++ with {bd}{bc} as baseline")
            elif delta_mk > 0:
                print(f"  ⚠️  Marginal improvement over Min-K%++")
            else:
                print(f"  ❌ No candidate beats Min-K%++")

        # ── Known tracker results for context ────────────────────────
        print(f"\n  ═══ TRACKER CONTEXT (known results) ═══")
        print(f"  Logit-only family:")
        print(f"    Min-K%++ (EXP02):      0.5770")
        print(f"    Mean loss (EXP01):     0.5807")
        print(f"    SURP (EXP16):          0.5884")
        print(f"    CDD entropy slope:     0.6292  ← current logit-only SOTA")
        print(f"  Beyond logit-only:")
        print(f"    Gradient (EXP41):      0.6539")
        print(f"    AttenMIA (EXP43):      0.6642")
        print(f"    memTrace RF (EXP50):   0.6908")
        print(f"    LUMIA-fast (full):     0.7805  ← overall SOTA")

        print("\n" + "=" * 70)

        # ── Save ─────────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"EXP56_logit_baselines_{ts}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\n[EXP56] Results saved to {out_path}")
        print(f"[EXP56] Total errors during extraction: {ext._err}")
        print(f"[EXP56] Done.\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10   # 10% = 10K samples
        max_length = 512
        output_dir = "results"
        seed = 42

    print(f"\n[Config] Model:   {Args.model_name}")
    print(f"[Config] Dataset: {Args.dataset}")
    print(f"[Config] Sample:  {Args.sample_fraction*100:.0f}%")
    print(f"[Config] MaxLen:  {Args.max_length}")

    Experiment(Args).run()
