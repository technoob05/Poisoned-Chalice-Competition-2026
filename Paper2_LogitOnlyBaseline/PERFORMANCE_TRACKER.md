# Paper 2 — ESP-Cal Performance Tracker

> Last updated: **2026-03-05**  
> Model (PC): `bigcode/starcoder2-3b` · Model (WikiMIA/MIMIR): `EleutherAI/pythia-2.8b-deduped`  
> GPU: H100 80 GB · Framework: PyTorch 2.8.0+cu126

---

## Poisoned Chalice ★

`sample_fraction=0.10` → 9 999 rows · 245 s · ~41 rows/s

### Overall AUROC (averaged across 5 languages)

| Rank | Method | AUROC | Scale | Notes |
|------|--------|------:|-------|-------|
| 1 | `z_esp_slope` ⭐ | **0.6274** | Scale-1 only | Best single feature |
| 2 | `esp_slope` | 0.6255 | none | Raw slope |
| 3 | `loss_slope` | 0.6252 | none | Very competitive |
| 4 | `surprise_accel` | 0.6021 | none | Novel trajectory signal |
| 5 | `h_curvature` | 0.5994 | none | 2nd-order entropy |
| — | `minkpp_50` | 0.5462 | none | Weak on code |
| — | `signal_esp` (full cal) | 0.5031 | Scale 1+2+3 | ⚠️ Hurts at 10% |

### Per-Language AUROC (`signal_esp`, Scale 1+2+3)

| Go | Java | Python | Ruby | Rust | Avg |
|----|------|--------|------|------|-----|
| 0.5209 | **0.5669** | 0.5204 | 0.5029 | 0.5391 | 0.5031 |

### Ablation (PC, 10%)

| Step | Config | AUROC |
|------|--------|------:|
| (a) raw `esp_slope` | no calibration | 0.6255 |
| **(b) +Scale 1** | z-norm per token | **0.6274** ← best |
| (c) +Scale 2+3 | pos-bucket + domain | 0.5031 ⚠️ |
| (d) Min-K%++ baseline | — | 0.5218 |
| (e) Loss baseline | — | 0.5065 |

> **Finding**: Scale 2+3 hurts at 10% due to insufficient data for stable bucket statistics.  
> **Hypothesis**: Will recover at `sample_fraction=1.0` (full ~100k rows/language).

---

## WikiMIA

Model: `pythia-2.8b-deduped` · No calibration

| Split | n | Best Signal | AUROC |
|-------|---|-------------|------:|
| len-32 | 776 | `minkpp_50` | 0.6437 |
| len-64 | 542 | `minkpp_10` | 0.6454 |
| len-128 | 250 | `signal_mink` (Min-K%) | 0.6617 |
| len-256 | 82 | `signal_mink` (Min-K%) | **0.7008** |

---

## MIMIR

Model: `pythia-2.8b-deduped` · 9 subsets

| Subset | Best Signal | AUROC | Run date |
|--------|-------------|------:|----------|
| arxiv | — | — | — |
| c4 | — | — | — |
| dm_mathematics | — | — | — |
| github | — | — | — |
| hackernews | — | — | — |
| pile_cc | — | — | — |
| pubmed_central | — | — | — |
| temporal_wiki | — | — | — |
| wikipedia | — | — | — |

> ⚠️ MIMIR ran (2026-03-05) but numeric output was truncated in Kaggle log. Re-run needed.

---

## Experiment Scripts Status

| Script | Status | Last run | Notes |
|--------|--------|----------|-------|
| `kaggle_exp_00_Baseline.py` | ✅ ready | — | Loss, Min-K%, Min-K%++, Zlib, SURP |
| `kaggle_exp_01_ESP_NoCal.py` | ✅ ready | — | ESP slopes, no calibration |
| `kaggle_exp_02_ESPCal.py` | ✅ ran | 2026-03-05 | 3-scale calibration + ablation |
| `kaggle_exp_03_SurpriseTraj.py` | ✅ ready | — | Trajectory features (surprise_drop, accel…) |
| `kaggle_exp_04_Combined.py` | ✅ ready | — | All features + 5 ensembles |

---

## Action Items

- [ ] Re-run `kaggle_exp_02_ESPCal.py` with `sample_fraction=1.0` → validate Scale 2+3
- [ ] Try `sample_fraction=0.5` → find crossover point where Scale 2+3 helps
- [ ] Run `kaggle_exp_03_SurpriseTraj.py` → collect trajectory AUROC
- [ ] Run `kaggle_exp_04_Combined.py` → collect ensemble AUROC (esp. `ens_all`)
- [ ] Fix MIMIR output truncation → re-run and fill MIMIR table above
- [ ] Update LaTeX tables in `latex/` with full-run numbers
- [ ] Target: PC AUROC ≥ 0.70 for competition submission

---

## Literature Targets

| Dataset | Reported SOTA | Our best (so far) | Gap |
|---------|---------------|-------------------|-----|
| WikiMIA len-128 | ~0.72 | 0.6617 | −0.058 |
| WikiMIA len-256 | ~0.75 | 0.7008 | −0.049 |
| MIMIR C4 | ~0.69 | — | — |
| Poisoned Chalice | — (competition) | 0.6274 | — |
