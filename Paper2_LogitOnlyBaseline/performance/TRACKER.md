# ESP-Cal — Performance Tracker

> **Paper 2**: Entropy Slope + Multi-Scale Calibration for MIA  
> **Method**: Grey-box, logit-only, 3-scale z-normalization  
> **Last updated**: _(auto-updated after each run)_

---

## Model Registry

### WikiMIA Models (19)
| # | Model | Family | Params | Status |
|---|-------|--------|--------|--------|
| 1 | `EleutherAI/pythia-160m-deduped` | Pythia | 160M | ⬜ |
| 2 | `EleutherAI/pythia-410m-deduped` | Pythia | 410M | ⬜ |
| 3 | `EleutherAI/pythia-1.4b-deduped` | Pythia | 1.4B | ⬜ |
| 4 | `EleutherAI/pythia-2.8b-deduped` | Pythia | 2.8B | ⬜ |
| 5 | `EleutherAI/pythia-6.9b-deduped` | Pythia | 6.9B | ⬜ |
| 6 | `EleutherAI/pythia-12b-deduped` | Pythia | 12B | ⬜ |
| 7 | `EleutherAI/gpt-neo-125m` | GPT-Neo | 125M | ⬜ |
| 8 | `EleutherAI/gpt-neo-1.3B` | GPT-Neo | 1.3B | ⬜ |
| 9 | `EleutherAI/gpt-neo-2.7B` | GPT-Neo | 2.7B | ⬜ |
| 10 | `EleutherAI/gpt-neox-20b` | GPT-NeoX | 20B | ⬜ |
| 11 | `facebook/opt-125m` | OPT | 125M | ⬜ |
| 12 | `facebook/opt-1.3b` | OPT | 1.3B | ⬜ |
| 13 | `facebook/opt-2.7b` | OPT | 2.7B | ⬜ |
| 14 | `facebook/opt-6.7b` | OPT | 6.7B | ⬜ |
| 15 | `facebook/opt-13b` | OPT | 13B | ⬜ |
| 16 | `facebook/opt-30b` | OPT | 30B | ⬜ |
| 17 | `state-spaces/mamba-130m-hf` | Mamba | 130M | ⬜ |
| 18 | `state-spaces/mamba-1.4b-hf` | Mamba | 1.4B | ⬜ |
| 19 | `state-spaces/mamba-2.8b-hf` | Mamba | 2.8B | ⬜ |

### MIMIR Models (10)
| # | Model | Family | Params | Status |
|---|-------|--------|--------|--------|
| 1–6 | Pythia suite (160M–12B) | Pythia | — | ⬜ |
| 7–10 | GPT-Neo/NeoX (125M–20B) | GPT-Neo | — | ⬜ |

### BookMIA Models (7)
| # | Model | Params | Status |
|---|-------|--------|--------|
| 1–7 | Pythia + GPT-Neo + OPT | 2.8B–20B | ⬜ |

---

## Benchmark Results

### Poisoned Chalice (Code MIA) — `bigcode/starcoder2-3b`

| Signal | Go | Java | Python | Ruby | Rust | **Avg** |
|--------|---:|-----:|-------:|-----:|-----:|--------:|
| **signal_esp** (Ours) | — | — | — | — | — | **—** |
| signal_h_drop | — | — | — | — | — | — |
| signal_loss | — | — | — | — | — | — |
| neg_mean_loss | — | — | — | — | — | — |
| minkprob_20 | — | — | — | — | — | — |
| surp | — | — | — | — | — | — |
| zlib_ratio | — | — | — | — | — | — |

### Ablation: Calibration Scales (Poisoned Chalice)

| Condition | AUROC |
|-----------|------:|
| (a) ESP raw — no calibration | — |
| (b) + Scale 1 (token z-norm) | — |
| (c) + Scale 2 (position z-norm) | — |
| (d) + Scale 3 (domain z-norm) = **ESP-Cal** | **—** |

### WikiMIA — AUROC per model × length

| Model | len32 | len64 | len128 | len256 | **Avg** |
|-------|------:|------:|-------:|-------:|--------:|
| pythia-160m-deduped | — | — | — | — | — |
| pythia-410m-deduped | — | — | — | — | — |
| pythia-1.4b-deduped | — | — | — | — | — |
| pythia-2.8b-deduped | — | — | — | — | — |
| pythia-6.9b-deduped | — | — | — | — | — |
| pythia-12b-deduped | — | — | — | — | — |
| gpt-neo-125m | — | — | — | — | — |
| gpt-neo-1.3B | — | — | — | — | — |
| gpt-neo-2.7B | — | — | — | — | — |
| gpt-neox-20b | — | — | — | — | — |
| opt-125m | — | — | — | — | — |
| opt-1.3b | — | — | — | — | — |
| opt-2.7b | — | — | — | — | — |
| opt-6.7b | — | — | — | — | — |
| opt-13b | — | — | — | — | — |
| opt-30b | — | — | — | — | — |
| mamba-130m-hf | — | — | — | — | — |
| mamba-1.4b-hf | — | — | — | — | — |
| mamba-2.8b-hf | — | — | — | — | — |

### MIMIR — AUROC per model × domain

| Model | wikipedia | github | pile_cc | pubmed | arxiv | dm_math | hackernews | **Avg** |
|-------|----------:|-------:|--------:|-------:|------:|--------:|-----------:|--------:|
| pythia-160m-deduped | — | — | — | — | — | — | — | — |
| pythia-2.8b-deduped | — | — | — | — | — | — | — | — |
| pythia-6.9b-deduped | — | — | — | — | — | — | — | — |
| pythia-12b-deduped | — | — | — | — | — | — | — | — |
| gpt-neo-2.7B | — | — | — | — | — | — | — | — |
| gpt-neox-20b | — | — | — | — | — | — | — | — |

### BookMIA — AUROC per model

| Model | AUROC | Status |
|-------|------:|--------|
| pythia-2.8b-deduped | — | ⬜ |
| pythia-6.9b-deduped | — | ⬜ |
| pythia-12b-deduped | — | ⬜ |
| gpt-neo-2.7B | — | ⬜ |
| gpt-neox-20b | — | ⬜ |
| opt-6.7b | — | ⬜ |
| opt-13b | — | ⬜ |

---

## Method Details

### ESP Score Computation
```
For each token t:
    H(t) = -Σ_v p(v|x_{<t}) log p(v|x_{<t})    (per-token entropy)
ESP score = slope(H(t) vs t)                       (linear fit)
Members → steeper decline → more negative slope
```

### Multi-Scale Calibration
| Scale | What | How |
|-------|------|-----|
| 1 (Token) | Per-sample z-norm of entropy | `z = (H - μ_H) / σ_H` |
| 2 (Position) | By sequence-length bucket | `z = (score - μ_bucket) / σ_bucket` |
| 3 (Domain) | Per language/domain | `z = (score - μ_domain) / σ_domain` |

---

## Run Log

| Date | Benchmark | Models | GPU | Time | Notes |
|------|-----------|--------|-----|------|-------|
| — | — | — | — | — | _(no runs yet)_ |
