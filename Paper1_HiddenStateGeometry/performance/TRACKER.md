# MultiGeo-MIA — Performance Tracker

> **Paper 1**: Multi-Axis Hidden-State Geometry for Membership Inference  
> **Method**: 4-axis unsupervised white-box (magnitude, dimensionality, dynamics, routing)  
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

### BookMIA Models (7)
| # | Model | Params | Status |
|---|-------|--------|--------|
| 1 | `EleutherAI/pythia-2.8b-deduped` | 2.8B | ⬜ |
| 2 | `EleutherAI/pythia-6.9b-deduped` | 6.9B | ⬜ |
| 3 | `EleutherAI/pythia-12b-deduped` | 12B | ⬜ |
| 4 | `EleutherAI/gpt-neo-2.7B` | 2.7B | ⬜ |
| 5 | `EleutherAI/gpt-neox-20b` | 20B | ⬜ |
| 6 | `facebook/opt-6.7b` | 6.7B | ⬜ |
| 7 | `facebook/opt-13b` | 13B | ⬜ |

---

## Benchmark Results

### Poisoned Chalice (Code MIA) — `bigcode/starcoder2-3b`

| Signal | Go | Java | Python | Ruby | Rust | **Avg** |
|--------|---:|-----:|-------:|-----:|-----:|--------:|
| magnitude | — | — | — | — | — | — |
| dimensionality | — | — | — | — | — | — |
| dynamics | — | — | — | — | — | — |
| routing | — | — | — | — | — | — |
| **multigeo_4axis** | — | — | — | — | — | **—** |
| loss (baseline) | — | — | — | — | — | — |

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

### MIMIR — AUROC per model × domain (Avg over 7 domains)

| Model | wikipedia | github | pile_cc | pubmed | arxiv | dm_math | hackernews | **Avg** |
|-------|----------:|-------:|--------:|-------:|------:|--------:|-----------:|--------:|
| pythia-160m-deduped | — | — | — | — | — | — | — | — |
| pythia-410m-deduped | — | — | — | — | — | — | — | — |
| pythia-1.4b-deduped | — | — | — | — | — | — | — | — |
| pythia-2.8b-deduped | — | — | — | — | — | — | — | — |
| pythia-6.9b-deduped | — | — | — | — | — | — | — | — |
| pythia-12b-deduped | — | — | — | — | — | — | — | — |
| gpt-neo-125m | — | — | — | — | — | — | — | — |
| gpt-neo-1.3B | — | — | — | — | — | — | — | — |
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

## Per-Axis Contribution Analysis

| Axis | Description | Signal | Expected Polarity |
|------|-------------|--------|-------------------|
| Magnitude | Mid-layer hidden-state L2 norm | `signal_magnitude = -‖h_mid‖` | Members: lower norm |
| Dimensionality | Effective rank (SVD) of H matrix | `signal_dimensionality = -eff_rank` | Members: lower rank |
| Dynamics | Layer cascade drift (cosine dist) | `signal_dynamics = -mean_drift` | Members: less drift |
| Routing | Attention entropy | `signal_routing = -mean_entropy` | Members: sharper attention |

## Run Log

| Date | Benchmark | Models | GPU | Time | Notes |
|------|-----------|--------|-----|------|-------|
| — | — | — | — | — | _(no runs yet)_ |
