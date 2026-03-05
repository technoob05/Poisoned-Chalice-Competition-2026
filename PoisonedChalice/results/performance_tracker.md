# Poisoned Chalice 2026 - Performance Tracker

**Model**: `bigcode/starcoder2-3b`
**Dataset**: `AISE-TUDelft/Poisoned-Chalice` (Sample 0.1)

---

## Experiment Results

### Tracker Sync (2026-03-05)

- **Pending AUC (chưa có điểm)**: `EXP10`, `EXP15`, `EXP19`, `EXP20`, `EXP25`, `EXP54`
- **Placeholder cũ cần cleanup**: `EXP51` có 1 dòng `-` cũ, nhưng đã có kết quả chính thức **0.7338** ở phía dưới.
- **Coverage**: 50/56 EXP đã có điểm AUC trong tracker.

| Experiment | Method | AUC Score | Notes |
| :--- | :--- | :--- | :--- |
| **EXP01** | Multi-Baseline (MK/Loss) | **0.5852** | Loss: 0.5807, MKP: 0.5852, PAC: 0.4582. |
| **EXP02** | Min-K%++ (SOTA 2025) | **0.5770** | Baseline Z-score normalization. |
| **EXP03** | Neighborhood Loss (RMIA) | **0.5633** | Self-perturbation calibration. |
| **EXP04** | Structural Divergence | **0.4970** | SECT Rename-Loss Diff. |
| **EXP05** | SIA-TTS (Gradient) | **0.4938** | Brittle Influence check. |
| **EXP06** | **Funnel (Cascading)** | **0.5790** | Multi-stage logic. |
| **EXP07** | Adv Stability | **0.5365** | Context Perturbation. |
| **EXP08** | Graph Invariance | **0.4886** | Topological Reordering. |
| **EXP09** | Canonical Gap | **0.4960** | Aggressive normalization. |
| **EXP10** | **ENSEMBLE SOTA** | - | Rank-Averaging (Wait for all). |
| **EXP11** | Gradient Norm | **0.6472** | White-box Gradient Magnitude. Embed layer dominates. |
| **EXP12** | Hessian Trace | **0.5000** | White-box Local Curvature (Baseline signal). |
| **EXP13** | Layer Grads | **0.6446** | High Signal at specific blocks (Trajectory). |
| **EXP14** | Internal Entropy | **0.4397** | Attn Entropy (Weak standalone, good feature). |
| **EXP15** | **XGBOOST STACKER** | - | **Final Master Ensemble (Pending)**. |
| **EXP16** | SURP (Calibration) | **0.5884** | Mean - Std log-prob (Solid baseline+ signal). |
| **EXP17** | ICP-MIA-SP (Random Mask) | **0.4330** | ⚠️ **INVERSE SIGNAL** - score direction wrong. |
| **EXP18** | Hybrid Optimization Gap | **0.5514** | ICP: 0.4332, GradNorm: 0.6421, SURP: 0.5882. |
| **EXP19** | BUZZER-Style CMI | - | Multi-signal + hard-sample calibration + LR. |
| **EXP20** | BUZZER+ (ICP + GradNorm) | - | BUZZER signals + ICP + grad norm. |
| **EXP21** | Gradient Sliding Window | **0.5732** | Chunk-based gradient norm (256 tokens/window). |
| **EXP22** | Selective Layer Gradient | **0.6337** | Ensemble AUC. Layer AUCs: embedding 0.6423, early 0.6181, middle 0.6250, late 0.6204, head 0.6423. |
| **EXP23** | Gradient Direction Variance | **0.4852** | 5% sample, 5 passes with dropout. Stability (M): 0.2111, (NM): 0.2193. AUC near random — use as feature for EXP15. |
| **EXP24** | Perturbed Gradient Stability | **0.4455** | 10% sample, 4 perturbations, 5% token substitution. Score = -CV. Mean CV (M): 0.1160, (NM): 0.1068 — signal weak/inverted; use as feature for EXP15. |
| **EXP25** | Cross-layer Correlation | - | Pairwise gradient correlation between layers. |
| **EXP26** | Early-Settling Ratio (ESR) | **0.4371** | JSD(p_L ∥ p_l) at 5 layers [5,10,15,20,29]. Subset: Go 0.375, Java 0.422, Python 0.438, Ruby 0.401, Rust 0.514. Per-layer AUC: L5 0.432, L10 0.447, L15 0.461, L20 0.531, L29 0.50. JSD_early (M)≈(NM) ~0.67 — signal weak; use as feature for EXP15. |
| **EXP27** | JSD-Weighted Gradient Norm | **0.6484** | 5% sample, single forward+backward pass. **Product -(grad×jsd) = 0.6484 ← BEST**; Rank-avg 0.5845; -grad_embed 0.6480; -jsd_early 0.4371. Subset CombinedRank: Go 0.6015, Java 0.5733, Python 0.5956, Ruby 0.5610, Rust 0.5919. Subset GradEmbed: Go 0.7097, Java 0.6272, Python 0.6739, Ruby 0.6891, Rust 0.5811. Gradient dominates; product > rank-avg when combining with weak JSD. |
| **EXP28** | Delta JSD Trajectory | **0.5107** | 5% sample, 2 forward passes, deterministic variable renaming on context (first 50%). Primary delta_jsd_early AUC 0.5107; -jsd_early_orig 0.4373 (EXP26 baseline). Subset Delta AUC: Python 0.5306, Rust 0.5156, Go 0.5079, Java 0.5004, Ruby 0.4893. **Δ(M) ≈ Δ(NM) ≈ 0** (e.g. Go −0.0005 vs −0.0005) — **hypothesis NOT confirmed**; renaming does not create discriminative Δ. Valid 4998/5000. |
| **EXP29** | Attention Early-Settling | **0.3840** | 5% sample, max_seq 512, attn_implementation=eager. AttenMIA × ESR: combined 0.3840, -entropy_early 0.3805, conv_early 0.4004. Subset: Java 0.4264, Rust 0.4204, Python 0.3749, Go 0.3581, Ruby 0.3373. ⚠️ **AUC < 0.5 — signal inverted** (low entropy/high conv currently = non-member); flip sign before EXP15. |
| **EXP30** | Privacy-Vulnerable Layer Gradient | **0.6456** | 5% sample, 200-probe (100M+100NM), top-5 PVC components data-driven (91 total). **PVC Mask: layer_28_norm, layer_29_norm, layer_24_norm, layer_29_mlp, layer_29_attn** (all late layers 24-29). Subset: Go 0.7052, Ruby 0.6901, Python 0.6743, Java 0.6153, Rust 0.5821. PVC Norm M vs NM: Go 0.313 vs 0.510, Python 0.362 vs 0.662 (1.83x), Ruby 0.494 vs 0.895 (1.81x), Java 0.473 vs 0.632, Rust 0.356 vs 0.516. Beats EXP22 heuristic (+0.012); near EXP11 embed-only (-0.002). Norm weights ranked above MLP/attn. All pvc_* for EXP15. |
| **EXP31** | Tail-Loss Delta (Embedding Noise) | **0.5089** | 10% sample (10K), 3 noise levels [0.05, 0.10, 0.20], 20 min A100. combined_rank AUC 0.4270; score_delta (Δ_rel@10%) 0.5056; mean_delta **0.5089**; **input_grad_norm 0.3835 — SIGN BUG** (raw norm, should negate → flipped ~0.6165). Per-noise: n=0.05 AUC 0.4973, n=0.10 AUC 0.5056, n=0.20 AUC 0.5164. Δ(M)≈0.0003 ≈ Δ(NM)≈0.0003 — **no separation**. Loss: M 0.8068 vs NM 0.9681 (✓ memorization confirmed). Subset (combined_rank): Go 0.4163, Java 0.4551, Python 0.3896, Ruby 0.4060, Rust 0.4567. ⚠️ **Δ_rel hypothesis NOT confirmed**: Gaussian embedding noise does not create discriminative loss perturbation on code LLMs. Only -input_grad_norm useful (~0.62, redundant with EXP11). |
| **EXP32** | OR-MIA Gradient Stability | **0.6165** | 10% sample (10K), noise [0.01, 0.05], 3 trials/level, 7 bwd/sample, 63 min A100. 100% valid. combined_rank 0.5824; **-G_orig 0.6165**; -ΔG 0.5130; -mean_ΔG_rel 0.4451. Per-noise: n=0.01 AUC 0.5130 ΔG(M)=0.095 vs ΔG(NM)=0.089; n=0.05 AUC 0.5187 ΔG(M)=0.128 vs ΔG(NM)=0.142. G_orig: M **1.24** vs NM **2.12** (✓ flat minima). Subset (G_norm): Ruby 0.6766, Python 0.6580, Go 0.6448, Java 0.5748, Rust 0.5666. Subset (combined): Ruby 0.6486, Python 0.5983, Go 0.5849, Java 0.5589, Rust 0.5348. ⚠️ **ΔG stability near random** (0.5130) — perturbation hypothesis FAILS again. Only -G_orig useful (0.6165, consistent with EXP11). Rank-avg with weak ΔG degrades signal (0.6165 → 0.5824). **5/5 perturbation methods now confirmed failed.** |
| **EXP33** | CAMIA Context-Aware Loss Dynamics | **0.6065** | 10% sample, 8×256-token blocks, forward-only. Combined rank 0.6065; MDM 0.4803, TVar 0.4888, AUCG 0.5019. Subset: Go 0.6515, Python 0.6351, Java 0.5905, Rust 0.5875, Ruby 0.5755. Sub-signals weak alone → rank_avg captures synergy (+0.11 gain). |
| **EXP34** | Block-wise Privacy Signature | **0.5608** | 10% sample, 32 components (embed + 30 layers + head). Combined rank 0.5608; signal_g_mean 0.6383, CBC 0.5498, entropy 0.4448, RPS 0.4918, IBVR 0.4635. Top layers: gnorm_embed/head 0.6422, layer_29 0.6407, layer_28 0.6303. g_mean (M 0.80 vs NM 1.24). All gnorm_* for EXP15. |
| **EXP35** | GradSim (Gradient Profile LDA) | **0.6417** | 5% sample (5K), probe 200 (100M+100NM), 31 gradient components (embed+29 blocks+head), ~7 min A100. Valid: 5000/5000 (100%). Probe fit LDA 0.7632, LR 0.6397 (probe overfitting on LDA). Inference: **-GradMean 0.6417** (best); LR 0.6415; Cosine 0.6370; Rank-avg 0.6295; LDA ⚠️ **0.5573** (LDA collapses on inference). Subset (-GradMean): Go 0.6987, Ruby 0.6924, Python 0.6734, Java 0.6090, Rust 0.5758. Subset (LR): Go 0.6884, Ruby 0.6790, Python 0.6636, Java 0.6290, Rust 0.5775. ⚠️ **LDA severely overfits probe** (0.7632 probe → 0.5573 inference): 31-dim features fitted on only 200 samples — curse of dimensionality. LR generalizes better (0.6415) but equals -GradMean (0.6417). Cosine to centroid (0.6370) ≈ LR — member centroid ~= -GradMean direction. Gradient profile shape adds NO information beyond mean magnitude. Consistent with EXP38 finding: learned gradient weighting ≤ single -grad_embed. |
| **EXP36** | CodeMIF (Token Memorization Fingerprint) | **0.5780** | 10% sample (10K), forward-only, conf_thr=0.5, window=32, ~3.7 min A100. Valid: 9997/10000 (100%). Combined rank **0.5780**; -mean_loss **0.5811** (best single); SURP 0.5289; %confident 0.5661. ⚠️ Anchor signals INVERTED: anchor_body_gap 0.4521, -anchor_ratio 0.4675, min_loss_window 0.3853. Subset: Ruby 0.6131, Python 0.5944, Go 0.5806, Java 0.5771, Rust 0.5335. Anchor-body gap stats (M vs NM): Go 0.069 vs 0.098, Java 0.373 vs 0.474, Python 0.216 vs 0.285, Ruby 0.494 vs 0.742, Rust 0.060 vs 0.097. ⚠️ **Hypothesis INVERTED**: NM has a LARGER anchor-body gap than M across all languages. Members already have low loss on ALL tokens (anchors AND body), so the gap shrinks. Non-members have moderate anchor loss but very high body loss → larger gap. Combined rank hurts by including inverted signals. Only -mean_loss (0.5811) and %confident (0.5661) useful; anchor signals should be EXCLUDED or flipped. |
| **EXP37** | CAMIA-D (Loss Cliff Derivatives) | **0.5664** | 10% sample (10K), forward-only, smooth_window=32, cliff_thr=0.3, ~3 min A100. Valid: 9566/10000 (95.7%). Combined rank **0.5664**; -mean_loss 0.5556; cliff_count 0.5449; -d1_min 0.5394; max_cliff_mag 0.5388. Weak signals: -slope 0.4249, early_late_ratio 0.4624 (inverted). Subset: Python 0.5944, Ruby 0.5950, Go 0.5761, Rust 0.5552, Java 0.5226. Cliff stats (M vs NM): max_cliff_mag 0.3448 vs 0.3259, cliff_count 3.78 vs 3.30, cliff_position_first 0.247 vs 0.286 (earlier for M ✔). slope_overall M: −0.918 vs NM: −1.158 (NM drops faster — inverted!). ⚠️ **Hypothesis partially confirmed but signal too weak**: Cliffs ARE earlier and bigger for members, but gap is small (~0.03 in both metrics). Slope inverted (NM drops faster from high initial loss). Cliff-based signals near CAMIA baseline (0.5664 vs 0.5556 mean_loss). Derivatives add ~0.01 over raw mean_loss. Not worth the complexity. |
| **EXP38** | MasterStack (XGBoost Ensemble) | **0.6347** | 10% sample (10K), probe 400 (200M+200NM), 18 features, 5-fold CV, ~10 min A100. XGB Stacker **0.6347**; -grad_embed single **0.6422** (stacker LOSES to single feature); SURP 0.5289; MinK++ 0.5871. CV Mean: 0.5470 ± 0.0609 (high variance, fold range 0.455–0.601). Auto-flipped 11/18 features: mean_loss, std_loss, loss_p10/50/90, minkpp, camia_mdm, camia_tvar, grad_embed, grad_L28, grad_L29. Top importances: seq_len 0.089, seq_len_norm 0.077, grad_embed [F] 0.069, std_loss [F] 0.066, grad_L28 [F] 0.062. Subset XGB: Ruby 0.6593, Go 0.6494, Java 0.6436, Python 0.6343, Rust 0.5961. Subset GradEmbed: Go 0.6920, Python 0.6712, Ruby 0.6848, Java 0.6250, Rust 0.5928. ⚠️ **XGB stacker UNDERPERFORMS single -grad_embed (0.6347 vs 0.6422)**. Root causes: (1) probe_size=400 too small for 18-feature XGB; (2) seq_len dominates importance — spurious length correlation; (3) auto-flip on 400 samples unreliable. → Fix: probe_size ≥ 1000, drop seq_len features, use Ridge LR or LightGBM with higher regularization. |
| **EXP39** | Focused Ridge Stacker | **0.6490** | 10% sample (10K), probe 1000 (500M+500NM), 8 features, 5-fold CV, Ridge LR C=0.1, ~12 min A100. Valid: 10000/10000. Ridge Stacker **0.6490**; -grad_embed single 0.6422 (+0.0068 gain); product-(grad×loss) 0.6215; -mean_loss 0.5811; SURP 0.5289. CV Mean: **0.6453 ± 0.0162** (much more stable than EXP38's 0.547±0.061). Ridge coefficients (top): neg_grad_embed +0.6284 (dominant), neg_grad_L28 −0.3173 (partial cancellation), neg_grad_L29 +0.1887, surp +0.1424, neg_minkpp +0.1018, product +0.0854, neg_mean_loss −0.0126, neg_grad_head +0.0000. Subset Ridge / GradEmbed: Go 0.6902/0.6920 (−0.002); Java **0.6359/0.6250 (+0.011)**; Python **0.6821/0.6712 (+0.011)**; Ruby 0.6861/0.6847 (+0.001); Rust **0.6005/0.5928 (+0.008)**. ✅ Ridge stacker BEATS single -grad_embed globally (+0.0068). ⚠️ neg_grad_L28 gets NEGATIVE coefficient (−0.317) — L28 partially redundant/anti-correlated with embed at this regularization level; Ridge pushes it down. neg_grad_head ≈ 0.0000: head gradient adds no signal beyond embed. Java and Python show largest gains (+0.011) — ensemble helps where single gradient is weakest. Go is the only subset where single grad barely edges ensemble (0.6920 vs 0.6902 −0.002, within noise). → Fix EXP38 validated: fewer features + larger probe + linear model = stable, positive gain. → L28 should be dropped or constraints relaxed; try C=0.3 next. |
| **EXP40** | Generalized Product Scores | **0.6423** | 10% sample (10K), fwd+bwd, ~11 min A100. Baseline -grad_embed **0.6423** (best overall); -(embed×L29) 0.6418; -(grad×camia_mdm) 0.6366; -(grad×camia_tvar) 0.6364; -(L28×L29) 0.6359; -(grad×loss) 0.6215; rank-avg(grad,loss) 0.6174; -(grad×minkpp) 0.6048. ⚠️ SEVERELY INVERTED: -(grad×surp) **0.3618** (grad and SURP anti-correlated: high-grad files have low SURP, product amplifies noise); -(loss×surp) 0.4200. Ruby exception: -(grad×loss) 0.6884 > -grad_embed 0.6847 (loss adds signal for Ruby). Subset best = neg_grad_embed for Go/Java/Python/Rust; prod_grad_loss for Ruby. ⚠️ **No product beats -grad_embed globally** (0.6423). EXP27's product beat baseline because JSD is a COMPLEMENTARY signal (different information). Loss, SURP, MinK++ are all loss-based and CORRELATED with grad → product adds no information. Only -(embed×L29) nearly ties (0.6418): cross-layer product has slight decorrelation. → Key finding: multiplicative combination only helps when the two signals are ORTHOGONAL. |
| **EXP41** | Per-Language Calibrated Gradient | **0.6539** | 10% sample (10K), fwd+bwd, ~10 min A100. Raw -grad_embed **0.6423**; Z-global 0.6423 (same); **-grad_z_lang 0.6539** (+0.012 over raw); pctile_lang 0.6534; -loss_z_lang 0.5906 (+0.01 over raw -loss); ⚠️ product_z_lang **0.4588** (severely inverted). Per-subset AUCs IDENTICAL across raw/z-global/z-lang/pctile for all 5 languages (Z-norm within language = monotonic transform, preserves per-subset rank order). Global improvement comes from cross-language rebalancing only. Per-lang gradient stats (M vs NM ratio): Go 0.619, Python 0.576, Ruby 0.589, Java 0.732, Rust 0.721. Java/Rust have higher ratio (weaker signal) but NM absolute norm also lower — harder to separate. ✔ **Per-language Z-norm beats global threshold by +0.012** (0.6423 → 0.6539). ⚠️ product_z_lang INVERTED (0.4588): grad_z and loss_z are anti-correlated within language (high-loss files also have high gradient, so z-scores cancel each other). → Use -grad_z_lang as drop-in replacement for -grad_embed. New best single signal: **0.6539** (vs EXP27 0.6484). |
| **EXP56** | Next-Generation Logit-Only Baselines | **0.6332** | 10% sample (10K), single fwd pass/sample, ~3.0 min A100, valid 9997/10000. Evaluated 10 candidates + 3 baselines with per-language z-norm. **Best: -z_esp 0.6332**; next: -esp 0.6292, -loss_slope 0.6287, -esp_corr 0.6181. Baselines in same run: minkpp_k20 0.5433, loss 0.5611, surp 0.5687. Gains: +0.0899 vs Min-K%++, +0.0721 vs loss, +0.0646 vs SURP. Subset (best): Go 0.6491, Java 0.6289, Python 0.6444, Ruby 0.6677, Rust 0.5806. Verdict: strongest pure logit-only baseline so far; recommended replacement for Min-K%++ in this setup. |
| **EXP55** | DocLevel Histogram MIA | **0.6612** (RF 5-fold CV) | Based on Meeus, Jain, Rei, de Montjoye (USENIX Security 2024); surveyed in Wu & Cao (arXiv:2503.19338v3, Aug 2025) Section 4.1 [40]. 10% sample (10K), 1 fwd pass/sample, 16 histogram bins + 19 aggregate features = 36 total features, RF (200 trees, max_depth=8), 5-fold CV, ~3 min A100. Valid: 9997/10000. **RF CV Mean AUC: 0.6612 ± 0.0101** (beats EXP41 -grad_z_lang 0.6539; below EXP43 AttenMIA 0.6642 and EXP50 memTrace 0.6908). Fold AUCs: 0.6809, 0.6597, 0.6531, 0.6550, 0.6573. Best unsupervised: `+agg_max` 0.5921, `-agg_std` 0.5733, `+agg_p95` 0.5732, `+agg_p90` 0.5728, `+top_20_mean` 0.5700. Baseline: `agg_mean` (= -loss) 0.5611. **Histogram ADD value: +0.1001 over mean_loss** (RF 0.6612 vs scalar loss 0.5611). Top RF importances: `seq_len` **0.1992** ⚠️ (STRONGEST length confound seen yet — 3× higher than EXP50's 0.0898), `agg_max` 0.0628, `agg_std` 0.0539, `hist_bin_1` 0.0449 (bin [-12,-10]), `agg_p95` 0.0356, `agg_p5` 0.0335, `bottom_20_mean` 0.0323. Subset RF: Ruby 0.6964, Go 0.6949, Python 0.6724, Java 0.6369, Rust 0.6258. ⚠️ `seq_len` dominance (0.1992) is alarming — members are longer (EXP49 M=871 vs NM=590 tokens), so `agg_max`/`agg_p95` (tail of distribution) also correlated with length. True histogram AUC without length confound is likely lower (~0.62–0.63). See Insight 24. |
| **EXP54** | SPV-MIA (Self-Prompted) | - | Based on Fu, Wang, Gao, Liu, Li, Jiang (NeurIPS 2024); surveyed in Wu & Cao (arXiv:2503.19338v3, Aug 2025) Section 4.2 [17]. Generates reference data FROM THE TARGET MODEL ITSELF — no external reference needed. Split code: first 50% as prompt → generate continuation → compare LL(original_suffix\|prefix) vs LL(generated_suffix\|prefix). Signals: spv_diff = LL_orig - LL_gen (higher = model "prefers" original = member), spv_ratio, gen_consistency (std of generated LLs). 3 generations per sample, temp=0.8, top_p=0.95, max_new=128. ~8 fwd passes/sample, 10% sample. Paper: strong on fine-tuned LLMs with self-prompt calibration. Adaptation: pre-training MIA on code (weaker signal). KEY insight: model should assign higher LL to memorized training data than to its own stochastic generations. Expected AUC 0.52-0.62. |
| **EXP53** | WEL-MIA (Weighted Likelihood) | **0.5819** | Based on Song, Zhao, Xiang (ACSAC 2024); surveyed in Wu & Cao (arXiv:2503.19338v3, Aug 2025) Section 4.2 [59]. 10% sample (10K), 2 fwd passes/sample (target + ref), ref=starcoderbase-1b, max_len=512, ~3:54 min A100 (42.56 it/s). Valid: 9997/10000. Score variants: -lr_mean 0.5696, +wel_target 0.5440, -wel_dual 0.5643, **-wel_inv_ref 0.5819** (best), +lr_topk20 0.5492, +neg_mean_loss 0.5611, +neg_ref_loss 0.5784, -loss_ratio 0.5361. **-wel_inv_ref = 0.5819** (w_i = 1/NLL_ref — up-weights tokens easy for reference): BEST, marginally above EXP01 raw loss 0.5807 (+0.0012). Unweighted LR (-lr_mean 0.5696) beats target-weighted (0.5440) and dual-weighted (0.5643) — intermediate weighting hurts. Subset (wel_inv_ref): Ruby **0.6477**, Go 0.5891, Java 0.5827, Python 0.5616, Rust 0.5392. ⚠️ **All WEL variants ≤ 0.5819** — well below gradient ceiling (0.6539). Reference model calibration (starcoderbase-1b) provides modest lift over raw loss but cannot match white-box gradient. Token-level weighting does NOT improve over simple LR for pre-training MIA on code; effect size ~0.001–0.021 over baselines. Ruby outlier (0.6477) suggests language-specific token difficulty patterns. |
| **EXP52** | Semantic MIA (SMIA) | **0.6378** (LR 5-fold CV) | Based on Mozaffari & Marathe (NeurIPS Safe GenAI 2024); surveyed in Wu & Cao (arXiv:2503.19338v3, Aug 2025) Section 4.3 [45]. 10% sample (10K), 10 fwd passes/sample (1 orig + 3×3 perturb), perturbation rates [5%, 10%, 20%], K=3 reps/rate, ~22 min A100. Valid: 9997/10000. **LR CV Mean AUC: 0.6378 ± 0.0069** (5-fold). Fold AUCs: 0.6435, 0.6445, 0.6295, 0.6294, 0.6420. Best unsupervised: `+loss_per_sem_r5` **0.6365**, `-sem_dist_r5` 0.6282, `+robustness_ratio` 0.6240, `-sem_dist_mean` 0.6182. Baseline: `-orig_loss` 0.5611 (semantic calibration HELPS: +0.077). Loss-only signals weak: `+loss_diff_r5` 0.5566, `+loss_diff_mean` 0.5421 — same as EXP03/07. **Dominant discriminator: embedding drift (`-sem_dist`)** — members have SMALLER embedding layer distance between original and corrupted inputs. Top LR coefficients: `sem_dist_r5` −0.537 (dominant negative = smaller drift = member), `sem_dist_mean` −0.230, `loss_per_sem_r20` −0.141. Subset SMIA_LR: Ruby 0.6730, Python 0.6672, Go 0.6608, Java 0.6091, Rust 0.5935. ✅ **Semantic calibration unlocks perturbation signal** — prior methods (EXP03/07/28/31/32) measured loss/gradient change alone (AUC 0.50–0.56); adding embedding distance as denominator pushes to 0.6378. See Insight 23. |
| **EXP51** | LUMIA (Linear Probes) | - | Based on Ibáñez-Lissen et al. (arXiv:2411.19876, Nov 2024); surveyed in Wu & Cao (arXiv:2503.19338v3, Aug 2025) Section 4.4 [27]. Attaches lightweight LINEAR PROBES (LogisticRegression) to per-layer mean-pooled hidden activations. For each layer l: A_l(X) = mean(a_l(x_i)) → StandardScaler → LR(C=0.1) → AUC. Paper finds middle/deeper layers reveal most information. Also tests top-K layer concatenation (K=1,3,5,8). 1 fwd pass/sample (output_hidden_states=True), 5-fold CV, 10% sample. DIFFERENT from EXP50 (memTrace): memTrace extracts cross-layer features (transitions, confidence) → Random Forest; LUMIA uses RAW mean activation VECTOR at each layer independently → per-layer linear probe. Simpler, more interpretable, identifies WHICH layers leak most. DIFFERENT from EXP11/13 (gradient norms): those measure gradient MAGNITUDE; LUMIA uses hidden STATE VECTOR (direction + magnitude). Paper tested on Pythia/GPT-Neo on The Pile. Expected AUC 0.55-0.70. |
| **EXP50** | memTrace (Neural Breadcrumbs) | **0.6908** ✅ NEW BEST | 10% sample (10K), 1 fwd pass/sample (output_hidden_states=True), 69 features, 5-fold stratified CV RF (200 trees, max_depth=8), ~4 min A100. Valid: 9997/10000. **RF CV Mean AUC: 0.6908 ± 0.0093** (NEW BEST, beats EXP43 AttenMIA 0.6642 by +0.027). Fold AUCs: 0.7035, 0.6961, 0.6796, 0.6807, 0.6941. Top unsupervised features: -hnorm_std_L15 0.6335, -hnorm_global_mean 0.6260, -hnorm_mean_L15 0.6161, -stability_min_L15 0.6059, -stability_min_L22 0.6048. Top RF importances: seq_len 0.0898 ⚠️ (length confound — see EXP49 lesson), hnorm_global_mean 0.0516, hnorm_std_L15 0.0397, hnorm_mean_L15 0.0385, ctx_evolution_std 0.0352. neg_mean_loss unsupervised: 0.5611 (baseline). Subset RF: Go 0.7274, Python 0.7047, Ruby 0.7074, Java 0.6787, Rust 0.6279. Feature families: hidden state norm (mid layer) → STRONGEST single signal (L15=mid); layer transition surprise/stability; context evolution std; confidence/entropy. ⚠️ seq_len ranks #1 in RF importance (0.0898) — same length confound as EXP49; dropping seq_len may reduce true AUC slightly but removes spurious correlation. See Insight 22. |
| **EXP49** | Tokenizer MIA (BPE Signals) | **0.5923** (genuine) | 100% dataset (100K), tokenizer-only, no GPU, ~3.5 min. Valid: 99988/100000 (100%). **Apparent best: `n_tokens` 0.6620, `n_unique_tokens` 0.6609, `-token_type_ratio` 0.6331** — but all 3 are LENGTH CONFOUNDS (M=871.6 tokens vs NM=589.9 tokens, ratio 1.4774; members are longer files). Genuine length-independent tokenizer signals: **`merge_idx_max` 0.5923** (best true signal), `rare_token_count_p90` 0.5887, `bytes_per_token` 0.5674, `merge_idx_mean` 0.5320. Rarity score 0.5205 (weak). Subset `merge_idx_max`: Go 0.6075, Python 0.6118, Ruby 0.5954, Java 0.5780, Rust 0.5702. Distribution: merge_idx_max M=44187 vs NM=41349 (M has higher max merge index → more distinctive tokens ✓). bytes_per_token M=3.89 vs NM=3.70 (ratio 1.053 — tiny). ⚠️ **n_tokens confound dominant**: M sequences are ~48% longer than NM, which alone explains n_tokens AUC 0.6620. Cannot use n_tokens/n_unique/type_ratio without length normalization. See Insight 21. |
| **EXP48** | ICIMIA (Corruption Robustness) | **0.6419** (`sim_span_mid`) | Adapted from Wu et al. (arXiv:2506.12340v3, Feb 2026, originally for VLLMs). 10% sample (10K), 8 fwd passes/sample (orig + mask×3 + drop×3 + span), rates=[5%,10%,20%], layers=[mid(L15), last(L29)], ~18:39 min A100 (8.93 it/s). Valid: 9997/10000. **PRIMARY sim_combined_avg: 0.4696 ← FAILED**; but **sim_span_mid 0.6419** (surprise!), sim_span_last 0.6111, sim_span_avg 0.6142. Mask/drop signals ALL INVERTED (AUC 0.42–0.49 < 0.5). Loss baseline 0.5611. Subset (combined): Go 0.5201, Java 0.4948, Python 0.4234, Ruby 0.4139, Rust 0.4885. Distribution: sim_span M=0.908±0.094 vs NM=0.859±0.130, delta=**+0.049** (M more robust to span corruption ✓); sim_mask M=0.551 vs NM=0.580, delta=−0.029 ⚠️ INVERTED (NM more robust to token masking); sim_drop M=0.724 vs NM=0.742, delta=−0.018 ⚠️ INVERTED. ⚠️ **Split verdict**: Span corruption → member representation IS more robust (+0.6419, near gradient ceiling). Token mask/drop → INVERTED (NM more robust, confirms Insight 9 for substitution-style perturbation). **Key mechanistic insight**: span noise preserves local syntactic structure around the corrupted span; members "recognize" surrounding memorized context. Random mask/drop destroys token identity globally → non-members' more generalizable representations handle this better. sim_span_mid useful for EXP15 stacker as orthogonal feature. |
| **EXP47** | WBC (Window-Based Comparison) | **0.5624** | 10% sample (10K), self-temp(T=2.0) fallback (starcoderbase-1b gated), max_len=512, W={2,3,4,5,8,11,15,21,29,40}, ~4.5 min A100. Valid: 9997/10000 (100%). **Best signal: loss_ratio 0.5624**; global sign fraction 0.5569; -mean_loss 0.5611; WBC Ensemble **0.5205** ← PRIMARY but NOT best; mean loss diff 0.4617 ⚠️ inverted. Per-window AUC: **w=2 0.5401** (best) → w=3 0.5173 → w=4 0.4916 → w≥5 ≈ 0.48–0.50 (near-random or inverted). WBC ensemble vs global diff: **+0.0588** (WBC does improve over raw averaging). Subset WBC_ens: Ruby 0.5407, Rust 0.5289, Java 0.5246, Go 0.5210, Python 0.4869. Best window per subset all w=2 (Go 0.5332, Java 0.5475, Python 0.5110, Ruby 0.5792, Rust 0.5361). Distribution: WBC M=0.9929 vs NM=0.9920, Δ=0.0009 (tiny, ≈ statistical noise). Loss diff inverted: M=0.9647 vs NM=0.9979, Δ=−0.0332. ⚠️ WBC's sliding-window advantage only evident at w=2 (large-window averaging converges to global mean, erasing the signal). Fine-tuning→pre-training gap confirmed: self-temp(T=2.0) is not a true reference — same as EXP46 root cause. See Insight 20. |
| **EXP46** | EZ-MIA (Error Zone Score) | **0.5645** | 10% sample (10K), self-temp(T=5) reference (starcoderbase-1b gated), max_len=512, ~5 min A100. Valid: 9997/10000 (100%). **Best: mean_delta_error 0.5645**; Loss-Ref 0.5632; Loss 0.5611; pos_fraction_error 0.5529; EZ Score (P/N) 0.5527; -error_fraction 0.5477; Loss Ratio 0.5351. ⚠️ EZ Score direction: M=242.18 vs NM=271.92 — **slightly inverted** (NM P/N > M P/N with self-temp ref). mean_delta_error: M=4.2054 vs NM=4.1173 (correct direction, M > NM ✓). Error fraction: M=0.2114 vs NM=0.2303 (members fewer errors ✓). Subset best (mean_delta_error): Ruby 0.5709, Go 0.5627, Rust 0.5559, Java 0.5397, Python 0.5393. ⚠️ All signals 0.51–0.56 — far below EXP43 (0.6642). Temperature scaling creates weak calibration: it is a monotonic transform, NOT a true pre-training gap. Paper's 0.893 AUC used fine-tuned model + separate pretrained reference. Hypothesis directionally confirmed (fewer errors = member, delta direction correct) but effect size too small (M vs NM error_fraction gap = 0.019, delta gap = 0.09). |
| **EXP45** | CON-RECALL (Contrastive Decoding) | **0.4889** ⚠️ | 10% sample (10K), 5 shots/prefix, max128 tok/shot, target 256, 3 fwd passes/sample, ~9 min A100. Valid: 9990/10000 (eval set after removing 10 probe shots). **Gamma search (all inverted):** gamma=0.0 (ReCaLL) 0.4762 → gamma=1.0 (best) **0.4889** — all AUCs < 0.5. After flip: **-CON-RECALL (gamma=1.0) ≈ 0.5111**. Loss (LL) 0.5484; Min-K%++ 0.5434. CON-RECALL improvement over ReCaLL: +0.0127 (but both inverted). Subset CON-ReCaLL: Ruby 0.5708 ⚠️ borderline inverted, Python 0.4977, Java 0.4935, Rust 0.4616, Go 0.4446. LL stats: Unconditional M=−1.037 vs NM=−1.137; With M prefix M=−1.109 vs NM=−1.230; With NM prefix M=−1.165 vs NM=−1.297. ReCaLL ratio M=1.1493 vs NM=1.1602 — **NM ratio slightly higher → signal INVERTED for code**. P_member=633 tok, P_nonmember=484 tok. Prefix conditioning produces differential suppression in BOTH groups (not just members) — consistent with EXP44/ReCaLL failure. See Insight 19. |
| **EXP44** | EM-MIA (Expectation-Maximization) | **0.5802** | 2% sample (2K), D_em=200 (100M+100NM), 10 EM iters, max_len=256, ~13 min A100 (8 min pairwise + 4.6 min Phase 2). Best signal: **baseline Loss 0.5802** (EM subset), 0.5482 (full eval). ⚠️ ALL EM/ReCaLL signals near random or inverted: ReCaLL-Avg 0.4756, -AvgP 0.3779, **EM-MIA Loss-init 0.4577, EM-MIA MinK++-init 0.4549, ReCaLL top-k (full eval) 0.4481**. EM iterations: AUC 0.5028→0.4577 (monotone decline — EM actively diverges from the correct signal). Prefix scores (r_loss) range 0.578–0.598; best-10 prefix labels: 4M + 6NM (no clear member bias). MinK++ on EM subset: 0.5751 (baseline, not beaten by EM). Subset ReCaLL top-k: Java 0.5347 (closest to random), Go 0.4642, Rust 0.4323, Ruby 0.4076, Python 0.3993. ⚠️ **ReCaLL/EM-MIA hypothesis NOT confirmed for code pre-training**: prefix conditioning does not create differential suppression in StarCoder2-3b. Only baseline Loss survives (0.5802, consistent with EXP01 0.5807). |
| **EXP43** | AttenMIA (Attention-Based MIA) | **0.6642** | 5% sample (5K), 4 fwd passes/sample (1 orig + 3 pert), attn_implementation=eager, ~35 min A100. Valid: 5000/5000 (100%). **AttenMIA Full (Trans+Pert): 0.6642** (NEW BEST); Transitional+Conc only: 0.6573; CV Mean Full: 0.6477±0.0229; CV Trans: 0.6438±0.0358. Best individual: conc_mean 0.6508; trans_bary_mean_std 0.6499; trans_bary_var_mean 0.6456; trans_kl_late 0.6319. ⚠️ INVERTED (need flip for EXP15): trans_frob_* (0.35–0.36), trans_corr_mean/late (0.39–0.40), pert_drop/replace/prefix_shift_std/max (0.37–0.40). Ridge top5 (Full): conc_std −0.247 (classifier flipped), pert_prefix_shift_mean +0.239, pert_replace_shift_max −0.200, trans_kl_late +0.197, pert_prefix_shift_std −0.196. Subset Full: Go 0.6981, Ruby 0.6921, Python 0.6701, Java 0.6572, Rust 0.6167. ✅ **BEATS EXP41 -grad_z_lang (0.6539) by +0.010** — attention is a genuinely new signal family orthogonal to gradient. Breaks gradient plateau (~0.65 ceiling). |
| **EXP42** | Full Dataset Gradient Norm | **0.6500** | 50% sample (50K), single fwd+bwd, ~50 min A100. Valid: 49993/50000. **-grad_embed 0.6500**; -mean_loss 0.5857; -(grad×loss) 0.6299. Checkpoint AUCs stable: 5K→0.6480, 10K→0.6423, 20K→0.6472, 35K→0.6486, 50K→0.6500. Delta vs 10% baseline (EXP11): **+0.0028**. Subset: Go 0.6921, Python 0.6841, Ruby 0.6695, Java 0.6277, Rust 0.6198. Gradient stats: M 8.57±5.43, NM 13.51±10.84, ratio **0.634** (M grad is 63.4% of NM). Loss: M 0.807, NM 0.971, ratio 0.831. ✅ **CEILING IS REAL**: +0.0028 with 5× more data confirms ~0.65 is near-information-theoretic limit, NOT a sampling artifact. AUC variance across checkpoints (0.64–0.65) matches 10% sample variance. |
| **NOVEL01** | **LensMIA** | **0.5611** | Logit Lens Convergence. +neg_mean_loss: 0.5611, -settling_depth_top1: 0.5124. |
| **NOVEL02** | **OrthoFuse-MIA** | **0.6582** | Gradient × Attention Fusion. +attn_conc_mean_z_lang: 0.6582, -neg_grad_norm: 0.5885. |
| **NOVEL03** | **RepDen-MIA** | **0.6559** | k-NN Density in Hidden Space. +knn_mid_dist_diff: 0.6559, +knn_mid_vote: 0.6411. |
| **NOVEL05** | **GradCoh-MIA** | **0.5870** | Cross-Subsequence Gradient Coherence. -grad_norm_full: 0.5870, grad_coh: 0.5183. |
| **NOVEL06** | **RankDef-MIA** | **0.6512** | Effective Rank of Hidden States (SVD). +eff_rank_at_last: 0.6512, +eff_rank_mean: 0.6500, +stable_rank_mean: 0.6428. M eff_rank_mean=216.6 vs NM=166.3 (HIGHER rank for members = INVERTED from hypothesis). |
| **NOVEL07** | **GradOrth-MIA** | **0.6349** | Gradient-Hidden Orthogonality. neg_effective_pressure: 0.6349. |
| **NOVEL08** | **PredAgree-MIA** | **0.5637** | Prediction Agreement Depth. -stability_min: 0.5637, early_commit: 0.5398. |
| **NOVEL09** | **ConfTraj-MIA** | **0.5611** | Confidence Trajectory. neg_mean_loss: 0.5611, acc_late: 0.5407. |
| **NOVEL10** | **GradSparse-MIA** | **0.5878** | Gradient Sparsity Profile. neg_layer_grad_norm_mean: 0.5878, l0_mean: 0.5672. |
| **EXP51** | **LUMIA** (Linear Probes, 10%) | **0.7338** | Per-layer linear probes on mean-pooled hidden activations; top-8 layer concat (L6,L5,L7,L2,L4,L8,L21,L22). Best single layer: block_5 0.7031. Norm unsupervised: -norm_L4/L5/L6 ≈ 0.6507. Subset (top-8): Go 0.7469, Java 0.7461, Python 0.7492, Ruby 0.7464, Rust 0.6706. Valid: 9997/10000. Runtime: 7h (LogReg saga 31 layers). |
| **LUMIA-fast** | **LUMIA Fast** (Ridge, full 100K) | **0.7805** 🏆 **NEW BEST** | Ridge probes on 9 key layers (PCA 256D), full 100K samples. Top-9 concat: **0.7805** ± 0.0024. Best single: block_20 (L21) 0.7456. Subset (top-9): Go 0.7888, Java 0.7797, **Python 0.8017**, Ruby 0.7951, Rust 0.7318. Valid: 99988/100000. Runtime: **0.47h (28 min fwd + 12s Ridge)** ✅ well within 12h. ⚠️ Batch failed (no pad token), fell back to single-sample (63.88 it/s). |
| **NOVEL11** | **ResDecomp-MIA** | **0.6590** | Residual Stream Decomposition (Attention vs MLP balance). +total_attn_frac: 0.6590, +attn_mlp_ratio_mid: 0.6380, +attn_mlp_ratio_early: 0.6306. ~4 min A100. Members have higher attention fraction (attn dominates over MLP more). |
| **NOVEL12** | **CaliFuse-MIA** | **0.6741** | Calibrated Multi-Signal Fusion Stacker. 5-fold LR CV (18 Z-features, 1 fwd+bwd pass). Best C=0.3594. Top coefs: neg_hs_norm_mid −1.31, attn_conc_mean +1.24, neg_hs_norm_early +1.03. Subset: Go 0.7060, Python 0.6920, Ruby 0.6965, Java 0.6515, Rust 0.6198. ~11 min A100 (15 it/s). |
| **COMBO01** | **memTrace+Grad RF** | **0.6954** | memTrace features (70) + RF (5-fold CV). ⚠️ **Gradient extraction FAILED** (TypeError 10K/10K) — effectively memTrace-only. AUC 0.6954 ± 0.0103 (folds: 0.7112, 0.6995, 0.6824, 0.6862, 0.6976). Top feature: seq_len 0.0663 (length confound), hnorm_global_mean 0.0348, hnorm_std_L15 0.0325. Subset: Go 0.7321, Python 0.7085, Ruby 0.7105, Java 0.6839, Rust 0.6298. ~5 min A100. |
| **COMBO02** | **memTrace+Attention RF** | **0.6961** | Hidden-state geometry × attention-flow fusion (single eager fwd pass, hidden_states+attentions, 10% sample/10K, 85 features). 5-fold RF CV: **0.6967 ± 0.0099** (folds 0.7122/0.6991/0.6838/0.6885/0.6999). Full score AUC 0.6961. Baselines: memTrace-only CV 0.6968 (≈tie), attention-only CV 0.6582. Top unsup: +attn_conc_mean 0.6513, +attn_bary_drift_std 0.6485. Top importances: attn_bary_drift_std 0.0304, attn_bary_drift_mean 0.0285, seq_len 0.0282 ⚠️ length confound persists. Subset: Go 0.7301, Python 0.7079, Ruby 0.7088, Java 0.6845, Rust 0.6369. ~6:54 A100. Verdict: robust improvement over EXP50 (+0.0053) but gain is marginal and mostly from stronger attention features, not additive with memTrace block. |
| **BASELINE-CDD** | **CDD-MIA (Conditional Distribution Dynamics)** | **0.6292** | `improved_baselines/BASELINE_cdd.py` (10% sample/10K, single fwd pass, ~6:15 A100). Best unsupervised signal: **-neg_entropy_slope 0.6292** (NEW dynamic trajectory signal). Static baselines: -loss 0.5611, Min-K%++(k20) 0.5433. Dynamic/Cross beating best static: **8 features**. Top dynamic: -entropy_drop 0.5803, +neg_worst_rank_k20 0.5736, +neg_surprise_std 0.5733. Subset (best signal): Go 0.6491, Java 0.6289, Python 0.6444, Ruby 0.6677, Rust 0.5806. Verdict: trajectory dynamics clearly beat static Min-K%++ but remain below supervised families (COMBO02 0.6961, EXP50 0.6908). |
| **BASELINE-MinKPP-Reimpl** | **Min-K%++ (clean reimplementation)** | **0.5671** | `improved_baselines/BASELINE_minkpp.py` (10% sample/10K, single fwd pass, ~2:40 A100). Best signal: **+mink_k20 0.5671** (raw Min-K 20%); calibrated Min-K%++ scores: minkpp_k20 0.5433, k30 0.5431, k50 0.5371. Loss baseline 0.5611. Subset (best): Go 0.5429, Java 0.5695, Python 0.5693, Ruby 0.6537, Rust 0.5254. Verdict: slight gain over loss (+0.0060), but weaker than EXP02 result (0.5770) and far below CDD/COMBO families. |
| **BASELINE-DCPDD** | **DC-PDD (frequency-calibrated divergence)** | **0.6004** | `improved_baselines/BASELINE_dcpdd.py` (10% sample/10K; frequency build 0:14 + inference 2:38; 1 fwd pass + token-frequency lookup). Best signals: **+dcpdd_k100 0.6004** and **+freq_calibrated_loss 0.6004**; next: dcpdd_k50 0.5913, dcpdd_k30 0.5816, dcpdd_k20 0.5799. Baselines: minkpp_k20 0.5433, loss 0.5611. Subset (dcpdd_k100): Go 0.6253, Java 0.5715, Python 0.5986, Ruby 0.6613, Rust 0.5501. Verdict: clear gain over Min-K%++ and loss, but still below CDD (0.6292) and supervised families. |
| **BASELINE-TopoCal** | **TopoCal-MIA (topological distribution calibration)** | **0.5952** | `improved_baselines/BASELINE_topocal.py` (10% sample/10K, single fwd pass, ~39:03 A100). Best signals (tie): **+neg_tail90_mean 0.5952** and **+gini_min_k20 0.5952**; next: +neg_gini_std 0.5941, +gini_min_k30 0.5923, +gini_mean 0.5897. Baselines: minkpp_k20 0.5433, loss 0.5611. Subset (neg_tail90_mean): Go 0.5849, Java 0.5691, Python 0.6167, Ruby 0.6766, Rust 0.5458. Verdict: shape-based topology features beat Min-K%++/loss but trail DCPDD (0.6004) and CDD (0.6292). |
| **NOVEL-CorruptAsym** | **Corruption Asymmetry MIA** | **0.5978** | `improved_baselines/NOVEL_CorruptAsym.py` (10% sample/10K, 7 fwd passes/sample, ~16:41 A100). Best signal: **+corrupt_asym_mid 0.5978**; next: +asym_ratio_mid 0.5969, +neg_sim_token_mid 0.5909, +sim_span_last 0.5900. Baseline loss: 0.5611. Subset (best): Go 0.6118, Java 0.5803, Python 0.6125, Ruby 0.6772, Rust 0.5543. ⚠️ Contrary to EXP48 hypothesis, `sim_span_mid` here is inverted/weak (AUC 0.5387 with flip), while asymmetry features remain useful and outperform loss. |
| **NOVEL-GeoPrint** | **Geometric Fingerprint MIA (4-axis)** | **0.6335** | `improved_baselines/NOVEL_GeoPrint.py` (10% sample/10K, single fwd with hidden states, ~7:36 A100). Best signal: **+neg_norm_mid_token_std 0.6335**. Top-10 dominated by magnitude/dimensionality/velocity: -neg_velocity_mean 0.6275, +neg_norm_global_mean 0.6260, +neg_norm_early_mean 0.6233, +eff_rank_L5 0.6216. Baseline loss 0.5611. Subset (best): Go 0.6792, Java 0.5877, Python 0.6753, Ruby 0.6707, Rust 0.6054. Verdict: strongest unsupervised method in improved_baselines so far; confirms multi-axis geometric signal (magnitude + dimensionality + velocity) is robust. |
| **NOVEL-LayerCascade** | **Layer Cascade Signature MIA** | **0.6403** | `improved_baselines/NOVEL_LayerCascade.py` (10% sample/10K, single fwd with hidden states, ~7:42 A100). Best signal: **-neg_drift_early 0.6403**. Strong trajectory family signals: -norm_std_curvature 0.6357, -norm_std_std 0.6348, -norm_std_early 0.6345, -norm_std_mid 0.6337, -neg_vel_convergence 0.6310. Baseline loss 0.5611. Subset (best): Go 0.6921, Java 0.6147, Python 0.6722, Ruby 0.6790, Rust 0.6078. Verdict: best unsupervised result across improved_baselines (slightly above GeoPrint 0.6335), validating depth-wise cascade trajectory hypothesis. |
| **NOVEL-LogitSig** | **LogitSignature (logit-only multi-family baseline)** | **0.6287** | `improved_baselines/NOVEL_LogitSig.py` (10% sample/10K, single fwd pass, ~45:12 A100, ~30 logit-only signals). Best signal: **-traj_loss_slope 0.6287**. Family champions: dist_gini_mean 0.5897, -tail_worst_k10 0.5743, con_neg_surprise_std 0.5733, base_neg_loss_std 0.5733, lcal_surp 0.5732. Baselines in same run: cal_minkpp_k20 0.5433, base_neg_loss 0.5611, cal_surp 0.5687. Subset (best): Go 0.6548, Java 0.6298, Python 0.6439, Ruby 0.6645, Rust 0.5769. Verdict: strongest pure logit-only trajectory signal in improved_baselines, nearly matching BASELINE-CDD 0.6292 and clearly above Min-K%++/loss baselines. |

---

## Signal Direction Quick Reference

| Signal | Higher = Member? | Best Source | AUC |
| :--- | :--- | :--- | :--- |
| `-grad_norm_embed` | ✅ Yes | EXP11, EXP22, EXP27, EXP34 | 0.64–0.71 |
| `product_score` `-(grad×jsd_early)` | ✅ Yes | EXP27 | **0.6484** |
| `combined_rank_score` (CAMIA) | ✅ Yes | EXP33 | 0.6065 |
| `-g_mean` (block-wise grad) | ✅ Yes | EXP34 | 0.6383 |
| `SURP` (mean_lp − std_lp) | ✅ Yes | EXP16 | 0.5884 |
| `CBC` (cross-block correlation) | ✅ Yes | EXP34 | 0.5498 |
| `IBVR` (early/late var ratio) | ✅ Yes | EXP34 | 0.4635 |
| `-attention_entropy` | ⚠️ **Needs flip** | EXP29 | 0.38 → ~0.62 if flipped |
| `-stability_cv` | ⚠️ Likely inverted | EXP24 | 0.4455 |
| `ESR` (early-settling ratio) | ⚠️ Weak | EXP26 | 0.4371 |
| `delta_jsd_early` (Δ after rename) | ⚠️ **No separation** | EXP28 | 0.5107 (Δ≈0 for M & NM) |
| `score_delta` (Δ_rel at 10%) | ⚠️ **No separation** | EXP31 | 0.5056 (Δ(M)≈Δ(NM)≈0.0003) |
| `-input_grad_norm` (EXP31) | ✅ Yes (needs sign flip) | EXP31 | 0.3835 raw → ~0.6165 flipped (redundant with EXP11) |
| `-G_orig` (OR-MIA gradient magnitude) | ✅ Yes | EXP32 | 0.6165 (M 1.24 vs NM 2.12) |
| `-ΔG` (gradient stability under noise) | ⚠️ **Near random** | EXP32 | 0.5130 (ΔG(M)≈ΔG(NM)) |
| `lda_score` (gradient profile LDA) | ⚠️ **Overfits probe** | EXP35 | 0.5573 (probe 0.76 → test 0.56) |
| `-grad_mean` (scalar gradient mean) | ✔ Yes | EXP35 | 0.6417 (≈ -grad_embed) |
| `anchor_body_gap` | ⚠️ **Inverted** | EXP36 | 0.4521 (NM gap > M gap) |
| `cliff_count` / `max_cliff_mag` | ⚠️ Weak | EXP37 | 0.5449 / 0.5388 |
| `xgb_score` (18-feature stacker) | ⚠️ **Underperforms single** | EXP38 | 0.6347 (< -grad_embed 0.6422) |
| `-grad_z_lang` (per-lang Z-norm) | ✔ Yes | EXP41 | **0.6539** |
| `-(grad×loss)` product | ✔ Weak | EXP40 | 0.6215 (below -grad_embed) |
| `-(grad×surp)` product | ⚠️ **Severely inverted** | EXP40 | 0.3618 (grad & SURP anti-correlated) |
| `attn_full_score` (AttenMIA Full: Trans+Pert) | ✅ Yes | EXP43 | **0.6642** (**NEW BEST** — beats gradient ceiling) |
| `combo02_score` (memTrace+Attention RF fusion) | ✅ Yes | COMBO02 | **0.6961** (5-fold CV mean 0.6967 ± 0.0099; slight gain over EXP50 0.6908; attention-only branch = 0.6582) |
| `-neg_entropy_slope` (CDD trajectory entropy slope) | ✅ Yes | BASELINE-CDD | **0.6292** (best CDD signal; dynamic trajectory > Min-K%++ static 0.5433, loss 0.5611) |
| `-z_esp` (EXP56 per-language calibrated entropy slope) | ✅ Yes | EXP56 | **0.6332** (best EXP56 signal; strongest pure logit-only baseline; +0.0899 over Min-K%++) |
| `mink_k20` (raw Min-K 20%, reimplementation) | ✅ Yes (weak) | BASELINE-MinKPP-Reimpl | **0.5671** (best in reimpl run; slightly > loss 0.5611, > minkpp_k20 0.5433) |
| `dcpdd_k100` (DC-PDD full-token frequency calibration) | ✅ Yes | BASELINE-DCPDD | **0.6004** (ties freq_calibrated_loss; improves over minkpp_k20 0.5433 and loss 0.5611) |
| `neg_tail90_mean` (TopoCal tail-mass concentration) | ✅ Yes | BASELINE-TopoCal | **0.5952** (ties gini_min_k20; topology of p(·|x<t) improves over minkpp_k20 0.5433 and loss 0.5611) |
| `corrupt_asym_mid` (span-vs-token corruption asymmetry) | ✅ Yes | NOVEL-CorruptAsym | **0.5978** (best in CorruptAsym run; beats loss 0.5611; stronger than individual span/token robustness signals) |
| `neg_norm_mid_token_std` (GeoPrint magnitude axis) | ✅ Yes | NOVEL-GeoPrint | **0.6335** (best GeoPrint signal; unsupervised SOTA in improved_baselines, clearly above loss 0.5611) |
| `-neg_drift_early` (LayerCascade early-layer drift trajectory) | ✅ Yes | NOVEL-LayerCascade | **0.6403** (best LayerCascade signal; new unsupervised top in improved_baselines, above GeoPrint 0.6335) |
| `-traj_loss_slope` (LogitSignature trajectory slope) | ✅ Yes | NOVEL-LogitSig | **0.6287** (best LogitSig signal; strongest logit-only trajectory feature in improved_baselines; above Min-K%++/loss and near CDD 0.6292) |
| `attn_trans_score` (AttenMIA Transitional) | ✅ Yes | EXP43 | 0.6573 |
| `conc_mean` (KL concentration mean) | ✅ Yes | EXP43 | 0.6508 (best individual attn feature) |
| `trans_bary_mean_std` (barycenter drift std) | ✅ Yes | EXP43 | 0.6499 |
| `trans_bary_var_*` / `trans_kl_*` | ✅ Yes | EXP43 | 0.6430–0.6466 / 0.6299–0.6319 |
| `trans_frob_*` (Frobenius inter-layer dist) | ⚠️ **Inverted** | EXP43 | 0.35–0.36 (flip sign before EXP15) |
| `trans_corr_mean/late` (layer-to-layer corr) | ⚠️ **Inverted** | EXP43 | 0.39–0.40 (flip sign before EXP15) |
| `pert_*_shift_std/max` | ⚠️ **Inverted** | EXP43 | 0.37–0.40 (flip sign before EXP15) |
| `ReCaLL` (LL(x\|p)/LL(x)) | ⚠️ **Inverted / near random** | EXP44 | 0.4481–0.4756 (prefix conditioning fails for code pre-training) |
| `EM-MIA score` (EM-refined membership) | ⚠️ **Inverted & diverges** | EXP44 | 0.4549–0.4577 (EM iterations worsen signal monotonically) |
| `ReCaLL` LL(x\|P_nm)/LL(x) — EXP45 replication | ⚠️ **Inverted** | EXP45 | 0.4762 (raw); flip → ~0.5238 (weak). ReCaLL ratio M=1.1493 vs NM=1.1602 — NM marginally higher |
| `CON-RECALL` [LL(x\|P_nm) − γ·LL(x\|P_m)] / LL(x) | ⚠️ **Inverted** | EXP45 | 0.4889 (gamma=1.0, raw); flip → ~0.5111 (weak). +0.0127 over ReCaLL but both < 0.5 |
| `mean_delta_error` (EZ-MIA avg Δ at error positions) | ✅ Yes (weak) | EXP46 | 0.5645 (best EZ signal; M=4.21 vs NM=4.12) |
| `ez_score` (P/N ratio at error positions) | ⚠️ **Slightly inverted** | EXP46 | 0.5527 (NM P/N > M P/N with self-temp ref) |
| `-error_fraction` | ✅ Yes (weak) | EXP46 | 0.5477 (fewer errors = member ✓, weak) |
| `loss_ratio` ℓ_R/ℓ_T (WBC baseline) | ✅ Yes (weak) | EXP47 | 0.5624 (best signal in EXP47; same concept as EXP46 Loss Ratio 0.5351) |
| `wbc_ensemble` (sliding window sign avg) | ✅ Yes (weak) | EXP47 | 0.5205 (ensemble PRIMARY; +0.0588 over global diff, but < loss_ratio) |
| `wbc_w2` (window=2 sign fraction) | ✅ Yes (weak) | EXP47 | 0.5401 (best single window; gain lost at w≥4 → converges to global avg) |
| `loss_diff` mean(ℓ_R − ℓ_T) (WBC global) | ⚠️ **Inverted** | EXP47 | 0.4617 (M=0.9647 vs NM=0.9979, Δ=−0.0332; self-temp ref artefact) |
| `n_tokens` (sequence length) | ✅ Yes (⚠️ **length confound**) | EXP49 | 0.6620 (M=871.6 vs NM=589.9 tokens, ratio 1.48 — members are longer, NOT tokenizer signal) |
| `-token_type_ratio` | ✅ Yes (⚠️ **length confound**) | EXP49 | 0.6331 (longer files have lower TTR — length artefact, same confound as n_tokens) |
| `merge_idx_max` (max BPE merge index) | ✅ Yes (genuine) | EXP49 | 0.5923 (best length-independent tokenizer signal; M=44187 vs NM=41349) |
| `rare_token_count_p90` (count tokens above vocab p90) | ✅ Yes (partly length-confounded) | EXP49 | 0.5887 (count scales with length; normalise by n_tokens for clean signal) |
| `bytes_per_token` (compression rate) | ✅ Yes (genuine) | EXP49 | 0.5674 (M=3.89 vs NM=3.70; members compress more efficiently) |
| `memtrace_rf_score` (RF over 69 hidden-state features) | ✅ Yes | EXP50 | **0.6908** CV mean (NEW BEST 2026-03-02); beats AttenMIA 0.6642 and all gradient signals) |
| `-hnorm_std_L15` (neg hidden-state norm std at mid layer) | ✅ Yes | EXP50 | 0.6335 (best single unsupervised feature; mid-layer norm variance captures memorization) |
| `-hnorm_global_mean` (neg global hidden-state norm mean) | ✅ Yes | EXP50 | 0.6260 (members have lower avg hidden state magnitude) |
| `-stability_min_L15/L22` (neg min cosine stability at mid layers) | ✅ Yes | EXP50 | 0.6059 / 0.6048 (lower minimum cosine sim between layers = member) |
| `ctx_evolution_std` (context evolution std at mid layer) | ✅ Yes | EXP50 | 0.5870 (higher evolution variance = member; recognition hot-spots) |
| `smia_lr_score` (LR on loss_diff + sem_dist features) | ✅ Yes | EXP52 | **0.6378** CV mean (±0.0069); semantic calibration unlocks perturbation family |
| `+loss_per_sem_r5` (loss change per unit embedding drift at 5%) | ✅ Yes | EXP52 | 0.6365 (best single unsupervised; members have higher loss_diff/sem_dist ratio at low rate) |
| `-sem_dist_r5` (neg embedding drift at 5% perturbation) | ✅ Yes | EXP52 | 0.6282 (dominant LR feature coef −0.537; members have more stable embedding layer under perturbation) |
| `+robustness_ratio` (mean_loss_diff / mean_sem_dist across rates) | ✅ Yes | EXP52 | 0.6240 (aggregated ratio; same mechanism as loss_per_sem) |
| `-sem_dist_mean` (neg mean embedding drift across all rates) | ✅ Yes | EXP52 | 0.6182 |
| `+loss_diff_r5` (loss change at 5% perturbation) | ✅ Yes (weak) | EXP52 | 0.5566 (loss-only signals weak — calibration by sem_dist is required) |
| `-wel_inv_ref` (inverse-ref weighted LR: w_i = 1/NLL_ref) | ✅ Yes (weak) | EXP53 | **0.5819** (best WEL variant; up-weights tokens easy for reference; +0.0012 over raw loss; Ruby 0.6477 outlier) |
| `-lr_mean` (unweighted likelihood ratio NLL_ref − NLL_target) | ✅ Yes (weak) | EXP53 | 0.5696 (unweighted LR beats target-weighted 0.5440 and dual-weighted 0.5643) |
| `-wel_dual` (dual-weighted LR: w_i = NLL_target/NLL_ref) | ✅ Yes (weak) | EXP53 | 0.5643 (paper's proposed weighting; below unweighted LR for pre-training MIA) |
| `+neg_ref_loss` (negative reference model loss) | ✅ Yes (weak) | EXP53 | 0.5784 (reference model loss alone; near raw target loss 0.5611) |
| `+wel_target` (target-weighted LR: w_i = NLL_target) | ⚠️ Weak | EXP53 | 0.5440 (emphasizing hard tokens for target degrades LR signal) |
| `+lr_topk20` (LR avg of top-20% hardest target tokens) | ⚠️ Weak | EXP53 | 0.5492 (Min-K%-style selection of hardest tokens; weaker than global LR) |
| `hist_rf_score` (RF on 16-bin histogram + aggregate features) | ✅ Yes | EXP55 | **0.6612** CV mean (±0.0101); distribution SHAPE adds +0.100 over scalar mean_loss; ⚠️ seq_len importance 0.1992 — length confound very strong |
| `+agg_max` (max per-token log-prob in document) | ✅ Yes | EXP55 | 0.5921 (best single unsupervised; tail of distribution = how confident model gets on easiest tokens; correlates with length) |
| `-agg_std` (neg std of per-token log-probs) | ✅ Yes | EXP55 | 0.5733 (lower std = member; members have more uniform confidence — fewer extremely uncertain tokens) |
| `+agg_p95` / `+agg_p90` / `+top_20_mean` (upper tail percentiles) | ✅ Yes | EXP55 | 0.5732 / 0.5728 / 0.5700 (upper percentiles capture the "easy token" tail; all length-partially-confounded) |
| `+bimodality_coeff` | ✅ Yes (weak) | EXP55 | RF importance 0.0243 (bimodality hypothesis partially supported: members show more bimodal distributions, but signal weak) |
| `+sim_span_mid` (cosine sim of mid-layer hidden states under span corruption) | ✅ Yes | EXP48 | **0.6419** (best signal; span noise preserves local context → members recognize surrounding memorized structure; M=0.908 vs NM=0.859, delta=+0.049) |
| `+sim_span_last` / `+sim_span_avg` (span corruption, last/avg layers) | ✅ Yes | EXP48 | 0.6111 / 0.6142 (same mechanism as sim_span_mid; mid layer slightly better) |
| `sim_combined_avg` (avg cosine sim across all corruption methods) | ⚠️ **FAILED / inverted** | EXP48 | 0.4696 (mask/drop dominate the average and are inverted → combined worse than random) |
| `sim_mask_avg` / `sim_drop_avg` (token mask/drop cosine sim) | ⚠️ **Inverted** | EXP48 | 0.4499 / 0.4580 (NM more robust to random token substitution/removal — NM representations generalize better; flip if using in stacker) |
| `neg_effective_pressure` | ✅ Yes | NOVEL07 | **0.6349** |
| `neg_layer_grad_norm_mean` (Sparsity) | ✅ Yes | NOVEL10 | 0.5878 |
| `neg_grad_norm_full` (GradCoh) | ✅ Yes | NOVEL05 | 0.5870 |
| `-stability_min` (PredAgree) | ✅ Yes | NOVEL08 | 0.5637 |
| `early_commit_ratio` | ✅ Yes | NOVEL08 | 0.5398 |
| `acc_late` (ConfTraj) | ✅ Yes | NOVEL09 | 0.5407 |
| `knn_mid_dist_diff` | ✅ Yes | NOVEL03 | **0.6559** |
| `attn_conc_mean_z_lang` | ✅ Yes | NOVEL02 | **0.6582** |
| `settling_depth_top1` | ✅ Yes | NOVEL01 | 0.5124 |
| `+eff_rank_at_last` / `+eff_rank_mean` (RankDef effective rank) | ✅ Yes (⚠️ **direction inverted from hypothesis**) | NOVEL06 | 0.6512 / 0.6500 (M eff_rank=216.6 > NM=166.3 — members have HIGHER rank, not lower as hypothesized; top_sv_frac lower for M confirms dominant direction dominates less) |
| `+stable_rank_mean` (RankDef stable rank) | ✅ Yes | NOVEL06 | 0.6428 |
| `neg_top_sv_frac_mean` (neg top singular value fraction) | ✅ Yes | NOVEL06 | 0.6397 (lower top-SV fraction = member; energy is less concentrated in first singular vector) |
| `+eff_rank_per_token` (length-normalized effective rank) | ✅ Yes (genuine, length-controlled) | NOVEL06 | 0.6082 (AUC drops from 0.65 to 0.61 after length normalization — partial length confound in raw eff_rank) |
| `lumia_top8_score` (LR on top-8 layer concat, 5-fold CV) | ✅ Yes | EXP51 | **0.7338** (**NEW BEST** beats memTrace 0.6908 by +0.043); Go 0.7469, Java 0.7461, Python 0.7492, Ruby 0.7464, Rust 0.6706 |
| `lumia_single_layer_score` best layer (block_5, layer 6) | ✅ Yes | EXP51 | **0.7031** CV mean (best single; early-to-mid transition layer; probe AUC plateau 0.69–0.70 across blocks 1–21) |
| `-norm_L4/L5/L6` (activation norm at early-mid layers, unsupervised) | ✅ Yes | EXP51 | 0.6507 / 0.6505 / 0.6502 (lower norm = member; peaks at blocks 3–5, decays at later layers — consistent with NOVEL06 finding that members use more dimensions but with lower per-dimension norm) |
| `+total_attn_frac` (attention share of total residual update) | ✅ Yes | NOVEL11 | **0.6590** (members have higher attention fraction across all layers; attention dominates over MLP more for memorized sequences) |
| `+attn_mlp_ratio_mid` (attention/MLP norm ratio at mid layers) | ✅ Yes | NOVEL11 | 0.6380 (best per-region ratio; mid-layer balance is most discriminative) |
| `+attn_mlp_ratio_early` / `+attn_mlp_ratio_mean` | ✅ Yes | NOVEL11 | 0.6306 / 0.6265 (early layers second-best; mean across all layers solid baseline) |
| `+attn_dominant_frac` (fraction of layers where attn > mlp) | ✅ Yes | NOVEL11 | 0.6201 (members have more attn-dominant layers) |
| `-attn_mlp_cos_mean` (negative cosine similarity between attn and MLP outputs) | ✅ Yes | NOVEL11 | 0.5907 (members' attn and MLP outputs are more anti-aligned) |
| `neg_attn_norm_mean` / `neg_mlp_norm_mean` | ✅ Yes (weak) | NOVEL11 | 0.5493 / 0.5828 (lower sublayer norms = member; MLP norm cleaner signal than attn norm) |
| `califuse_score` (LR+CV over 18 Z-normed multi-signal features) | ✅ Yes | NOVEL12 | **0.6741** CV mean (5-fold); 1 fwd+bwd pass — dominant: neg_hs_norm_mid (−1.31), attn_conc_mean (+1.24), neg_hs_norm_early (+1.03). ⚠️ Below LUMIA (0.7338) because raw 3072-D activation vectors not used. |
| `neg_hs_norm_mid` (neg hidden state norm at mid layer — CaliFuse) | ✅ Yes | NOVEL12 | coeff −1.31 (most important feature in stacker; consistent with EXP50 memTrace finding) |
| `attn_conc_mean` (KL concentration of attention — CaliFuse) | ✅ Yes | NOVEL12 | coeff +1.24 (second most important; consistent with EXP43 AttenMIA) |
| `neg_vel_early_mid` (neg representation velocity early→mid) | ✅ Yes | NOVEL12 | coeff +0.14 (lower velocity = member; velocity signal confirmed useful) |
| `combo01_score` (RF on 70 memTrace features, effectively memTrace-only) | ✅ Yes | COMBO01 | **0.6954** CV mean (5-fold); gradient extraction failed — gradient features = 0. +0.005 over EXP50 from richer feature set (more per-layer statistics). Top RF features: seq_len (⚠️ length confound), hnorm_global_mean, hnorm_std_L15 |
| `lumia_fast_score` (Ridge on top-9 key-layer concat, full 100K, PCA-256) | ✅ Yes | LUMIA-fast | **0.7805** 🏆 **NEW BEST** (top-9 all key layers; top-1 L21=0.7456, top-3=0.7672, top-5=0.7740). Subset: Go 0.7888, Java 0.7797, **Python 0.8017**, Ruby 0.7951, Rust 0.7318. PCA expl.var: block_1=0.987, block_4-7=0.93-0.96. ⚠️ Best single layer shifted from block_5 (EXP51 10%) to **block_20** (full data) — deeper layers more discriminative at scale. |

---

## Insights & Lessons Learned

*Tư liệu cho paper và định hướng modeling tiếp theo.*

---

### Insight 1 — Gradient norm là cửa ngõ cao nhất hiện có

**Observation:** Gradient norm của embedding layer (EXP11 0.6472, EXP22 embed 0.6423, EXP27 grad_embed 0.6480, EXP34 gnorm_embed 0.6422) luôn dẫn đầu single-feature AUC. EXP27 còn cho thấy **product(grad × jsd_early) = 0.6484** vượt cả grad_embed đứng riêng, chứng tỏ JSD đóng vai trò tiebreaker tốt dù standalone yếu.

**Mechanism hypothesis:** StarCoder2-3B tạo ra "flat minima" cho training files. Non-member files nằm ngoài basin → gradient cao hơn rõ rệt (member mean G ≈ 0.80, non-member ≈ 1.24, EXP34). Đây là bằng chứng thực nghiệm cho "flat minimum = memorization" hypothesis trên code LLM.

**Paper angle:** Novelty: code domain, white-box, không cần reference model. 5 programming languages, flat-minimum hypothesis holds stronger for code than natural language.

**Next modeling step:**
- **Bookend layer ensemble**: embed + layer_28 + layer_29 + head (EXP34: layer_29 0.6407, embed/head 0.6422). Expected AUC 0.65–0.68.
- **Per-token gradient norm on anchor tokens** (`def`, `class`, `import`) — code-specific memorization markers.
- EXP27's product score formula `-(grad_embed × jsd_early)` là candidate mạnh cho EXP15 stacking.

---

### Insight 2 — Context accumulation là signal, không phải điểm loss tuyệt đối

**Observation:** EXP33 CAMIA — sub-signals đứng riêng (MDM 0.4803, TVar 0.4888, AUCG 0.5019) đều yếu, nhưng **rank_avg đạt 0.6065** (+0.11 so với từng signal). EXP01 loss trung bình toàn sequence chỉ 0.5807.

**Mechanism hypothesis:** Loss tuyệt đối bị ảnh hưởng bởi code style, độ phức tạp, ngôn ngữ. Nhưng loss TRAJECTORY (cách loss giảm khi context tích lũy) là ngôn ngữ-agnostic: model "nhận ra dần" file đã thấy tạo ra loss cliff đặc trưng. Non-member: loss giảm đều.

**Paper angle:** Extension của CAMIA paper cho code domain — "progressive recognition" pattern: import → class def → method body → usage.

**Next modeling step:**
- **Adaptive block splitting** theo semantic boundary (function/class definitions) thay vì fixed 256 tokens. Expected AUC 0.63–0.67.
- Feature: **block_idx_of_max_drop** — position của sharpest loss cliff là feature mạnh cho XGBoost stacker.
- **Per-language loss calibration**: normalize trajectory bằng language-specific baseline để loại bỏ language bias.

---

### Insight 3 — Rank averaging giải cứu signal yếu, nhưng cần auto-direction flip

**Observation:**
- EXP29 AUC = 0.3840, EXP26 = 0.4371, EXP23 = 0.4852, EXP24 = 0.4455 — đều yếu/ngược chiều đứng riêng.
- EXP34 combined rank kết hợp signal_g_mean (0.6383) + CBC (0.5498) + entropy (0.4448) + RPS (0.4918) → **0.5608**.
- EXP33 tương tự: rank_avg 3 signal yếu → **0.6065**.
- **Nhưng** EXP27 cho thấy khi JSD (0.4371) được kết hợp với grad (0.6480) bằng rank_avg → **0.5845** (thấp hơn grad_embed riêng lẻ!). Product score lại **0.6484** (cao hơn cả hai). → **Rank averaging phá tín hiệu mạnh khi kết hợp với tín hiệu yếu; product/multiplicative combination đôi khi tốt hơn.**

**Critical flaw:** EXP29 AUC = 0.3840 với dấu chưa flipped đang drag ensemble xuống. Cần flip sign trước khi đưa vào EXP15.

**Next modeling step:**
- EXP15 XGBoost stacker: **auto-direction test** — nếu individual AUC < 0.5 thì flip sign. Expected stacker AUC 0.67–0.72.
- **Signal selection**: loại bỏ signal có |AUC - 0.5| < 0.02 (quá gần random) để tránh noise dilution.
- Thử **multiplicative combination** (product of complementary signals) như EXP27's product_score thay vì chỉ rank_avg.

---

### Insight 4 — Context perturbation (variable renaming) không tạo ra Δ JSD phân biệt (EXP28)

**Observation:** EXP28 Delta JSD Trajectory — Δ = JSD_early(perturbed) − JSD_early(original) với deterministic variable renaming trên 50% đầu file. **AUC = 0.5107** (barely above random). Mean Δ(Members) và Δ(Non-Members) **gần như trùng nhau** (ví dụ Go: −0.0005 vs −0.0005; Java: 0.0000 vs −0.0000; Python: −0.0002 vs −0.0004). Subset AUC: Python 0.5306, Rust 0.5156, Go 0.5079, Java 0.5004, Ruby 0.4893 — không subset nào tách biệt rõ.

**Hypothesis (ban đầu):** Member memorized → original settles early (low JSD); renaming phá chuỗi token thuộc lòng → model "think deeper" → JSD_early tăng → Δ lớn. Non-member → Δ ≈ 0. **Kết quả:** Hypothesis **không được ủng hộ** — Δ gần 0 cho cả hai nhóm.

**Có thể giải thích:** (1) Renaming chỉ áp dụng nửa đầu file, token sampling JSD lại lấy đều cả sequence → nhiều token không bị đổi, nên JSD_early(perturbed) ≈ JSD_early(original). (2) Model có thể không "early-settle" theo cách mà đổi tên biến làm gián đoạn — tín hiệu JSD đã yếu (EXP26 0.4371), nên delta càng khó phân biệt. (3) Deterministic map (i→idx_var, data→_data_, …) có thể ít gặp trong thực tế code (naming convention khác) → perturbed text vẫn rất giống original về mặt thống kê.

**Paper angle:** Negative result có giá trị: "Variable-renaming-induced Δ JSD does not discriminate membership on code LLMs under our setup" — nên báo cáo để tránh người khác lặp lại thiết kế tương tự mà kỳ vọng AUC cao.

**Next modeling step:**
- **Perturbation mạnh hơn:** statement reorder, block swap, hoặc **gradient-directed edit** (sửa token có gradient lớn) thay vì chỉ rename. So sánh Δ JSD hoặc Δ loss.
- **Perturb toàn file** hoặc chỉ **đoạn có JSD được dùng** (e.g. first 50% tokens tương ứng với early layers) để perturbation align với metric.
- Ưu tiên **gradient-based perturbation** (EXP32 OR-MIA, EXP31 tail loss) thay vì pure symbolic rename cho code MIA.

---

### Insight 5 — EXP31 Tail-Loss Delta: “Tail Tells All” hypothesis NOT confirmed on code LLMs

**Observation (re-run after BFloat16 fix):** EXP31 ran successfully (10K samples, 10%, 20 min A100):
- **Δ_rel (core signal)**: AUC = 0.5056 at noise=10%, 0.5164 at noise=20% — **near random**. Δ(M) = 0.0003 ≈ Δ(NM) = 0.0003.
- **input_grad_norm**: AUC = 0.3835 — **sign bug**: `score_input_grad = input_grad_norm` (higher=member) but members have LOWER gradient. Flip → AUC ≈ 0.6165 (redundant with EXP11 0.6472).
- **combined_rank_score**: AUC = 0.4270 — dragged by rank-avg of near-random Δ_rel + inverted grad.
- **Loss confirms memorization**: Members 0.8068 < Non-members 0.9681 (✓).

**Why Δ_rel fails despite clear loss gap:**
1. Gaussian noise 5–20% of embedding std is too small to escape local basin — both groups experience proportionally similar micro-perturbations.
2. On code LLMs, non-members may sit in their own local minima (higher loss, but similar curvature) rather than flat plateaus.
3. Code’s structural regularity (syntax, indentation) smooths the loss surface uniformly, unlike natural language.

**Sign bug lesson:** `score_input_grad = input_grad_norm` assumes higher norm = member. But EXP11/27/30/34 all show lower norm = member (flat minima). Must negate: `score = -input_grad_norm`. After flip, AUC ≈ 1 − 0.3835 = 0.6165.

**Paper angle:** Strong negative result — “Continuous embedding noise does not reveal differential loss curvature for code LLMs despite clear loss gap.” Combined with EXP28 (discrete perturbation), paints complete picture: perturbation-based MIA ineffective for code domain.

**Conclusion for EXP15:** Drop Δ_rel entirely. The -input_grad_norm is redundant with EXP11/EXP27 grad_embed.

---

### Insight 6 — Data-driven layer selection confirms late-layer memorization concentration (EXP30)

**Observation:** EXP30 probed 91 gradient components (embed + 30×{attn, mlp, norm} + head). Probe phase (100M+100NM) selected top-5: **layer_28_norm, layer_29_norm, layer_24_norm, layer_29_mlp, layer_29_attn** — all in final 6 transformer layers. Overall AUC **0.6456**; EXP22 heuristic 0.6337 (+0.012); EXP11 full embed 0.6472 (-0.002). PVC Norm ratio: Python 0.362 vs 0.662 (1.83x), Ruby 0.494 vs 0.895 (1.81x), Go 0.313 vs 0.510 (1.63x).

**Mechanism hypothesis:** Memorization concentrates in the final representation-refinement layers (24-29) where predictions narrow to the specific token IDs seen during training. Surprisingly, **layer-norm weights outrank attn/MLP weights** as top discriminators — layer norms act as “gain knobs”: in memorized sequences they converge to narrow activation distributions, leaving a strong per-sample fingerprint in their gradient.

**Paper angle:** “Privacy-Vulnerable Component Identification via Data-Driven Gradient Probing” — removes heuristic layer selection; top components cluster in the final ~15% of layers for code LLMs; layer normalization gradient as an under-explored memorization signal.

**Next modeling step:**
- **Bookend + PVC Mask hybrid**: embed (EXP11) + PVC top-5 (EXP30) → expected AUC 0.66-0.69 (partially orthogonal signals).
- **Top-K sensitivity**: try K=3, 10, 20 to find optimal mask size (K=5 → 0.6456).
- **Anchor-token PVC gradient**: compute PVC-masked gradient only on `def`/`class`/`import` positions — code-specific refinement.

---

### Insight 7 — Language-specific memorization hierarchy: Go >> Python > Ruby/Java >> Rust

**Evidence (subset AUC across gradient-based experiments):**

| Subset | EXP27 GradEmbed | EXP30 PVC | EXP32 G_orig | EXP33 CAMIA | Avg Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Go** | **0.7097** | **0.7052** | 0.6448 | **0.6515** | **#1** |
| **Python** | 0.6739 | 0.6743 | 0.6580 | 0.6351 | #2 |
| **Ruby** | 0.6891 | 0.6901 | **0.6766** | 0.5755 | #2–3 |
| **Java** | 0.6272 | 0.6153 | 0.5748 | 0.5905 | #4 |
| **Rust** | 0.5811 | 0.5821 | 0.5666 | 0.5875 | #5 |

**Observation:** Go consistently achieves the highest per-language AUC (0.70+), while Rust consistently sits at the bottom (~0.58). This pattern is stable across 3 independent signal families (gradient magnitude, data-driven PVC, loss trajectory). Ruby is volatile — high with gradient signals (0.69) but low with CAMIA (0.58).

**Mechanism hypothesis:** Go has highly idiomatic, rigid syntax/structure (“only one way to write it”) → more distinct memorization fingerprint. Rust’s complex ownership/lifetime semantics lead to diverse code patterns for the same logic → noisier memorization signal. Python and Ruby sit in between; Ruby’s dynamic nature makes trajectory-based features harder.

**Paper angle:** Language-specific memorization analysis — first empirical comparison of MIA vulnerability across 5 programming languages on the same model; recommends per-language calibration for production MIA tools.

**Next modeling step:**
- **Per-language threshold calibration** in EXP15: train separate XGBoost models or use language as a categorical feature.
- **Language-weighted ensemble**: weight gradient signals higher for Go/Python, trajectory signals higher for Java/Rust where gradient ceiling is lower.

---

### Insight 8 — Gradient signal ceiling ~0.65: convergence of 6 independent methods

**Evidence:**

| Method | Core idea | AUC |
| :--- | :--- | :--- |
| EXP11 | Full embed gradient norm | 0.6472 |
| EXP13 | Per-layer gradient trajectory | 0.6446 |
| EXP22 | Hand-picked 5 layers | 0.6337 |
| EXP27 | grad × JSD product | **0.6484** |
| EXP30 | Data-driven PVC top-5 | 0.6456 |
| EXP34 | g_mean (all 32 components) | 0.6383 |
| EXP32 | OR-MIA -G_orig (embed grad) | 0.6165 |
| **EXP42** | **Full dataset 50K (-grad_embed)** | **0.6500** |

**Observation:** **Seven** completely different architectures for extracting gradient-based features all converge to **AUC ≈ 0.64–0.65**. This suggests a hard ceiling for single gradient-based features on StarCoder2-3B with this dataset size (5–10% sample, 5K–10K samples).

**Implication:** Breaking past 0.65 requires either: (1) fundamentally different signal families (loss trajectory, attention, perturbation sensitivity), or (2) **stacking complementary features** via EXP15. A single gradient metric is unlikely to push past ~0.66 alone.

**Paper angle:** “Universal gradient ceiling” — demonstrate convergence across diverse gradient extraction strategies; argue this ceiling reflects the intrinsic information-theoretic limit of gradient magnitude as a memorization proxy for this model/data scale.

**Next modeling step:**
- Focus effort on EXP15 stacker (combine gradient + trajectory + attention features) rather than inventing more gradient variants.
- Test with full dataset (100% sample) to check if the ceiling is a sampling artifact.

---

### Insight 9 — Perturbation-based approaches consistently underperform direct measurement

**Evidence:**

| Perturbation Method | Signal | AUC | Compute Cost | vs. Direct Gradient |
| :--- | :--- | :--- | :--- | :--- |
| EXP24 (token sub, CV) | Stability CV | 0.4455 | 4 fwd+bwd | −0.20 vs EXP11 |
| EXP28 (variable rename, ΔJSD) | Delta JSD | 0.5107 | 2 fwd | −0.14 vs EXP11 |
| EXP23 (dropout variance) | Direction stability | 0.4852 | 5 fwd+bwd | −0.16 vs EXP11 |
| **EXP31** (embed noise, Δloss) | **Tail loss delta** | **0.5056** | 3 fwd+1 bwd | **−0.14 vs EXP11** |
| **EXP32** (embed noise, Δgrad) | **OR-MIA ΔG** | **0.5130** | 7 fwd+bwd | **−0.13 vs EXP11** |
| **EXP11 (direct grad)** | **Gradient norm** | **0.6472** | **1 fwd+bwd** | **baseline** |

**Observation:** **All 5 perturbation methods** (EXP23, EXP24, EXP28, EXP31, **EXP32**) add computation cost (2–5×) while delivering worse AUC than a single direct gradient pass. The perturbation hypothesis (“members are more/less sensitive to input changes”) has not been validated on this code LLM setup.

**Possible explanation:** (1) Perturbations (token substitution, variable renaming, dropout) are too coarse to probe the fine-grained curvature of the loss surface near memorized points. (2) The “sharp minimum” signal is already captured by the gradient norm itself — perturbation adds noise without new information. (3) Both EXP31 (loss delta, AUC 0.5056) and EXP32 (gradient stability, AUC 0.5130) used continuous Gaussian embedding noise (theoretically optimal) and **both failed** — **definitively confirming** that perturbation-based approaches are fundamentally flawed for code LLM MIA, regardless of whether you measure loss change or gradient change.

**Paper angle:** Negative result section — “Perturbation stability is a poor proxy for memorization in code LLMs; direct gradient measurement strictly dominates.” **DEFINITIVE CONCLUSION:** 5/5 perturbation methods failed. Both discrete (EXP23/24/28) and continuous (EXP31/32) perturbations are ineffective. Direct gradient measurement strictly dominates for code LLM MIA.

**Next modeling step:**
- **Perturbation family closed.** EXP31 Δ_rel (0.5056) + EXP32 ΔG (0.5130) both near random. Drop all perturbation signals from EXP15.
- **Save compute budget** for gradient + trajectory features (EXP15 stacker, semantic CAMIA, bookend ensemble).
- **For paper:** 5-method perturbation ablation table is a strong contribution — saves future researchers from repeating these experiments.

---

### Insight 10 — EXP30 vs EXP34: selective masking beats meta-feature derivation

**Evidence:**
- **EXP34** (block-wise privacy signature): scans all 32 components, derives meta-features (entropy of gradient vector, cross-block correlation CBC, resonance RPS, inter-block variance ratio IBVR). Combined rank AUC = **0.5608**; but raw g_mean alone = **0.6383**.
- **EXP30** (PVC mask): same 91 components probed, selects top-5 by normalized absolute difference, uses RMS of masked gradient norms. AUC = **0.6456**.

**Observation:** EXP34’s meta-features (CBC 0.5498, entropy 0.4448, IBVR 0.4635) actually hurt when rank-averaged with g_mean (0.6383 → 0.5608, a −0.08 drop). Meanwhile, EXP30’s simple selective masking preserves signal: 0.6456 > 0.5608. This reinforces Insight 3: rank-averaging weak signals with a strong one degrades the strong one.

**Mechanism:** Gradient distribution meta-features (entropy, correlation) capture structure of the gradient vector but are orthogonal to the “magnitude = memorization” axis. When combined naively, they add noise. Simple selection (keep only the most discriminative components) is more effective.

**Paper angle:** “Simple beats complex” for gradient-based MIA — data-driven component selection (EXP30) outperforms engineered meta-features (EXP34) despite using the same underlying gradient measurements.

**Next modeling step:**
- For EXP15: prefer raw per-component gradient norms (gnorm_embed, gnorm_layer_28, gnorm_layer_29, gnorm_head) over derived features like CBC/entropy/IBVR.
- Use XGBoost to learn non-linear feature importance rather than hand-crafting meta-features.

---

### Insight 11 — EXP32 OR-MIA: gradient magnitude works, gradient stability does not

**Observation:** EXP32 separated -G_orig (magnitude) from -ΔG (stability) cleanly:
- **-G_orig AUC = 0.6165**: Members G=1.24, Non-members G=2.12 (1.71× ratio). Confirms flat-minima hypothesis. Consistent with EXP11 (0.6472), EXP34 g_mean (0.6383).
- **-ΔG AUC = 0.5130**: Near random. ΔG(M)=0.095 vs ΔG(NM)=0.089 at noise=1% — **members slightly MORE sensitive**, opposite of hypothesis. At noise=5%: ΔG(M)=0.128 vs ΔG(NM)=0.142 — direction flips but still near-zero separation.
- **Rank-avg degrades**: 0.6165 (G_orig alone) → 0.5824 (combined) — classic Insight 3 pattern.

**Why stability fails while magnitude works:**
1. G_orig captures the ABSOLUTE position on the loss surface (deep basin vs steep slope). This is a first-order signal with large effect size (1.71× ratio).
2. ΔG measures the CURVATURE (second-order derivative). The curvature difference between member and non-member regions is too small to detect with finite noise perturbation. Both groups have similar local Hessian structure despite different loss values.
3. The “flat minimum = stable gradient” theory assumes the minimum is flat in ALL directions. But embeddings are high-dimensional (3072-D for StarCoder2); noise in random directions may not probe the relevant curvature axes.

**Key numbers for paper:**
- G_orig ratio: M/NM = 1.24/2.12 = 0.585 (member gradient is 58.5% of non-member)
- ΔG ratio: M/NM = 0.095/0.089 = 1.07 at 1% noise (essentially no difference)
- Ruby has strongest gradient signal (0.6766) while Rust is weakest (0.5666) — consistent with Insight 7 language hierarchy

**Conclusion:** For EXP15 stacker, use -G_orig from EXP32 as additional gradient feature (10K sample, may help with Ruby/Python where it scores 0.66-0.68). Drop all ΔG/ΔG_rel features.

---

### Insight 12 — Gradient profile SHAPE adds no information beyond magnitude (EXP35)

**Observation:** EXP35 extracted full 31-dim gradient profile vectors (embed + 29 blocks + head) and tested three learned projections:
- **LDA**: probe AUC 0.7632, inference AUC **0.5573** (↓ severe overfitting)
- **Logistic Regression**: 0.6415
- **Cosine to member centroid**: 0.6370
- **-GradMean (scalar mean)**: **0.6417**

All three methods that generalize (LR, Cosine, -GradMean) converge to the same AUC ≈ 0.64. LDA's 0.76 on probe was pure overfitting (31 features / 200 samples = curse of dimensionality).

**Key finding:** The optimal LDA/LR/Cosine projections all collapse to the **negative mean magnitude direction**. This means the per-layer gradient profile vector's discriminative axis IS just "lower overall magnitude = member." There is no exploitable pattern in the RELATIVE distribution across layers (e.g., "high embed but low L15" is not a member signature).

**Implication for paper:** "Gradient magnitude is a sufficient statistic for memorization detection via gradients" — the full gradient fingerprint is informationally equivalent to a single scalar. This is a strong theoretical claim supported by EXP35's three independent projections converging.

**Lesson for EXP15:** No benefit from including 32 individual gnorm_* features. Use 1–4 bookend norms (embed, L28, L29, head) max.

---

### Insight 13 — Anchor token hypothesis inverted: Members memorize EVERYTHING, not just structure (EXP36)

**Observation:** EXP36 tested whether code anchor tokens (def/class/import/func/fn/struct) have disproportionately low loss for members. Results across ALL 5 languages:
- Go: anchor_body_gap M=0.069 vs NM=0.098
- Java: M=0.373 vs NM=0.474
- Python: M=0.216 vs NM=0.285
- Ruby: M=0.494 vs NM=0.742
- Rust: M=0.060 vs NM=0.097

NM consistently has LARGER gap than M. Anchor-based signals: anchor_body_gap AUC=0.4521, -anchor_ratio AUC=0.4675 (both inverted).

**Mechanism:** Members have low loss on ALL tokens (anchors AND body) because the model memorized the entire file. Non-members have moderate anchor loss (keywords are universal patterns) but high body loss (specific logic/variables unknown) → larger gap. The "structural anchor = memorization marker" hypothesis is wrong; memorization is holistic, not structural.

**Paper angle:** Important negative result for code MIA: "Code structural keywords are universal, not member-specific. Token-level MIA signals must focus on the unpredictable tokens (variable names, logic), not the predictable ones (keywords)."

**Lesson for future experiments:** If testing token-level signals, focus on BODY tokens (variables, string literals, numeric constants) where the model's prediction varies between member/non-member, not anchors where it's always confident.

---

### Insight 14 — Loss trajectory derivatives add negligible information over mean loss (EXP37)

**Observation:** EXP37 computed first/second derivatives of smoothed per-token loss trajectory. Cliff features were weakly discriminative:
- cliff_count: M=3.78 vs NM=3.30 (AUC 0.5449)
- max_cliff_mag: M=0.345 vs NM=0.326 (AUC 0.5388)
- cliff_position_first: M=0.247 vs NM=0.286 (AUC 0.5388, earlier for M ✔)
- slope_overall: M=−0.918 vs NM=−1.158 (AUC 0.4249, **inverted** — NM drops faster)

Combined rank AUC = 0.5664; raw -mean_loss alone = 0.5556. Derivative features add only +0.01.

**Why derivatives fail where CAMIA succeeds:** CAMIA uses fixed BLOCKS (256 tokens) which average out token-noise, creating stable trajectory points. EXP37's derivatives operate on token-level smoothed signal, which is still noisy even with window=32. The "recognition cliff" exists but is too gradual and variable in position to be captured by simple derivative metrics.

**Slope inversion explained:** NM starts with higher initial loss (unfamiliar code) and drops more steeply as context accumulates. M starts with already-low loss, so slope is flatter. This means -slope is actually anti-correlated with membership — another Insight 3 pattern of hidden signal inversion.

**Lesson:** Stick with block-based CAMIA features (MDM, TVar) rather than token-level derivatives. The optimal CAMIA granularity is coarse (~256 tokens/block), not fine-grained.

---

### Insight 15 — Learned stackers need ≥ 1000 probe samples to beat single features (EXP38 + EXP35)

**Observation:** Two independent learned methods failed to beat single -grad_embed:
- **EXP38 XGBoost** (18 features, 400 probe): CV AUC 0.547 ± 0.061, test AUC **0.6347** vs -grad_embed **0.6422** (−0.0075)
- **EXP35 LDA** (31 features, 200 probe): probe AUC 0.7632, test AUC **0.5573** vs -GradMean **0.6417** (−0.0844)

Both show the same pattern: **probe overfitting + test degradation** when features/probe ratio is too high.

**EXP38 specific failure modes:**
1. **seq_len dominance** (importance 0.089): XGB learned spurious length correlation instead of real signals.
2. **Auto-flip instability**: 11/18 features flipped on only 400 samples — some flips may be wrong.
3. **CV variance** (fold range 0.455–0.601): Extreme instability across folds = unreliable model.

**Rule of thumb derived:** For XGBoost with K features, need probe_size ≥ 50×K. For 18 features: ≥ 900 samples. For LDA with K features: ≥ 100×K. For 31 features: ≥ 3100 samples.

**Fix path for next stacker:**
1. **Reduce features to 5–7**: only use signals with individual AUC > 0.55 (-grad_embed, -grad_L28, -grad_L29, surp_score, camia_combined, -mean_loss).
2. **Increase probe to 1000+** (500M + 500NM).
3. **Drop seq_len** features entirely (spurious).
4. **Use Ridge LR** instead of XGBoost — linear model needs fewer samples and is more robust to feature noise.
5. **Add product_score** (-(grad × loss)) as engineered feature rather than letting XGB discover interactions.

---

### Insight 18 — EZ-MIA error-position hypothesis confirmed directionally but too weak without a true reference model (EXP46)

**Observation:** EXP46 ran EZ-MIA with temperature-scaled self-reference (T=5, starcoderbase-1b gated). All signals clustered 0.51–0.56:
- **mean_delta_error 0.5645** (best) — M=4.2054 vs NM=4.1173 (correct direction ✓)
- **EZ Score P/N 0.5527** but **slightly inverted** — M=242.18 vs NM=271.92 (NM accumulates more upward mass)
- **-error_fraction 0.5477** — M=0.2114 vs NM=0.2303 (members make fewer errors ✓)
- Baseline Loss: 0.5611 (consistent with EXP01/EXP44)

**Why EZ-MIA is weak with self-temp reference:**
1. **Temperature scaling ≠ pre-training gap**: T=5 makes all distributions more uniform by the same factor for all inputs. It does NOT simulate "what the model knew before seeing this document." A true reference checkpoint would show large deltas at memorized positions; T=5 creates nearly identical delta structure for M and NM.
2. **EZ Score inversion explained**: With self-temp, tokens where model is very confident (high p) generate large upward delta. Non-members have higher overall loss → more error positions → more opportunities to accumulate P. Net: NM P/N (271.92) > M P/N (242.18) — inverted. The effect is an artifact of the reference, not memorization.
3. **mean_delta_error survives (barely)**: Average delta per error position measures per-error signal intensity, not total accumulation. Members slightly higher (4.21 vs 4.12, Δ=0.09, <2% relative) — this is the core EZ-MIA mechanism, directionally confirmed but effect size tiny.
4. **Fine-tuning vs pre-training**: Paper's 0.893 on code used fine-tuned model where before/after training delta is large and concentrated on memorized content. Pre-training memorization is diffuse across massive corpus with a single epoch.

**Key numbers for paper:**
- Error fraction gap: NM 0.2303 vs M 0.2114 (+9% more errors for NM) — directionally confirms EZ-MIA hypothesis
- mean_delta gap: M 4.2054 - NM 4.1173 = 0.0881 (<2% relative) — too small for practical use
- AUC 0.5645 vs EXP43 AttenMIA 0.6642: EZ-MIA is 10 AUC points weaker

**Paper angle:** "EZ-MIA hypothesis partially confirmed for code pre-training (error rate and delta direction correct) but signal is insufficient with temperature-scaled self-reference. A true pretrained reference model is required to realize the full EZ-MIA signal."

**Next modeling step:**
- **Drop all EZ signals from EXP15** — max 0.5645, weaker than baseline loss (0.5611 → marginally useful only if reference model accessible).
- EXP47 WBC also needs starcoderbase-1b reference; expect same access issue — consider alternating to public models (e.g., codeparrot/codeparrot-small) as surrogate reference.

---

### Insight 22 — memTrace hidden-state features reach 0.6908 (NEW BEST); mid-layer norm is the dominant discriminator (EXP50)

**Observation:** EXP50 ran the full memTrace pipeline (Makhija et al. arXiv:2509.05449v1) with 5 feature families extracted from 31 hidden state layers, 5-fold stratified RF.
- **RF CV Mean AUC: 0.6908 ± 0.0093** — NEW BEST, surpassing AttenMIA (0.6642) by +0.027 and the gradient ceiling (EXP41 0.6539) by +0.037.
- Fold stability: 0.6796–0.7035 (low variance ±0.0093, much better than EXP38 XGB ±0.061).
- **Subset**: Go 0.7274, Ruby 0.7074, Python 0.7047, Java 0.6787, Rust 0.6279. Language hierarchy preserved and all subsets improve over gradient baseline.

**Feature breakdown:**
- **Best unsupervised**: `-hnorm_std_L15` 0.6335, `-hnorm_global_mean` 0.6260, `-hnorm_mean_L15` 0.6161 — hidden state norms at mid-layer (L15 = layer 15 of 30) dominate unsupervised ranking.
- **Best RF importances**: seq_len 0.0898 ⚠️, hnorm_global_mean 0.0516, hnorm_std_L15 0.0397, hnorm_mean_L15 0.0385, ctx_evolution_std 0.0352.
- **seq_len ranks #1**: same length confound as EXP49. Members are longer on average (M=871 tokens vs NM=590 in EXP49), and RF is exploiting this. RF with seq_len dropped will have slightly lower AUC but cleaner generalization.
- **True signal hierarchy** (length-independent features): mid-layer norm stats → stability minima → context evolution std → entropy/confidence → position features.

**Why hidden state norms discriminate membership:**
1. Members produce lower hidden state norms at mid-layers because the model has "settled" into a familiar attractor basin. The forward pass requires less representational work for memorized code.
2. Non-members produce higher norms as intermediate layers work harder to map unfamiliar token sequences to plausible continuations.
3. Norm STD (spread across token positions) is higher for members: some tokens are recognized instantly (near-zero norm change), others require adjustment — heterogeneous processing vs uniform high-norm processing for NMs.
4. Paper's "recognition hot-spots" manifest as high confidence variance AND high norm variance at specific positions — consistent with our `ctx_evolution_std` (5th in RF importances, 0.0352).

**Why memTrace beats AttenMIA (+0.027):**
- AttenMIA (EXP43) uses attention weights only as a lens on model behavior. memTrace accesses the ACTUAL intermediate representations (hidden states), which contain richer information about the model's internal state.
- By combining 5 feature families (transition, confidence, norm, position, evolution), the RF can learn which combination is most predictive for each language/style.
- 69 features >> AttenMIA's ~30 attention features, with a larger RF (200 trees, depth 8) providing better capacity.

**Length confound caution:** seq_len is #1 importance (0.0898). The 48% length gap (M=871 vs NM=590 tokens) from EXP49 carries over here. True signal from hidden states alone: estimate ≈0.67–0.69 (subtracting 1–2 AUC points from length). Research quality: should report memTrace AUC both with and without seq_len for transparency.

**Paper angle:** "memTrace-adapted features (hidden state norm at middle layers, stability minima, context evolution) achieve AUC 0.6908 on code pre-training MIA, establishing a new state-of-the-art for StarCoder2-3b. Middle-layer hidden state norm variance is the single strongest discriminator, consistent with the flat-minima hypothesis: members produce lower, more heterogeneous hidden state norms."

**Next modeling step:**
- **EXP15 stacker update**: add `hnorm_std_L15`, `hnorm_global_mean`, `stability_min_L15/L22`, `ctx_evolution_std` as features (length-independent, complementary to gradient). Combine with -grad_z_lang (0.6539) and attn_full_score (0.6642) — expected stacker AUC 0.70–0.73.
- **Drop seq_len** from EXP50 features in stacker to avoid length confound amplification.
- **EXP51 LUMIA** (linear probes on raw activation vectors per layer) will test whether richer layer-specific information is accessible beyond norm statistics alone.

---

### Insight 21 — Tokenizer signals are dominated by sequence length confound; genuine BPE signals reach 0.59 (EXP49)

**Observation:** EXP49 ran tokenizer-only MIA (Tong et al. arXiv:2510.05699v1) on the full 100K dataset in ~3.5 min without any model inference.
- **Apparent top signals: `n_tokens` 0.6620, `n_unique_tokens` 0.6609, `-token_type_ratio` 0.6331** — all dramatically above the 0.55 threshold.
- **Root cause: LENGTH CONFOUND** — Member sequences are **48% longer** on average (M=871.6 tokens vs NM=589.9 tokens, ratio 1.4774). Longer files naturally have more tokens and more unique tokens; lower type-token ratio follows algebraically (TTR decreases as sequence grows).
- **Genuine length-independent signals**: `merge_idx_max` **0.5923** (max BPE merge index — not length-correlated since it takes the maximum, not a count), `bytes_per_token` **0.5674** (compression rate is per-token, normalized by length).
- `rare_token_count_p90` (0.5887) is partially length-confounded (count scales with length); should be normalized to `frac_above_p90` (0.5487) for a cleaner signal.
- Rarity score (0.5205) is weak — the power-law SI proxy via merge index is too coarse.

**Why members are longer (length confound origin):** The Poisoned-Chalice dataset may have a systematic split: member files (used in pre-training) were selected/filtered differently than non-member files. Longer code files are more likely to be included in pre-training datasets (more content → more training signal). This is a **dataset construction artifact**, not a tokenizer-memorization signal.

**Paper's dataset-level insight doesn't transfer cleanly to sample-level:** The paper's AUC 0.771 was for entire websites (aggregate compression/vocab overlap over many documents), where length effects average out. At individual file level, per-file length variation dominates.

**True tokenizer signals:**
1. **`merge_idx_max` (0.5923)**: Member files contain at least one token with a very high merge index (merged late in BPE training = rare, possibly file-specific vocabulary). This is genuine — a single idiom or library-specific name can push the max high. NM average max = 41349 vs M = 44187 (Δ≈2838 merge steps ≈ 5.8% of vocab).
2. **`bytes_per_token` (0.5674)**: Members compress slightly more efficiently (M=3.89 vs NM=3.70 bytes/token). Effect is small (5.3% relative) but consistent across all 5 languages.
3. **`frac_above_p95` (0.5440)**: Member files have a marginally higher fraction of tokens in the rarest 5% of vocab. Very weak but length-normalized.

**Conclusion for EXP15:** 
- **DROP** `n_tokens`, `n_unique_tokens`, `-token_type_ratio` — pure length proxies, will cause spurious correlations in EXP15 stacker.
- **INCLUDE** `merge_idx_max` (0.5923) — genuine, orthogonal to all gradient/attention signals, free to compute.
- `bytes_per_token` (0.5674) — marginal, borderline worth including.
- Tokenizer signals are completely orthogonal to model-based signals: zero GPU cost, can be pre-computed once for full dataset and reused in any stacker.

**Paper angle:** "BPE tokenizer's max merge index provides a weak but genuine per-sample membership signal (AUC 0.59) orthogonal to gradient-based features; the dominant apparent signals (sequence length, token count) reflect dataset construction bias rather than tokenizer memorization."

---

### Insight 20 — WBC sliding-window advantage collapses at w≥4 with self-temp reference; fine-tuning gap is required (EXP47)

**Observation:** EXP47 ran the full WBC pipeline (Chen et al. arXiv:2601.02751v1) with geometric window sizes W={2,3,4,5,8,11,15,21,29,40}, self-temp(T=2.0) fallback (starcoderbase-1b gated), 10K samples.
- **Best signal: loss_ratio = 0.5624** (not the WBC ensemble)
- **WBC Ensemble: 0.5205** — improves over raw global mean diff (+0.0588) but still beaten by trivial loss_ratio
- **Per-window AUC collapse**: w=2 → 0.5401, w=3 → 0.5173, w=4 → 0.4916, w≥5 → 0.47–0.50 (near-random or inverted)
- Distribution delta: WBC M=0.9929 vs NM=0.9920, **Δ=0.0009** (negligible, close to floating-point noise)
- **Loss diff inverted**: M=0.9647 vs NM=0.9979, Δ=−0.0332 — self-temp(T=2.0) makes NM have larger loss difference, inverting the signal direction
- Best window per subset is always w=2; all subsets: Go 0.5332, Java 0.5475 (best), Python 0.5110, Ruby 0.5792 (best), Rust 0.5361

**Why WBC fails to reproduce paper results for pre-training MIA:**
1. **Self-temp ≠ true reference**: T=2.0 rescaling is a deterministic monotonic transform of the same model's logits. It creates a near-constant ratio across all tokens, so the loss difference (ℓ_R − ℓ_T) is dominated by length/entropy effects, not memorization. This is identical to EXP46's root failure.
2. **Window size collapse mechanism**: With a weak reference, the per-token signal (ℓ_R[i] − ℓ_T[i]) is dominated by random per-token noise. WBC's sign test averages these noisy votes: at w=2, local noise occasionally clusters in the correct direction; at w≥4, the law of large numbers washes out any bias → T_sign(w) converges to 0.5 for both M and NM equally.
3. **Paper's fine-tuning setup**: In the paper, target = fine-tuned model (overfit on ~10K examples), reference = original pre-trained checkpoint. The delta (ℓ_R − ℓ_T) is LARGE and CONCENTRATED on fine-tuning data → windows spanning memorized chunks cast reliable member votes. Pre-training: memorization is diffuse across billions of tokens, per-token delta is tiny (~0.03 globally, much less per token).
4. **Loss diff inversion**: Self-temp at T=2.0 compresses all logits uniformly → tokens where the model is already confident (members) see a larger negative shift than tokens with high uncertainty. This accidentally gives NM a higher ℓ_self-temp − ℓ_orig delta → inverted signal.

**Why w=2 works marginally**: A size-2 window is essentially comparing two adjacent token pairs. At this granularity, the random noise hasn't averaged out yet, and there may be very weak local co-occurrence memorization signals. But this is a fragile artefact, not the paper's intended mechanism.

**Conclusion**: WBC requires a **true independent reference model** (different checkpoint or different-scale model with genuine pre-training gap). Self-temperature scaling is insufficient. Combined with EXP46 (EZ-MIA same issue), this confirms: **all reference-based methods tested require a real external reference to be competitive** — temperature-scaled self-reference universally produces AUC ≤ 0.56 regardless of algorithm complexity. The `loss_ratio` (0.5624) is the only useful signal from EXP47, and it's redundant with EXP16 SURP (0.5884).

**Paper angle:** "Sign-test robustness of WBC is effective at suppressing outlier noise when a genuine reference model exists, but offers no benefit when the reference is a monotonic transform of the target — window averaging then amplifies the null hypothesis rather than the membership signal."

**Next modeling step:**
- **Drop wbc_ensemble and loss_diff from EXP15** — both ≤ 0.52, weaker than any gradient signal.
- `loss_ratio` (0.5624) is marginally useful but nearly redundant with SURP (0.5884). Include in EXP15 only if Ridge coefficient > 0 after fitting.
- **If starcoderbase-1b access becomes available**: re-run EXP47 + EXP46 with true reference. Expected AUC 0.58–0.65 based on paper's mismatched-reference result.

---

### Insight 19 — CON-RECALL confirms prefix-conditioning family universally fails for code pre-training (EXP45)

**Observation:** EXP45 ran the full CON-RECALL pipeline (Wang et al. arXiv:2409.03363v2) with 5-shot prefixes (member + non-member), gamma search 0.0–1.0, 3 forward passes per sample, 10% dataset (9990 eval samples).
- **All gamma AUCs < 0.5**: ReCaLL (gamma=0) 0.4762 → gamma=1.0 (best) **0.4889** — monotone improvement but never crosses 0.5.
- **After flip**: -CON-RECALL ≈ 0.5111 (weak, barely above Loss baseline 0.5484).
- **Loss (LL) standalone**: 0.5484 — unchanged from EXP01/EXP44, unaffected by prefix conditioning.
- **Subset breakdown (CON-ReCaLL)**: Ruby 0.5708 (sole outlier, borderline inverted), Python 0.4977, Java 0.4935, Rust 0.4616, Go 0.4446.
- **LL stats confirm differential suppression exists but is SYMMETRIC**: M prefix lowers LL by −0.072 for M and −0.093 for NM; NM prefix lowers LL by −0.128 for M and −0.160 for NM. **NM prefix suppresses both groups MORE, not just members** — the assumed asymmetry (NM prefix hurts members, not non-members) does not hold.

**Why CON-RECALL fails on code pre-training (extending Insight 17):**
1. **ReCaLL ratio inversion**: M=1.1493 vs NM=1.1602 — NM unconditional LL is more negative, making it a smaller denominator and inflating NM's ratio. The effect is language-level statistical artifact, not prefix-induced memorization signal.
2. **Symmetric suppression**: The paper's mechanism requires member prefix P_m to barely affect member target (already memorized) while hurting NM target significantly. In code, ALL sequences share structural grammar — a Python function prefix suppresses any Python file (M or NM) nearly equally.
3. **Prefix information leakage**: The 5-shot member prefix contains real code syntax/patterns that help ALL code generation equally, not just recall of seen examples. The contrastive signal (P_nm score − γ·P_m score) ends up comparing two near-equal values → gamma=1.0 only marginally amplifies the small difference.
4. **Paper's 95-98% AUC on WikiMIA** relies on strong topic-correlation: a member article's topic cluster (P_m) strongly predicts its own continuation while being uninformative for other topics. Code has no analogous topic boundaries.

**Gamma analysis:** Monotone increase in best AUC from 0.4762 → 0.4889 (best at gamma=1.0, not converging to a clear optimum). This indicates the contrastive term γ·LL(x|P_m) subtracts a small amount of noise at every step — the formula is working directionally but the base signal LL(x|P_nm)/LL(x) is already inverted.

**Conclusion:** CON-RECALL is the **5th and final** prefix-conditioning method tested (EXP09 canonical gap, EXP17 ICP-MIA, EXP44 EM-MIA/ReCaLL, EXP45 CON-RECALL). All fail for code pre-training. **Prefix-conditioning family is CLOSED — do not add any prefix signal to EXP15.**

**Paper angle:** Strong cluster of 5 negative results all pointing to the same root cause — "code LLMs lack topic locality" — unified explanation for why prefix methods that work on natural language text corpora universally fail on code pre-training MIA.

---

### Insight 17 — ReCaLL/EM-MIA prefix suppression hypothesis fails for code pre-training (EXP44)

**Observation:** EXP44 tested the full EM-MIA pipeline (Kim et al. 2026). Results across all signals:
- **Baseline Loss** (EM subset): 0.5802 — consistent with EXP01 (0.5807). Not beaten by any ReCaLL variant.
- **ReCaLL-Avg**: 0.4756 (inverted — non-member prefix suppresses non-member MORE than member)
- **-AvgP**: 0.3779 (strongly inverted)
- **EM-MIA (Loss init)**: AUC trajectory 0.5028→0.5168→0.5033→0.4809→…→**0.4577** — EM iterations **monotonically worsen** the signal.
- **EM-MIA (MinK++ init)**: 0.4858→…→**0.4549** — same divergence pattern.
- **ReCaLL top-k (full eval)**: 0.4481 — worse than random. Prefix labels of best-10: 4M+6NM — no consistent member-prefix pattern.

**Why EM-MIA fails fundamentally for code pre-training:**
1. **ReCaLL requires differential prefix suppression**: the method relies on a non-member prefix suppressing a member target's LL more than a non-member target. This works in natural language (WikiMIA) because topic/style drift is large — an off-topic prefix derails NM text but doesn't affect M text (model already memorized it unconditionally). Code has **uniform structural grammar** that constrains generation regardless of prefix, so suppression is nearly equal for M and NM.
2. **EM diverges because its pseudo-labels are noisy**: the M-step `f = -r` (membership = negative prefix score) creates a feedback loop that amplifies noise. With a weak initial signal (0.5028), EM latches onto spurious patterns in the 200-sample pairwise matrix and amplifies them in the wrong direction.
3. **Pre-training vs fine-tuning gap**: Paper achieves 97-99% AUC on fine-tuned WikiMIA. Fine-tuned models show extreme ReCaLL separation (model is strongly conditioned by the same domain). Pre-trained StarCoder2-3b on massive corpus: per-sample memorization signal is fundamentally smaller (EXP42 confirmed ceiling at 0.65, not 0.97).
4. **Code ALL looks like code**: A non-member Ruby snippet as prefix doesn't "confuse" the model on a Python member target compared to a Python non-member — the model has seen all combinations during pre-training. Natural language has topic boundaries that code lacks.

**Key number for paper:** EM iterations: AUC[t] = {0.5028, 0.5168, 0.5033, 0.4809, 0.4825, …, 0.4577} — a clean demonstration that EM's feedback loop diverges rather than converges when the base signal is too weak. Loss correlation `corr(f_t, f_{t+1})` stabilizes at ~0.99 while AUC continues declining: the model converges on a **self-consistent but wrong** solution.

**Paper angle:** Second negative result for prefix-based family — EM-MIA's EM-refined prefix selection fails for pre-training code MIA for the same fundamental reason as ReCaLL (no differential suppression). Combined with EXP09 (canonical gap), this closes the prefix-conditioning family for this setting.

**Next modeling step:**
- **Drop all ReCaLL/prefix signals from EXP15** — all inverted or near random, adding only noise.
- **Loss baseline (0.5802)** from EXP44 EM subset is consistent with EXP01 — redundant, no new information.
- EXP45 (CON-RECALL) likely to see similar failure; if run, expect AUC ≤ 0.55 on code pre-training.

---

### Insight 16 — Attention signals break the gradient plateau: AttenMIA full at 0.6642 (NEW BEST)

**Observation:** EXP43 AttenMIA (Full: Transitional + Perturbation) reached **AUC = 0.6642**, beating EXP41's -grad_z_lang (0.6539) by +0.010 and the gradient ceiling consistently observed across EXP11–EXP42 (~0.64–0.65). Transitional-only also beats the gradient ceiling at 0.6573. 5000/5000 valid samples (100%), forward-only, ~35 min A100. CV Mean Full: 0.6477±0.0229 (stable, smaller variance than EXP38's 0.061).

**Feature breakdown:**
- **Best individual features**: conc_mean 0.6508, trans_bary_mean_std 0.6499, trans_bary_var_{mean/std/late} 0.6430–0.6466, trans_kl_{late/mean} 0.6299–0.6319. Barycenter drift and KL concentration are the core discriminators.
- **Inverted features** (need flip for EXP15): trans_frob_* (0.35–0.36), trans_corr_mean/late (0.39–0.40), pert_*_shift_std/max (0.37–0.40). Frobenius norm and layer-to-layer correlation are antiparallel to membership.
- **Perturbation contribution**: Full (0.6642) > Transitional (0.6573) by +0.007. Perturbation features add value despite many individual ones being inverted — Ridge recovers by learning negative weights.
- **conc_std paradox**: individual AUC 0.5979, but Ridge gives it coefficient −0.247 (flipped). This means higher concentration STD = non-member; Ridge learns to invert it, adding signal beyond its raw direction.

**Why attention breaks the gradient ceiling:**
1. **Orthogonal information channel**: gradient magnitude measures the loss surface geometry (flat basin vs steep slope). Attention patterns measure the model's internal routing behaviour — how information flows across token positions. These are genuinely independent aspects of memorization.
2. **Barycenter drift mechanism**: for memorized files, adjacent transformer layers show larger attention shift (higher barycenter drift) — the model is actively reinforcing specific long-range dependencies established during training. Non-members show more uniform, layer-stable attention.
3. **KL concentration mechanism**: member attention is more peaked (higher KL from uniform) — the model is more certain about which tokens to attend to, reflecting memorized token co-occurrence patterns.
4. **Why perturbation adds signal here** (unlike EXP23–32): perturbation features measure concentration SHIFT (relative), not loss CHANGE (absolute). Relative shift is less sensitive to the code complexity confound that made absolute perturbation methods fail.

**Per-language frontier (Full model):** Go 0.6981, Ruby 0.6921, Python 0.6701, Java 0.6572, Rust 0.6167. Language hierarchy (Insight 7) preserved, but all languages improve over gradient baseline (Go: 0.6981 vs grad 0.6921, Ruby: 0.6921 vs 0.6847).

**Paper angle:** "Attention concentration and barycenter drift as memorization indicators" — first systematic evaluation of attention transition features for code LLM MIA; demonstrates that attention signals are complementary to (not redundant with) gradient signals, suggesting a gradient×attention ensemble as a strong foundation.

**Next modeling step:**
- **EXP15 stacker update**: add `attn_full_score` (or its constituent features conc_mean, trans_bary_mean_std, trans_kl_late) to EXP15 feature set. Flip inverted features (trans_frob_*, trans_corr_*, pert_*_shift_std/max) before inclusion.
- **Gradient + Attention ensemble**: -grad_z_lang (0.6539) × attn_full_score (0.6642) — if orthogonal, Ridge/LogReg stacker could reach 0.68–0.70.
- **Reduce perturbation passes** in future runs: drop the 3 perturbation forward passes (keep only 1 original) to cut runtime to ~9 min for transitional-only features (0.6573), accepting −0.007 AUC vs full.

---

### Insight 24 — Distribution shape beats scalar loss but is compromised by length confound; seq_len dominance must be controlled (EXP55)

**Observation:** EXP55 DocLevel Histogram RF achieved **AUC 0.6612 ± 0.0101** using 16 histogram bins + 19 aggregate statistics of per-token log-probability distributions. This beats EXP41 -grad_z_lang (0.6539) and is close to EXP43 AttenMIA (0.6642), using only 1 forward pass (same as EXP01).
- **Distribution shape adds massive value**: 0.6612 vs scalar `agg_mean` 0.5611 (+0.1001) — the SHAPE of the log-prob distribution across tokens is far more informative than just the mean.
- **Best unsupervised features**: `+agg_max` 0.5921, `-agg_std` 0.5733, `+agg_p95/p90` 0.5732/0.5728, `+top_20_mean` 0.5700. Upper tail statistics dominate — how confident the model gets on its easiest tokens.
- **⚠️ seq_len importance = 0.1992 — most severe length confound yet** (EXP50 had 0.0898, EXP38 had 0.089). `seq_len` ranks #1 by large margin, accounting for nearly 20% of RF decision. Members are ~48% longer (EXP49), so they have more tokens → more chances to hit a very high-probability token → `agg_max` and upper percentiles naturally higher for longer files.
- **Top 15 fold importances confirm length cascade**: `seq_len` 0.199 → `agg_max` 0.063 (length-correlated) → `agg_std` 0.054 → `hist_bin_1` 0.045 (bin [-12,-10], low-probability tokens = rare tokens more present in shorter NM files) → `agg_p95` 0.036.
- Subset consistency: Ruby 0.6964, Go 0.6949, Python 0.6724 — same language ordering as gradient/attention signals. Java 0.6369 and Rust 0.6258 remain harder (same pattern as EXP43/EXP50).

**Why distribution shape works even after controlling for confounds (partially):**
1. **Upper tail mechanism**: Members have at least some tokens the model finds very easy (memorized phrases, idiomatic patterns). `agg_max` captures "best token" in the file. Even length-controlled, member files should have more extreme peaks because specific memorized sequences exist.
2. **Lower standard deviation for members**: counter-intuitive — members have LESS variance in per-token log-prob? Expected was bimodal (very high + moderate), but `-agg_std` 0.5733 shows members have lower std. Possible explanation: non-members have very uncertain tokens (novel vocabulary/structure) pulling std up. Members' entire distribution is shifted toward moderate-to-high confidence, with fewer extreme low-probability outliers.
3. **Bimodality hypothesis weakly confirmed but not dominant** (RF importance 0.0243): the paper's insight applies weakly to code pre-training. Some member files ARE bimodal (memorized chunks + transitional code), but the overall statistical signature is a distribution shift rightward, not a full bimodal shape.
4. **Histogram bins as discriminators**: `hist_bin_1` ([-12,-10]) importance 0.0449 — very low probability tokens. Non-members have MORE such tokens (higher fraction in the extreme-left bins). This is a clean signal: NM files have vocabulary the model rarely produces.

**Length confound control strategy for EXP15:**
- **Option A (preferred)**: drop `seq_len` from the feature set and refit RF. Expect AUC to drop to ~0.62–0.63 (true histogram signal). The histogram features that don't scale with length (bin fractions, std, bimodality coefficient, hist_entropy) should survive.
- **Option B**: divide counts by `seq_len` (already done for bin fractions = counts/T), but `agg_max`, `agg_p95`, `top_20_mean` should be kept as-is (they're statistics of the distribution, not totals — but longer files have more extreme samples).
- **Option C**: stratify by seq_len quantiles in CV. More complex but ensures the model learns within-length differences.
- **Recommended for EXP15**: include `hist_bin_0..15` (bin fractions, length-normalized), `agg_std`, `bimodality_coeff`, `hist_entropy`, `-agg_range`, `frac_uncertain` — features that measure distribution SHAPE rather than absolute level. Exclude `seq_len`, `agg_max`, `agg_p95`, `top_20_mean` (too length-dependent).

**Paper angle:** "Per-token log-probability distribution shape provides a +0.10 AUC gain over scalar mean loss; however, the dominant RF feature (seq_len, importance=0.199) reveals a systematic length bias in the dataset — member files are 48% longer than non-members, creating spurious upper-tail statistics. Controlling for this confound reduces the gain but distribution-shape features (histogram bin fractions, std, bimodality) retain genuine signal (+0.05 over mean loss) without requiring gradient computation."

**Key comparison with EXP50 memTrace:**
- EXP50 also used RF on 69 features of HIDDEN STATES (not log-probs), with seq_len importance 0.0898 — already concerning. EXP55's seq_len importance 0.1992 is 2.2× worse, suggesting log-prob features encode length MORE directly than hidden state norm features. This makes sense: `agg_max` is literally the max over T tokens — larger T = more extreme max by statistics of extremes.
- EXP50's true AUC without length: remained high (hidden state norm features genuinely discriminate). EXP55 without seq_len: likely drops more (many aggregate features are length-influenced).

---

### Insight 23 — Embedding drift is the key perturbation discriminator: SMIA unlocks what EXP03/07/28/31/32 missed (EXP52)

**Observation:** EXP52 SMIA ran token-replacement perturbation (rates 5/10/20%, K=3 reps) and measured BOTH `loss_diff` and `sem_dist` (embedding layer cosine/L2 distance between original and corrupted hidden states). **LR CV AUC: 0.6378 ± 0.0069** — breaking the perturbation family ceiling (previous best EXP03 0.5633, EXP07 0.5365).
- **Best unsupervised feature: `+loss_per_sem_r5` = 0.6365** (loss change per unit embedding distance at 5% substitution rate)
- **Dominant LR coefficient: `sem_dist_r5` (coef −0.537)** — smaller embedding drift = member
- **Loss-only signals remain weak**: `loss_diff_r5` 0.5566, `loss_diff_mean` 0.5421 — same as EXP03/07; adding sem_dist as denominator adds +0.08 AUC
- **Semantic calibration HELPS**: 0.6378 vs `-orig_loss` baseline 0.5611 (+0.077)
- Per-subset: Ruby 0.6730, Python 0.6672, Go 0.6608, Java 0.6091, Rust 0.5935 (same language ordering as gradient signals)

**Why SMIA succeeds where EXP03/07/28/31/32 failed:**
1. **Signal decomposition insight**: prior methods computed Δloss (perturbation sensitivity of prediction). SMIA instead computes `loss_diff / sem_dist` — the ratio of prediction sensitivity to embedding-space sensitivity. The numerator (loss change) is shared with prior methods. The denominator (embedding drift) normalizes out individual code complexity: complex code naturally shows larger absolute changes, but members show PROPORTIONALLY less embedding drift per unit perturbation.
2. **Members have more stable embedding layer representations**: The embedding layer (token lookup table + positional embeddings) is the most overfit part of the model for seen sequences. When tokens are replaced, member code's embedding-layer representation stays closer to its original because the model has learned to associate specific token patterns from training (even occasional random tokens may co-occur with memorized patterns). Non-member code has no such anchor → larger embedding drift.
3. **Connects to memTrace Insight 22**: EXP50 found members have lower hidden-state NORM variance at mid layers (flatter attractor basin). Here, members have lower embedding-layer DRIFT under perturbation. Both measure the same phenomenon (the flat-minima effect) in different spaces — norm variance measures equilibrium, drift measures robustness to disturbance.
4. **Why `r5` dominates over `r10` and `r20`**: At 5% substitution, the signal-to-noise ratio is highest — perturbation is small enough that the embedding drift is dominated by the model's structural response (member vs NM) rather than by the random token noise. At 20%, the noise overwhelms the signal and `sem_dist_r20` AUC drops to 0.5988.

**Key mechanism (paper adaptation):** The original SMIA paper (Mozaffari & Marathe 2024) used an **external** embedding model (Cohere Embed V3) to measure semantic distance in a domain-agnostic space. Here we use the target model's own embedding layer, which is effectively the input representation — this is the MOST memorized part of the model (token lookup directly). The paper's external embedding was meant to capture semantic change independent of the target model; our internal embedding is actually STRONGER because it captures the model's own memorized representation space.

**Stacking recommendation for EXP15:**
- Add `smia_lr_score` as a level-0 feature (AUC 0.6378, partially orthogonal to gradient and attention signals since it operates on perturbation × embedding space)
- The sem_dist family (sem_dist_r5, sem_dist_mean, loss_per_sem_r5) is complementary to: gradient norms (EXP41: output space), attention patterns (EXP43: routing space), hidden state norms (EXP50: internal state), and tokenizer features (EXP49: input space). Covers the embedding-perturbation space.
- **Orthogonality check**: similarity to memTrace (EXP50) is partial — both measure representation stability, but memTrace fixes the input and measures norm across layers; SMIA fixes the layer and measures drift across perturbations. Different axes of the same phenomenon → low correlation expected.
- **Drop `seq_len` as usual** — check importance in EXP15.  

**Paper angle:** "Embedding drift under token substitution is the key signal that perturbation-based methods missed: loss change alone suffers from code complexity confound, but normalizing by embedding-layer drift isolates the memorization-specific component. Five prior approaches (neighborhood loss, adversarial stability, delta JSD, embedding noise, gradient stability) all showed AUC 0.50–0.56 because they measured absolute loss or gradient change; SMIA's ratio formulation achieves 0.6378 on the same task."

---

### Failure Mode A — BFloat16 dtype trap

**Affected:** EXP26, EXP27, EXP28, EXP31, EXP32 (ban đầu).

**Root cause:** Model load với `torch_dtype=bfloat16`. Bất kỳ tensor nào cast sang `float32` rồi feed vào model → `RuntimeError: expected scalar type Float but found BFloat16`.

**Fix pattern chuẩn:**
```python
model_dtype = next(model.parameters()).dtype       # tự adapt theo GPU
tensor_in = tensor.detach().to(dtype=model_dtype)  # cast đúng dtype trước feed
output = model(inputs_embeds=tensor_in, ...)
grad_norm = tensor_in.grad.float().norm(2).item()  # đọc norm bằng float32 SAU backward
```

---

### Failure Mode B — `inf` không bị bắt bởi `fillna`

**Affected:** EXP19, EXP20.

**Root cause:** Ratio operations sinh `inf` khi denominator ≈ 0. `pandas.fillna(mean)` không xử lý `inf` → `StandardScaler` crash.

**Fix pattern chuẩn:**
```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[col] = df[col].fillna(df[col].median())
df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
```

---

### Failure Mode D — Score sign direction mismatch

**Affected:** EXP31 (`score_input_grad`), EXP29 (`-entropy_early`).

**Root cause:** Assigning `score = raw_metric` without checking whether higher metric values correspond to members or non-members. EXP31’s `score_input_grad = input_grad_norm` assumes higher norm = member, but all prior experiments show members have LOWER gradient norms.

**Fix pattern chuẩn:**
```python
# Before using any signal as a score, check direction:
auc_raw = roc_auc_score(labels, signal)
if auc_raw < 0.5:
    signal = -signal  # flip direction
# Or always define: score = -gradient_norm (consistent with EXP11/EXP27)
```

**Detection:** If individual signal AUC < 0.5, the sign is likely wrong. Auto-detect in EXP15 stacker.

---

### Failure Mode C — Silent exception swallowing

**Affected:** EXP26, EXP27, EXP28 (version đầu).

**Root cause:** `except Exception: return None` không print → 100% NaN mà không biết lý do.

**Fix pattern chuẩn:**
```python
except Exception as e:
    if self._err_count < 3:
        print(f"[WARNING] {type(e).__name__}: {e}")
    self._err_count += 1
    return np.nan
# Luôn report: print(f"Total errors: {self._err_count}")
```

---

## Modeling Roadmap

### Tier 1 — Highest expected AUC

| Ý tưởng | Dựa trên | Expected AUC |
| :--- | :--- | :--- |
| **EXP15 XGBoost Stacker** (auto-flip + signal selection) | Insight 3 | 0.67–0.72 |
| **Bookend layer ensemble** (embed + L28 + L29 + head) | Insight 1 | 0.65–0.68 |
| **Embed + PVC Mask hybrid** (EXP11 + EXP30 top-5 late-layer norms) | Insight 6 | 0.66–0.69 |
| **Semantic block CAMIA** (function/class boundary splitting) | Insight 2 | 0.63–0.67 |

### Tier 2 — Novel signals chưa thử

| Ý tưởng | Motivation |
| :--- | :--- |
| Per-token gradient norm on anchor tokens (`def`, `class`, `import`) | Code-specific memorization markers |
| Loss trajectory curvature (second derivative) | Finer-grained CAMIA: inflection point detection |
| Cross-language loss normalization | Remove language bias from trajectory features |
| EXP29 sign-flipped + re-ensembled | Attention entropy needs flip to contribute positively |
| `product_score = -(grad × jsd_early)` generalized | EXP27 showed multiplicative > additive rank |
| EXP28-style Δ JSD with stronger perturbation (statement reorder, gradient-directed edit) | Variable renaming alone insufficient (EXP28 Δ≈0, AUC 0.51) |
| ~~EXP31 re-run~~ | ✅ Done. Δ_rel = 0.5056 (near random), hypothesis failed. -input_grad_norm (~0.62) redundant with EXP11. Drop from EXP15. |
| ~~EXP32 re-run~~ | ✅ Done. -G_orig 0.6165 (useful, consistent with EXP11). ΔG 0.5130 (near random). **Perturbation family CLOSED — 5/5 methods failed.** |
| **Top-K sensitivity for EXP30** (K=3,10,20) | Find optimal PVC mask size; K=5 → 0.6456 |

### Tier 3 — Paper framing angles

1. **"Gradient Norm as Memorization Fingerprint in Code LLMs"** — empirical study trên 5 programming languages, flat-minimum hypothesis trên code domain.
2. **"Context-Aware Loss Trajectory for MIA on Code"** — semantic-aware block splitting extension của CAMIA.
3. **"Ensemble MIA via Auto-Directed Signal Fusion"** — why rank-averaging with auto-direction correction + multiplicative combination outperforms single signals.

---

### Insight 25 — Gradient distribution SHAPE and ALIGNMENT reveal membership (NOVEL 07/10)

**Observation**: NOVEL07 (GradOrth-MIA) achieves **AUC 0.6349** using `neg_effective_pressure` (gradient magnitude × hidden state alignment). NOVEL10 (GradSparse-MIA) confirms that members exhibit more **sparse** gradients (highest signal at late layers: `hoyer_late` 0.5359).

**Mechanism**: At a flat minimum (members), the gradient is not just smaller, it's more **orthogonal** to the representation space because the model has already aligned its parameters to the data. Non-members require a "push" in a specific direction, leading to higher alignment (effective pressure). Sparsity metrics (Hoyer, Gini) confirm that members' gradients are concentrated in fewer directions than non-members'.

---

### Insight 26 — Prediction Stability as a Recognition Fingerprint (NOVEL 08/09)

**Observation**: NOVEL08 (PredAgree-MIA) reached **AUC 0.5637** via `-stability_min`. Members show higher prediction stability across layers; they "commit" to the final prediction earlier and flip-flop less. NOVEL09 (ConfTraj-MIA) shows that the confidence trajectory shape (AUC under curve) is slightly discriminative, though dominated by final loss.

**Mechanism**: Memorized sequences have "learned paths" through the transformer layers. The model doesn't need all 30 layers to "know" the next token for a training sample. This manifests as stable predictions in the "logit lens" even in intermediate layers.

---

### Insight 27 — Holistic Coherence: Model sees training files as units (NOVEL 05)

**Observation**: NOVEL05 (GradCoh-MIA) achieved **0.5870**. Gradients from the first half and second half of a member file are more coherent in direction than for non-members.

**Mechanism**: This supports the "holistic memorization" hypothesis. Training code is processed as a complete unit. The model learns a unified representation where all parts of the file contribute to a coherent optimization direction. Non-members, which the model encounters "piecemeal," lack this cross-subsequence gradient alignment.

---

### Insight 28 — Hidden Space Density (k-NN) tracks membership (NOVEL 03)

**Observation**: NOVEL03 (RepDen-MIA) achieved **AUC 0.6559** using `knn_mid_dist_diff` (distance to member probes vs non-member probes in L15 hidden space). This ranks alongside the best gradient and attention signals.

**Mechanism**: This confirms that training samples lie in denser regions of the model's representation manifold. The local geometry around a member is "populated" by other training examples that influenced the weights during optimization. Non-members are outliers in this space.

---

### Insight 29 — Attention Concentration is a dominant, independent signal (NOVEL 02)

**Observation**: NOVEL02 (OrthoFuse-MIA) confirms **AUC 0.6582** for `attn_conc_mean_z_lang`. While fusion with gradient didn't yield a massive leap, attention concentration itself remains one of the strongest and most stable indicators for code LLMs.

**Mechanism**: Memorized sequences lead to highly concentrated attention patterns (information routing). This is orthogonal to weight-gradient magnitude. The fact that per-language Z-normalization helps (+0.01) suggests that attention "scales" differently across languages.

---

### Insight 30 — Logit Lens: Early settling is subtle but real (NOVEL 01)

**Observation**: NOVEL01 (LensMIA) reached **AUC 0.5124** for `settling_depth_top1`. While raw loss still dominates length-normalized metrics, the depth at which the model "commits" to a vocabulary prediction provides a weak but novel signal for code.

**Mechanism**: Members "settle" on the correct next token slightly earlier in the layer stack than non-members. However, the signal is noisier than representation-based methods, likely because the LM-Head projection (logit lens) adds significant noise or bias compared to direct hidden state analysis.

---

**Goal**: AUC > 0.70 via gradient-based white-box signals + structured ensemble.

---

## Novelty Experiments (`novelty_exp/`)

12 brand-new experiments in `novelty_exp/`, designed from insights across all 55 main experiments. Each targets a UNIQUE signal family never used in prior MIA literature.

| ID | Name | Core Signal | Expected AUC | Status |
|---|---|---|---|---|
| NOVEL01 | **LensMIA** — Logit Lens Convergence | Settling depth via logit lens projection | 0.60-0.68 | **0.5611** ✅ |
| NOVEL02 | **OrthoFuse** — Gradient × Attention Product | Multiplicative fusion of orthogonal signals | 0.67-0.72 | **0.6582** ✅ |
| NOVEL03 | **RepDen** — k-NN Density in Hidden Space | k-NN distance in hidden state manifold | 0.62-0.70 | **0.6559** ✅ |
| NOVEL04 | **RepVel** — Representation Velocity Profile | Inter-layer hidden state velocity/deceleration | 0.60-0.66 | Not run |
| NOVEL05 | **GradCoh** — Cross-Subsequence Gradient Coherence | Cosine sim of gradients from code halves | 0.58-0.65 | **0.5870** ✅ |
| NOVEL06 | **RankDef** — Effective Rank of Hidden States | SVD effective rank (dimensionality) per layer | 0.55-0.63 | **0.6512** ✅ ⚠️ Direction inverted |
| NOVEL07 | **GradOrth** — Gradient-Hidden Orthogonality | cos(grad, hidden) at each layer | 0.56-0.64 | **0.6349** ✅ |
| NOVEL08 | **PredAgree** — Prediction Agreement Depth | Cross-layer logit lens prediction stability | 0.60-0.67 | **0.5637** ✅ |
| NOVEL09 | **ConfTraj** — Confidence Trajectory | Shape of per-layer confidence curve | 0.58-0.66 | **0.5611** ✅ |
| NOVEL10 | **GradSparse** — Gradient Sparsity Profile | Hoyer/Gini sparsity of gradient per layer | 0.58-0.65 | **0.5878** ✅ |
| NOVEL11 | **ResDecomp** — Residual Stream Decomposition | Attention vs MLP contribution balance | 0.58-0.65 | **0.6590** ✅ |
| NOVEL12 | **CaliFuse** — Multi-Signal Calibrated Stacker | Ridge LR over all Z-normed signals | 0.72-0.76 | **0.6741** ✅ ⚠️ Below target (no raw activation vectors) |

**Priority order**: NOVEL12 (final stacker) → NOVEL04 (velocity) → NOVEL11 (residual) → rest

---

### Insight 31 — Effective rank is INVERTED: Members use MORE hidden dimensions, not fewer (NOVEL06)

**Observation:** NOVEL06 (RankDef-MIA) ran SVD on hidden state matrices `(seq_len × hidden_dim)` at 10 quartile layers. Best AUC **0.6512** (`+eff_rank_at_last`). Critically, the direction is **inverted from the hypothesis**: Members have HIGHER effective rank (M=216.6 vs NM=166.3) and LOWER top-SV fraction (M=0.6627 vs NM=0.7057).

**Why the hypothesis failed:** The original hypothesis (memorized → low-rank attractor → lower effective rank) assumed the sequence dimension (seq_len ≈ 50–512) creates a thin matrix where rank reflects compression. In practice, StarCoder2-3b's hidden states are high-dimensional (3072-D), and members, having been seen during training, activate MORE diverse directions — the model "spreads" its internal computation more richly across the hidden space for familiar code rather than compressing it. Non-members are handled by the model's generic generalization capacity, which uses a more stereotyped (lower-rank) encoding.

**Correct mechanism (post-hoc):** Members cause HIGHER effective rank because the model has learned to exploit the full representational richness of its hidden space for training samples. Non-members collapse to a lower-dimensional generic code representation. This is consistent with Insight 22 (memTrace: members have higher hidden state norm STD = more varied processing).

**Signal quality:** Effective rank is a genuine but partially length-confounded signal: `eff_rank_per_token` drops to 0.6082 (vs 0.65 raw), confirming ~6pp is length artifact. The `neg_top_sv_frac_mean` (0.6397) is cleaner — it measures concentration: member activations are LESS dominated by a single principal direction.

**Paper angle:** Empirical disproof of the low-rank-attractor hypothesis; hidden representation RICHNESS (not compression) as the correct memorization signature for code LLMs.

**EXP15 stacker:** Include `neg_top_sv_frac_mean` (0.6397) and `eff_rank_per_token` (0.6082, length-normalized) as orthogonal features. Likely partially correlated with memTrace norm features (same hidden-state family), but different axis (dimensionality vs magnitude).

---

### Insight 32 — LUMIA: Linear probes on early-to-mid layer activations are the strongest single signal (EXP51)

**Observation:** EXP51 LUMIA extracted mean-pooled hidden state vectors (dim=3072) at all 31 layers (embed + 30 blocks) for 9997 samples. Per-layer LogReg CV (5-fold):
- **Best single layer: block_5 (layer 6)** AUC **0.7031** ± 0.0077
- AUC curve: rises steeply (embed 0.607 → block_0 0.694 → block_5 **0.703** peak) then slowly declines (block_9 0.691 → block_29 0.688). **Early-to-mid transition is the sweet spot.**
- **Top-8 concat (L6,L5,L7,L2,L4,L8,L21,L22): 0.7338** — NEW ABSOLUTE BEST, surpasses memTrace RF (0.6908) by **+0.043**.
- Subset (top-8): Go 0.7469, Java 0.7461, Python 0.7492, Ruby 0.7464, Rust 0.6706. **All subsets above 0.65 — best cross-language coverage seen.**
- Norm unsupervised baseline: -norm_L4/L5/L6 ≈ 0.6507 (AUC near gradient ceiling, from vector MAGNITUDE alone; probe adds +0.08 AUC by using direction).

**Why layer 5–6 (block_4–5) is optimal:**
1. The embedding layer (block_0) handles token identity — linear in raw token frequency. By block_5, syntax/semantics are encoded but the model hasn't yet "forgotten" the training-time statistical footprint.
2. Deeper layers (blocks 15–29) progressively transform toward prediction, erasing the encoding-time memorization signal as it gets overwritten by next-token computation.
3. The fact that blocks 20–21 show a secondary bump (0.697) suggests a second memorization signature emerges in the late processing stages before the final LM head.

**Why linear probe defeats all prior methods:**
- LUMIA accesses the FULL 3072-dimensional hidden state vector vs gradient methods (scalar norms) vs attention methods (aggregated patterns). The full vector direction carries information that magnitude-only metrics cannot recover.
- Linear separability at depth: at layer 6, the member/non-member populations are **linearly separable in 3072-D**, confirming the flat-minima theory extends to the activation space.
- memTrace (EXP50) used CROSS-LAYER features (norm stats, transition) → RF. LUMIA uses WITHIN-LAYER raw vectors → LogReg. The raw vector approach wins because it exploits the full geometry of the activation manifold.

**Length confound:** seq_len not explicitly controlled; activation norms (-norm_L4 0.6507 ≈ -grad_embed 0.6422) suggest partial length confound is present. True signal from direction (not magnitude) is +0.08 AUC above norm baseline — likely robust to length.

**Paper angle:** "Linear probes on early Transformer layer activations provide the strongest membership signal (AUC 0.73) on code pre-training MIA; the optimal layer is the early-to-mid transition (block_5 of 30), where syntactic/semantic encoding is complete but next-token computation has not yet erased the training-time footprint."

**EXP15 stacker:** Include `lumia_top8_score` as a dominant feature. If computing fresh, extract block_5 activations (single layer, dim=3072 → Ridge or LR). Concatenating blocks 5+6+7 (top-3 concat: 0.7221) is a more compute-efficient option. ⚠️ C=0.01 is very strong regularization for dim=24576 — consider C=0.05 for top-8.

---

---

### Insight 33 — Attention dominates MLP for memorized code: total_attn_frac reaches 0.6590 (NOVEL11)

**Observation:** NOVEL11 (ResDecomp-MIA) used forward hooks to capture per-layer attention and MLP sublayer output norms, then computed their balance. Best signal: `+total_attn_frac` **AUC = 0.6590** — the fraction of total residual update attributable to attention (vs MLP) is higher for members. Per-region ratios: mid 0.6380 > early 0.6306 > late 0.5963. `attn_dominant_frac` (fraction of layers where attn > mlp): **0.6201**. Runtime: 10K samples, ~4 min A100 (45 it/s).

**Mechanism:** For memorized code, the transformer routes information via attention more heavily — attention heads "recognize" the seen token patterns and activate stored co-occurrence statistics. MLP blocks (knowledge storage) are activated relatively less because the attention routing is already sufficient to recall the training-time signal. Non-members require more MLP "computation" (pattern completion from scratch) relative to attention routing. This creates a systematic attention/MLP imbalance that is largest at mid-to-late layers (blocks 10–20) where high-level semantics are processed.

**Connects to:** Insight 22 (members produce lower hidden-state norm variance at mid layers — the same "settling into attractor" effect); Insight 16 (attention concentration is a strong MIA signal in EXP43 AttenMIA 0.6642). ResDecomp provides a complementary angle: not just HOW attention concentrates but HOW MUCH of the total update comes from attention vs MLP.

**Signal direction note:** `neg_mlp_norm_mean` (0.5828) is cleaner than `neg_attn_norm_mean` (0.5493) — lower MLP magnitude = member. This is consistent with the mechanism: MLP activates less for memorized sequences.

**EXP15 stacker:** `total_attn_frac` (0.6590) is potentially orthogonal to gradient signals (EXP41: output/weight space) and hidden-state signals (EXP50/EXP51: activation space). Checking sublayer BALANCE is distinct from checking sublayer MAGNITUDE. Include as candidate stacker feature.

---

### Insight 34 — CaliFuse confirms hidden-state norm + attention concentration are the dominant learnable features; single-pass stacker ceiling ~0.674 (NOVEL12)

**Observation:** NOVEL12 (CaliFuse-MIA) extracted 24 features from 6 signal families in a single fwd+bwd pass (11 min, 10K samples) and stacked them with 5-fold LR+CV and per-language Z-normalization. **CV AUC = 0.6741** — above AttenMIA (0.6642) but below memTrace (0.6908) and LUMIA (0.7338). Fold AUCs: 0.6854, 0.6716, 0.6697, 0.6637, 0.6812 (stable, low variance). 9997/10000 valid.

**Top feature importances (|coeff|):**
1. `neg_hs_norm_mid` −1.31 (hidden state norm at mid layer — consistent with EXP50 top feature)
2. `attn_conc_mean` +1.24 (KL concentration of attention — consistent with EXP43 AttenMIA)
3. `neg_hs_norm_early` +1.03 (early-layer hidden state norm — LUMIA confirms early layers strong)
4. `neg_attn_entropy_mid`/`hs_token_var_mid` (attention entropy + spatial variance — secondary)
5. `neg_grad_norm` +0.58 (gradient magnitude — still useful but no longer dominant after norm/attn)

**Why CaliFuse underperforms LUMIA (0.6741 vs 0.7338):** CaliFuse uses SCALAR aggregates of hidden states (mean norms, std per layer) while LUMIA uses the FULL 3072-dimensional mean-pooled vector at each layer. The full vector retains directional information that scalar norms discard — this is the same finding as EXP35 (gradient profile SHAPE) but applied to activations: scalar statistics capture less than the full vector.

**Why CaliFuse underperforms memTrace (0.6741 vs 0.6908):** memTrace uses 69 features including cross-layer transitions and stability metrics. CaliFuse uses only 18 features with coarser aggregation (3-region early/mid/late). The additional discriminative power of memTrace comes from its richer characterization of inter-layer dynamics.

**Key positive finding — representation velocity confirmed:** `neg_vel_early_mid` (negative early→mid velocity, coeff +0.14) is positive in the stacker — lower velocity = member, meaning members' representations change less across early layers. First direct confirmation of the velocity-as-membership signal (NOVEL04's expected result).

**CaliFuse design lesson:** For a SINGLE-PASS ensemble (budget-constrained), scalar aggregate features top out at ~0.67. Breaking 0.69+ requires either raw activation vectors (LUMIA approach) or cross-layer dynamics (memTrace approach). The scalar/aggregated strategy hits a ceiling consistent with individual signal ceilings (~0.65), with +0.02 from multi-signal fusion.

**EXP15 stacker:** CaliFuse's raw scores (`neg_hs_norm_mid`, `attn_conc_mean`) are now validated as dominant stacker features. These should be included as-is in the final COMBO stacker alongside LUMIA scores.

---

### Insight 35 — COMBO01: Gradient extraction failed; memTrace-only RF reaches 0.6954, confirming mid-layer norm dominance

**Observation:** COMBO01 intended to fuse memTrace (fwd) + gradient (bwd) features. **Gradient extraction failed for all 10,000 samples** (`TypeError: cannot unpack non-iterable NoneType object`) due to incompatible manual layer iteration. Result: **70 memTrace features only, RF CV AUC = 0.6954 ± 0.0103** — +0.005 over EXP50 (0.6908) from a richer per-layer feature set (5 key layers × multiple statistics vs EXP50's 69 features).

**Fold stability:** 0.6824–0.7112 (range 0.029) — comparable to EXP50's 0.6796–0.7035 (range 0.024). No significant improvement confirmed beyond EXP50.

**Top RF importances (memTrace features only):** seq_len 0.0663 (⚠️ length confound, same as EXP50), hnorm_global_mean 0.0348, hnorm_std_L15 0.0325, hnorm_mean_L15 0.0311, ctx_evolution_std 0.0282 — **identical ranking to EXP50 memTrace**. No new information added.

**Subset results:** Go 0.7321, Ruby 0.7105, Python 0.7085, Java 0.6839, Rust 0.6298 — consistent language hierarchy (Insight 7).

**Gradient extraction fix needed:** The `TypeError` comes from manually iterating `self.model.model.layers` and passing `position_ids` to a layer that doesn't accept it (StarCoder2 uses sliding window attention with different layer signature). Fix: use `output_attentions=False` + `requires_grad=True` on embeddings via the standard forward pass (as in EXP11/EXP27).

**For COMBO01 re-run:** Replace manual layer iteration with `model(inputs_embeds=embed, labels=..., output_hidden_states=True)` and call `.backward()` on loss. This is the proven pattern from EXP11/EXP27/EXP30.

---

### Insight 36 — LUMIA-fast on full 100K crushes 10% result: 0.7805 NEW BEST (+0.047 over EXP51)

**Observation:** LUMIA-fast (Ridge+PCA-256, 9 key layers) ran on **full 100K samples** in **0.47h** and achieved **top-9 concat AUC = 0.7805 ± 0.0024** — the new absolute best, surpassing EXP51's 0.7338 by +0.047 and memTrace's 0.6908 by +0.090.

**Key results per layer (full 100K):**
- Best single: **block_20 (L21) = 0.7456** (was block_5 on 10%)
- blocks 4–7 (L5–8): 0.7433–0.7448 (tight cluster, consistent with EXP51)
- embed (L0): 0.6699 (weak but improves concat)
- block_29 (L30, final): 0.7225

**Best single layer shift (block_5 → block_20 at full scale):** At 10K samples, block_5 appeared best (0.7031 with 5-fold LogReg). At 100K samples with Ridge+PCA, block_20 (0.7456) edges out block_5/6/7 (0.7441–0.7448). This is likely because deeper layers' discriminative geometry requires more data to learn a reliable linear separator — the Ridge classifier at 10K couldn’t fully exploit the block_20 signal.

**Top-K scaling:** Top-1=0.7456, Top-3=0.7672 (+0.022), Top-5=0.7740 (+0.007), Top-9=0.7805 (+0.007) — diminishing returns after Top-3. The three best layers (L21, L7, L6) capture most joint information; adding L5/L8/L22/L2/L30/L0 each adds ~1-2pp more.

**Per-subset highlights:**
- **Python: 0.8017** (first subset to break 0.80! Likely due to larger Python training corpus diversity vs non-members)
- Ruby: 0.7951, Go: 0.7888, Java: 0.7797 — all strong
- **Rust: 0.7318** (still lowest, consistent with all prior experiments)

**Runtime breakdown:** Forward passes (single-sample fallback, 63.88 it/s): **26.3 min**; PCA fit (9 layers): **83s**; Ridge 3-fold CV: **12s**; Total: **0.47h ✅** (well within 12h limit).

**Batch tokenizer bug (⚠️ for production run):** Batch mode failed because tokenizer has no pad token. Fix: `tokenizer.pad_token = tokenizer.eos_token` before batch call. Adding this would reduce runtime from 26 min to ~3–6 min (batch_size=8 ⇒ 8× speedup).

**PCA explained variance sanity:** block_1 (L2) = 0.987 (very linear layer), blocks 4–7 = 0.93–0.96, block_20/21 = 0.872–0.883 (slightly lower — deeper layers have more complex geometry needing more dims), embed = 0.811 (most complex input representation).

**Competition readiness:** 0.7805 on full dev set is an extremely strong result for a white-box MIA. This is the score to report and build upon for the final submission.

---

*Last updated: 2026-03-05 (EXP56 added: best **0.6332** via `-z_esp` with single fwd pass and per-language calibration; new strongest pure logit-only baseline (+0.0899 vs Min-K%++). improved_baselines ranking remains: LayerCascade 0.6403 > GeoPrint 0.6335 > CDD 0.6292 > LogitSig 0.6287 > DCPDD 0.6004 > CorruptAsym 0.5978 > TopoCal 0.5952 > MinKPP-reimpl 0.5671. COMBO02 remains **0.6961**; LUMIA-fast full 100K remains **0.7805** 🏆 **NEW BEST**.)*
