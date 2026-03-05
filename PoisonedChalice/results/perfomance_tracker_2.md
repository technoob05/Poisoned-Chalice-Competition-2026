# Poisoned Chalice 2026 - Tracker 2 (Pending/Run Status)

**Synced at**: 2026-03-05  
**Source**: `results/performance_tracker.md`

---

## 1) EXP chưa có AUC (chưa cập nhật điểm)

| EXP | Script | Status | Note |
| :--- | :--- | :--- | :--- |
| **EXP10** | `experiments/EXP10_ensemble_sota.py` | ⏳ Pending | Tracker vẫn để `-` |
| **EXP15** | `experiments/EXP15_xgboost_meta_stacking.py` | ⏳ Pending | Tracker vẫn để `-` |
| **EXP19** | `experiments/EXP19_buzzer_cmi.py` | ⏳ Pending | Tracker vẫn để `-` |
| **EXP20** | `experiments/EXP20_buzzer_plus.py` | ⏳ Pending | Tracker vẫn để `-` |
| **EXP25** | `experiments/EXP25_cross_layer_correlation.py` | ⏳ Pending | Tracker vẫn để `-` |
| **EXP54** | `experiments/EXP54_spvmia_self_prompted.py` | ⏳ Pending | Tracker vẫn để `-` |

---

## 2) Dòng cần dọn trong tracker

| EXP | Tình trạng |
| :--- | :--- |
| **EXP51** | Có 1 dòng mô tả cũ để `-`, nhưng bên dưới đã có kết quả chính thức **0.7338**. |

---

## 3) Tổng quan coverage

- Tổng số EXP trong dải `EXP01`–`EXP56`: **56**
- Đã có AUC: **50**
- Chưa có AUC: **6** (`EXP10`, `EXP15`, `EXP19`, `EXP20`, `EXP25`, `EXP54`)

---

## 4) Checklist cập nhật nhanh (copy/paste)

- [ ] Chạy `EXP10_ensemble_sota.py`
- [ ] Chạy `EXP15_xgboost_meta_stacking.py`
- [ ] Chạy `EXP19_buzzer_cmi.py`
- [ ] Chạy `EXP20_buzzer_plus.py`
- [ ] Chạy `EXP25_cross_layer_correlation.py`
- [ ] Chạy `EXP54_spvmia_self_prompted.py`
- [ ] Sau khi có kết quả, cập nhật lại `results/performance_tracker.md`
- [ ] Dọn dòng placeholder `EXP51` cũ (AUC `-`) để tránh nhầm

---

## 5) Coverage của thư mục `novelty_exp`

- Tổng file: **19**
- Đã phản ánh rõ trong tracker (có dòng kết quả/entry): **14**
- Chưa có entry kết quả riêng trong bảng chính: **5**

**Đã có trong tracker**

- `EXP51_LUMIA_fast.py` (đang được ghi dưới dạng `LUMIA-fast` + `EXP51`)
- `NOVEL01_logit_lens_mia.py`
- `NOVEL02_orthofuse_grad_attn.py`
- `NOVEL03_knn_representation_density.py`
- `NOVEL05_gradient_coherence.py`
- `NOVEL06_effective_rank_mia.py`
- `NOVEL07_gradient_orthogonality.py`
- `NOVEL08_prediction_agreement.py`
- `NOVEL09_confidence_trajectory.py`
- `NOVEL10_gradient_sparsity.py`
- `NOVEL11_residual_decomposition.py`
- `NOVEL12_califuse_stacker.py`
- `COMBO01_memtrace_gradient.py`
- `COMBO02_memtrace_attention.py` *(đã cập nhật: AUC **0.6961**, CV mean **0.6967 ± 0.0099**)*

**Chưa thấy entry kết quả riêng trong bảng chính**

- `NOVEL04_representation_velocity.py` *(mới thấy xuất hiện như feature trong insight/stacker, chưa có dòng AUC standalone)*
- `COMBO03_memtrace_logitlens.py`
- `COMBO04_memtrace_histogram.py`
- `COMBO05_memtrace_mega.py`
- `COMBO06_memtrace_xgboost_meta.py`

---

## 6) Coverage của thư mục `improved_baselines`

- Tổng file: **9**
- Đã phản ánh trong tracker: **8/9**

**Đã phản ánh**

- `BASELINE_minkpp.py` *(đã có `EXP02 Min-K%++` + entry riêng `BASELINE-MinKPP-Reimpl` AUC **0.5671**)*
- `BASELINE_cdd.py` *(đã có `BASELINE-CDD` trong tracker, best unsupervised AUC **0.6292**)*
- `BASELINE_dcpdd.py` *(đã có `BASELINE-DCPDD` trong tracker, best unsupervised AUC **0.6004**)*
- `BASELINE_topocal.py` *(đã có `BASELINE-TopoCal` trong tracker, best unsupervised AUC **0.5952**)*
- `NOVEL_CorruptAsym.py` *(đã có `NOVEL-CorruptAsym` trong tracker, best AUC **0.5978** với `corrupt_asym_mid`)*
- `NOVEL_GeoPrint.py` *(đã có `NOVEL-GeoPrint` trong tracker, best AUC **0.6335** với `neg_norm_mid_token_std`)*
- `NOVEL_LayerCascade.py` *(đã có `NOVEL-LayerCascade` trong tracker, best AUC **0.6403** với `-neg_drift_early`)*
- `NOVEL_LogitSig.py` *(đã có `NOVEL-LogitSig` trong tracker, best AUC **0.6287** với `-traj_loss_slope`)*

**Chưa thấy entry riêng trong tracker**

- `BASELINE_multiscale.py`
