# Paper Outline (初稿模板)

## Title (working)
- Cross-site Generalization and Calibration for Chest X-ray Triage Across Public Datasets

## 1. Introduction
- 背景：多站点设备差异 → 泛化/校准问题
- 缺口：跨站点失校准 + 性能不稳定
- 贡献梗概：系统评估 + 简单可复现的校准/域适应 + 临床分诊模拟

## 2. Related Work
- Cross-site generalization in CXR
- Calibration in medical imaging
- Simple domain adaptation / per-site thresholds
- Clinical triage simulation / decision support

## 3. Methods
### 3.1 Datasets & Splits
- NIH / CheXpert / Open-i / MIMIC-CXR
- 固定 splits（见 `data/splits/`），metadata（见 `data/metadata/`）
### 3.2 Model & Training
- Backbone (e.g., ResNet) + loss，训练配置（`config/model/*.yaml` + `config/exp/`）
### 3.3 Evaluation
- Metrics: AUROC, AUPRC, ECE, Brier, group-wise metrics (site/device)
### 3.4 Calibration
- Temperature Scaling, Isotonic；可靠性曲线绘制
### 3.5 Domain Adaptation / Thresholding
- 简单可复现的方法（per-site BN/微调/阈值）
### 3.6 Clinical Triage Simulation
- 基于打分/阈值的危急片优先队列，输出等待时间/漏诊率

## 4. Results
### 4.1 Baseline Internal/External Performance
- Overall + per-site/device
### 4.2 Calibration
- 全局/分站点 ECE，可靠性图
### 4.3 Domain Adaptation / Thresholding
- 对比均值/方差/ECE
### 4.4 Clinical Impact
- 等待时间、漏诊率、决策曲线

## 5. Discussion
- 发现与意义；泛化/校准挑战；简单方法的收益

## 6. Limitations & Future Work
- 数据偏差、标注噪声、大模型/更强DA的潜力

## 7. Conclusion
- 总结与临床部署前景

## Appendix
- 详细配置、更多图表、消融

---
产出物存放约定：
- 图表：`paper/figs/`，表格：`paper/tables/`，一页纸：`paper/clinical_impact_one_pager.md`
- 每个主要实验的运行方式：参考 `config/exp/` + `scripts/*`，结果归档 `experiments/`
