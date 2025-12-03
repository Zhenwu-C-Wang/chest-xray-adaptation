# chest-xray-adaptation
This project develops a deep learning model for cross-site generalization and calibration in chest X-ray triage. It aims to enhance diagnostic accuracy across different devices and reduce critical image waiting times, with code for domain adaptation, calibration, and external validation using datasets like NIH ChestX-ray14 and CheXpert.

## Paper-oriented workflow (框架提要)
- 目标故事线：跨站点泛化 + 概率校准 + 简单域适应/分层阈值，在临床分诊场景中降低危急片等待时间。
- 主要结果对应的 pipeline：
  - Baseline 内/外部验证：AUROC/AUPRC，按站点/设备分层稳定性。
  - 校准：全局/分站点 ECE，可靠性曲线（温度缩放/等距回归）。
  - 轻量域适应/分层阈值：对比方差与 ECE 改善。
  - 临床模拟：分诊阈值下危急片等待时间、漏诊率。
- 结构约定：
  - `config/exp/`：论文每个主要实验的配置（可直接复现实验）。
  - `experiments/expX_*/`：实际跑出的指标、日志、曲线（归档可复现）。
  - `notebooks/`：仅用于出图/出表，不跑训练。
  - `paper/outline.md`：论文大纲，`paper/` 下放最终图表和一页纸。
