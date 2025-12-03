# Notebooks 指南

- 仅用于“读结果 → 出图/出表”，不在 notebook 中做训练/大规模计算。
- 建议对应论文主要图表建立独立 notebook：
  - `01_eda_datasets.ipynb`：站点/设备/标签分布
  - `02_fig_site_wise_performance.ipynb`：各站点/设备性能箱线图/条形图
  - `03_fig_calibration_curves.ipynb`：可靠性曲线、ECE 对比
  - `04_fig_clinical_impact.ipynb`：分诊等待时间/决策曲线
- 数据来源：`experiments/expX_*/` 输出的 metrics/logs/curves。
