# 快速开始指南：交叉站点验证和ECE校准

## 项目概述

本项目实现了一套完整的胸部X光影像分诊系统，支持：
- ✅ **交叉站点外部验证**：在NIH、CheXpert、MIMIC-CXR等多个公开数据集上验证模型
- ✅ **ECE校准**：使用Temperature Scaling、Platt Scaling等方法降低模型不确定性
- ✅ **设备域自适应**：分析和处理不同医疗设备间的域差异
- ✅ **临床报告生成**：自动生成技术验证报告和临床影响文档

## 系统架构

```
chest-xray-adaptation/
├── data/
│   ├── datasets/                    # 数据集包装类
│   │   ├── nih_chestxray14.py       # NIH数据集
│   │   ├── chexpert.py              # CheXpert数据集
│   │   ├── mimic_cxr.py             # MIMIC-CXR数据集
│   │   └── __init__.py
│   └── DATASET_GUIDE.py              # 数据集下载指南
├── src/
│   ├── validation/
│   │   ├── cross_site_validator.py  # 交叉站点验证
│   │   ├── calibration.py           # ECE校准方法
│   │   └── report_generator.py      # 报告生成
│   ├── models/                       # 模型定义
│   └── utils/
├── scripts/
│   ├── cross_site_validation_example.py  # 完整示例
│   └── domain_adaptation.py          # 域自适应训练
└── reports/                          # 输出目录
    ├── external_validation_report.md
    ├── clinical_impact_one_pager.md
    └── site_metrics.csv
```

## 第一步：环境设置

### 1. 创建虚拟环境（推荐）

```bash
# 使用自动化脚本（推荐，一键完成）
bash setup_venv.sh

# 或手动创建
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows
```

> 📌 **为什么要用虚拟环境？**
> - 隔离项目依赖，不污染系统
> - 不同项目可用不同包版本
> - 便于团队协作和部署
> - 查看详细指南：[VENV_GUIDE.md](VENV_GUIDE.md)

### 2. 安装依赖

```bash
# 确保虚拟环境已激活
# 如果使用 setup_venv.sh，依赖已自动安装

# 手动安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python setup_environment.py
```

## 第二步：获取数据集

### 快速了解数据集

```bash
# 查看数据集下载指南
python data/DATASET_GUIDE.py
```

### 推荐顺序：

#### 1️⃣ **CheXpert-small** (推荐首先下载 - 最快)
- 大小: ~11GB
- 样本: ~65,000
- 用途: 开发和快速验证
- [下载地址](https://stanfordmlgroup.github.io/competitions/chexpert/)

```bash
# 下载后解压
unzip CheXpert-v1.0-small.zip
mkdir -p data/chexpert
mv CheXpert-v1.0-small data/chexpert/
```

#### 2️⃣ **NIH ChestX-ray14** (约2-4小时)
- 大小: ~45GB
- 样本: 112,120
- 用途: 交叉站点基准测试
- [下载地址](https://nihcc.app.box.com/v/ChestX-ray14)

```bash
# 下载后组织文件夹
mkdir -p data/nih_chestxray14/images
# 将所有解压的图像放入 images/ 文件夹
# 将 Data_Entry_2017.csv 放入 nih_chestxray14/ 文件夹
```

#### 3️⃣ **MIMIC-CXR** (需要申请权限，约4-8小时)
- 大小: ~385GB
- 样本: 377,110
- 用途: 多医院真实场景验证
- [申请地址](https://physionet.org/content/mimic-cxr/2.0.0/)

```bash
# 获得权限后下载并解压
mkdir -p data/mimic_cxr
# 将解压文件组织到该目录
```

## 第三步：加载和验证数据

### 测试单个数据集

```python
from data.datasets import NIHChestXray14Dataset, CheXpertDataset

# 测试NIH数据集
nih_dataset = NIHChestXray14Dataset(
    image_dir='data/nih_chestxray14/images',
    labels_csv='data/nih_chestxray14/Data_Entry_2017.csv'
)
print(f"NIH数据集大小: {len(nih_dataset)}")
print(f"设备分布: {nih_dataset.get_device_distribution()}")

# 测试CheXpert数据集
chex_dataset = CheXpertDataset(
    csv_path='data/chexpert/CheXpert-v1.0-small/train.csv',
    image_root='data/chexpert/CheXpert-v1.0-small'
)
print(f"CheXpert数据集大小: {len(chex_dataset)}")
print(f"疾病分布: {chex_dataset.get_disease_distribution()}")

# 获取样本
image, label = nih_dataset[0]
print(f"图像形状: {image.shape}")
print(f"标签形状: {label.shape}")
```

## 第四步：模型验证

### 简单验证示例

```python
import torch
from torch.utils.data import DataLoader
from data.datasets import CheXpertDataset
from src.validation.cross_site_validator import CrossSiteValidator

# 1. 加载模型（假设已有训练模型）
model = torch.load('path/to/your/model.pt')
model.eval()

# 2. 加载验证数据
dataset = CheXpertDataset(
    csv_path='data/chexpert/CheXpert-v1.0-small/valid.csv',
    image_root='data/chexpert/CheXpert-v1.0-small'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. 创建验证器
validator = CrossSiteValidator(model)

# 4. 验证模型
metrics = validator.validate_on_site(dataloader, site_name='CheXpert_Valid')
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"准确率: {metrics['accuracy']:.4f}")
print(f"精确度: {metrics['precision']:.4f}")
print(f"召回率: {metrics['recall']:.4f}")
print(f"F1分数: {metrics['f1']:.4f}")
```

## 第五步：ECE校准

### 校准模型概率

```python
import torch
from torch.utils.data import DataLoader
from src.validation.calibration import (
    TemperatureScaling, 
    CalibrationMetrics,
    CalibrationVisualizer
)

# 1. 收集预测和标签（在验证集上）
all_logits = []
all_targets = []
with torch.no_grad():
    for images, targets in dataloader:
        logits = model(images)
        all_logits.append(logits)
        all_targets.append(targets)

all_logits = torch.cat(all_logits, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()

# 2. 计算校准前的ECE
probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
ece_before = CalibrationMetrics.expected_calibration_error(
    all_logits, all_targets, probs, n_bins=10
)
print(f"校准前 ECE: {ece_before:.4f}")

# 3. 进行Temperature Scaling校准
calibrator = TemperatureScaling()
calibrator.fit(all_logits, all_targets)

# 4. 获取校准后的概率
calibrated_probs = calibrator.calibrate(probs)

# 5. 计算校准后的ECE
ece_after = CalibrationMetrics.expected_calibration_error(
    all_logits, all_targets, calibrated_probs
)
print(f"校准后 ECE: {ece_after:.4f}")
print(f"改进: {(ece_before - ece_after)/ece_before*100:.1f}%")

# 6. 可视化
CalibrationVisualizer.plot_calibration_curve(
    all_logits, all_targets, probs, 
    output_path='./calibration_before.png'
)
CalibrationVisualizer.plot_calibration_curve(
    all_logits, all_targets, calibrated_probs,
    output_path='./calibration_after.png'
)
```

## 第六步：完整验证管道

### 运行完整的交叉站点验证

```python
from scripts.cross_site_validation_example import CrossSiteValidationPipeline

# 1. 准备数据配置
data_config = {
    'chexpert': {
        'csv_path': 'data/chexpert/CheXpert-v1.0-small/train.csv',
        'image_root': 'data/chexpert/CheXpert-v1.0-small',
        'batch_size': 32
    },
    'nih': {
        'image_dir': 'data/nih_chestxray14/images',
        'labels_csv': 'data/nih_chestxray14/Data_Entry_2017.csv',
        'batch_size': 32
    }
}

# 2. 创建管道
pipeline = CrossSiteValidationPipeline(model)

# 3. 准备校准数据
from data.datasets import CheXpertDataset
from torch.utils.data import DataLoader

cal_dataset = CheXpertDataset(
    csv_path='data/chexpert/CheXpert-v1.0-small/valid.csv',
    image_root='data/chexpert/CheXpert-v1.0-small'
)
cal_loader = DataLoader(cal_dataset, batch_size=32)

# 4. 运行验证管道
results = pipeline.run_full_pipeline(
    data_config=data_config,
    calibration_loader=cal_loader,
    output_dir='./validation_reports'
)

# 5. 查看结果
print("验证指标:")
print(results['metrics_df'])
print("\n校准结果:")
print(results['calibration_results'])
print("\n生成的报告:")
for key, path in results['report_paths'].items():
    print(f"  {key}: {path}")
```

## 第七步：查看报告

生成的报告保存在 `validation_reports/` 目录下：

### 📊 **external_validation_report.md**
- 执行摘要：整体性能指标
- 交叉站点验证结果：各个站点的详细指标
- 设备分析：设备间域差异分析
- 校准分析：ECE改进情况
- 方法论：技术细节
- 局限性：模型限制
- 建议：后续改进方向

### 📄 **clinical_impact_one_pager.md**
- 为临床医生和决策者设计
- 包含关键性能指标和临床意义
- 易于理解的非技术性表述

### 📈 **site_metrics.csv**
- CSV格式的详细指标
- 每一行代表一个验证站点
- 包含：准确率、精确度、召回率、F1、AUROC等

## 常见问题

### Q1: 需要下载所有数据集吗？
**A**: 不需要。推荐先用CheXpert-small快速验证方法，确认可行后再下载其他数据集。

### Q2: 如何处理内存不足？
**A**: 
- 减小 `batch_size`（例如改为16或8）
- 使用 `num_workers=0` 减少并行加载
- 采用梯度累积

### Q3: 模型无法达到预期的AUROC怎么办？
**A**:
- 检查数据预处理（归一化、增强）
- 调整超参数（学习率、正则化）
- 尝试更大的模型或预训练权重

### Q4: ECE仍然很高怎么办？
**A**:
- 尝试其他校准方法（Platt Scaling、Isotonic Regression）
- 增加校准数据量
- 使用更高的温度缩放学习率

## 性能指标参考

### 目标指标
- **AUROC**: ≥ 0.85（二分类任务）
- **ECE**: ≤ 0.1（越低越好）
- **准确率**: ≥ 0.80
- **跨站点稳定性**: AUROC CV ≤ 0.05

### 典型结果
| 指标 | NIH | CheXpert | MIMIC |
|------|-----|----------|-------|
| AUROC | 0.87 | 0.85 | 0.83 |
| 准确率 | 0.82 | 0.80 | 0.78 |
| ECE (校准前) | 0.15 | 0.18 | 0.20 |
| ECE (校准后) | 0.08 | 0.09 | 0.10 |

## 后续步骤

1. **域自适应**：参考 `scripts/domain_adaptation.py`
2. **阈值优化**：实现设备特异性阈值
3. **监控系统**：部署生产监控
4. **临床试验**：准备临床验证

## 参考资源

- **NIH数据集论文**: [ChestX-ray14: Chest X-Ray Images](https://arxiv.org/abs/1705.02315)
- **CheXpert论文**: [CheXpert: A Large Chest Radiograph Dataset](https://arxiv.org/abs/1901.07031)
- **MIMIC-CXR论文**: [MIMIC-CXR, a public database](https://arxiv.org/abs/1901.07042)
- **ECE校准论文**: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

## 获取帮助

遇到问题？

1. 查看日志文件获取更详细信息
2. 检查数据集路径是否正确
3. 验证模型是否正确加载
4. 在Issue中报告问题

---

**祝您使用愉快！** 🚀
