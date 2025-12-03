"""
公开数据集下载和配置指南
支持NIH ChestX-ray14、CheXpert、MIMIC-CXR
"""

import os
from pathlib import Path


class DatasetGuide:
    """
    公开胸部X光数据集配置指南
    """
    
    @staticmethod
    def get_nih_chestxray14_guide() -> str:
        return """
# NIH ChestX-ray14 数据集下载指南

## 基本信息
- **数据集名称**: ChestX-ray14
- **数据来源**: National Institutes of Health (NIH)
- **图像数量**: 112,120张胸部X光图像
- **诊断标签**: 14种疾病诊断
- **许可证**: 公开数据集（CC0许可证）
- **访问方式**: 无需申请，直接下载

## 详细步骤

### 步骤 1: 访问数据集
访问链接: https://nihcc.app.box.com/v/ChestX-ray14

### 步骤 2: 下载标签文件
文件名: Data_Entry_2017.csv
- 包含所有112,120张图像的标签
- CSV格式，可直接加载

### 步骤 3: 下载图像文件
图像文件分为12个压缩包：
- images_001.tar.gz - images_012.tar.gz
- 总大小: ~45GB
- 推荐使用高速网络或VPN下载

### 步骤 4: 组织目录结构
```
chest-xray-adaptation/data/
├── nih_chestxray14/
│   ├── images/              # 解压后的图像目录
│   │   ├── 00000001_000.png
│   │   ├── 00000001_001.png
│   │   └── ...
│   ├── Data_Entry_2017.csv  # 标签文件
│   └── README.txt           # 数据集说明
```

### 步骤 5: 验证数据
```python
from data.datasets import NIHChestXray14Dataset

dataset = NIHChestXray14Dataset(
    image_dir='data/nih_chestxray14/images',
    labels_csv='data/nih_chestxray14/Data_Entry_2017.csv'
)
print(f"数据集大小: {len(dataset)}")  # 应该是 112120
```

## 诊断标签说明
1. Atelectasis (肺不张)
2. Cardiomegaly (心脏肥大)
3. Consolidation (实变)
4. Edema (肺水肿)
5. Effusion (胸腔积液)
6. Emphysema (肺气肿)
7. Fibrosis (肺纤维化)
8. Hernia (疝)
9. Infiltration (浸润)
10. Mass (肿块)
11. Nodule (结节)
12. Pleural Thickening (胸膜增厚)
13. Pneumonia (肺炎)
14. Pneumothorax (气胸)

## 数据特点
- 年龄范围: 1-95 岁
- 性别分布: 相对均衡
- 设备: 多家医院不同型号X光机
- 图像质量: 差异较大（真实临床数据特点）

## 推荐用途
- 跨站外部验证的基准数据集
- 设备泛化能力评估
- 模型在真实临床数据上的表现

---
"""
    
    @staticmethod
    def get_chexpert_guide() -> str:
        return """
# CheXpert 数据集下载指南

## 基本信息
- **数据集名称**: CheXpert: A Large Chest Radiograph Dataset
- **数据来源**: Stanford Machine Learning Group
- **图像数量**: 223,648张胸部X光图像
- **诊断标签**: 14种疾病诊断
- **特点**: 包含不确定性标注（-1）
- **访问方式**: 需要填写使用协议

## 详细步骤

### 步骤 1: 注册并获取下载链接
访问: https://stanfordmlgroup.github.io/competitions/chexpert/

需要：
1. 填写注册表单
2. 同意数据使用协议
3. 获取下载链接（通过邮件）

### 步骤 2: 下载数据集
CheXpert 提供两个版本：
- **CheXpert-v1.0-small** (~11GB): 推荐用于快速原型开发
- **CheXpert-v1.0** (~439GB): 完整数据集

建议下载small版本进行开发。

### 步骤 3: 解压文件
```bash
# 解压small版本
unzip CheXpert-v1.0-small.zip

# 结果目录结构
CheXpert-v1.0-small/
├── train/
│   ├── patient00000/
│   ├── patient00001/
│   └── ...
├── valid/
│   ├── patient61999/
│   └── ...
└── ...
```

### 步骤 4: 组织目录结构
```
chest-xray-adaptation/data/
├── chexpert/
│   ├── CheXpert-v1.0-small/
│   │   ├── train/
│   │   ├── valid/
│   │   ├── train.csv
│   │   └── valid.csv
│   └── README.txt
```

### 步骤 5: 验证数据
```python
from data.datasets import CheXpertDataset

dataset = CheXpertDataset(
    csv_path='data/chexpert/CheXpert-v1.0-small/train.csv',
    image_root='data/chexpert/CheXpert-v1.0-small'
)
print(f"数据集大小: {len(dataset)}")
```

## 数据特点

### 诊断标签
- Enlarged Cardiomediastinum (纵隔扩大)
- Cardiomegaly (心脏肥大)
- Airspace Opacity (气腔混浊)
- Lung Lesion (肺病变)
- Edema (肺水肿)
- Consolidation (实变)
- Pneumonia (肺炎)
- Atelectasis (肺不张)
- Pneumothorax (气胸)
- Pleural Effusion (胸腔积液)
- Pleural Other (其他胸膜病变)
- Fracture (骨折)
- Support Devices (支持装置)

### 标注说明
- 1: 正样本（诊断明确）
- -1: 不确定样本（可能存在诊断）
- 0: 负样本（诊断排除）
- NaN: 未标注

## 处理不确定标注
```python
# 方法1: 视为正样本
dataset = CheXpertDataset(
    csv_path='...',
    image_root='...',
    uncertain_as_positive=True  # -1 → 0.5
)

# 方法2: 视为负样本
dataset = CheXpertDataset(
    csv_path='...',
    image_root='...',
    uncertain_as_positive=False  # -1 → 0
)
```

## 推荐用途
- 多标签分类任务
- 处理不确定性的研究
- 大规模医学影像学习

---
"""
    
    @staticmethod
    def get_mimic_cxr_guide() -> str:
        return """
# MIMIC-CXR 数据集下载指南

## 基本信息
- **数据集名称**: MIMIC-CXR-JPG, A Large Publicly Available Database of Labeled Chest Radiographs
- **数据来源**: MIT-LCP PhysioNet
- **图像数量**: 377,110张胸部X光图像
- **患者数**: 63,685名患者
- **诊断标签**: 14种疾病诊断
- **特点**: 多医院、包含医学记录、设备元数据
- **访问方式**: 需要完成培训和签署协议

## 详细步骤

### 步骤 1: 获取PhysioNet访问权限
访问: https://physionet.org/

需要：
1. 创建PhysioNet账户
2. 完成 CITI "Data or Specimens Only Research" 培训
3. 获取批准（通常需要1-2个工作日）

### 步骤 2: 申请MIMIC-CXR访问权限
访问: https://physionet.org/content/mimic-cxr/2.0.0/

条件：
- 已完成CITI培训
- 同意数据使用协议
- 学术或医疗机构从业者

### 步骤 3: 下载数据集
需要下载的文件：
1. mimic-cxr-2.0.0-chexpert.csv.gz (标签文件，~1.7GB)
2. mimic-cxr-2.0.0-metadata.csv.gz (元数据，~500MB)
3. 图像文件（多个压缩包，~385GB）

建议使用`wget`或`aria2`等工具加速下载。

### 步骤 4: 解压文件
```bash
# 解压CSV文件
gunzip mimic-cxr-2.0.0-chexpert.csv.gz
gunzip mimic-cxr-2.0.0-metadata.csv.gz

# 解压图像文件（多个包）
# 使用tar解压
tar -xzf mimic-cxr-jpg-2.0.0.tar.gz
```

### 步骤 5: 组织目录结构
```
chest-xray-adaptation/data/
├── mimic_cxr/
│   ├── mimic-cxr-2.0.0-chexpert.csv
│   ├── mimic-cxr-2.0.0-metadata.csv
│   ├── files/
│   │   ├── p10/
│   │   │   └── p10000xxx/
│   │   │       └── s5xxxx/
│   │   │           └── 00000.jpg
│   │   ├── p11/
│   │   └── ...
│   └── README.txt
```

### 步骤 6: 验证数据
```python
from data.datasets import MIMICCXRDataset

dataset = MIMICCXRDataset(
    csv_path='data/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv',
    image_root='data/mimic_cxr'
)
print(f"数据集大小: {len(dataset)}")  # 应该是 377110
```

## 数据特点

### 多医院数据
- 来自Beth Israel Deaconess Medical Center
- 多个放射科部门
- 真实临床数据

### 元数据
- 患者ID
- 研究ID
- 设备类型
- 成像参数
- 电子健康记录链接

### 跨站验证优势
- 包含多家医院数据
- 设备和协议差异
- 真实临床多样性

## 诊断标签
与CheXpert标签相同，包括：
- Atelectasis
- Cardiomegaly
- Consolidation
- 等其他13个诊断

## 关键优势
1. **多站点数据**: 理想的跨站外部验证源
2. **完整的EHR链接**: 支持临床结果研究
3. **大规模**: 充分的样本量
4. **高质量**: 由经验丰富的放射科医生标注

## 推荐用途
- 跨医院泛化能力评估
- 域自适应研究
- 真实临床场景建模

---
"""
    
    @staticmethod
    def create_dataset_config() -> dict:
        """
        创建推荐的数据集配置
        """
        return {
            'nih_chestxray14': {
                'name': 'NIH ChestX-ray14',
                'size': '112,120 images',
                'download_time': '~2-4 hours',
                'disk_space': '~45GB',
                'recommended_use': 'Cross-site validation baseline',
                'url': 'https://nihcc.app.box.com/v/ChestX-ray14',
                'license': 'CC0 (Public Domain)'
            },
            'chexpert': {
                'name': 'CheXpert',
                'size': '223,648 images (small: ~65,000)',
                'download_time': '~2-8 hours',
                'disk_space': '~11GB (small) / ~439GB (full)',
                'recommended_use': 'Development and validation',
                'url': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
                'license': 'Stanford License Agreement'
            },
            'mimic_cxr': {
                'name': 'MIMIC-CXR',
                'size': '377,110 images',
                'download_time': '~4-8 hours',
                'disk_space': '~385GB',
                'recommended_use': 'Multi-site generalization testing',
                'url': 'https://physionet.org/content/mimic-cxr/2.0.0/',
                'license': 'PhysioNet Credentialed Health Data License'
            }
        }
    
    @staticmethod
    def print_complete_guide():
        """
        打印完整指南
        """
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  公开胸部X光数据集完整配置指南                            ║
╚════════════════════════════════════════════════════════════════════════════╝

""")
        
        print(DatasetGuide.get_nih_chestxray14_guide())
        print(DatasetGuide.get_chexpert_guide())
        print(DatasetGuide.get_mimic_cxr_guide())
        
        print("""
# 快速比较表

| 特性 | NIH ChestX-ray14 | CheXpert | MIMIC-CXR |
|------|-----------------|----------|-----------|
| 图像数 | 112K | 223K | 377K |
| 诊断标签 | 14 | 14 | 14 |
| 多医院数据 | 否 | 否 | 是 ✓ |
| 需要申请 | 否 | 是 | 是 |
| 下载难度 | 简单 | 中等 | 中等 |
| 推荐用途 | 基准 | 开发 | 多站点验证 |
| 磁盘空间 | 45GB | 11GB(S) | 385GB |

# 建议工作流程

1. **开发阶段**: 使用 CheXpert-small
   - 快速迭代
   - 验证方法
   - 原型开发

2. **验证阶段**: 使用 NIH ChestX-ray14
   - 跨站基准测试
   - 评估泛化能力
   - 设备差异分析

3. **部署阶段**: 使用 MIMIC-CXR
   - 多医院数据验证
   - 真实临床场景
   - 最终模型评估

---
""")


if __name__ == '__main__':
    DatasetGuide.print_complete_guide()
