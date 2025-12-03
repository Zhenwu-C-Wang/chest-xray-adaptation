# 项目结构说明

本项目为胸部X光域自适应项目，采用清晰、模块化的设计结构。以下是完整的项目结构说明。

## 目录树

```
chest-xray-adaptation/
│
├── README.md                          # 项目说明文档
├── LICENSE                            # 开源许可证
├── requirements.txt                   # 项目依赖
├── .gitignore                         # Git忽略文件
├── PROJECT_STRUCTURE.md               # 本项目结构说明文件
│
├── config/                            # 配置文件
│   ├── config.yaml                    # 设备配置和模型超参数
│   └── training_config.yaml           # 训练相关配置
│
├── data/                              # 数据处理模块
│   ├── __init__.py
│   ├── raw_data/                      # 原始数据集（数据链接或脚本）
│   ├── preprocessing/                 # 数据预处理
│   │   ├── __init__.py
│   │   └── preprocess.py              # 图像预处理、增强、设备分层脚本
│   └── splits/                        # 数据划分
│       ├── __init__.py
│       └── split_dataset.py           # 将数据划分为训练、验证、测试集
│
├── src/                               # 源代码模块
│   ├── __init__.py
│   ├── models/                        # 模型定义
│   │   ├── __init__.py
│   │   └── resnet.py                  # ResNet、DenseNet等模型实现
│   ├── training/                      # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py                 # 模型训练器类
│   ├── validation/                    # 验证和评估
│   │   ├── __init__.py
│   │   └── evaluator.py               # 模型评估器类（AUROC、ECE等指标）
│   └── utils/                         # 辅助工具
│       ├── __init__.py
│       └── utils.py                   # 通用工具函数
│
├── scripts/                           # 可执行脚本
│   ├── train.py                       # 训练脚本
│   ├── evaluate.py                    # 评估脚本
│   ├── calibration.py                 # 模型校准脚本
│   └── domain_adaptation.py           # 域自适应训练脚本
│
└── tests/                             # 单元测试
    ├── test_model.py                  # 模型测试
    ├── test_preprocessing.py          # 数据预处理测试
    └── test_utils.py                  # 工具函数测试
```

## 详细说明

### 1. 配置文件 (`config/`)

#### `config.yaml`
存储设备特定的配置：
- 数据集设置（大小、归一化参数）
- 多设备配置（阈值、预处理方式）
- 图像预处理参数

#### `training_config.yaml`
存储训练相关的超参数：
- 优化器配置（学习率、权重衰减）
- 批次大小、训练周期数
- 学习率调度策略
- 正则化参数
- 早停策略

### 2. 数据模块 (`data/`)

#### `raw_data/`
- 存放原始数据集或数据链接
- 如果数据量大，可存放下载脚本

#### `preprocessing/`
- `preprocess.py`: 实现图像预处理、增强、设备分层等功能

#### `splits/`
- `split_dataset.py`: 将数据按比例划分为训练、验证、测试集

### 3. 源代码 (`src/`)

#### `models/`
- `resnet.py`: 包含ResNet18、DenseNet121等模型定义

#### `training/`
- `trainer.py`: 包含Trainer类，管理训练过程（train_epoch、train、validate等方法）

#### `validation/`
- `evaluator.py`: 包含Evaluator类，计算各种评估指标（准确率、F1、AUROC、ECE）

#### `utils/`
- `utils.py`: 包含通用工具函数
  - `set_seed()`: 设置随机种子
  - `get_device()`: 获取计算设备
  - `ImageDataset`: 图像数据集类
  - `get_data_transforms()`: 获取数据变换
  - `create_data_loader()`: 创建数据加载器
  - `save_checkpoint()` / `load_checkpoint()`: 模型检查点管理

### 4. 脚本 (`scripts/`)

#### `train.py`
- 主训练脚本
- 用法: `python scripts/train.py --config config/training_config.yaml`

#### `evaluate.py`
- 评估脚本
- 用法: `python scripts/evaluate.py --model checkpoints/model.pth`

#### `calibration.py`
- 模型概率校准脚本
- 支持Temperature scaling和Platt scaling方法

#### `domain_adaptation.py`
- 域自适应训练脚本
- 用于跨设备的域适应训练

### 5. 测试 (`tests/`)

#### `test_model.py`
- 测试模型输出形状和推理过程

#### `test_preprocessing.py`
- 测试数据预处理和增强功能

#### `test_utils.py`
- 测试工具函数的正确性

## 使用工作流

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python data/preprocessing/preprocess.py --input raw_data --output data/processed
python data/splits/split_dataset.py --input data/processed --output data/splits
```

### 3. 模型训练
```bash
python scripts/train.py --config config/training_config.yaml
```

### 4. 模型评估
```bash
python scripts/evaluate.py --model checkpoints/best_model.pth
```

### 5. 模型校准
```bash
python scripts/calibration.py --model checkpoints/best_model.pth --method temperature
```

### 6. 域自适应
```bash
python scripts/domain_adaptation.py --source-domain device1 --target-domain device2
```

## 关键特性

### 模块化设计
- 清晰的职责划分，便于维护和扩展
- 每个模块可独立测试

### 配置驱动
- 使用YAML配置文件，灵活调整参数
- 无需修改代码即可改变超参数

### 可扩展性
- 易于添加新模型
- 易于支持新的设备或数据集
- 易于实现新的域适应方法

### 最佳实践
- 使用预训练模型加速收敛
- 实现详细的评估指标
- 支持模型检查点保存和恢复
- 包含完整的单元测试

## 扩展建议

### 1. 添加新模型
在 `src/models/` 中创建新文件，实现模型类

### 2. 添加新评估指标
在 `src/validation/evaluator.py` 中扩展Evaluator类

### 3. 添加新的域适应方法
在 `scripts/` 中创建新脚本

### 4. 添加CI/CD
创建 `.github/workflows/` 目录，添加自动化测试和部署

### 5. 添加Docker支持
创建 `Dockerfile` 确保环境一致性

## 依赖管理

项目主要依赖：
- **PyTorch**: 深度学习框架
- **scikit-learn**: 机器学习工具
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Pillow**: 图像处理
- **PyYAML**: 配置文件解析

查看 `requirements.txt` 了解详细版本信息。
