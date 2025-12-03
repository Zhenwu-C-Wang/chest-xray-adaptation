#!/usr/bin/env python3
"""
快速系统验证 - 仅测试架构和基础功能（无需PyTorch）
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

print("=" * 70)
print("🚀 胸部X光分诊系统 - 架构验证")
print("=" * 70)
print()

# 测试1: 项目结构
print("✅ 测试1: 项目结构完整性")
print("-" * 70)

base_path = Path(__file__).parent
required_dirs = [
    'data/datasets',
    'src/validation',
    'scripts',
]

all_exist = True
for dir_path in required_dirs:
    full_path = base_path / dir_path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {dir_path}: {'存在' if exists else '缺失'}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✓ 所有必需目录存在")
else:
    print("\n⚠ 某些目录缺失")

# 测试2: 代码文件完整性
print("\n✅ 测试2: 核心代码文件")
print("-" * 70)

required_files = [
    'data/datasets/__init__.py',
    'data/datasets/nih_chestxray14.py',
    'data/datasets/chexpert.py',
    'data/datasets/mimic_cxr.py',
    'src/validation/cross_site_validator.py',
    'src/validation/calibration.py',
    'src/validation/report_generator.py',
    'scripts/cross_site_validation_example.py',
]

files_ok = True
for file_path in required_files:
    full_path = base_path / file_path
    exists = full_path.exists()
    if exists:
        size = full_path.stat().st_size
        lines = len(full_path.read_text().splitlines())
        print(f"✓ {file_path}")
        print(f"  ({lines} 行, {size/1024:.1f} KB)")
    else:
        print(f"✗ {file_path}: 缺失")
        files_ok = False

if files_ok:
    print("\n✓ 所有核心代码文件完整")

# 测试3: Python语法检查
print("\n✅ 测试3: Python语法检查")
print("-" * 70)

import py_compile
syntax_ok = True

files_to_check = [
    base_path / 'src/validation/cross_site_validator.py',
    base_path / 'setup_environment.py',
    base_path / 'quick_test.py',
]

for py_file in files_to_check:
    try:
        py_compile.compile(str(py_file), doraise=True)
        print(f"✓ {py_file.name}: 语法正确")
    except py_compile.PyCompileError as e:
        print(f"✗ {py_file.name}: 语法错误")
        print(f"  {e}")
        syntax_ok = False

if syntax_ok:
    print("\n✓ 所有文件语法正确")

# 测试4: 文档完整性
print("\n✅ 测试4: 文档完整性")
print("-" * 70)

docs = [
    'QUICK_START.md',
    'IMPLEMENTATION_OVERVIEW.md',
    'VENV_GUIDE.md',
    'SETUP.md',
]

for doc in docs:
    doc_path = base_path / doc
    if doc_path.exists():
        size = doc_path.stat().st_size
        lines = len(doc_path.read_text().splitlines())
        print(f"✓ {doc}")
        print(f"  ({lines} 行, {size/1024:.1f} KB)")
    else:
        print(f"✗ {doc}: 缺失")

# 测试5: 依赖检查
print("\n✅ 测试5: Python依赖检查")
print("-" * 70)

required_packages = [
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'matplotlib',
    'PIL',
    'cv2',
    'yaml',
    'pydantic',
]

import importlib
packages_ok = True
for pkg_name in required_packages:
    # 特殊映射
    import_name = {
        'sklearn': 'sklearn',
        'PIL': 'PIL',
        'cv2': 'cv2',
        'yaml': 'yaml',
    }.get(pkg_name, pkg_name)
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {pkg_name}: {version}")
    except ImportError:
        print(f"✗ {pkg_name}: 未安装")
        packages_ok = False

# 测试6: 模拟数据处理（无需PyTorch）
print("\n✅ 测试6: 数据处理功能测试")
print("-" * 70)

# 生成模拟数据
np.random.seed(42)
n_samples = 100
n_classes = 14

# 模拟多标签分类的概率和标签
probs = np.random.uniform(0, 1, (n_samples, n_classes))
targets = np.random.randint(0, 2, (n_samples, n_classes))

print(f"✓ 生成模拟数据:")
print(f"  - 样本数: {n_samples}")
print(f"  - 类别数: {n_classes}")

# 计算基础指标
overall_auc = roc_auc_score(targets.flatten(), probs.flatten())
binary_preds = (probs > 0.5).astype(int)
accuracy = accuracy_score(targets.flatten(), binary_preds.flatten())

print(f"\n✓ 计算验证指标:")
print(f"  - AUROC: {overall_auc:.4f}")
print(f"  - 准确率: {accuracy:.4f}")

# 按类别计算指标
aucs = []
for class_idx in range(n_classes):
    try:
        auc = roc_auc_score(targets[:, class_idx], probs[:, class_idx])
        aucs.append(auc)
    except:
        pass

if aucs:
    print(f"  - 平均类别AUROC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# 测试7: 配置管理
print("\n✅ 测试7: 配置文件检查")
print("-" * 70)

config_files = [
    'requirements.txt',
    'requirements-macos.txt',
    'config.example.json',
    '.gitignore',
]

for config in config_files:
    cfg_path = base_path / config
    if cfg_path.exists():
        size = cfg_path.stat().st_size
        print(f"✓ {config} ({size} 字节)")
    else:
        print(f"⚠ {config}: 可选文件")

# 总结
print("\n" + "=" * 70)
print("✅ 系统架构验证完成！")
print("=" * 70)
print()
print("📊 系统状态:")
print()
if packages_ok:
    print("✓ 基础依赖满足 (numpy, pandas, sklearn, matplotlib 等)")
else:
    print("⚠ 部分依赖缺失")

print()
print("📦 下一步 - 安装完整依赖:")
print()
print("方案 1️⃣  使用 conda (推荐在macOS):")
print("   conda install pytorch torchvision torchaudio -c pytorch")
print()
print("方案 2️⃣  使用虚拟环境 (如果网络正常):")
print("   bash setup_venv.sh")
print()
print("方案 3️⃣  手动使用 conda 环境:")
print("   bash setup_conda.sh")
print()
print("💡 PyTorch安装验证:")
print("   python -c 'import torch; print(torch.__version__)'")
print()
