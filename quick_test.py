#!/usr/bin/env python3
"""
快速测试脚本 - 验证核心功能（无需真实PyTorch模型）
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

print("=" * 70)
print("🚀 胸部X光分诊系统 - 快速测试")
print("=" * 70)
print()

# 测试1: 导入检查
print("✅ 测试1: 核心模块导入")
print("-" * 70)
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.validation.calibration import CalibrationMetrics, TemperatureScaling
    print("✓ CalibrationMetrics 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 生成模拟数据
print("\n✅ 测试2: 生成模拟验证数据")
print("-" * 70)

# 模拟单标签多分类的验证结果
np.random.seed(42)
n_samples = 100
n_classes = 4

logits = np.random.randn(n_samples, n_classes)
targets = np.random.randint(0, n_classes, n_samples)
exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

print(f"✓ 生成{n_samples}个样本的模拟数据")
print(f"  - Logits shape: {logits.shape}")
print(f"  - Targets shape: {targets.shape}")
print(f"  - 概率范围: [{probs.min():.4f}, {probs.max():.4f}]")

# 测试3: ECE计算
print("\n✅ 测试3: ECE（期望校准误差）计算")
print("-" * 70)

try:
    metrics = CalibrationMetrics()
    ece = metrics.expected_calibration_error(probs, targets, n_bins=10)
    mce = metrics.maximum_calibration_error(probs, targets, n_bins=10)
    brier = metrics.brier_score(probs, targets)
    
    print(f"✓ ECE计算成功: {ece:.4f}")
    print(f"✓ MCE计算成功: {mce:.4f}")
    print(f"✓ Brier Score: {brier:.4f}")
    
    if ece <= 0.3:
        print("✓ ECE 在合理范围内")
    else:
        print(f"⚠ ECE较高（{ece:.4f}），需要校准")
except Exception as e:
    print(f"✗ ECE计算失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Temperature Scaling
print("\n✅ 测试4: Temperature Scaling 校准")
print("-" * 70)

try:
    calibrator = TemperatureScaling()
    calibrator.fit(logits, targets)
    calibrated_probs = calibrator.calibrate(logits)
    
    print(f"✓ Temperature Scaling 拟合成功")
    print(f"  - 温度参数: {calibrator.temperature:.4f}")
    print(f"  - 校准后概率范围: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")
    
    # 计算校准前后的ECE
    ece_before = metrics.expected_calibration_error(probs, targets)
    ece_after = metrics.expected_calibration_error(calibrated_probs, targets)
    improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
    
    print(f"  - ECE改进: {improvement:.1f}% ({ece_before:.4f} → {ece_after:.4f})")
except Exception as e:
    print(f"✗ Temperature Scaling 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 验证框架
print("\n✅ 测试5: 多站点验证框架")
print("-" * 70)

try:
    from src.validation.cross_site_validator import CrossSiteValidator
    
    # 生成模拟多站点数据
    sites_data = {}
    for site_idx in range(3):
        site_name = f"Site_{site_idx+1}"
        site_logits = np.random.randn(50, n_classes)
        exp_logits = np.exp(site_logits - site_logits.max(axis=1, keepdims=True))
        site_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        site_targets = np.random.randint(0, n_classes, 50)
        sites_data[site_name] = {
            'probs': site_probs,
            'targets': site_targets
        }
    
    print(f"✓ 生成 {len(sites_data)} 个站点的模拟数据")
    
    # 计算每个站点的指标
    for site, data in sites_data.items():
        site_ece = metrics.expected_calibration_error(data['probs'], data['targets'])
        print(f"  - {site}: ECE = {site_ece:.4f}")
        
except Exception as e:
    print(f"✗ 多站点验证失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 报告生成
print("\n✅ 测试6: 报告生成框架")
print("-" * 70)

try:
    from src.validation.report_generator import ExternalValidationReportGenerator
    
    gen = ExternalValidationReportGenerator()
    print("✓ 报告生成器初始化成功")
    print("✓ 支持的方法:")
    methods = [
        m for m in dir(gen) 
        if not m.startswith('_') and callable(getattr(gen, m))
    ]
    for method in methods[:5]:
        print(f"  - {method}")
    if len(methods) > 5:
        print(f"  ... 和 {len(methods)-5} 个其他方法")
    
except Exception as e:
    print(f"✗ 报告生成失败: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 70)
print("✅ 测试完成！")
print("=" * 70)
print()
print("📋 后续步骤:")
print()
print("1️⃣  安装PyTorch:")
print("   conda install pytorch torchvision torchaudio -c pytorch")
print()
print("2️⃣  下载数据集:")
print("   python data/DATASET_GUIDE.py")
print()
print("3️⃣  运行完整验证:")
print("   python scripts/cross_site_validation_example.py")
print()
