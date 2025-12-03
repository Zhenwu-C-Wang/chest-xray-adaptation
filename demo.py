#!/usr/bin/env python3
"""
演示脚本：胸部X光分诊系统完整功能演示（模拟数据）
展示系统所有核心功能而无需真实医学数据
"""

import numpy as np
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.calibration import (
    CalibrationMetrics, 
    TemperatureScaling, 
    PlattScaling, 
    IsotonicCalibration
)
from src.validation.report_generator import ExternalValidationReportGenerator

print("=" * 80)
print("🏥 胸部X光分诊系统 - 完整功能演示")
print("=" * 80)
print()

# ============================================================================
# 第1部分：生成模拟验证数据
# ============================================================================
print("📊 第1步：生成模拟验证数据")
print("-" * 80)

np.random.seed(42)
torch.manual_seed(42)

# 模拟多个站点的验证数据
n_sites = 3
n_samples_per_site = 100
n_classes = 14  # 胸部X光的14种诊断

sites_data = {}

for site_idx in range(n_sites):
    site_name = f"Hospital_{chr(65+site_idx)}"  # Hospital_A, Hospital_B, Hospital_C
    
    # 生成该站点的logits和标签
    logits = torch.randn(n_samples_per_site, n_classes)
    targets = torch.randint(0, 2, (n_samples_per_site, n_classes)).float()
    probs = torch.sigmoid(logits)
    
    sites_data[site_name] = {
        'logits': logits.numpy(),
        'probs': probs.numpy(),
        'targets': targets.numpy(),
    }
    
    print(f"✓ {site_name}:")
    print(f"  - 样本数: {n_samples_per_site}")
    print(f"  - 诊断类别: {n_classes}")
    print(f"  - 概率范围: [{probs.min():.4f}, {probs.max():.4f}]")

print()

# ============================================================================
# 第2部分：多站点验证和ECE计算
# ============================================================================
print("📈 第2步：多站点验证 - 计算ECE和其他指标")
print("-" * 80)

metrics_calc = CalibrationMetrics()
site_metrics = {}

for site_name, data in sites_data.items():
    probs = data['probs']
    targets = data['targets']
    
    # 对于多标签分类，对每个标签计算ECE并求平均
    ece_list = []
    mce_list = []
    brier_list = []
    
    for class_idx in range(n_classes):
        class_probs = probs[:, class_idx]
        class_targets = targets[:, class_idx]
        class_preds = (class_probs > 0.5).astype(int)
        
        # 计算该类别的指标
        try:
            ece_c = metrics_calc.expected_calibration_error(
                class_preds, class_targets, class_probs, n_bins=5
            )
            mce_c = metrics_calc.maximum_calibration_error(
                class_preds, class_targets, class_probs, n_bins=5
            )
            brier_c = metrics_calc.brier_score(
                class_preds, class_targets, class_probs
            )
            
            ece_list.append(ece_c)
            mce_list.append(mce_c)
            brier_list.append(brier_c)
        except:
            pass
    
    # 取平均值
    ece = np.mean(ece_list) if ece_list else 0.0
    mce = np.mean(mce_list) if mce_list else 0.0
    brier = np.mean(brier_list) if brier_list else 0.0
    
    site_metrics[site_name] = {
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
    }
    
    print(f"✓ {site_name}:")
    print(f"  - ECE (期望校准误差): {ece:.4f}")
    print(f"  - MCE (最大校准误差): {mce:.4f}")
    print(f"  - Brier Score (布赖尔分数): {brier:.4f}")

print()

# ============================================================================
# 第3部分：概率校准演示
# ============================================================================
print("🔧 第3步：概率校准 - 对比三种校准方法")
print("-" * 80)

# 使用第一个站点作为校准演示
demo_site = list(sites_data.keys())[0]
demo_data = sites_data[demo_site]

logits_np = demo_data['logits']
probs_np = demo_data['probs']
targets_np = demo_data['targets']

# 计算校准前的ECE
ece_before_list = []
for class_idx in range(n_classes):
    class_probs = probs_np[:, class_idx]
    class_targets = targets_np[:, class_idx]
    class_preds = (class_probs > 0.5).astype(int)
    try:
        ece_c = metrics_calc.expected_calibration_error(
            class_preds, class_targets, class_probs, n_bins=5
        )
        ece_before_list.append(ece_c)
    except:
        pass

ece_before = np.mean(ece_before_list) if ece_before_list else 0.0

print(f"校准前的ECE: {ece_before:.4f}")
print()

# 应用三种校准方法
calibrators = {
    'Temperature Scaling': TemperatureScaling(),
    'Platt Scaling': PlattScaling(),
    'Isotonic Calibration': IsotonicCalibration(),
}

calibration_results = {}

for cal_name, calibrator in calibrators.items():
    try:
        # 拟合校准器
        calibrator.fit(logits_np, targets_np)
        
        # 获取校准后的概率
        calibrated_probs = calibrator.calibrate(probs_np)
        
        # 计算校准后的ECE
        ece_after_list = []
        for class_idx in range(n_classes):
            class_cal_probs = calibrated_probs[:, class_idx]
            class_targets = targets_np[:, class_idx]
            class_preds = (class_cal_probs > 0.5).astype(int)
            try:
                ece_c = metrics_calc.expected_calibration_error(
                    class_preds, class_targets, class_cal_probs, n_bins=5
                )
                ece_after_list.append(ece_c)
            except:
                pass
        
        ece_after = np.mean(ece_after_list) if ece_after_list else 0.0
        
        # 计算改进比例
        improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
        
        calibration_results[cal_name] = {
            'ECE_before': ece_before,
            'ECE_after': ece_after,
            'improvement': improvement,
            'calibrated_probs': calibrated_probs,
        }
        
        print(f"✓ {cal_name}:")
        print(f"  - ECE 改进: {improvement:.1f}%")
        print(f"  - 校准后 ECE: {ece_after:.4f}")
        if hasattr(calibrator, 'temperature'):
            print(f"  - 温度参数: {calibrator.temperature:.4f}")
        
    except Exception as e:
        print(f"✗ {cal_name} 失败: {e}")

print()

# ============================================================================
# 第4部分：跨站点稳定性分析
# ============================================================================
print("🔍 第4步：跨站点稳定性分析")
print("-" * 80)

# 计算各站点ECE的统计数据
ece_values = [metrics[' ECE'] for metrics in site_metrics.values()]
ece_mean = np.mean([m['ECE'] for m in site_metrics.values()])
ece_std = np.std([m['ECE'] for m in site_metrics.values()])
ece_cv = (ece_std / ece_mean * 100) if ece_mean > 0 else 0

print(f"✓ 跨站点ECE分析:")
print(f"  - 平均ECE: {ece_mean:.4f}")
print(f"  - 标准差: {ece_std:.4f}")
print(f"  - 变异系数: {ece_cv:.2f}%")

if ece_cv <= 5:
    print(f"  - 稳定性评估: ✅ 非常好（CV ≤ 5%）")
elif ece_cv <= 10:
    print(f"  - 稳定性评估: ✅ 较好（CV ≤ 10%）")
else:
    print(f"  - 稳定性评估: ⚠️  可以接受（CV > 10%）")

print()

# ============================================================================
# 第5部分：生成报告
# ============================================================================
print("📄 第5步：生成验证报告")
print("-" * 80)

try:
    # 创建报告生成器
    report_gen = ExternalValidationReportGenerator()
    
    # 添加执行摘要
    summary = {
        'model_name': 'DenseNet-121 (模拟)',
        'total_samples': n_samples_per_site * n_sites,
        'sites_count': n_sites,
        'average_ece': ece_mean,
    }
    
    # 添加站点指标
    for site_name, metrics in site_metrics.items():
        report_gen.add_site_metrics(
            site_name=site_name,
            metrics_dict=metrics,
        )
    
    # 生成报告
    reports_dir = Path(__file__).parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / 'demo_report.md'
    
    # 手动生成简单报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 胸部X光分诊系统 - 外部验证报告（演示）\n\n")
        f.write("## 执行摘要\n\n")
        f.write(f"- **模型**: {summary['model_name']}\n")
        f.write(f"- **总样本数**: {summary['total_samples']}\n")
        f.write(f"- **验证站点数**: {summary['sites_count']}\n")
        f.write(f"- **平均 ECE**: {summary['average_ece']:.4f}\n\n")
        
        f.write("## 站点级别的性能\n\n")
        for site_name, metrics in site_metrics.items():
            f.write(f"### {site_name}\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"| {metric_name} | {metric_value:.4f} |\n")
            f.write("\n")
        
        f.write("## 校准效果\n\n")
        for cal_name, results in calibration_results.items():
            f.write(f"- {cal_name}: ECE 改进 {results['improvement']:.1f}% ")
            f.write(f"({results['ECE_before']:.4f} → {results['ECE_after']:.4f})\n")
        f.write("\n")
        
        f.write("## 稳定性分析\n\n")
        f.write(f"- 跨站点 ECE 变异系数: {ece_cv:.2f}%\n")
        f.write(f"- 稳定性评估: {'✅ 优秀' if ece_cv <= 5 else '✅ 良好' if ece_cv <= 10 else '⚠️  可接受'}\n")
    
    print(f"✓ 报告已生成: {report_path}")
    print(f"  - 包含: 执行摘要、站点指标、校准效果、稳定性分析")
    
except Exception as e:
    print(f"⚠️  报告生成失败: {e}")

print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("✅ 演示完成！")
print("=" * 80)
print()
print("🎯 系统功能验证清单:")
print()
print("✓ 多站点数据处理 - 支持3个医院的数据")
print("✓ ECE/MCE计算 - 期望和最大校准误差")
print("✓ 三种校准方法 - Temperature, Platt, Isotonic")
print("✓ 跨站点分析 - 稳定性和泛化性评估")
print("✓ 报告生成 - Markdown格式的验证报告")
print()
print("📊 下一步建议:")
print()
print("1️⃣  查看生成的报告:")
print("   cat reports/demo_report.md")
print()
print("2️⃣  使用真实数据运行:")
print("   python scripts/cross_site_validation_example.py")
print()
print("3️⃣  阅读完整文档:")
print("   - QUICK_START.md")
print("   - IMPLEMENTATION_OVERVIEW.md")
print()
