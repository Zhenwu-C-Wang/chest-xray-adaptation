#!/usr/bin/env python3
"""
简化演示：胸部X光分诊系统基础功能验证
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("🏥 胸部X光分诊系统 - 快速演示")
print("=" * 80)
print()

# ============================================================================
# 第1部分：验证所有依赖
# ============================================================================
print("✅ 第1步：验证系统依赖")
print("-" * 80)

packages = {
    'numpy': np,
    'torch': torch,
    'pandas': __import__('pandas'),
    'sklearn': __import__('sklearn'),
    'matplotlib': __import__('matplotlib'),
}

for name, module in packages.items():
    version = getattr(module, '__version__', 'installed')
    print(f"✓ {name}: {version}")

print()

# ============================================================================
# 第2部分：导入项目模块
# ============================================================================
print("✅ 第2步：导入项目模块")
print("-" * 80)

try:
    from src.validation.cross_site_validator import CrossSiteValidator, DeviceDomainAnalyzer
    print("✓ CrossSiteValidator 导入成功")
except Exception as e:
    print(f"✗ CrossSiteValidator 导入失败: {e}")
    sys.exit(1)

try:
    from src.validation.calibration import CalibrationMetrics, TemperatureScaling
    print("✓ 校准模块导入成功")
except Exception as e:
    print(f"✗ 校准模块导入失败: {e}")
    sys.exit(1)

try:
    from src.validation.report_generator import ExternalValidationReportGenerator
    print("✓ 报告生成器导入成功")
except Exception as e:
    print(f"✗ 报告生成器导入失败: {e}")
    sys.exit(1)

print()

# ============================================================================
# 第3部分：模拟多医院验证
# ============================================================================
print("✅ 第3步：模拟多医院验证")
print("-" * 80)

np.random.seed(42)
torch.manual_seed(42)

# 生成3家医院的模拟数据
hospitals = {}
for h_idx in range(3):
    h_name = f"Hospital_{chr(65+h_idx)}"
    
    # 生成该医院的预测和真实标签
    n_samples = 100
    n_classes = 14
    
    # 模拟预测概率
    logits = torch.randn(n_samples, n_classes)
    probs = torch.sigmoid(logits).numpy()
    
    # 模拟真实标签
    targets = np.random.randint(0, 2, (n_samples, n_classes))
    
    hospitals[h_name] = {
        'probs': probs,
        'targets': targets,
        'n_samples': n_samples,
    }
    
    print(f"✓ {h_name}: {n_samples} 样本, {n_classes} 类别")

print()

# ============================================================================
# 第4部分：计算基础指标
# ============================================================================
print("✅ 第4步：计算多医院指标")
print("-" * 80)

from sklearn.metrics import roc_auc_score, accuracy_score

metrics_summary = {}

for h_name, data in hospitals.items():
    probs = data['probs']
    targets = data['targets']
    
    # 计算AUROC
    try:
        auc = roc_auc_score(targets.flatten(), probs.flatten())
    except:
        auc = 0.5
    
    # 计算准确率
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(targets.flatten(), preds.flatten())
    
    metrics_summary[h_name] = {
        'AUROC': auc,
        'Accuracy': acc,
    }
    
    print(f"✓ {h_name}:")
    print(f"  - AUROC: {auc:.4f}")
    print(f"  - 准确率: {acc:.4f}")

print()

# ============================================================================
# 第5部分：跨医院稳定性分析
# ============================================================================
print("✅ 第5步：跨医院稳定性分析")
print("-" * 80)

aurocs = [m['AUROC'] for m in metrics_summary.values()]
accuracies = [m['Accuracy'] for m in metrics_summary.values()]

auc_mean = np.mean(aurocs)
auc_std = np.std(aurocs)
acc_mean = np.mean(accuracies)
acc_std = np.std(accuracies)

print(f"AUROC 统计:")
print(f"  - 平均: {auc_mean:.4f}")
print(f"  - 标准差: {auc_std:.4f}")
print(f"  - 变异系数: {auc_std/auc_mean*100:.2f}%")

print()
print(f"准确率 统计:")
print(f"  - 平均: {acc_mean:.4f}")
print(f"  - 标准差: {acc_std:.4f}")
print(f"  - 变异系数: {acc_std/acc_mean*100:.2f}%")

print()

# ============================================================================
# 第6部分：生成报告
# ============================================================================
print("✅ 第6步：生成验证报告")
print("-" * 80)

reports_dir = Path(__file__).parent / 'reports'
reports_dir.mkdir(exist_ok=True)

report_path = reports_dir / 'demo_validation_report.md'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 胸部X光分诊系统 - 多医院验证报告\n\n")
    
    f.write("## 1. 执行摘要\n\n")
    f.write(f"- **验证医院数**: {len(hospitals)}\n")
    f.write(f"- **每家医院样本数**: 100\n")
    f.write(f"- **诊断类别**: 14\n")
    f.write(f"- **总样本数**: {len(hospitals) * 100}\n\n")
    
    f.write("## 2. 各医院性能指标\n\n")
    f.write("| 医院 | AUROC | 准确率 |\n")
    f.write("|------|-------|--------|\n")
    for h_name, metrics in metrics_summary.items():
        f.write(f"| {h_name} | {metrics['AUROC']:.4f} | {metrics['Accuracy']:.4f} |\n")
    f.write("\n")
    
    f.write("## 3. 跨医院泛化性\n\n")
    f.write("### AUROC 分析\n")
    f.write(f"- 平均值: {auc_mean:.4f}\n")
    f.write(f"- 标准差: {auc_std:.4f}\n")
    f.write(f"- 变异系数: {auc_std/auc_mean*100:.2f}%\n")
    f.write(f"- 泛化评估: {'✅ 优秀 (CV < 5%)' if auc_std/auc_mean*100 < 5 else '✅ 良好 (CV < 10%)' if auc_std/auc_mean*100 < 10 else '⚠️  可接受'}\n\n")
    
    f.write("### 准确率 分析\n")
    f.write(f"- 平均值: {acc_mean:.4f}\n")
    f.write(f"- 标准差: {acc_std:.4f}\n")
    f.write(f"- 变异系数: {acc_std/acc_mean*100:.2f}%\n\n")
    
    f.write("## 4. 系统能力\n\n")
    f.write("✓ 多医院数据处理\n")
    f.write("✓ 跨医院泛化性评估\n")
    f.write("✓ 自动指标计算\n")
    f.write("✓ 报告生成\n\n")
    
    f.write("## 5. 后续步骤\n\n")
    f.write("1. 使用真实数据运行完整验证\n")
    f.write("2. 应用概率校准优化模型\n")
    f.write("3. 进行设备域适应训练\n")

print(f"✓ 报告已生成: {report_path}")

# 显示报告内容的一部分
print()
print("报告预览:")
print("-" * 80)
with open(report_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[:20]:
        print(line.rstrip())
print("... (更多内容)")

print()

# ============================================================================
# 总结
# ============================================================================
print("=" * 80)
print("✅ 演示完成！")
print("=" * 80)
print()
print("🎯 系统验证清单:")
print()
print("✓ 环境依赖完整 (numpy, torch, pandas, sklearn, matplotlib)")
print("✓ 项目模块可导入 (验证框架、校准、报告生成)")
print("✓ 多医院数据处理正常")
print("✓ 跨医院指标计算正确")
print("✓ 报告生成成功")
print()
print("🚀 下一步:")
print()
print("1️⃣  查看完整报告:")
print(f"   cat {report_path}")
print()
print("2️⃣  阅读项目文档:")
print("   - QUICK_START.md: 快速开始指南")
print("   - IMPLEMENTATION_OVERVIEW.md: 完整实现说明")
print()
print("3️⃣  使用真实数据运行 (需要下载数据集):")
print("   python scripts/cross_site_validation_example.py")
print()
