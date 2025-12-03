#!/bin/bash
# 校准示例：在验证集拟合温度缩放/等距回归，再评估外部集
# 提示：当前仓库的完整校准流水线需按实际数据/loader 调整，本脚本示意入口。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

MODEL_PATH=experiments/exp_internal_baseline/best_model.pth
CALIB_DATA=data   # 校准用的验证集（ImageFolder 或自定义 loader）

if [ ! -f "$MODEL_PATH" ]; then
  echo "未找到模型 $MODEL_PATH ，请先运行 run_train_baseline.sh"
  exit 1
fi

python scripts/cross_site_validation_example.py  # 如需真实校准，请在脚本中接入实际 DataLoader
echo "提示：如要真实校准，请在 cross_site_validation_example.py 中用你的校准/评估 DataLoader 替换示例代码。"
