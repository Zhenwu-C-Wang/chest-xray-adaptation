#!/bin/bash
# 外部验证示例（对应 exp_external_nih 配置）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

MODEL_PATH=experiments/exp_internal_baseline/best_model.pth
DATA_DIR=data  # 替换为 NIH/CheXpert/MIMIC 的 ImageFolder 或自定义 DataLoader 入口

if [ ! -f "$MODEL_PATH" ]; then
  echo "未找到模型 $MODEL_PATH ，请先运行 run_train_baseline.sh"
  exit 1
fi

python scripts/evaluate.py \
  --model "$MODEL_PATH" \
  --data-dir "$DATA_DIR" \
  --arch resnet18 \
  --img-size 224
