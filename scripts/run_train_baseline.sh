#!/bin/bash
# 训练基线模型（内部开发集），与论文实验 exp_internal_baseline 对应
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

# 使用 ImageFolder 结构的数据；示例使用 dummy metadata/splits
python scripts/train.py \
  --data-dir data \
  --arch resnet18 \
  --img-size 224 \
  --val-split 0.1 \
  --checkpoint-dir experiments/exp_internal_baseline

echo "完成：模型已保存到 experiments/exp_internal_baseline/best_model.pth"
