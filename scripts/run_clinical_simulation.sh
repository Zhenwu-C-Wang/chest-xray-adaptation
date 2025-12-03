#!/bin/bash
# 临床分诊模拟占位脚本，需在 src/clinical/triage_simulation.py 落地后填充。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD

echo "占位：请实现 src/clinical/triage_simulation.py 后，在此调用生成等待时间/漏诊率结果。"
