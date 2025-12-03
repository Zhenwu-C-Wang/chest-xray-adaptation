#!/bin/bash
# macOS Conda 环境设置脚本
# 如果 pip 安装失败，使用此脚本通过 conda 创建环境

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="chest-xray"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      使用 Conda 创建环境                                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 环境信息"
echo "  项目目录: $PROJECT_DIR"
echo "  Conda 环境名: $ENV_NAME"
echo ""

# 检查是否已存在环境
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Conda 环境已存在: $ENV_NAME"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "使用现有环境"
        echo ""
        echo "激活环境:"
        echo "  conda activate $ENV_NAME"
        exit 0
    fi
fi

# 创建 conda 环境
echo "📦 创建 Conda 环境..."
conda create -n $ENV_NAME python=3.13 -y
echo "✅ Conda 环境创建成功"
echo ""

# 激活环境
echo "🔄 激活环境..."
source activate $ENV_NAME
echo "✅ 环境已激活"
echo ""

# 安装 PyTorch
echo "📥 安装 PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y
echo "✅ PyTorch 安装成功"
echo ""

# 安装其他依赖
echo "📦 安装其他依赖..."
pip install -r "$PROJECT_DIR/requirements-macos.txt" -q
echo "✅ 依赖安装成功"
echo ""

# 验证安装
echo "🔍 验证安装..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'  Pandas: {pandas.__version__}')"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ 设置完成！                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 后续步骤:"
echo ""
echo "1️⃣  激活环境:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2️⃣  验证环境:"
echo "   python setup_environment.py"
echo ""
echo "3️⃣  开始开发:"
echo "   python scripts/cross_site_validation_example.py"
echo ""
