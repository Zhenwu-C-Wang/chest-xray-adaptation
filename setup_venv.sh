#!/bin/bash
# 虚拟环境设置脚本
# 自动创建、激活和配置Python虚拟环境

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      Python 虚拟环境设置                                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 系统信息"
echo "  Python 版本: $PYTHON_VERSION"
echo "  项目目录: $PROJECT_DIR"
echo "  虚拟环境: $VENV_DIR"
echo ""

# 检查虚拟环境是否已存在
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        echo "使用现有虚拟环境"
        echo ""
        echo "激活虚拟环境:"
        echo "  source $VENV_DIR/bin/activate"
        exit 0
    fi
fi

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python3 -m venv "$VENV_DIR"
echo "✅ 虚拟环境创建成功"
echo ""

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source "$VENV_DIR/bin/activate"
echo "✅ 虚拟环境已激活"
echo ""

# 升级pip
echo "📥 升级 pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✅ pip 升级成功"
echo ""

# 检测操作系统
OS_TYPE="$(uname -s)"
echo "🖥️  操作系统: $OS_TYPE"
echo ""

# 在macOS上安装PyTorch
if [ "$OS_TYPE" = "Darwin" ]; then
    echo "🍎 检测到 macOS - 使用官方PyTorch安装方式..."
    echo "📦 安装 PyTorch..."
    
    # 尝试多个源
    pip install torch torchvision torchaudio 2>/dev/null || \
    pip install -i https://pypi.tsinghua.edu.cn/simple torch torchvision torchaudio 2>/dev/null || \
    pip install -i https://mirrors.aliyun.com/pypi/simple torch torchvision torchaudio
    
    echo "✅ PyTorch 安装成功"
    echo ""
    
    # 安装其他依赖
    if [ -f "$PROJECT_DIR/requirements-macos.txt" ]; then
        echo "📦 安装其他项目依赖..."
        pip install -r "$PROJECT_DIR/requirements-macos.txt" -i https://pypi.tsinghua.edu.cn/simple 2>/dev/null || \
        pip install -r "$PROJECT_DIR/requirements-macos.txt"
        echo "✅ 依赖安装成功"
        echo ""
    fi
else
    # Linux 或其他系统
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        echo "📦 安装项目依赖..."
        pip install -r "$PROJECT_DIR/requirements.txt" -i https://pypi.tsinghua.edu.cn/simple 2>/dev/null || \
        pip install -r "$PROJECT_DIR/requirements.txt"
        echo "✅ 依赖安装成功"
        echo ""
    fi
fi

# 验证安装
echo "🔍 验证安装..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ⚠️  PyTorch 未安装"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  ⚠️  NumPy 未安装"
python -c "import pandas; print(f'  Pandas: {pandas.__version__}')" 2>/dev/null || echo "  ⚠️  Pandas 未安装"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   ✅ 设置完成！                              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 后续步骤:"
echo ""
echo "1️⃣  激活虚拟环境 (如果尚未激活):"
echo "   source venv/bin/activate"
echo ""
echo "2️⃣  验证环境:"
echo "   python setup_environment.py"
echo ""
echo "3️⃣  开始开发:"
echo "   python scripts/cross_site_validation_example.py"
echo ""
echo "💡 常用命令:"
echo "   激活: source venv/bin/activate"
echo "   退出: deactivate"
echo "   查看包: pip list"
echo "   更新包: pip install --upgrade package_name"
echo ""
