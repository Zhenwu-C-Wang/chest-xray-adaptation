#!/bin/bash
# macOS上推荐的项目运行方式
# 由于PyTorch安装复杂性，推荐使用Conda Base环境

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        胸部X光分诊系统 - 快速启动脚本                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检查当前环境
VENV_ACTIVE=$(python -c "import sys; print(sys.prefix != sys.base_prefix)" 2>/dev/null)
CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV

echo "📊 当前环境检查"
echo "  conda环境: $CONDA_DEFAULT_ENV"
echo "  虚拟环境活跃: $VENV_ACTIVE"
echo ""

# 推荐方案
echo "🎯 推荐方案: 使用 Conda Base 环境"
echo ""
echo "如果你在 venv 中，请先退出:"
echo "  deactivate"
echo ""
echo "然后确保在 base conda 环境中:"
echo "  conda activate base"
echo ""

# 检查PyTorch
echo "🔍 检查 PyTorch..."
python -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} 已安装')
    print(f'✅ CUDA 可用: {torch.cuda.is_available()}')
except ImportError:
    print('❌ PyTorch 未安装')
    print('请运行: conda install pytorch torchvision torchaudio -c pytorch')
" 2>&1

echo ""
echo "📋 验证所有依赖:"
python verify_system.py 2>&1 | head -30

echo ""
echo "✅ 准备就绪！"
echo ""
echo "运行完整系统:"
echo "  python scripts/cross_site_validation_example.py"
echo ""
