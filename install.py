#!/usr/bin/env python3
"""
一站式安装和验证脚本
快速检查和安装项目依赖
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def install_dependencies():
    """安装依赖"""
    print_header("安装依赖包")
    
    try:
        print("检查 pip...")
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        
        print("更新 pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                      capture_output=True, check=True)
        
        print("安装项目依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def verify_installation():
    """验证安装"""
    print_header("验证安装")
    
    packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'cv2',
        'PIL'
    ]
    
    all_ok = True
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            all_ok = False
    
    return all_ok


def check_gpu():
    """检查GPU"""
    print_header("GPU检查")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   设备: {torch.cuda.get_device_name(0)}")
            print(f"   数量: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  CUDA 不可用（将使用CPU）")
            return False
    except Exception as e:
        print(f"⚠️  GPU检查失败: {e}")
        return False


def check_project_structure():
    """检查项目结构"""
    print_header("项目结构检查")
    
    required_dirs = [
        'data/datasets',
        'src/validation',
        'src/models',
        'scripts',
        'reports'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ (创建中...)")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║     胸部X光分诊系统 - 安装和验证脚本                               ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # 步骤1: 项目结构
    print("\n[1/4] 检查项目结构...")
    check_project_structure()
    
    # 步骤2: 安装依赖
    print("\n[2/4] 安装依赖...")
    if not install_dependencies():
        print("\n❌ 依赖安装失败")
        sys.exit(1)
    
    # 步骤3: 验证安装
    print("\n[3/4] 验证安装...")
    if not verify_installation():
        print("\n⚠️  某些包未安装")
    
    # 步骤4: 检查GPU
    print("\n[4/4] 检查GPU...")
    check_gpu()
    
    # 完成
    print_header("安装完成")
    
    print("""
✅ 系统已准备就绪！

下一步：

1. 查看快速开始指南
   python -c "from pathlib import Path; print(Path('QUICK_START.md').read_text()[:500])"

2. 检查数据集配置
   python data/DATASET_GUIDE.py

3. 运行示例验证
   python scripts/cross_site_validation_example.py

详细信息请参考：
  • QUICK_START.md - 快速开始指南
  • IMPLEMENTATION_OVERVIEW.md - 完整实现说明
  • PROJECT_STATUS.md - 项目状态总结
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中止安装")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        sys.exit(1)
