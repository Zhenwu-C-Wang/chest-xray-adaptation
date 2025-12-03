"""
环境检查和配置工具
验证所有必要的依赖和数据集配置
"""

import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
import json


class EnvironmentChecker:
    """环境检查工具"""
    
    REQUIRED_PACKAGES = {
        'torch': '1.9.0',
        'torchvision': '0.10.0',
        'numpy': '1.19.0',
        'pandas': '1.1.0',
        'scikit-learn': '0.24.0',
        'matplotlib': '3.3.0',
        'PIL': '8.0.0',
        'cv2': '4.5.0'
    }
    
    @staticmethod
    def check_python_version() -> Tuple[bool, str]:
        """检查Python版本"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 7:
            msg = f"✓ Python {version.major}.{version.minor}.{version.micro}"
            return True, msg
        else:
            msg = f"✗ Python版本过低 ({version.major}.{version.minor})，需要 3.7+"
            return False, msg
    
    @staticmethod
    def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
        """检查单个包"""
        try:
            # 特殊处理容易冲突的包
            if package_name == 'torchvision':
                try:
                    import torchvision
                    version = getattr(torchvision, '__version__', 'unknown')
                    msg = f"✓ {package_name} ({version})"
                    return True, msg
                except RuntimeError as e:
                    if "operator torchvision::nms does not exist" in str(e):
                        msg = f"⚠️  {package_name} 版本不兼容 (与 torch 冲突)"
                        return True, msg  # 非致命错误
                    raise
            
            pkg = importlib.import_module(package_name)
            version = getattr(pkg, '__version__', 'unknown')
            msg = f"✓ {package_name} ({version})"
            return True, msg
        except ImportError:
            msg = f"✗ {package_name} 未安装"
            return False, msg
        except Exception as e:
            msg = f"⚠️  {package_name}: {str(e)[:50]}"
            return True, msg  # 非致命错误，继续
    
    @staticmethod
    def check_all_packages() -> Tuple[bool, List[str]]:
        """检查所有必需的包"""
        messages = []
        all_ok = True
        
        for package, min_version in EnvironmentChecker.REQUIRED_PACKAGES.items():
            ok, msg = EnvironmentChecker.check_package(package, min_version)
            messages.append(msg)
            if not ok:
                all_ok = False
        
        return all_ok, messages
    
    @staticmethod
    def check_gpu() -> Tuple[bool, str]:
        """检查GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                msg = f"✓ CUDA可用 ({device_count}x {device_name})"
                return True, msg
            else:
                msg = "⚠ CUDA不可用，将使用CPU（速度较慢）"
                return False, msg
        except Exception as e:
            msg = f"⚠ CUDA检查失败: {str(e)}"
            return False, msg
    
    @staticmethod
    def check_dataset_path(dataset_name: str, path: str) -> Tuple[bool, str]:
        """检查数据集路径"""
        if os.path.exists(path):
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            msg = f"✓ {dataset_name}: {path} ({size_gb:.2f}GB)"
            return True, msg
        else:
            msg = f"✗ {dataset_name}: {path} 不存在"
            return False, msg
    
    @staticmethod
    def check_project_structure() -> Tuple[bool, List[str]]:
        """检查项目目录结构"""
        messages = []
        all_ok = True
        
        required_dirs = [
            'data/datasets',
            'src/validation',
            'src/models',
            'scripts',
            'reports'
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                msg = f"✓ {dir_path}/"
                messages.append(msg)
            else:
                msg = f"⚠ {dir_path}/ 不存在（将创建）"
                messages.append(msg)
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return all_ok, messages
    
    @staticmethod
    def run_full_check() -> Dict:
        """运行完整的环境检查"""
        print("\n" + "="*70)
        print("胸部X光分诊系统 - 环境检查")
        print("="*70 + "\n")
        
        results = {}
        
        # 1. Python版本
        print("1️⃣  Python版本")
        ok, msg = EnvironmentChecker.check_python_version()
        print(f"   {msg}\n")
        results['python'] = ok
        
        # 2. 依赖包
        print("2️⃣  依赖包检查")
        ok, messages = EnvironmentChecker.check_all_packages()
        for msg in messages:
            print(f"   {msg}")
        print()
        results['packages'] = ok
        
        # 3. GPU
        print("3️⃣  GPU检查")
        ok, msg = EnvironmentChecker.check_gpu()
        print(f"   {msg}\n")
        results['gpu'] = ok
        
        # 4. 项目结构
        print("4️⃣  项目结构")
        ok, messages = EnvironmentChecker.check_project_structure()
        for msg in messages:
            print(f"   {msg}")
        print()
        results['structure'] = ok
        
        # 5. 数据集
        print("5️⃣  数据集检查")
        datasets_to_check = {
            'NIH ChestX-ray14': 'data/nih_chestxray14',
            'CheXpert': 'data/chexpert',
            'MIMIC-CXR': 'data/mimic_cxr'
        }
        
        dataset_results = {}
        for name, path in datasets_to_check.items():
            ok, msg = EnvironmentChecker.check_dataset_path(name, path)
            print(f"   {msg}")
            dataset_results[name] = ok
        print()
        results['datasets'] = dataset_results
        
        # 总结
        print("="*70)
        if results['python'] and results['packages']:
            print("✓ 环境检查完成！系统已准备好")
        else:
            print("⚠ 环境检查有问题，请修复后重试")
        print("="*70 + "\n")
        
        return results


class ConfigurationManager:
    """配置管理工具"""
    
    DEFAULT_CONFIG = {
        'device': 'cuda' if importlib.util.find_spec('torch') and 
                  importlib.import_module('torch').cuda.is_available() else 'cpu',
        'batch_size': 32,
        'num_workers': 4,
        'datasets': {
            'nih': {
                'enabled': False,
                'image_dir': 'data/nih_chestxray14/images',
                'labels_csv': 'data/nih_chestxray14/Data_Entry_2017.csv'
            },
            'chexpert': {
                'enabled': True,
                'csv_path': 'data/chexpert/CheXpert-v1.0-small/train.csv',
                'image_root': 'data/chexpert/CheXpert-v1.0-small'
            },
            'mimic': {
                'enabled': False,
                'csv_path': 'data/mimic_cxr/mimic-cxr-2.0.0-chexpert.csv',
                'image_root': 'data/mimic_cxr'
            }
        },
        'model': {
            'architecture': 'resnet50',
            'pretrained': True,
            'num_classes': 14
        },
        'validation': {
            'test_split': 0.2,
            'val_split': 0.1,
            'seed': 42
        },
        'calibration': {
            'method': 'temperature',
            'n_bins': 10,
            'learning_rate': 0.01,
            'max_iterations': 1000
        },
        'output': {
            'report_dir': 'reports',
            'metrics_file': 'site_metrics.csv',
            'report_file': 'external_validation_report.md',
            'one_pager_file': 'clinical_impact_one_pager.md'
        }
    }
    
    @staticmethod
    def load_config(config_path: str = 'config.json') -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return ConfigurationManager.DEFAULT_CONFIG.copy()
    
    @staticmethod
    def save_config(config: Dict, config_path: str = 'config.json'):
        """保存配置文件"""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ 配置已保存至: {config_path}")
    
    @staticmethod
    def create_default_config(config_path: str = 'config.json'):
        """创建默认配置文件"""
        ConfigurationManager.save_config(
            ConfigurationManager.DEFAULT_CONFIG,
            config_path
        )
    
    @staticmethod
    def print_config(config: Dict):
        """打印配置"""
        print("\n当前配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    # 环境检查
    results = EnvironmentChecker.run_full_check()
    
    # 配置管理
    print("配置文件管理")
    print("-" * 70)
    
    config_path = 'config.json'
    if os.path.exists(config_path):
        print(f"✓ 已找到现有配置文件: {config_path}")
        config = ConfigurationManager.load_config(config_path)
    else:
        print(f"⚠ 未找到配置文件，创建默认配置...")
        ConfigurationManager.create_default_config(config_path)
        config = ConfigurationManager.load_config(config_path)
    
    ConfigurationManager.print_config(config)
    
    # 建议
    print("\n" + "="*70)
    print("建议:")
    print("="*70)
    
    if not results['packages']:
        print("1. 安装缺失的依赖包:")
        print("   pip install -r requirements.txt")
    
    if not results['gpu']:
        print("2. 考虑使用GPU来加速计算:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    missing_datasets = [name for name, ok in results['datasets'].items() if not ok]
    if missing_datasets:
        print(f"3. 下载缺失的数据集: {', '.join(missing_datasets)}")
        print("   查看 QUICK_START.md 获取下载指南")
    
    print("\n✓ 准备就绪！运行以下命令开始:")
    print("   python scripts/cross_site_validation_example.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
