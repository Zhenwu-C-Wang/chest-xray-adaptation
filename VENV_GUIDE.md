# Python 虚拟环境 (venv) 使用指南

## 为什么需要虚拟环境？

虚拟环境提供了**隔离的Python环境**，解决以下问题：

- ✅ **依赖隔离**: 项目依赖不会影响系统其他项目
- ✅ **版本管理**: 不同项目可以使用不同版本的包
- ✅ **环境重现**: 可以精确复现开发环境
- ✅ **团队协作**: 确保所有开发者使用相同的依赖版本
- ✅ **部署安全**: 生产环境依赖清晰明确
- ✅ **系统保护**: 不污染系统Python环境

## 快速开始

### 方式1: 使用自动化脚本（推荐）

```bash
# 一键创建并配置虚拟环境
bash setup_venv.sh

# 脚本会自动:
# 1. 创建 venv 目录
# 2. 激活虚拟环境
# 3. 升级 pip
# 4. 安装依赖
# 5. 验证安装
```

### 方式2: 手动创建

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

## 虚拟环境文件结构

```
venv/
├── bin/                    # 可执行文件
│   ├── python             # Python 解释器
│   ├── pip                # pip 包管理器
│   ├── activate           # 激活脚本 (macOS/Linux)
│   └── ...
├── include/               # C头文件
├── lib/                   # 安装的包
│   └── python3.x/
│       └── site-packages/
├── pyvenv.cfg            # 配置文件
└── ...
```

## 常用命令

### 激活和退出

```bash
# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 退出虚拟环境 (所有平台)
deactivate
```

### 包管理

```bash
# 查看已安装的包
pip list

# 安装包
pip install package_name

# 安装特定版本
pip install package_name==1.2.3

# 升级包
pip install --upgrade package_name

# 卸载包
pip uninstall package_name

# 安装依赖文件中的所有包
pip install -r requirements.txt

# 更新依赖文件（记录当前环境的所有包）
pip freeze > requirements.txt
```

### 验证和诊断

```bash
# 查看Python版本
python --version

# 查看Python位置（应该在 venv 目录）
which python

# 验证特定包
python -c "import torch; print(torch.__version__)"

# 运行环境检查
python setup_environment.py

# 查看虚拟环境信息
python -m venv --help
```

## .gitignore 配置

**重要**: 不要把虚拟环境文件夹上传到Git！

```gitignore
# 虚拟环境
venv/
env/
.venv/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 最佳实践

### ✅ 应该做

```bash
# 1. 项目开始时创建虚拟环境
python3 -m venv venv

# 2. 每次工作前激活
source venv/bin/activate

# 3. 定期更新 requirements.txt
pip freeze > requirements.txt

# 4. 将 requirements.txt 提交到 Git
git add requirements.txt
git commit -m "Update dependencies"

# 5. 安装新包时更新 requirements.txt
pip install new_package
pip freeze > requirements.txt

# 6. 工作完成后退出虚拟环境
deactivate
```

### ❌ 不应该做

```bash
# ❌ 不要在系统Python中安装包
pip install package_name  # 会污染系统环境

# ❌ 不要提交虚拟环境文件夹
git add venv/  # venv 会很大（几百MB）

# ❌ 不要手动修改 venv 目录
rm -rf venv/lib/...  # 这样做很危险

# ❌ 不要在虚拟环境外安装依赖
# 必须先激活虚拟环境再安装
```

## 常见问题

### Q1: 虚拟环境占用空间很大？

**A**: 这是正常的。虚拟环境通常占用几百MB到1GB。

```bash
# 查看虚拟环境大小
du -sh venv/

# 如果需要节省空间，可以删除并重新创建
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q2: 切换Python版本？

**A**: 重新创建虚拟环境

```bash
# 使用特定Python版本创建
python3.9 -m venv venv

# 或者使用 pyenv 管理多个Python版本
```

### Q3: 虚拟环境"坏了"？

**A**: 删除并重新创建

```bash
deactivate
rm -rf venv
bash setup_venv.sh  # 或手动重新创建
```

### Q4: 在IDE中配置虚拨环境？

**A**: VS Code 示例

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

### Q5: 团队协作如何同步环境？

**A**: 使用 requirements.txt

```bash
# 开发者A: 更新依赖
pip install new_package
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add new dependency"
git push

# 开发者B: 拉取并同步
git pull
pip install -r requirements.txt  # 自动安装所有依赖
```

## 项目特定配置

### 本项目的建议流程

```bash
# 1. 克隆项目
git clone https://github.com/user/chest-xray-adaptation.git
cd chest-xray-adaptation

# 2. 创建虚拟环境
bash setup_venv.sh

# 3. 虚拟环境已自动激活，可以直接使用
python setup_environment.py
python scripts/cross_site_validation_example.py

# 4. 每次新开终端需要激活
source venv/bin/activate

# 5. 完成工作后退出
deactivate
```

### 依赖更新流程

```bash
# 添加新依赖
source venv/bin/activate
pip install new_package

# 更新 requirements.txt
pip freeze > requirements.txt

# 提交更改
git add requirements.txt
git commit -m "Add: new_package for feature X"
```

## 虚拨环境 vs 其他工具对比

| 工具 | 用途 | 复杂度 | 推荐度 |
|------|------|--------|--------|
| **venv** | 项目隔离 | ⭐ | ⭐⭐⭐⭐⭐ |
| **virtualenv** | 增强venv | ⭐⭐ | ⭐⭐⭐ |
| **conda** | 完整环境管理 | ⭐⭐⭐ | ⭐⭐⭐ |
| **poetry** | 依赖管理 | ⭐⭐⭐ | ⭐⭐ |
| **pipenv** | 依赖管理 | ⭐⭐ | ⭐⭐ |

**本项目推荐**: `venv` (简单够用) + `requirements.txt` (清晰明确)

## 系统特定说明

### macOS/Linux

```bash
# 确保有 Python3
python3 --version

# 创建虚拟环境
python3 -m venv venv

# 激活
source venv/bin/activate

# 激活后命令行会显示 (venv)
(venv) $ python --version
```

### Windows

```bash
# 创建虚拈环境
python -m venv venv

# 激活 (PowerShell)
venv\Scripts\Activate.ps1

# 激活 (Command Prompt)
venv\Scripts\activate.bat

# 激活后显示 (venv)
(venv) C:\path\to\project>
```

## 故障排除

### 问题1: "command not found: python3"

```bash
# 检查Python安装
which python3
python3 --version

# 如果未安装，需要安装 Python
# macOS: brew install python3
# Linux: sudo apt-get install python3
# Windows: https://www.python.org/downloads/
```

### 问题2: "Permission denied" 激活脚本

```bash
# 添加执行权限
chmod +x venv/bin/activate

# 然后激活
source venv/bin/activate
```

### 问题3: 激活后 pip install 还是装在系统Python

```bash
# 确保虚拨环境已激活
source venv/bin/activate

# 验证使用的是虚环境中的 pip
which pip  # 应该显示 .../venv/bin/pip

# 如果不是，手动激活
source venv/bin/activate

# 重新启动终端
```

## 参考资源

- [Python venv 官方文档](https://docs.python.org/3/library/venv.html)
- [Python 包管理最佳实践](https://python-docs-samples.readthedocs.io/en/latest/environment-setup.html)
- [requirements.txt 格式说明](https://pip.pypa.io/en/latest/reference/requirements-file-format/)

## 总结

| 步骤 | 命令 |
|------|------|
| 创建 | `python3 -m venv venv` |
| 激活 | `source venv/bin/activate` |
| 安装 | `pip install -r requirements.txt` |
| 更新 | `pip freeze > requirements.txt` |
| 退出 | `deactivate` |
| 删除 | `rm -rf venv` |

---

**现在就创建虚拟环境，享受隔离、安全、可重现的开发体验！** 🚀
