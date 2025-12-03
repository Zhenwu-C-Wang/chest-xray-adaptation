# 🚀 快速设置指南

## ⚡ 一键设置（推荐）

如果这是您第一次使用此项目，运行这个命令自动创建和配置虚拟环境：

```bash
bash setup_venv.sh
```

这会自动：
1. 创建 Python 虚拈环境
2. 激活虚拈环境
3. 升级 pip
4. 安装所有依赖
5. 验证安装

完成后，您可以立即开始使用项目！

## 📋 后续步骤

### 第一次使用
```bash
# 虚拨环境已由 setup_venv.sh 激活
python setup_environment.py          # 检查环境
python data/DATASET_GUIDE.py        # 查看数据集信息
```

### 后续开发（每次打开新终端）
```bash
# 激活虚拨环境
source venv/bin/activate

# 现在可以开始工作
python scripts/cross_site_validation_example.py
```

### 完成后
```bash
# 退出虚拨环境
deactivate
```

## 📚 详细文档

- **[VENV_GUIDE.md](VENV_GUIDE.md)** - 虚拨环境完整指南
- **[QUICK_START.md](QUICK_START.md)** - 项目快速开始
- **[IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)** - 完整技术文档

## ⚠️ 常见问题

**Q: 为什么要用虚拨环境？**
A: 虚拨环境隔离项目依赖，防止版本冲突，确保可重现的开发环境。

**Q: 虚拨环境占空间很大吗？**
A: 通常 500MB-1GB，但不要上传到 Git。`.gitignore` 已配置忽略 `venv/`。

**Q: setup_venv.sh 不工作？**
A: 确保您在项目根目录，且有执行权限。查看 [VENV_GUIDE.md](VENV_GUIDE.md) 了解详情。

## 🎯 现在就开始！

```bash
# 1. 执行一键设置
bash setup_venv.sh

# 2. 按照屏幕提示操作（通常自动完成）

# 3. 开始开发
python setup_environment.py
```

---

有问题？查看 [VENV_GUIDE.md](VENV_GUIDE.md) 或 [QUICK_START.md](QUICK_START.md)
