#!/usr/bin/env python
"""
训练脚本
使用 ImageFolder 数据结构训练分类模型，并保存最佳检查点。
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.models.resnet import ResNet18, DenseNet121
from src.training.trainer import Trainer
from src.utils.utils import set_seed, get_device, get_data_transforms, save_checkpoint


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    """根据名称构建模型"""
    arch = arch.lower()
    if arch in ("resnet18", "resnet"):
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    if arch in ("densenet121", "densenet"):
        return DenseNet121(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"不支持的模型架构: {arch}")


def load_datasets(data_dir: Path, img_size: int, val_split: float):
    """基于 ImageFolder 加载训练/验证集"""
    transforms = get_data_transforms(img_size=img_size)

    train_root = data_dir / "train"
    val_root = data_dir / "val"

    if val_root.exists():
        train_dataset = datasets.ImageFolder(train_root, transform=transforms["train"])
        val_dataset = datasets.ImageFolder(val_root, transform=transforms["val"])
    else:
        full_dataset = datasets.ImageFolder(train_root, transform=transforms["train"])
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        # 验证集需要使用验证变换
        val_dataset.dataset.transform = transforms["val"]

    return train_dataset, val_dataset


def main(args):
    """主训练函数"""

    config = load_config(args.config)
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 32)
    num_epochs = args.num_epochs or training_cfg.get("num_epochs", 10)
    num_workers = training_cfg.get("num_workers", 4)
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    img_size = config.get("dataset", {}).get("img_size", args.img_size)
    val_split = args.val_split

    set_seed(training_cfg.get("seed", 42))
    device = get_device(use_cuda=training_cfg.get("device", "cuda") == "cuda")

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists():
        raise FileNotFoundError(f"未找到训练数据目录: {data_dir/'train'} (使用 ImageFolder 结构)")

    train_dataset, val_dataset = load_datasets(data_dir, img_size=img_size, val_split=val_split)
    num_classes = len(train_dataset.dataset.classes) if hasattr(train_dataset, "dataset") else len(train_dataset.classes)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(model, optimizer, criterion, device=device)

    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        print(
            f"[{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir / "best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练胸部X光模型（基于ImageFolder）")
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="训练配置文件路径"
    )
    parser.add_argument("--data-dir", type=str, default="data/", help="数据目录（需包含 train/ 与可选 val/）")
    parser.add_argument(
        "--arch", type=str, default="resnet18", choices=["resnet18", "densenet121"], help="模型架构"
    )
    parser.add_argument("--img-size", type=int, default=224, help="输入图像大小")
    parser.add_argument("--val-split", type=float, default=0.1, help="当没有 val/ 目录时，从 train/ 划分验证比例")
    parser.add_argument("--pretrained", action="store_true", help="使用torchvision预训练权重")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--num-epochs", type=int, default=None, help="训练轮数，优先于配置文件")

    args = parser.parse_args()
    main(args)
