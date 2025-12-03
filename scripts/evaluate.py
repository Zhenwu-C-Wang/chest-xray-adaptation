#!/usr/bin/env python
"""
评估脚本
基于 ImageFolder 数据结构评估模型性能。
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets

from src.models.resnet import ResNet18, DenseNet121
from src.utils.utils import get_device, get_data_transforms, load_checkpoint
from src.validation.evaluator import Evaluator


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch in ("resnet18", "resnet"):
        return ResNet18(num_classes=num_classes, pretrained=False)
    if arch in ("densenet121", "densenet"):
        return DenseNet121(num_classes=num_classes, pretrained=False)
    raise ValueError(f"不支持的模型架构: {arch}")


def load_dataset(data_dir: Path, img_size: int):
    transforms = get_data_transforms(img_size=img_size)
    # 优先使用 test/，否则使用 val/，再否则使用 train/
    for split in ["test", "val", "train"]:
        candidate = data_dir / split
        if candidate.exists():
            dataset = datasets.ImageFolder(candidate, transform=transforms["val"])
            return dataset, split
    raise FileNotFoundError(f"未找到可用的数据目录 (期待 {data_dir}/test 或 val 或 train)")


def main(args):
    """主评估函数"""

    config = load_config(args.config)
    training_cfg = config.get("training", {})
    device = get_device(use_cuda=training_cfg.get("device", "cuda") == "cuda")

    print("开始评估...")
    print(f"模型路径: {args.model}")
    print(f"数据目录: {args.data_dir}")

    data_dir = Path(args.data_dir)
    dataset, split_used = load_dataset(data_dir, img_size=args.img_size)
    num_classes = len(dataset.classes)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    model = build_model(args.arch, num_classes=num_classes).to(device)
    load_checkpoint(args.model, model, optimizer=None, device=device)

    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(dataloader)

    print(f"使用 split: {split_used}")
    for k, v in metrics.items():
        if v is not None:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: None")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"指标已保存到: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估胸部X光模型")
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="训练配置文件路径"
    )
    parser.add_argument("--model", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data-dir", type=str, default="data/", help="数据目录")
    parser.add_argument(
        "--arch", type=str, default="resnet18", choices=["resnet18", "densenet121"], help="模型架构"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="评估批次大小")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像大小")
    parser.add_argument("--output-dir", type=str, default=None, help="保存指标的目录（如 experiments/exp_internal_baseline）")

    args = parser.parse_args()
    main(args)
