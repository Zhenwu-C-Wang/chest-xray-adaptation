#!/usr/bin/env python3
"""
下载/解压并重组数据为 ImageFolder 结构的辅助脚本。

特性：
- 可选下载（需要提供公开可访问的 URL，需自行确认许可/协议）
- 支持 zip / tar(.gz/.tgz) 解压
- 自动从类目录构建 train/val/test 划分
- 可通过 JSON 字符串指定类名映射（例如 {"PNEUMONIA": "pneumonia", "NORMAL": "normal"}）

用法示例：
1) 仅重组本地压缩包/目录：
   python scripts/prepare_open_data.py --source-path /path/to/dataset_root --output-dir data

2) 提供下载 URL（需确保无授权限制）：
   python scripts/prepare_open_data.py --download-url https://example.com/data.zip --output-dir data

注意：公开数据集可能要求注册/协议（如 NIH、CheXpert、MIMIC-CXR 等），请先在官网获取后再使用本脚本重组。
"""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(dest, "wb") as f:
            for data in r.iter_content(chunk_size=chunk):
                if data:
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r下载中: {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
    print()
    return dest


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(target_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path.suffix}")
    return target_dir


def find_class_dirs(root: Path) -> List[Path]:
    class_dirs = []
    for sub in root.rglob("*"):
        if sub.is_dir():
            if any((sub / f).suffix.lower() in IMAGE_EXTS for f in os.listdir(sub) if (sub / f).is_file()):
                class_dirs.append(sub)
    return class_dirs


def collect_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例之和必须为1"
    idx = list(range(n))
    import random

    random.shuffle(idx)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def copy_split(
    class_dir: Path,
    class_name: str,
    out_root: Path,
    ratios: Tuple[float, float, float],
):
    imgs = collect_images(class_dir)
    train_idx, val_idx, test_idx = split_indices(len(imgs), *ratios)
    mapping = [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    for split, ids in mapping:
        split_dir = out_root / split / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in ids:
            src = imgs[i]
            dst = split_dir / src.name
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="下载/解压/重组数据为 ImageFolder 结构")
    parser.add_argument("--download-url", type=str, default=None, help="数据压缩包的公开 URL（需自行确保许可）")
    parser.add_argument("--source-path", type=str, default=None, help="本地压缩包或目录路径")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录（将创建 train/val/test 子目录）")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--class-map", type=str, default=None, help='类名映射 JSON，如 {"PNEUMONIA": "pneumonia", "NORMAL": "normal"}')
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="cxr_prepare_"))
    source_path: Path

    try:
        if args.download_url:
            print(f"开始下载: {args.download_url}")
            fname = args.download_url.split("/")[-1]
            dest = work_dir / fname
            source_path = download_file(args.download_url, dest)
        elif args.source_path:
            source_path = Path(args.source_path).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"未找到 source-path: {source_path}")
        else:
            raise ValueError("需要提供 --download-url 或 --source-path 之一")

        # 如果是压缩包先解压
        extracted_root = work_dir / "extracted"
        if source_path.is_file():
            print(f"解压: {source_path}")
            extract_archive(source_path, extracted_root)
            search_root = extracted_root
        else:
            search_root = source_path

        # 发现类目录
        class_dirs = find_class_dirs(search_root)
        if not class_dirs:
            raise RuntimeError("未找到包含图像的类目录，请检查数据结构。")

        # 解析类名映射
        class_map: Dict[str, str] = {}
        if args.class_map:
            class_map = json.loads(args.class_map)

        ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

        print("开始重组为 ImageFolder ...")
        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            mapped = class_map.get(cls_name, cls_name)
            copy_split(cls_dir, mapped, out_root, ratios)
            print(f"  类别 {cls_name} -> {mapped}, 样本数: {len(collect_images(cls_dir))}")

        print(f"完成！数据已写入: {out_root} (train/val/test)")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
