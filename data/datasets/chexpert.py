"""
CheXpert 数据集集成模块
支持从公开数据源加载和预处理CheXpert数据集
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class CheXpertDataset(Dataset):
    """
    CheXpert 数据集类
    
    数据集来源：https://stanfordmlgroup.github.io/competitions/chexpert/
    
    包含223,648张胸部X光图像
    14个诊断标签（与NIH ChestX-ray14类似）
    包含不确定性标注（-1表示不确定）
    """
    
    # CheXpert 的主要诊断标签
    LABELS = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Airspace Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        uncertain_as_positive: bool = True,
        binary_classification: bool = False
    ):
        """
        Args:
            csv_path: CSV标签文件路径
            image_root: 图像根目录
            transform: 数据变换
            uncertain_as_positive: 不确定标注是否视为正样本（0.5）
            binary_classification: 是否转换为二分类
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.uncertain_as_positive = uncertain_as_positive
        self.binary_classification = binary_classification
        
        # 加载标签
        self.df = pd.read_csv(csv_path)
        self._preprocess_labels()
    
    def _preprocess_labels(self):
        """预处理标签，处理不确定值"""
        # 获取诊断列
        diagnosis_cols = [col for col in self.df.columns if col != 'Path']
        
        for col in diagnosis_cols:
            # 将不确定值 (-1) 处理为 0.5 或 0
            if self.uncertain_as_positive:
                self.df[col] = self.df[col].map({1: 1, -1: 0.5, 0: 0, np.nan: 0})
            else:
                self.df[col] = self.df[col].map({1: 1, -1: 0, 0: 0, np.nan: 0})
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        获取单个样本
        """
        row = self.df.iloc[idx]
        image_path = self.image_root / row['Path']
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        diagnosis_cols = [col for col in self.df.columns if col != 'Path']
        labels = row[diagnosis_cols].values.astype(np.float32)
        
        if self.binary_classification:
            # 转换为二分类：任何异常 vs 正常
            labels = 1 if np.sum(labels) > 0 else 0
            labels = np.array([labels], dtype=np.float32)
        
        return image, labels
    
    def get_disease_distribution(self) -> Dict[str, int]:
        """获取疾病分布"""
        diagnosis_cols = [col for col in self.df.columns if col != 'Path']
        distribution = {}
        
        for col in diagnosis_cols:
            # 统计正样本（不包括不确定的0.5）
            count = (self.df[col] == 1).sum()
            distribution[col] = int(count)
        
        return distribution
    
    def get_frontal_lateral_split(self) -> Tuple[List[int], List[int]]:
        """
        获取正位和侧位图像的索引划分
        
        假设 Path 列包含 'frontal' 或 'lateral' 的目录信息
        """
        frontal_indices = []
        lateral_indices = []
        
        for idx, path in enumerate(self.df['Path']):
            if 'frontal' in str(path).lower():
                frontal_indices.append(idx)
            elif 'lateral' in str(path).lower():
                lateral_indices.append(idx)
        
        return frontal_indices, lateral_indices
    
    @staticmethod
    def prepare_download_script() -> str:
        """准备数据集下载脚本说明"""
        script = """
# CheXpert 数据集下载说明

## 步骤 1: 注册并获取下载链接
访问: https://stanfordmlgroup.github.io/competitions/chexpert/

需要填写表单并同意使用条款

## 步骤 2: 下载数据集
CheXpert 数据集包含：
- train.csv
- valid.csv
- test.csv
- 对应的图像文件

## 步骤 3: 组织目录结构
chest-xray-adaptation/
├── data/
│   ├── chexpert/
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   ├── test.csv
│   │   ├── CheXpert-v1.0-small/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── ...

## 步骤 4: 处理不确定标注
CheXpert 中使用 -1 表示不确定诊断
可配置为：
- 视为正样本 (0.5)
- 视为负样本 (0)
- 排除
"""
        return script


class CheXpertDataModule:
    """
    CheXpert 数据模块
    """
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform_dict: Dict = None,
        split_type: str = 'standard'  # 'standard' 或 'frontal_lateral'
    ):
        """
        Args:
            csv_path: CSV标签文件路径
            image_root: 图像根目录
            transform_dict: 数据变换字典
            split_type: 数据划分方式
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.transform_dict = transform_dict or {}
        self.split_type = split_type
    
    def get_dataset(self, split: str = 'train') -> CheXpertDataset:
        """获取数据集"""
        transform = self.transform_dict.get(split, None)
        
        # 根据split选择对应的CSV文件
        if split == 'train':
            csv_file = os.path.join(os.path.dirname(self.csv_path), 'train.csv')
        elif split == 'val':
            csv_file = os.path.join(os.path.dirname(self.csv_path), 'valid.csv')
        else:
            csv_file = os.path.join(os.path.dirname(self.csv_path), 'test.csv')
        
        dataset = CheXpertDataset(
            csv_path=csv_file,
            image_root=self.image_root,
            transform=transform
        )
        
        return dataset
