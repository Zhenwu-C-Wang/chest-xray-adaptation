"""
MIMIC-CXR 数据集集成模块
支持从公开数据源加载和预处理MIMIC-CXR数据集
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR 数据集类
    
    数据集来源：https://physionet.org/content/mimic-cxr/2.0.0/
    
    包含377,110张胸部X光图像
    涵盖多家医院的数据（多站点）
    包括对应的医学记录和图像元数据
    """
    
    # MIMIC-CXR 的诊断标签
    LABELS = [
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Hernia',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pleural Thickening',
        'Pneumonia',
        'Pneumothorax',
        'No Finding'
    ]
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        split_type: str = 'default'  # 'default', 'by_hospital', 'by_equipment'
    ):
        """
        Args:
            csv_path: CSV标签文件路径
            image_root: 图像根目录
            transform: 数据变换
            split_type: 数据划分方式
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.split_type = split_type
        
        # 加载标签
        self.df = pd.read_csv(csv_path)
        self._preprocess()
    
    def _preprocess(self):
        """数据预处理"""
        # 标准化图像路径
        if 'path' in self.df.columns:
            self.df['image_path'] = self.df['path'].str.replace('mimic-cxr-jpg', 'files')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """获取单个样本"""
        row = self.df.iloc[idx]
        
        # 构建完整的图像路径
        image_path = self.image_root / row['image_path']
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        labels = self._extract_labels(row)
        
        return image, labels
    
    def _extract_labels(self, row) -> np.ndarray:
        """提取诊断标签"""
        labels = np.zeros(len(self.LABELS), dtype=np.float32)
        
        # 假设标签存储在单独的列中或以竖线分隔的字符串形式
        for i, label in enumerate(self.LABELS):
            if label in self.df.columns and row[label] == 1:
                labels[i] = 1.0
        
        return labels
    
    def get_hospital_distribution(self) -> Dict[str, int]:
        """获取医院分布"""
        if 'hospital' in self.df.columns or 'subject_id' in self.df.columns:
            return self.df['subject_id'].value_counts().head(10).to_dict()
        return {}
    
    def get_equipment_distribution(self) -> Dict[str, int]:
        """获取设备分布"""
        if 'equipment' in self.df.columns:
            return self.df['equipment'].value_counts().to_dict()
        return {}
    
    def get_imaging_type_distribution(self) -> Dict[str, int]:
        """获取成像类型分布（正位vs侧位）"""
        if 'ViewPosition' in self.df.columns:
            return self.df['ViewPosition'].value_counts().to_dict()
        return {}
    
    def get_cross_site_splits(self) -> Dict[str, List[int]]:
        """
        获取跨医院/站点的划分
        用于跨站外部验证
        """
        splits = {}
        
        if 'subject_id' in self.df.columns:
            unique_hospitals = self.df['subject_id'].unique()
            
            for hospital in unique_hospitals:
                indices = self.df[self.df['subject_id'] == hospital].index.tolist()
                splits[f'hospital_{hospital}'] = indices
        
        return splits
    
    @staticmethod
    def prepare_download_script() -> str:
        """准备数据集下载脚本说明"""
        script = """
# MIMIC-CXR 数据集下载说明

## 步骤 1: 获取访问权限
需要：
1. 在 PhysioNet 创建账户 (https://physionet.org/)
2. 完成 CITI Data or Specimens 培训
3. 在项目中签署数据使用协议

## 步骤 2: 下载数据集
访问: https://physionet.org/content/mimic-cxr/2.0.0/

需要下载：
- mimic-cxr-2.0.0-chexpert.csv.gz
- mimic-cxr-2.0.0-metadata.csv.gz
- 对应的JPG图像文件

## 步骤 3: 组织目录结构
chest-xray-adaptation/
├── data/
│   ├── mimic_cxr/
│   │   ├── mimic-cxr-2.0.0-chexpert.csv
│   │   ├── mimic-cxr-2.0.0-metadata.csv
│   │   ├── files/
│   │   │   ├── p10/
│   │   │   ├── p11/
│   │   │   └── ...

## 步骤 4: 重要特性
- 多医院数据（理想的跨站验证源）
- 包含设备元数据
- 支持成像类型划分（正位/侧位）
- 包含患者级别的标签
"""
        return script


class MIMICCXRDataModule:
    """
    MIMIC-CXR 数据模块
    """
    
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        metadata_csv: str = None,
        transform_dict: Dict = None
    ):
        """
        Args:
            csv_path: CheXpert标签CSV路径
            image_root: 图像根目录
            metadata_csv: 元数据CSV路径
            transform_dict: 数据变换字典
        """
        self.csv_path = csv_path
        self.image_root = image_root
        self.metadata_csv = metadata_csv
        self.transform_dict = transform_dict or {}
        
        # 加载元数据以便进行跨站分析
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)
        else:
            self.metadata = None
    
    def get_dataset(self, split: str = 'train') -> MIMICCXRDataset:
        """获取数据集"""
        transform = self.transform_dict.get(split, None)
        
        dataset = MIMICCXRDataset(
            csv_path=self.csv_path,
            image_root=self.image_root,
            transform=transform
        )
        
        return dataset
    
    def get_cross_site_validation_sets(self) -> Dict[str, List[int]]:
        """获取用于跨站验证的数据集划分"""
        return self._split_by_hospitals()
    
    def _split_by_hospitals(self) -> Dict[str, List[int]]:
        """按医院划分数据集"""
        if self.metadata is None or 'subject_id' not in self.metadata.columns:
            return {}
        
        splits = {}
        main_hospitals = self.metadata['subject_id'].value_counts().head(5).index
        
        for hospital in main_hospitals:
            indices = self.metadata[
                self.metadata['subject_id'] == hospital
            ].index.tolist()
            splits[f'hospital_{hospital}'] = indices
        
        return splits
