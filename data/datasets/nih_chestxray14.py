"""
NIH ChestX-ray14 数据集集成模块
支持从公开数据源加载和预处理NIH ChestX-ray14数据集
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class NIHChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 数据集类
    
    数据集来源：https://nihcc.app.box.com/v/ChestX-ray14
    
    包含114,799张胸部X光图像
    14个诊断标签：
    - 无发现
    - 房间隔缺损
    - 肺纤维化
    - 浸润
    - 肺门淋巴结肿大
    - 积液
    - 肺不张
    - 结节/肿块
    - 肺炎
    - 气胸
    - 巩固
    - 间质浸润
    - 心肥大
    - 胸膜增厚
    """
    
    # NIH ChestX-ray14 的14个诊断标签
    LABELS = [
        'No Finding',
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening'
    ]
    
    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform=None,
        multi_label: bool = True,
        binary_classification: bool = False
    ):
        """
        Args:
            image_dir: 图像目录路径
            labels_csv: 标签CSV文件路径
            transform: 数据变换
            multi_label: 是否使用多标签分类
            binary_classification: 是否转换为二分类（有病vs无病）
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.multi_label = multi_label
        self.binary_classification = binary_classification
        
        # 加载标签
        self.df = pd.read_csv(labels_csv)
        
        # 数据验证
        self._validate_data()
    
    def _validate_data(self):
        """验证数据的完整性"""
        print(f"加载 {len(self.df)} 条记录")
        print(f"图像目录: {self.image_dir}")
        
        # 检查图像文件是否存在
        missing_count = 0
        for image_name in self.df['Image Index'].head(10):
            image_path = self.image_dir / image_name
            if not image_path.exists():
                missing_count += 1
        
        if missing_count > 0:
            print(f"警告: 部分图像文件未找到 (检查了10个)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        获取单个样本
        
        Returns:
            image: 图像张量
            labels: 标签数组
        """
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        image_path = self.image_dir / image_name
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        if self.binary_classification:
            # 二分类：有病 vs 无病
            # 如果 'Finding Labels' 列为 'No Finding'，则标签为0，否则为1
            finding = row['Finding Labels']
            labels = 0 if finding == 'No Finding' else 1
            labels = np.array([labels], dtype=np.float32)
        else:
            # 多标签分类
            labels = self._extract_labels(row)
        
        return image, labels
    
    def _extract_labels(self, row) -> np.ndarray:
        """
        从行数据中提取标签向量
        
        Args:
            row: DataFrame 行数据
            
        Returns:
            标签二进制向量（14维）
        """
        labels = np.zeros(len(self.LABELS), dtype=np.float32)
        
        if 'Finding Labels' in row and pd.notna(row['Finding Labels']):
            findings = str(row['Finding Labels']).split('|')
            for i, label in enumerate(self.LABELS):
                if label in findings:
                    labels[i] = 1.0
        
        return labels
    
    @staticmethod
    def prepare_download_script() -> str:
        """
        准备数据集下载脚本说明
        """
        script = """
# NIH ChestX-ray14 数据集下载说明

## 步骤 1: 注册并下载数据集
访问: https://nihcc.app.box.com/v/ChestX-ray14

## 步骤 2: 下载标签文件
链接: https://nihcc.app.box.com/v/ChestX-ray14/file/519676641154

保存为: Data_Entry_2017.csv

## 步骤 3: 下载图像
数据集包含多个压缩文件，需要下载和解压
推荐使用提供的下载脚本

## 步骤 4: 组织目录结构
chest-xray-adaptation/
├── data/
│   ├── nih_chestxray14/
│   │   ├── images/
│   │   │   ├── 00000001_000.png
│   │   │   ├── 00000002_000.png
│   │   │   └── ...
│   │   └── Data_Entry_2017.csv
"""
        return script
    
    def get_device_distribution(self) -> Dict[str, int]:
        """获取设备分布信息"""
        if 'View Position' in self.df.columns:
            return self.df['View Position'].value_counts().to_dict()
        return {}
    
    def get_finding_distribution(self) -> Dict[str, int]:
        """获取诊断发现分布"""
        finding_counts = {}
        for label in self.LABELS:
            count = 0
            for finding_str in self.df['Finding Labels']:
                if label in str(finding_str):
                    count += 1
            finding_counts[label] = count
        return finding_counts


class NIHChestXray14DataModule:
    """
    NIH ChestX-ray14 数据模块
    便捷的数据加载和预处理接口
    """
    
    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform_dict: Dict = None
    ):
        """
        Args:
            image_dir: 图像目录
            labels_csv: 标签文件
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            transform_dict: 数据变换字典（'train', 'val', 'test'）
        """
        self.image_dir = image_dir
        self.labels_csv = labels_csv
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform_dict = transform_dict or {}
        
        # 验证比例
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "比例之和必须等于1.0"
    
    def get_dataset(self, split: str = 'train') -> NIHChestXray14Dataset:
        """
        获取数据集
        
        Args:
            split: 'train', 'val', 或 'test'
        """
        transform = self.transform_dict.get(split, None)
        dataset = NIHChestXray14Dataset(
            image_dir=self.image_dir,
            labels_csv=self.labels_csv,
            transform=transform
        )
        
        # 划分数据集
        indices = self._split_data(len(dataset))
        split_indices = indices[split]
        
        # 创建子集
        from torch.utils.data import Subset
        return Subset(dataset, split_indices)
    
    def _split_data(self, total_size: int) -> Dict[str, List[int]]:
        """
        划分数据集
        """
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
