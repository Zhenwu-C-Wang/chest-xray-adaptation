"""
工具函数模块
包含各种辅助工具函数
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


def set_seed(seed=42):
    """
    设置随机种子以确保可重现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda=True):
    """
    获取计算设备
    
    Args:
        use_cuda: 是否优先使用CUDA
        
    Returns:
        device: PyTorch设备对象
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ImageDataset(torch.utils.data.Dataset):
    """
    图像数据集类
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 数据变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]


def get_data_transforms(img_size=224):
    """
    获取数据变换
    
    Args:
        img_size: 图像大小
        
    Returns:
        dict: 包含训练和验证变换的字典
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'val': val_transform}


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        DataLoader: PyTorch数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """
    加载模型检查点
    
    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器
        device: 设备
        
    Returns:
        tuple: (model, optimizer, epoch)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {filepath}")
    
    return model, optimizer, epoch
