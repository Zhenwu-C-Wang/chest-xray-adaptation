"""
训练模块
包含模型训练的核心逻辑
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    模型训练器类
    负责模型训练过程的管理
    """
    
    def __init__(self, model, optimizer, loss_fn, device='cuda'):
        """
        Args:
            model: 神经网络模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失值
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            # 计算训练精度
            if output.dim() > 1:
                preds = output.argmax(dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += target.size(0)
            
        avg_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": train_acc}
    
    def train(self, train_loader, num_epochs, val_loader=None):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练周期数
            val_loader: 验证数据加载器（可选）
        """
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失值
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                if output.dim() > 1:
                    preds = output.argmax(dim=1)
                    total_correct += (preds == target).sum().item()
                    total_samples += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        val_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": val_acc}
