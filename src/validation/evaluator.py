"""
验证和评估模块
包含模型评估的各种指标计算
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    """
    模型评估器类
    计算各种评估指标
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 神经网络模型
            device: 计算设备
        """
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader):
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        self.model.eval()
        
        predictions = []
        probabilities = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # 获取预测类别和概率
                probs = torch.nn.functional.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                targets.extend(target.numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        targets = np.array(targets)
        
        # 计算各种指标
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted'),
        }
        
        # 如果是二分类，计算AUROC
        if len(np.unique(targets)) == 2:
            try:
                metrics['auroc'] = roc_auc_score(targets, probabilities[:, 1])
            except:
                metrics['auroc'] = None
        
        return metrics
    
    def calculate_calibration_error(self, test_loader):
        """
        计算期望校准误差 (ECE)
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            float: ECE值
        """
        self.model.eval()
        predictions = []
        probabilities = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                probs = torch.nn.functional.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.max(dim=1)[0].cpu().numpy())
                targets.extend(target.numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        targets = np.array(targets)
        
        # 计算ECE
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probabilities >= lower) & (probabilities < upper)
            if mask.sum() > 0:
                acc = (predictions[mask] == targets[mask]).mean()
                conf = probabilities[mask].mean()
                ece += mask.sum() / len(probabilities) * abs(acc - conf)
        
        return ece
