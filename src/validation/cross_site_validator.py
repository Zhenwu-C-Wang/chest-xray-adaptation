"""
跨站外部验证框架
用于评估模型在不同医院/设备/数据集上的泛化能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class CrossSiteValidator:
    """
    跨站外部验证器
    用于在多个站点/设备上验证模型泛化能力
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.results = {}
        self.site_metrics = {}
    
    def validate_on_site(
        self,
        data_loader,
        site_name: str,
        dataset_name: str = 'Unknown'
    ) -> Dict[str, float]:
        """
        在单个站点上进行验证
        
        Args:
            data_loader: 数据加载器
            site_name: 站点名称
            dataset_name: 数据集名称
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        predictions = []
        probabilities = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                # 获取预测
                probs = torch.nn.functional.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                targets.extend(target.numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        targets = np.array(targets)
        
        # 计算评估指标
        metrics = {
            'site': site_name,
            'dataset': dataset_name,
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
        }
        
        # 如果是二分类，计算AUROC
        if len(np.unique(targets)) == 2:
            try:
                metrics['auroc'] = roc_auc_score(targets, probabilities[:, 1])
            except:
                metrics['auroc'] = None
        
        # 存储结果用于后续分析
        self.site_metrics[site_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': targets
        }
        
        return metrics
    
    def validate_multi_sites(
        self,
        site_dataloaders: Dict[str, Any],
        dataset_names: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        在多个站点上进行验证
        
        Args:
            site_dataloaders: 站点名称到数据加载器的映射
            dataset_names: 站点名称到数据集名称的映射
            
        Returns:
            结果DataFrame
        """
        dataset_names = dataset_names or {}
        results = []
        
        for site_name, data_loader in site_dataloaders.items():
            dataset_name = dataset_names.get(site_name, 'Unknown')
            metrics = self.validate_on_site(data_loader, site_name, dataset_name)
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def analyze_device_performance(
        self,
        site_dataloaders: Dict[str, Any],
        device_info: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        分析不同设备间的性能差异
        
        Args:
            site_dataloaders: 站点数据加载器
            device_info: 设备信息映射 {site: {device_type, manufacturer, model}}
            
        Returns:
            设备性能分析结果
        """
        results = []
        
        for site_name, data_loader in site_dataloaders.items():
            metrics = self.validate_on_site(data_loader, site_name)
            
            # 添加设备信息
            if site_name in device_info:
                metrics.update(device_info[site_name])
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # 分析设备间差异
        analysis = {
            'results': results_df,
            'mean_metrics': results_df[['accuracy', 'auroc', 'f1']].mean(),
            'std_metrics': results_df[['accuracy', 'auroc', 'f1']].std(),
            'device_wise_performance': results_df.groupby('device_type')[
                ['accuracy', 'auroc', 'f1']
            ].agg(['mean', 'std'])
        }
        
        return analysis
    
    def get_cross_site_stability(self) -> Dict[str, float]:
        """
        计算跨站稳定性指标
        
        Returns:
            稳定性指标（方差、置信区间等）
        """
        if self.results is None or len(self.results) == 0:
            return {}
        
        stability = {
            'auroc_mean': self.results['auroc'].mean(),
            'auroc_std': self.results['auroc'].std(),
            'auroc_cv': self.results['auroc'].std() / self.results['auroc'].mean(),
            'accuracy_mean': self.results['accuracy'].mean(),
            'accuracy_std': self.results['accuracy'].std(),
            'f1_mean': self.results['f1'].mean(),
            'f1_std': self.results['f1'].std(),
            'num_sites': len(self.results),
        }
        
        return stability
    
    def generate_report(self, output_path: str = 'cross_site_validation_report.csv'):
        """生成跨站验证报告"""
        if self.results is not None:
            self.results.to_csv(output_path, index=False)
            print(f"报告已保存到: {output_path}")
        
        return self.results


class DeviceDomainAnalyzer:
    """
    设备域分析器
    分析不同设备之间的域差异
    """
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.feature_distributions = {}
    
    def extract_features(
        self,
        data_loader,
        device_name: str,
        layer_name: str = 'features'
    ) -> np.ndarray:
        """
        提取特定层的特征
        
        Args:
            data_loader: 数据加载器
            device_name: 设备名称
            layer_name: 特征层名称
            
        Returns:
            特征矩阵
        """
        self.model.eval()
        features = []
        
        # 注册hook以提取中间层特征
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # 这里需要根据实际模型结构调整
        # 示例：假设模型有 features 属性
        handle = None
        if hasattr(self.model, layer_name):
            layer = getattr(self.model, layer_name)
            handle = layer.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                _ = self.model(data)
                
                if activations:
                    # 展平并添加到特征列表
                    feat = activations[-1]
                    features.append(feat.reshape(feat.shape[0], -1))
                    activations.clear()
        
        if handle:
            handle.remove()
        
        if features:
            features = np.vstack(features)
            self.feature_distributions[device_name] = features
        
        return features
    
    def analyze_domain_shift(
        self,
        device_dataloaders: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析域转移
        计算设备间的特征分布差异
        
        Args:
            device_dataloaders: 设备名称到数据加载器的映射
            
        Returns:
            域转移指标
        """
        # 提取所有设备的特征
        all_features = {}
        for device_name, data_loader in device_dataloaders.items():
            features = self.extract_features(data_loader, device_name)
            all_features[device_name] = features
        
        # 计算特征间的距离（MMD或其他度量）
        shift_metrics = {}
        device_names = list(all_features.keys())
        
        for i in range(len(device_names)):
            for j in range(i+1, len(device_names)):
                device1, device2 = device_names[i], device_names[j]
                feat1 = all_features[device1]
                feat2 = all_features[device2]
                
                # 计算最大均值差异（MMD）
                mmd = self._compute_mmd(feat1, feat2)
                shift_metrics[f'{device1}_vs_{device2}'] = mmd
        
        return {
            'mmd_distances': shift_metrics,
            'feature_counts': {name: feats.shape[0] for name, feats in all_features.items()}
        }
    
    def _compute_mmd(self, X, Y, kernel='rbf', sigma=1.0) -> float:
        """
        计算最大均值差异 (MMD)
        
        Args:
            X, Y: 特征矩阵
            kernel: 核函数类型
            sigma: 高斯核参数
            
        Returns:
            MMD值
        """
        # 简化实现：使用欧氏距离
        if X.shape[0] == 0 or Y.shape[0] == 0:
            return 0.0
        
        # 计算均值
        mean_X = np.mean(X, axis=0)
        mean_Y = np.mean(Y, axis=0)
        
        # 计算欧氏距离
        mmd = np.linalg.norm(mean_X - mean_Y)
        
        return float(mmd)
    
    def visualize_distribution_shift(
        self,
        output_path: str = 'domain_shift_visualization.png'
    ):
        """
        可视化域转移
        """
        if len(self.feature_distributions) < 2:
            print("至少需要2个设备的特征分布")
            return
        
        from sklearn.decomposition import PCA
        
        # 合并所有特征并进行PCA降维
        all_features = np.vstack(list(self.feature_distributions.values()))
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        
        offset = 0
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.feature_distributions)))
        
        for (device_name, features), color in zip(
            self.feature_distributions.items(), colors
        ):
            size = features.shape[0]
            pca_features = features_2d[offset:offset+size]
            ax.scatter(pca_features[:, 0], pca_features[:, 1], 
                      label=device_name, alpha=0.6, color=color)
            offset += size
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('设备域差异 (PCA可视化)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"可视化已保存到: {output_path}")
        plt.close()
