"""
模型校准模块
实现ECE、MCE等校准指标及多种校准方法
"""

import numpy as np
from typing import Tuple, List
import torch
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt


class CalibrationMetrics:
    """
    校准评估指标
    """

    @staticmethod
    def _prepare_inputs(probabilities: np.ndarray, targets: np.ndarray, predictions=None):
        probs = np.array(probabilities)
        targets_arr = np.array(targets)

        if probs.ndim == 1:
            confidences = probs
            preds = predictions if predictions is not None else (probs >= 0.5).astype(int)
        else:
            confidences = probs.max(axis=1)
            preds = predictions if predictions is not None else probs.argmax(axis=1)
        return probs, targets_arr, preds, confidences

    @staticmethod
    def expected_calibration_error(
        probabilities: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray = None,
        n_bins: int = 10,
    ) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        """
        probs, targets_arr, preds, confidences = CalibrationMetrics._prepare_inputs(
            probabilities, targets, predictions
        )

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences >= lower) & (confidences < upper)

            if mask.sum() > 0:
                accuracy = (preds[mask] == targets_arr[mask]).mean()
                confidence = confidences[mask].mean()
                ece += mask.sum() / len(confidences) * abs(accuracy - confidence)

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        probabilities: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray = None,
        n_bins: int = 10,
    ) -> float:
        """
        计算最大校准误差 (Maximum Calibration Error)
        """
        probs, targets_arr, preds, confidences = CalibrationMetrics._prepare_inputs(
            probabilities, targets, predictions
        )

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0.0

        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences >= lower) & (confidences < upper)

            if mask.sum() > 0:
                accuracy = (preds[mask] == targets_arr[mask]).mean()
                confidence = confidences[mask].mean()
                error = abs(accuracy - confidence)
                max_error = max(max_error, error)

        return float(max_error)

    @staticmethod
    def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        计算Brier Score
        """
        probs = np.array(probabilities)
        targets_arr = np.array(targets)

        if probs.ndim == 1:
            return float(np.mean((probs - targets_arr) ** 2))

        n_classes = probs.shape[1]
        one_hot_targets = np.eye(n_classes)[targets_arr]
        return float(np.mean((probs - one_hot_targets) ** 2))


class CalibrationMethod:
    """
    模型校准基类
    """
    
    def fit(self, probabilities: np.ndarray, targets: np.ndarray):
        """拟合校准参数"""
        raise NotImplementedError
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率"""
        raise NotImplementedError


class TemperatureScaling(CalibrationMethod):
    """
    温度缩放 (Temperature Scaling)
    
    校准方法：p_calibrated = softmax(logits / T)
    其中T是温度参数
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, logits: np.ndarray, targets: np.ndarray, lr: float = 0.01, max_iter: int = 200):
        """
        拟合温度参数，使用简单的梯度下降最小化交叉熵。
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        log_temp = torch.nn.Parameter(torch.zeros(1))  # temp = softplus(log_temp)
        optimizer = torch.optim.SGD([log_temp], lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(max_iter):
            optimizer.zero_grad()
            temperature = torch.nn.functional.softplus(log_temp) + 1e-6
            scaled_logits = logits_tensor / temperature
            loss = loss_fn(scaled_logits, targets_tensor)
            loss.backward()
            optimizer.step()

        self.temperature = float(torch.nn.functional.softplus(log_temp).item())
        self.is_fitted = True

    def _calibrate_probs(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """应用温度缩放"""
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def calibrate(self, logits_or_probs: np.ndarray) -> np.ndarray:
        """
        校准概率。输入可以是logits或已经softmax后的概率。
        """
        if not self.is_fitted:
            return logits_or_probs

        arr = np.array(logits_or_probs)
        if arr.ndim == 2 and np.allclose(arr.sum(axis=1), 1, atol=1e-3):
            logits = np.log(arr + 1e-10)
        else:
            logits = arr

        return self._calibrate_probs(logits, self.temperature)


class PlattScaling(CalibrationMethod):
    """
    Platt缩放
    
    校准方法：p_calibrated = 1 / (1 + exp(A * logit + B))
    其中A和B是学习的参数
    """
    
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
        self.is_fitted = False
    
    def fit(self, probabilities: np.ndarray, targets: np.ndarray):
        """
        拟合Platt缩放参数
        
        Args:
            probabilities: 预测概率
            targets: 真实标签
        """
        # 将概率转换为logits
        logits = np.log(probabilities / (1 - probabilities + 1e-10) + 1e-10)
        
        # 使用逻辑回归拟合
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(logits.reshape(-1, 1), targets)
        
        self.A = lr.coef_[0][0]
        self.B = lr.intercept_[0]
        self.is_fitted = True
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        校准概率
        """
        if not self.is_fitted:
            return probabilities
        
        logits = np.log(probabilities / (1 - probabilities + 1e-10) + 1e-10)
        scaled_logits = self.A * logits + self.B
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))
        
        return np.clip(calibrated, 0, 1)


class IsotonicCalibration(CalibrationMethod):
    """
    等距回归校准 (Isotonic Regression Calibration)
    """
    
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, probabilities: np.ndarray, targets: np.ndarray):
        """
        拟合等距回归
        
        Args:
            probabilities: 预测概率
            targets: 真实标签
        """
        self.ir.fit(probabilities, targets)
        self.is_fitted = True
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """校准概率"""
        if not self.is_fitted:
            return probabilities
        
        return self.ir.predict(probabilities)


class CalibrationVisualizer:
    """
    校准可视化工具
    """
    
    @staticmethod
    def plot_calibration_curve(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
        output_path: str = 'calibration_curve.png'
    ):
        """
        绘制校准曲线
        
        Args:
            predictions: 预测标签
            targets: 真实标签
            probabilities: 预测概率
            n_bins: 分箱数量
            output_path: 输出路径
        """
        _, targets_arr, preds, confidences = CalibrationMetrics._prepare_inputs(
            probabilities, targets, predictions
        )
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        accuracies = []
        confs = []
        counts = []

        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences >= lower) & (confidences < upper)

            if mask.sum() > 0:
                accuracy = (preds[mask] == targets_arr[mask]).mean()
                confidence = confidences[mask].mean()

                accuracies.append(accuracy)
                confs.append(confidence)
                counts.append(mask.sum())
            else:
                accuracies.append(0)
                confs.append((lower + upper) / 2)
                counts.append(0)
        
        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 完美校准线
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # 实际校准曲线
        ax.plot(confs, accuracies, 'o-', label='Model Calibration', 
               linewidth=2, markersize=8)
        
        # 使用样本数作为大小
        sizes = [c / max(counts) * 100 for c in counts]
        ax.scatter(confs, accuracies, s=sizes, alpha=0.5)
        
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"校准曲线已保存到: {output_path}")
        plt.close()
    
    @staticmethod
    def plot_reliability_diagram(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
        output_path: str = 'reliability_diagram.png'
    ):
        """
        绘制可靠性图表
        """
        _, targets_arr, preds, confidences = CalibrationMetrics._prepare_inputs(
            probabilities, targets, predictions
        )
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        accuracies = []
        confs = []

        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidences >= lower) & (confidences < upper)

            if mask.sum() > 0:
                accuracy = (preds[mask] == targets_arr[mask]).mean()
                confidence = confidences[mask].mean()

                accuracies.append(accuracy)
                confs.append(confidence)

        # 绘制
        fig, ax = plt.subplots(figsize=(10, 8))

        # 完美校准
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2)

        # 直方图
        ax.bar(confs, accuracies, width=0.08, alpha=0.7, edgecolor='black', label='Model')

        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Reliability Diagram', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"可靠性图表已保存到: {output_path}")
        plt.close()
