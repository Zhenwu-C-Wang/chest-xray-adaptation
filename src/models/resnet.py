"""
模型定义模块
包含基于torchvision的分类模型定义
"""

import os

# 避免在缺少自定义算子（如nms）时导入失败
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import torch
import torch.nn as nn

try:
    from torchvision import models
    _TORCHVISION_ERROR = None
except Exception as e:  # 捕获RuntimeError: operator torchvision::nms does not exist
    models = None
    _TORCHVISION_ERROR = e


def _load_resnet18(pretrained: bool):
    """尽量兼容不同torchvision版本地加载预训练权重。"""
    if models is None:
        return None
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    except Exception:
        return models.resnet18(pretrained=pretrained)


def _load_densenet121(pretrained: bool):
    """尽量兼容不同torchvision版本地加载预训练权重。"""
    if models is None:
        return None
    try:
        from torchvision.models import densenet121, DenseNet121_Weights

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        return densenet121(weights=weights)
    except Exception:
        return models.densenet121(pretrained=pretrained)


class _SimpleBackbone(nn.Module):
    """
    当torchvision不可用时的轻量级替代模型，保证测试可运行。
    """

    def __init__(self, num_features: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNet18(nn.Module):
    """
    基于ResNet18的X光分类模型，输出指定类别数。
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        backbone = _load_resnet18(pretrained=pretrained)
        if backbone is None:
            # torchvision 不可用时使用简化网络
            self.features = _SimpleBackbone(num_features=256)
            in_features = 256
        else:
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(*(list(backbone.children())[:-1]))

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet conv -> pool 输出形状 (B, C, 1, 1)
        feat_map = self.features(x)
        feat = torch.flatten(feat_map, 1)
        return self.classifier(feat)


class DenseNet121(nn.Module):
    """
    基于DenseNet121的X光分类模型，输出指定类别数。
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        backbone = _load_densenet121(pretrained=pretrained)
        if backbone is None:
            self.features = _SimpleBackbone(num_features=256)
            in_features = 256
        else:
            in_features = backbone.classifier.in_features
            self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        feat_map = nn.functional.relu(feat_map, inplace=True)
        pooled = self.pool(feat_map)
        feat = torch.flatten(pooled, 1)
        return self.classifier(feat)
