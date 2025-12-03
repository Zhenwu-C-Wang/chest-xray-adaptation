"""
模型测试
"""

import unittest
import torch
from src.models.resnet import ResNet18, DenseNet121


class TestModels(unittest.TestCase):
    """模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_resnet18_output_shape(self):
        """测试ResNet18输出形状"""
        model = ResNet18(num_classes=2, pretrained=False).to(self.device)
        x = torch.randn(4, 3, 224, 224).to(self.device)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))
    
    def test_densenet121_output_shape(self):
        """测试DenseNet121输出形状"""
        model = DenseNet121(num_classes=2, pretrained=False).to(self.device)
        x = torch.randn(4, 3, 224, 224).to(self.device)
        output = model(x)
        self.assertEqual(output.shape, (4, 2))


if __name__ == '__main__':
    unittest.main()
