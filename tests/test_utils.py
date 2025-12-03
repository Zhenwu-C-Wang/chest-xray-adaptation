"""
工具函数测试
"""

import unittest
import torch
from src.utils.utils import set_seed, get_device


class TestUtils(unittest.TestCase):
    """工具函数测试类"""
    
    def test_set_seed(self):
        """测试随机种子设置"""
        set_seed(42)
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        self.assertTrue(torch.allclose(x1, x2))
    
    def test_get_device(self):
        """测试设备获取"""
        device = get_device(use_cuda=False)
        self.assertEqual(device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
