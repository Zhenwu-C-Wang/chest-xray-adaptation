"""
数据预处理测试
"""

import unittest
from src.utils.utils import get_data_transforms, ImageDataset


class TestPreprocessing(unittest.TestCase):
    """数据预处理测试类"""
    
    def test_data_transforms(self):
        """测试数据变换"""
        transforms = get_data_transforms(img_size=224)
        self.assertIn('train', transforms)
        self.assertIn('val', transforms)
    
    def test_image_dataset(self):
        """测试图像数据集"""
        image_paths = ['path/to/img1.jpg', 'path/to/img2.jpg']
        labels = [0, 1]
        dataset = ImageDataset(image_paths, labels)
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    unittest.main()
