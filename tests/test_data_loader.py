"""Tests unitaires — data_loader.py (PyTorch)"""
import unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from unittest.mock import MagicMock


class TestComputeClassWeights(unittest.TestCase):
    def _make_dataset(self, class_counts):
        ds = MagicMock()
        samples = []
        for cls_idx, count in enumerate(class_counts):
            samples.extend([(f'img_{cls_idx}_{i}.jpg', cls_idx) for i in range(count)])
        ds.samples = samples
        return ds

    def test_minority_class_gets_higher_weight(self):
        from data_loader import compute_class_weights
        # BACTERIA (idx=1) est majoritaire → poids plus faible
        ds = self._make_dataset([1341, 2538, 1329])
        weights = compute_class_weights(ds)
        self.assertLess(weights[1].item(), weights[0].item())
        self.assertLess(weights[1].item(), weights[2].item())

    def test_weights_are_positive(self):
        from data_loader import compute_class_weights
        ds = self._make_dataset([100, 300, 100])
        weights = compute_class_weights(ds)
        for w in weights:
            self.assertGreater(w.item(), 0)

    def test_returns_tensor(self):
        import torch
        from data_loader import compute_class_weights
        ds = self._make_dataset([200, 400, 200])
        weights = compute_class_weights(ds)
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape[0], 3)


class TestDataLoaderConfig(unittest.TestCase):
    def test_classes(self):
        from data_loader import CLASSES
        self.assertEqual(CLASSES, ['NORMAL', 'BACTERIA', 'VIRUS'])

    def test_img_size(self):
        from data_loader import IMG_SIZE
        self.assertEqual(IMG_SIZE, 224)

    def test_batch_size_default(self):
        from data_loader import BATCH
        self.assertEqual(BATCH, 32)


if __name__ == '__main__':
    unittest.main()
