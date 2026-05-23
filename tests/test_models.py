"""Tests unitaires — models.py (PyTorch)"""
import unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np


class TestCNNBaseline(unittest.TestCase):
    def setUp(self):
        from models import CNNBaseline
        self.model = CNNBaseline()

    def test_output_shape(self):
        """Sortie doit être (batch, 3)."""
        x   = torch.zeros(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.shape, (2, 3))

    def test_no_softmax_in_output(self):
        """CrossEntropyLoss attend des logits (pas de softmax appliqué)."""
        x   = torch.randn(4, 3, 224, 224)
        out = self.model(x)
        # Les logits peuvent être négatifs ou > 1
        self.assertFalse(torch.all(out >= 0) and torch.all(out <= 1))

    def test_param_count(self):
        """CNN baseline doit avoir environ 2M paramètres."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total, 5_000_000)
        self.assertGreater(total, 500_000)


class TestResNet50Transfer(unittest.TestCase):
    def test_phase1_frozen(self):
        """Phase 1 : le backbone doit être entièrement gelé."""
        from models import ResNet50Transfer
        model = ResNet50Transfer(unfreeze_top=0)
        for name, param in model.backbone.named_parameters():
            self.assertFalse(param.requires_grad,
                             f"Le paramètre backbone.{name} devrait être gelé en Phase 1")

    def test_phase1_head_trainable(self):
        """Phase 1 : la tête doit être entraînable."""
        from models import ResNet50Transfer
        model = ResNet50Transfer(unfreeze_top=0)
        head_params = list(model.head.parameters())
        self.assertTrue(any(p.requires_grad for p in head_params),
                        "La tête doit être entraînable en Phase 1")

    def test_phase2_some_unfrozen(self):
        """Phase 2 : au moins certains paramètres du backbone doivent être dégelés."""
        from models import ResNet50Transfer
        model = ResNet50Transfer(unfreeze_top=3)
        backbone_trainable = sum(p.numel() for p in model.backbone.parameters()
                                 if p.requires_grad)
        self.assertGreater(backbone_trainable, 0,
                           "Phase 2 : des paramètres backbone doivent être entraînables")

    def test_output_shape(self):
        """Sortie doit être (batch, 3) pour 3 classes."""
        from models import ResNet50Transfer
        model = ResNet50Transfer(unfreeze_top=0)
        model.eval()
        with torch.no_grad():
            x   = torch.zeros(1, 3, 224, 224)
            out = model(x)
        self.assertEqual(out.shape, (1, 3))

    def test_trainable_params_phase1_small_fraction(self):
        """Phase 1 : < 5% des params totaux entraînables."""
        from models import ResNet50Transfer
        model     = ResNet50Transfer(unfreeze_top=0)
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio     = trainable / total
        self.assertLess(ratio, 0.05,
                        f"Trop de paramètres entraînables en Phase 1 : {ratio:.1%}")

    def test_softmax_probabilities(self):
        """Après softmax, les probas doivent sommer à 1."""
        from models import ResNet50Transfer
        model = ResNet50Transfer(unfreeze_top=0)
        model.eval()
        with torch.no_grad():
            x     = torch.randn(4, 3, 224, 224)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)


class TestGetModel(unittest.TestCase):
    def test_get_cnn_baseline(self):
        from models import get_model, CNNBaseline
        model = get_model('cnn_baseline')
        self.assertIsInstance(model, CNNBaseline)

    def test_get_resnet50(self):
        from models import get_model, ResNet50Transfer
        model = get_model('resnet50')
        self.assertIsInstance(model, ResNet50Transfer)

    def test_unknown_arch_raises(self):
        from models import get_model
        with self.assertRaises(ValueError):
            get_model('unknown_architecture')


if __name__ == '__main__':
    unittest.main()
