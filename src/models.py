"""
models.py — PneumoCheck (PyTorch)
Architectures CNN Baseline et ResNet50 Transfer Learning.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, VGG16_Weights

N_CLASSES = 3


# ──────────────────────────────────────────────────────────────
# 1. CNN Baseline — From Scratch
# ──────────────────────────────────────────────────────────────
class CNNBaseline(nn.Module):
    """
    CNN 4 blocs Conv/BatchNorm/ReLU/MaxPool + tête Dense.
    Architecture :
        Conv(32) → BN → ReLU → MaxPool(2)
        Conv(64) → BN → ReLU → MaxPool(2)
        Conv(128)→ BN → ReLU → MaxPool(2)
        Conv(256)→ BN → ReLU → MaxPool(2)
        AdaptiveAvgPool → Flatten
        Linear(512) → ReLU → Dropout(0.5)
        Linear(3) → (CrossEntropyLoss inclut softmax)
    Paramètres : ~2.1M
    """
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.5):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            conv_block(3,   32),
            conv_block(32,  64),
            conv_block(64,  128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # → (256, 4, 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
# 2. ResNet50 Transfer Learning
# ──────────────────────────────────────────────────────────────
class ResNet50Transfer(nn.Module):
    """
    ResNet50 pré-entraîné ImageNet + tête de classification personnalisée.

    Architecture tête :
        GAP → Linear(512) → BN → ReLU → Dropout(0.5)
            → Linear(256) → BN → ReLU → Dropout(0.3)
            → Linear(3)

    Parameters
    ----------
    unfreeze_top : 0  = Phase 1 — base entièrement gelée
                   N  = Phase 2 — N dernières couches libérées
    """
    def __init__(self, n_classes: int = N_CLASSES,
                 unfreeze_top: int = 0,
                 dropout1: float = 0.5, dropout2: float = 0.3):
        super().__init__()

        # Backbone ResNet50 pré-entraîné
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features          # 2048

        # Supprimer la tête FC originale → GAP déjà inclus dans avgpool
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Geler / dégeler
        self._set_freeze(unfreeze_top)

        # Nouvelle tête
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(256, n_classes),
        )

    def _set_freeze(self, unfreeze_top: int):
        """Gèle tous les paramètres du backbone sauf les N derniers layers."""
        # D'abord tout geler
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_top > 0:
            # Lister les modules nommés du backbone en ordre
            all_layers = list(self.backbone.named_children())
            # Dégeler les N derniers
            for name, module in all_layers[-unfreeze_top:]:
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)    # (B, 2048)
        return self.head(features)     # (B, 3)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 3. VGG16 Transfer Learning (pour l'ensemble)
# ──────────────────────────────────────────────────────────────
class VGG16Transfer(nn.Module):
    """VGG16 pré-entraîné, base gelée + tête simple."""
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.5):
        super().__init__()
        backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Geler tout le backbone
        for param in backbone.parameters():
            param.requires_grad = False
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)


# ──────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────
def get_model(arch: str, unfreeze_top: int = 0, device: str = 'cpu'):
    """
    Retourne le modèle demandé sur le bon device.
    arch : 'cnn_baseline' | 'resnet50' | 'vgg16'
    """
    if arch == 'cnn_baseline':
        model = CNNBaseline()
    elif arch == 'resnet50':
        model = ResNet50Transfer(unfreeze_top=unfreeze_top)
    elif arch == 'vgg16':
        model = VGG16Transfer()
    else:
        raise ValueError(f"Architecture inconnue : {arch}")
    return model.to(device)
