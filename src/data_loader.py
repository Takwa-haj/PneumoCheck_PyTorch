"""
data_loader.py — PneumoCheck (PyTorch)
Chargement, augmentation et class weights avec torchvision.
"""
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

CLASSES  = ['NORMAL', 'BACTERIA', 'VIRUS']
IMG_SIZE = 224
BATCH    = 32


def get_transforms(augment: bool = True):
    """
    Transformations PyTorch pour train / val-test.
    Augmentation médicalement justifiée :
      ✅ RandomRotation(10)      : légère rotation de prise de vue
      ✅ RandomHorizontalFlip    : symétrie droite/gauche anatomiquement valide
      ✅ RandomResizedCrop       : zoom variable (distance patient-détecteur)
      ✅ ColorJitter brightness  : exposition radiographique variable
      ❌ RandomVerticalFlip      : anatomiquement impossible
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # stats ImageNet
        std =[0.229, 0.224, 0.225])

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, val_tf


def get_dataloaders(train_dir: str, val_dir: str, test_dir: str,
                    augment: bool = True, batch_size: int = BATCH,
                    num_workers: int = 2, seed: int = 42):
    """
    Retourne train_loader, val_loader, test_loader et les datasets.

    Parameters
    ----------
    train_dir, val_dir, test_dir : chemins vers les répertoires
    augment      : active l'augmentation sur train
    batch_size   : taille des mini-batches
    num_workers  : workers DataLoader (0 sur Windows)
    seed         : reproductibilité
    """
    import torch
    g = torch.Generator(); g.manual_seed(seed)

    train_tf, val_tf = get_transforms(augment)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              generator=g,   pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"Classes : {train_ds.class_to_idx}")
    print(f"Train   : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def compute_class_weights(dataset, device='cpu'):
    """
    Calcule les poids de classe pour compenser le déséquilibre BACTERIA.
    Retourne un tensor PyTorch prêt pour nn.CrossEntropyLoss(weight=...).
    """
    import torch
    labels = [label for _, label in dataset.samples]
    cw = compute_class_weight('balanced',
                               classes=np.unique(labels), y=labels)
    return torch.tensor(cw, dtype=torch.float32).to(device)
