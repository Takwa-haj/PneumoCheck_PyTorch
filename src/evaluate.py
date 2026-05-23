"""
evaluate.py — PneumoCheck (PyTorch)
Évaluation complète : rapport, matrice de confusion, courbes ROC.

Usage :
    python src/evaluate.py --model_path models/resnet50_best.pt
"""
import argparse, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.preprocessing import label_binarize

from data_loader import get_dataloaders
from models import get_model

CLASSES = ['NORMAL', 'BACTERIA', 'VIRUS']
COLORS  = ['#2ecc71', '#e74c3c', '#3498db']


def get_predictions(model, loader, device):
    """Retourne y_true, y_pred, y_prob pour tout le dataset."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return (np.array(all_labels),
            np.array(all_preds),
            np.vstack(all_probs))


def evaluate(model_path: str, test_dir: str, val_dir: str,
             train_dir: str, batch_size: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le modèle
    ckpt  = torch.load(model_path, map_location=device)
    model = get_model('resnet50', unfreeze_top=3, device=str(device))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modèle chargé : {model_path}  (ép. {ckpt.get('epoch','?')}, "
          f"val_acc={ckpt.get('val_acc',0):.4f})")

    _, _, test_loader, *_ = get_dataloaders(
        train_dir, val_dir, test_dir, augment=False, batch_size=batch_size)

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    # ── Rapport ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("Rapport de classification — ResNet50")
    print("="*55)
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Matrice de confusion ──────────────────────────────────
    os.makedirs('assets', exist_ok=True)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ['d', '.1%'],
        ['Matrice de confusion (effectifs)', 'Matrice de confusion (normalisée)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Vrai'); ax.set_xlabel('Prédit')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Courbes ROC ───────────────────────────────────────────
    Y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{cls}  (AUC={roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
    ax.set_xlabel('Taux de faux positifs'); ax.set_ylabel('Sensibilité')
    ax.set_title('Courbes ROC — One-vs-Rest', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--test_dir',   default='data/chest_xray/test')
    parser.add_argument('--val_dir',    default='data/chest_xray/val')
    parser.add_argument('--train_dir',  default='data/chest_xray/train')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model_path, args.test_dir, args.val_dir,
             args.train_dir, args.batch_size)
