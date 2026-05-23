"""
utils.py — PneumoCheck (PyTorch)
Fonctions utilitaires : courbes d'apprentissage, visualisations, résumés.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASSES = ['NORMAL', 'BACTERIA', 'VIRUS']
COLORS  = ['#2ecc71', '#e74c3c', '#3498db']


def plot_learning_curves(history: dict, title: str = 'Courbes d\'apprentissage',
                          phase_break: int = None, save_path: str = None):
    """
    Affiche les courbes train/val pour loss et accuracy.

    Parameters
    ----------
    history     : dict avec clés train_loss, val_loss, train_acc, val_acc
    phase_break : époque où le fine-tuning démarre (ligne verticale)
    save_path   : chemin optionnel pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairs = [
        ('train_loss', 'val_loss', 'Loss (Cross-Entropy)'),
        ('train_acc',  'val_acc',  'Accuracy'),
    ]
    for ax, (tr_k, vl_k, metric_title) in zip(axes, pairs):
        if tr_k not in history:
            continue
        ax.plot(history[tr_k], label='Train', linewidth=2, color='#1e88e5')
        ax.plot(history[vl_k], label='Val',   linewidth=2, color='#e53935', linestyle='--')
        ax.fill_between(range(len(history[tr_k])),
                        history[tr_k], history[vl_k],
                        alpha=0.06, color='orange', label='Gap')
        if phase_break is not None:
            ax.axvline(phase_break - 1, color='#43a047', linestyle=':',
                       linewidth=2, label=f'Fine-tuning (ép. {phase_break})')
        ax.set_title(f'{title} — {metric_title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Époque'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Affiche matrice de confusion en effectifs et normalisée."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, t in zip(
        axes, [cm, cm_norm], ['d', '.1%'],
        ['Matrice de confusion (effectifs)', 'Matrice de confusion (normalisée)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                    linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(t, fontsize=12, fontweight='bold')
        ax.set_ylabel('Vrai'); ax.set_xlabel('Prédit')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_model_summary(model, name: str = ''):
    """Affiche le nombre de paramètres entraînables / totaux."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n{'='*50}")
    print(f"Modèle : {name or model.__class__.__name__}")
    print(f"  Total paramètres      : {total:>12,}")
    print(f"  Entraînables          : {trainable:>12,}  ({trainable/total*100:.1f}%)")
    print(f"  Gelés                 : {frozen:>12,}  ({frozen/total*100:.1f}%)")
    print(f"{'='*50}\n")


def print_performance_table(results: dict):
    """
    results = {'Model': {'acc':0.95, 'f1':0.94, 'params':'23M', 'best':True}, ...}
    """
    print(f"\n{'Modèle':<38} {'Accuracy':>9}  {'F1-Score':>9}  {'Params':>8}")
    print("─" * 70)
    for name, r in results.items():
        marker = ' ✅' if r.get('best') else ''
        print(f"{name:<38} {r['acc']:>8.1%}  {r['f1']:>9.4f}  {r['params']:>8}{marker}")
