"""
train.py — PneumoCheck (PyTorch)
Script d'entraînement principal (CLI).

Usage :
    python src/train.py --model resnet50 --epochs 50 --batch_size 32
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import get_dataloaders, compute_class_weights
from models import get_model


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               scheduler, device, epochs, save_path, patience=6):
    best_val_acc = 0.
    no_improve   = 0
    history      = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss); history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc);   history['val_acc'].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_acc': vl_acc}, save_path)
            print(f"  ✓ Meilleur modèle sauvegardé ({vl_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  EarlyStopping déclenché à l'époque {epoch}")
                break

    print(f"\nMeilleure val_accuracy : {best_val_acc:.4f}")
    return history, best_val_acc


def main(args):
    device = get_device()
    print(f"Device : {device}")
    os.makedirs('models', exist_ok=True)

    train_loader, val_loader, _, train_ds, _, _ = get_dataloaders(
        args.train_dir, args.val_dir, args.test_dir,
        augment=True, batch_size=args.batch_size)

    cw = compute_class_weights(train_ds, device)

    if args.model == 'cnn_baseline':
        model = get_model('cnn_baseline', device=str(device))
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        print(f"\n=== CNN Baseline — {sum(p.numel() for p in model.parameters()):,} params ===")
        train_loop(model, train_loader, val_loader, criterion, optimizer,
                   scheduler, device, args.epochs,
                   save_path='models/cnn_baseline.pt')

    elif args.model == 'resnet50':
        # ── Phase 1 : base gelée ──────────────────────────────
        print("\n" + "="*55)
        print("  Phase 1 — Base ResNet50 gelée (LR=1e-4)")
        print("="*55)
        model = get_model('resnet50', unfreeze_top=0, device=str(device))
        print(f"Paramètres entraînables : {model.trainable_params():,} / {model.total_params():,}")

        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        _, best_p1 = train_loop(model, train_loader, val_loader, criterion,
                                 optimizer, scheduler, device,
                                 epochs=min(15, args.epochs),
                                 save_path='models/resnet50_phase1.pt')

        # ── Phase 2 : fine-tuning top-3 layer groups ─────────
        print("\n" + "="*55)
        print("  Phase 2 — Fine-tuning top-3 layer groups (LR=1e-5)")
        print("="*55)
        model2 = get_model('resnet50', unfreeze_top=3, device=str(device))
        # Charger les poids Phase 1
        ckpt = torch.load('models/resnet50_phase1.pt', map_location=device)
        model2.load_state_dict(ckpt['model_state'], strict=False)
        print(f"Paramètres entraînables : {model2.trainable_params():,} / {model2.total_params():,}")

        criterion2 = nn.CrossEntropyLoss(weight=cw)
        optimizer2 = Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=1e-5)
        scheduler2 = ReduceLROnPlateau(optimizer2, factor=0.5, patience=3)

        _, best_p2 = train_loop(model2, train_loader, val_loader, criterion2,
                                  optimizer2, scheduler2, device,
                                  epochs=args.epochs,
                                  save_path='models/resnet50_best.pt')

        print(f"\nGain fine-tuning : +{(best_p2 - best_p1)*100:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PneumoCheck — Entraînement PyTorch')
    parser.add_argument('--model',      default='resnet50',
                        choices=['cnn_baseline', 'resnet50'])
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_dir',  default='data/chest_xray/train')
    parser.add_argument('--val_dir',    default='data/chest_xray/val')
    parser.add_argument('--test_dir',   default='data/chest_xray/test')
    main(parser.parse_args())
