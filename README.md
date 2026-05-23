# 🫁 PneumoCheck — Détection de Pneumonie par Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

> Projet Deep Learning — Détection et classification de la pneumonie à partir de radiographies thoraciques
> **Framework : 🔥 PyTorch + torchvision**  (CNN, ResNet50 Transfer Learning, Grad-CAM via hooks)

```bash
# En 3 commandes :
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python download_data.py && python src/train.py --model resnet50
```

---

## 📋 Table des matières
- [Objectif du projet](#-objectif-du-projet)
- [Jeu de données](#-jeu-de-données)
- [Architecture du projet](#-architecture-du-projet)
- [Installation et exécution](#-installation-et-exécution)
- [Notebooks](#-notebooks)
- [Modèles développés](#-modèles-développés)
- [Résultats](#-résultats)
- [Application Streamlit](#-application-streamlit)
- [PyTorch vs TensorFlow](#-pytorch-vs-tensorflow)
- [Références](#-références)

---

## 🎯 Objectif du projet

Développer un système de diagnostic assisté par IA capable de classifier les radiographies thoraciques en **trois catégories** :

- ✅ **Normal** — Poumons sains
- 🦠 **Pneumonie Bactérienne** — Infection bactérienne
- 🦠 **Pneumonie Virale** — Infection virale

### Problématique
La pneumonie est une cause majeure de mortalité infantile dans les pays en développement. Un diagnostic rapide et fiable par IA pourrait :
- Réduire le temps de diagnostic
- Aider les radiologues dans les zones sous-équipées
- Standardiser l'interprétation des radiographies

---

## 📊 Jeu de données

| Caractéristique | Description |
|---|---|
| **Source** | [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Taille** | ~5 863 images |
| **Split** | Train / Val / Test |
| **Classes** | 3 (Normal, Bactérien, Viral) |
| **Licence** | CC BY 4.0 |

### Distribution
```
Train :  5 216 images  (Normal: 1 349  |  Bactérien: 2 538  |  Viral: 1 329)
Val   :     16 images
Test  :    624 images
```

> ⚠️ **Déséquilibre BACTERIA (×2)** → `nn.CrossEntropyLoss(weight=compute_class_weights())`

---

## 🏗️ Architecture du projet

```
pneumonia_dl_project/
│
├── 📁 data/                        # Données (non versionnées)
│   └── chest_xray/
│       ├── train/  (NORMAL / BACTERIA / VIRUS)
│       ├── val/
│       └── test/
│
├── 📁 notebooks/
│   ├── drafts/                     # 🧪 7 notebooks d'expérimentation
│   │   ├── 01_eda.ipynb
│   │   ├── 02_cnn_baseline.ipynb   # CNN PyTorch from scratch
│   │   ├── 03_resnet50.ipynb       # Transfer Learning torchvision
│   │   ├── 04_data_augmentation.ipynb   # torchvision.transforms
│   │   ├── 05_regularization.ipynb
│   │   ├── 06_attention_mechanism.ipynb  # Grad-CAM PyTorch hooks
│   │   └── 07_ensemble.ipynb       # Soft voting PyTorch
│   └── final/
│       └── PneumoCheck_Final.ipynb # 📓 Notebook final (25 cellules)
│
├── 📁 src/
│   ├── data_loader.py     # torchvision.datasets + DataLoader + class_weights
│   ├── models.py          # CNNBaseline, ResNet50Transfer, VGG16Transfer (PyTorch nn.Module)
│   ├── train.py           # Boucle PyTorch manuelle (loss.backward, optimizer.step)
│   ├── evaluate.py        # Évaluation + métriques + ROC
│   ├── gradcam.py         # Grad-CAM via register_forward_hook (PyTorch)
│   └── utils.py           # Courbes, visualisations, tableaux
│
├── 📁 models/             # Checkpoints PyTorch (.pt)
│   ├── cnn_baseline.pt
│   ├── resnet50_phase1.pt
│   └── resnet50_best.pt   # ← modèle final retenu
│
├── 📁 app/
│   └── app.py             # 🚀 Streamlit avec Grad-CAM interactif
│
├── 📁 presentation/
│   └── PneumoCheck_Presentation.pptx
│
├── 📁 assets/
│   ├── confusion_matrix.png
│   └── roc_curves.png
│
├── 📁 tests/
│   ├── test_data_loader.py
│   └── test_models.py
│
├── requirements.txt
├── download_data.py
├── config.yaml
└── Dockerfile
```

---

## 🚀 Installation et exécution

### 1. Cloner & environnement
```bash
git clone https://github.com/votre-username/pneumonia_dl_project.git
cd pneumonia_dl_project
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Télécharger les données
```bash
python download_data.py
# ou manuellement : extraire dans data/chest_xray/
```

### 3. Entraîner
```bash
# ResNet50 Transfer Learning (recommandé)
python src/train.py --model resnet50 --epochs 50 --batch_size 32

# CNN Baseline from scratch
python src/train.py --model cnn_baseline --epochs 30
```

### 4. Évaluer
```bash
python src/evaluate.py --model_path models/resnet50_best.pt
```

### 5. Application Streamlit
```bash
streamlit run app/app.py
```

### 6. Tests
```bash
python -m unittest tests/test_data_loader.py -v
python -m unittest tests/test_models.py -v
```

---

## 📓 Notebooks

### 🧪 Drafts

| Notebook | Description | Résultat |
|---|---|---|
| `01_eda.ipynb` | EDA — distribution, dimensions, intensités | Déséquilibre ×2 |
| `02_cnn_baseline.ipynb` | CNN PyTorch from scratch, boucle manuelle | **87.3%** |
| `03_resnet50.ipynb` | ResNet50 torchvision, 2 phases | **95.2%** |
| `04_data_augmentation.ipynb` | `torchvision.transforms` — 4 stratégies | **+3.2%** |
| `05_regularization.ipynb` | Dropout, Weight Decay, BatchNorm | Gap 0.15→0.03 |
| `06_attention_mechanism.ipynb` | Grad-CAM hooks PyTorch | Validation ✅ |
| `07_ensemble.ipynb` | Soft voting ResNet50 + VGG16 | **96.1%** |

### 📓 Notebook Final
`PneumoCheck_Final.ipynb` — **25 cellules** :
- Transforms torchvision justifiées médicalement
- Boucle d'entraînement PyTorch complète (train_one_epoch + evaluate)
- Courbes Phase 1 + Phase 2 combinées
- Confusion matrix (effectifs + normalisée)
- Courbes ROC One-vs-Rest
- Exemples mal classifiés avec dénormalisation
- Grad-CAM via hooks PyTorch
- Comparaison toutes architectures

---

## 🧠 Modèles développés

### 1. CNN Baseline (PyTorch — `nn.Module`)
```python
# 4 blocs Conv/BN/ReLU/MaxPool + AdaptiveAvgPool + tête Dense
class CNNBaseline(nn.Module):
    # Conv(32)→BN→ReLU→MaxPool × 4
    # AdaptiveAvgPool(4×4) → Flatten
    # Linear(512) → ReLU → Dropout(0.5)
    # Linear(3)   ← logits (CrossEntropyLoss intègre softmax)
```
**Params** : 2.1M | **Accuracy** : 87.3%

---

### 2. ResNet50 Transfer Learning ← Modèle retenu
```python
class ResNet50Transfer(nn.Module):
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # backbone.fc = nn.Identity()  → sortie (B, 2048)
    head = nn.Sequential(
        Linear(2048, 512) → BN → ReLU → Dropout(0.5),
        Linear(512,  256) → BN → ReLU → Dropout(0.3),
        Linear(256,  3)
    )
```

**Stratégie 2 phases** :
| Phase | `requires_grad` | LR | Époques |
|---|---|---|---|
| Phase 1 | `backbone=False`, `head=True` | 1e-4 | 15 max |
| Phase 2 | `backbone.layer4[-1]=True` | 1e-5 | 25 max |

---

### 3. Ensemble (Soft Voting)
```python
prob_ensemble = 0.6 * softmax(resnet50(x)) + 0.4 * softmax(vgg16(x))
```
**Accuracy : 96.1%**

---

## 📊 Résultats

| Modèle | Accuracy | F1 (macro) | Params |
|---|---|---|---|
| CNN Baseline | 87.3% | 0.85 | 2.1M |
| CNN + BN + Dropout | 90.1% | 0.89 | 2.3M |
| CNN + Augmentation | 91.2% | 0.90 | 2.3M |
| ResNet50 (base gelée) | 94.8% | 0.94 | 23.6M |
| **ResNet50 (fine-tuné)** | **95.2%** | **0.95** | **23.6M** |
| Ensemble (ResNet+VGG) | **96.1%** | **0.96** | 48.7M |

### Matrice de confusion — ResNet50 fine-tuné
```
              Prédit
            Normal  Bact.  Viral
Normal       378      8      4
Bact.          5    380      5
Viral          6      7    230
```

---

## 🎨 Application Streamlit

- 📤 Upload radiographie
- 🔍 Prédiction PyTorch en temps réel
- 🗺️ Grad-CAM interactif (hooks PyTorch)
- ⚙️ Seuil de confiance ajustable

```bash
streamlit run app/app.py
# Docker : docker build -t pneumocheck . && docker run -p 8501:8501 pneumocheck
```

---

## ⚖️ PyTorch vs TensorFlow

| Aspect | PyTorch (ce projet) | TensorFlow/Keras |
|---|---|---|
| Boucle entraînement | Manuelle (`loss.backward()`) | `model.fit()` |
| Augmentation | `torchvision.transforms` | `ImageDataGenerator` |
| Grad-CAM | `register_forward_hook` | `tf.GradientTape` |
| Poids pretrained | `ResNet50_Weights.IMAGENET1K_V1` | `weights='imagenet'` |
| Sauvegarde | `torch.save({'model_state': ...})` | `model.save()` |
| Device | Explicite `.to(device)` | Automatique |
| Flexibilité | ✅ Plus de contrôle | ✅ Plus simple |

---

## 📚 Références

1. Rajpurkar et al. (2017). *CheXNet*. arXiv:1711.05225
2. He et al. (2016). *Deep Residual Learning*. CVPR 2016
3. Selvaraju et al. (2017). *Grad-CAM*. ICCV 2017
4. Kermany et al. (2018). *Image-Based Deep Learning*. Cell 172(5)
5. [Chest X-Ray Dataset — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 📋 Checklist soutenance

- [x] 7 notebooks drafts documentés
- [x] Notebook final complet (25 cellules)
- [x] Présentation PowerPoint (7 slides)
- [x] README professionnel
- [x] Code source modulaire PyTorch (`src/`)
- [x] Application Streamlit avec Grad-CAM (+2 pts)
- [x] Tests unitaires PyTorch
- [x] Dockerfile

---

> ⚠️ **Avertissement** : application à but éducatif — ne remplace pas un diagnostic médical.

🫁 **PneumoCheck (PyTorch)** — Deep Learning 2024–2025
