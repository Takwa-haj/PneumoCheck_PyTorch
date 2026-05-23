"""
gradcam.py — PneumoCheck (PyTorch)
Implémentation Grad-CAM pour l'explicabilité des prédictions ResNet50.
Référence : Selvaraju et al., ICCV 2017.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CLASSES  = ['NORMAL', 'BACTERIA', 'VIRUS']
IMG_SIZE = 224

# Normalisation ImageNet
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class GradCAM:
    """
    Grad-CAM via hooks PyTorch.

    Algorithme :
      1. Forward pass → enregistre les activations de la couche cible
      2. Backward pass → enregistre les gradients
      3. αk = mean sur (H,W) des gradients pour chaque feature map k
      4. L_GradCAM = ReLU( Σk αk · Ak )
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None

        # Hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Génère la heatmap Grad-CAM.

        Parameters
        ----------
        img_tensor : (1, 3, H, W) normalisé
        class_idx  : classe cible (None = classe prédite)

        Returns
        -------
        heatmap : np.ndarray (H, W) dans [0, 1]
        """
        self.model.eval()
        img_tensor.requires_grad = True

        logits = self.model(img_tensor)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(probs.argmax(dim=1).item())

        # Backward pour la classe cible
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # αk = mean des gradients sur (H, W)
        grads   = self._gradients[0]              # (C, H, W)
        acts    = self._activations[0]            # (C, H, W)
        weights = grads.mean(dim=(1, 2))          # (C,)

        # L = ReLU(Σk αk · Ak)
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H, W)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalise dans [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, class_idx, probs[0].detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model):
    """Récupère la dernière couche Conv de ResNet50."""
    # Dans ResNet50 : layer4 → Bottleneck → conv3 (dernière conv avant avgpool)
    try:
        return model.backbone.layer4[-1].conv3
    except AttributeError:
        # Fallback : chercher la dernière Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv


def apply_colormap_overlay(orig_array: np.ndarray,
                            heatmap: np.ndarray,
                            alpha: float = 0.45) -> np.ndarray:
    """Superpose la heatmap JET sur l'image originale (uint8 RGB)."""
    h_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_colored = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)
    img_u8    = (np.clip(orig_array, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=-1)
    return (h_colored * alpha + img_u8 * (1 - alpha)).clip(0, 255).astype(np.uint8)


def gradcam_for_image(image_path: str, model: torch.nn.Module,
                       device: str = 'cpu') -> dict:
    """
    Pipeline complet Grad-CAM pour une image.

    Returns dict :
        prediction, confidence, probabilities, heatmap, overlay, original
    """
    target_layer = get_target_layer(model)
    gcam         = GradCAM(model, target_layer)

    # Charger et prétraiter
    orig_pil  = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    orig_arr  = np.array(orig_pil) / 255.
    inp       = PREPROCESS(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    heatmap, pred_idx, probs = gcam(inp)
    overlay = apply_colormap_overlay(orig_arr, heatmap)
    gcam.remove_hooks()

    return {
        'prediction'   : CLASSES[pred_idx],
        'confidence'   : float(probs[pred_idx]),
        'probabilities': {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
        'heatmap'      : heatmap,
        'overlay'      : overlay,
        'original'     : (orig_arr * 255).astype(np.uint8),
    }
