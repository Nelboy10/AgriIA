from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def get_default_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Trouve automatiquement la derniere couche convolutionnelle selon l'architecture."""
    arch = architecture.lower()
    if "resnet" in arch:
        # Dernier bloc résiduel de layer4
        if hasattr(model, "layer4"):
            return model.layer4[-1]
    elif "mobilenet" in arch:
        # Dernier bloc de features
        if hasattr(model, "features"):
            return model.features[-1]
    elif "simple_cnn" in arch or hasattr(model, "features"):
        # Chercher la dernière couche Conv2d dans features
        for layer in reversed(list(model.features)):
            if isinstance(layer, nn.Conv2d):
                return layer

    # Repli générique : chercher la dernière instance de Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError(f"Impossible de determiner la couche cible pour l'architecture: {architecture}")


class GradCAM:
    """Calculateur Grad-CAM (Gradient-weighted Class Activation Mapping)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, architecture: str = "resnet18"):
        self.model = model
        self.target_layer = target_layer or get_default_target_layer(model, architecture)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] est le gradient par rapport à la sortie du module
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        # Utilisation de register_full_backward_hook si disponible pour compatibilité PyTorch >= 2.0
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self._handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self):
        self.remove_hooks()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Genere une carte thermique normalisee 2D entre 0.0 et 1.0."""
        self.model.eval()
        self.activations = None
        self.gradients = None

        # S'assurer que le tenseur requiert des gradients
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Les activations ou gradients n'ont pas ete captures. Verifiez la couche cible.")

        # Pondération globale moyenne par canal (Global Average Pooling des gradients)
        weights = torch.mean(self.gradients[0], dim=(1, 2))  # Shape: [C]

        # Combinaison linéaire des cartes d'activation
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU pour ne conserver que les caractéristiques contribuant positivement
        cam = torch.relu(cam).cpu().numpy()

        # Normalisation Min-Max dans [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, Image.Image]:
    """
    Superpose la carte thermique sur l'image d'origine.
    Retourne (image_superposee, carte_thermique_rgb).
    """
    import matplotlib.cm as cm

    width, height = original_image.size

    # Redimensionner la heatmap aux dimensions de l'image originale
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((width, height), resample=Image.BICUBIC)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Appliquer une palette de couleurs (ex: jet / viridis)
    cmap = cm.get_cmap(colormap_name)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Retirer canal alpha
    heatmap_colored_uint8 = (heatmap_colored * 255).astype(np.uint8)
    heatmap_rgb_image = Image.fromarray(heatmap_colored_uint8)

    # Superposition pondérée
    orig_np = np.array(original_image.convert("RGB")).astype(np.float32)
    overlay_np = (1.0 - alpha) * orig_np + alpha * heatmap_colored_uint8.astype(np.float32)
    overlay_image = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    return overlay_image, heatmap_rgb_image
