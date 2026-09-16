from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image


def assess_image_quality(image: Image.Image, min_sharpness: float = 60.0) -> dict:
    """
    Controle la qualite de la photo de feuille sans dependance externe lourde :
    - Nettete via variance du filtre Laplacien (convolution 2D numpy pure)
    - Luminosite moyenne et contraste
    """
    rgb = image.convert("RGB")
    gray = np.array(rgb.convert("L"), dtype=np.float32)

    # Noyau Laplacien discret 3x3 pour detection de contour/nettete
    # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    h, w = gray.shape
    if h < 16 or w < 16:
        return {
            "is_valid": False,
            "sharpness": 0.0,
            "is_sharp": False,
            "brightness": 0.0,
            "contrast": 0.0,
            "reason": "Image trop petite pour analyse fiable.",
        }

    laplacian = (
        gray[1:-1, 0:-2]
        + gray[1:-1, 2:]
        + gray[0:-2, 1:-1]
        + gray[2:, 1:-1]
        - 4.0 * gray[1:-1, 1:-1]
    )
    sharpness_score = float(np.var(laplacian))

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    issues = []
    is_sharp = sharpness_score >= min_sharpness
    if not is_sharp:
        issues.append(f"Photo floue (score de netteté {sharpness_score:.1f} < seuil {min_sharpness:.1f}).")

    if brightness < 35.0:
        issues.append("Photo sous-exposee (trop sombre).")
    elif brightness > 230.0:
        issues.append("Photo surexposee (trop eblouissante).")

    if contrast < 18.0:
        issues.append("Contraste trop faible pour distinguer les symptomes.")

    return {
        "is_valid": len(issues) == 0,
        "sharpness": round(sharpness_score, 1),
        "is_sharp": is_sharp,
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "issues": issues,
    }


def compute_predictive_entropy(probabilities: np.ndarray) -> float:
    """Calcule l'entropie normalisee de Shannon (0 = certitude absolue, 1 = incertitude totale)."""
    k = len(probabilities)
    if k <= 1:
        return 0.0
    eps = 1e-9
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    max_entropy = np.log(k)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def run_tta_inference(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """
    Test-Time Augmentation (TTA) :
    Evalue la photo sous 5 transformations legeres (originale, miroirs, rotations).
    Retourne la distribution moyenne des probabilites et un score de stabilite [0.0 - 1.0].
    """
    variants = [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.transpose(Image.FLIP_TOP_BOTTOM),
        image.rotate(10, resample=Image.BILINEAR),
        image.rotate(-10, resample=Image.BILINEAR),
    ]

    tensors = [transform(v.convert("RGB")).unsqueeze(0).to(device) for v in variants]
    batch = torch.cat(tensors, dim=0)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probs = np.mean(probs, axis=0)
    # Mesure de concordance : ecart-type moyen inter-variantes
    std_per_class = np.std(probs, axis=0)
    stability = float(np.clip(1.0 - (np.mean(std_per_class) * 2.5), 0.0, 1.0))

    return mean_probs, stability


@dataclass
class SafeDiagnosisResult:
    status: str  # "HIGH_CONFIDENCE", "MODERATE_CONFIDENCE", "REJECTED"
    badge_label: str
    top_label: str
    top_confidence: float
    top_predictions: List[Tuple[str, float]]
    quality: dict
    stability_score: float
    entropy: float
    message: str


def safe_predict(
    model: torch.nn.Module,
    image: Image.Image,
    labels: Dict[str, str],
    transform,
    device: torch.device,
    min_confidence: float = 0.75,
    min_sharpness: float = 50.0,
    use_tta: bool = True,
) -> SafeDiagnosisResult:
    """
    Pipeline de diagnostic securise avec garde-fous agronomiques stricts.
    """
    # 1. Controle qualite de la photo
    quality = assess_image_quality(image, min_sharpness=min_sharpness)

    # 2. Inférence (standard ou avec TTA)
    if use_tta:
        probs, stability = run_tta_inference(model, image, transform, device)
    else:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        stability = 1.0

    entropy = compute_predictive_entropy(probs)
    sorted_indices = np.argsort(probs)[::-1]

    top_idx = int(sorted_indices[0])
    top_conf = float(probs[top_idx])
    top_label = labels.get(str(top_idx), f"Classe {top_idx}")

    top_predictions = [
        (labels.get(str(idx), f"Classe {idx}"), float(probs[idx]))
        for idx in sorted_indices[: min(3, len(labels))]
    ]

    # 3. Decision avec regles de rejet strictes
    if not quality["is_valid"]:
        status = "REJECTED"
        badge = "[REJET - Photo non conforme]"
        message = (
            f"Qualite d'image insuffisante pour garantir un diagnostic fiable : "
            f"{' ; '.join(quality['issues'])}. Veuillez reprendre une photo nette et bien eclairee."
        )
    elif top_conf < min_confidence or entropy > 0.65 or stability < 0.65:
        status = "MODERATE_CONFIDENCE"
        badge = "[ATTENTION - Certitude moderee]"
        message = (
            f"Le modele hesite (confiance: {top_conf:.1%}, stabilite: {stability:.1%}). "
            "Ce diagnostic doit etre valide par un inspecteur de terrain ou une seconde photo."
        )
    else:
        status = "HIGH_CONFIDENCE"
        badge = "[VALIDE - Certitude elevee]"
        message = (
            f"Diagnostic robuste confirme par les tests de stabilite (confiance: {top_conf:.1%}, "
            f"stabilite TTA: {stability:.1%})."
        )


    return SafeDiagnosisResult(
        status=status,
        badge_label=badge,
        top_label=top_label,
        top_confidence=top_conf,
        top_predictions=top_predictions,
        quality=quality,
        stability_score=stability,
        entropy=entropy,
        message=message,
    )
