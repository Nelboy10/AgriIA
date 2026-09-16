from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_plant_disease import build_model
from gradcam import GradCAM, overlay_heatmap_on_image


def _require_dependencies():
    global torch, Image, transforms, plt
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        missing = exc.name or "une dependance"
        raise SystemExit(
            f"Dependance manquante: {missing}. Installez les dependances avec: "
            "pip install -r requirements.txt"
        ) from exc


def explain(
    checkpoint_path: Path,
    labels_path: Path,
    image_path: Path,
    output_path: Path,
    target_class: int | None = None,
    alpha: float = 0.45,
    colormap: str = "jet",
):
    _require_dependencies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    architecture = checkpoint["architecture"]
    num_classes = checkpoint["num_classes"]
    model = build_model(
        architecture,
        num_classes,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Prétraitement d'image
    image_size = checkpoint.get("image_size", 224)
    normalize_mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    normalize_std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    original_image = Image.open(image_path).convert("RGB")
    tensor = transform(original_image).unsqueeze(0).to(device)

    # Inférence pour obtenir les scores et probabilités
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_values, top_indices = torch.topk(probabilities, k=min(3, len(labels)))
    pred_class_idx = int(top_indices[0].item())
    pred_confidence = float(top_values[0].item())
    pred_label = labels.get(str(pred_class_idx), f"Classe {pred_class_idx}")

    target_idx = target_class if target_class is not None else pred_class_idx
    target_label = labels.get(str(target_idx), f"Classe {target_idx}")

    # Calcul Grad-CAM
    cam_generator = GradCAM(model, architecture=architecture)
    heatmap = cam_generator.generate_heatmap(tensor, target_class=target_idx)
    cam_generator.remove_hooks()

    overlay_img, heatmap_img = overlay_heatmap_on_image(
        original_image, heatmap, alpha=alpha, colormap_name=colormap
    )

    # Création d'une figure comparative
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Photo d'origine")
    axes[0].axis("off")

    axes[1].imshow(heatmap_img)
    axes[1].set_title("Carte thermique d'activation")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title(f"Diagnostic: {pred_label}\nConfiance: {pred_confidence:.1%}")
    axes[2].axis("off")

    plt.suptitle(
        f"Explicabilité Grad-CAM — Détection focalisée sur: {target_label}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Prediction principale: {pred_label} ({pred_confidence:.1%})")
    print(f"Rapport d'explicabilite enregistre dans: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Explicabilite visuelle Grad-CAM pour le diagnostic de maladies.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Chemin vers best_model.pt")
    parser.add_argument("--labels", type=Path, required=True, help="Chemin vers label_map.json")
    parser.add_argument("--image", type=Path, required=True, help="Chemin vers la photo a analyser")
    parser.add_argument("--output", type=Path, default=Path("outputs/gradcam_report.png"), help="Fichier de sortie de l'explication")
    parser.add_argument("--target-class", type=int, default=None, help="Index specifique de classe a expliquer (par defaut la plus probable)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Transparence de la heatmap sur la photo (0.0 a 1.0)")
    parser.add_argument("--colormap", default="jet", help="Palette matplotlib (jet, viridis, magma, etc.)")
    args = parser.parse_args()

    explain(
        checkpoint_path=args.checkpoint,
        labels_path=args.labels,
        image_path=args.image,
        output_path=args.output,
        target_class=args.target_class,
        alpha=args.alpha,
        colormap=args.colormap,
    )


if __name__ == "__main__":
    main()
