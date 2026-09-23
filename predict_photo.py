import argparse
import json
from pathlib import Path

from train_plant_disease import build_model


def _require_prediction_dependencies():
    global torch, Image, transforms

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        missing = exc.name or "une dependance"
        raise SystemExit(
            f"Dependance manquante: {missing}. Installez les dependances avec: "
            "pip install -r requirements.txt"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Prediction d'une maladie de plante depuis une photo.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--explain", action="store_true", help="Genere une carte thermique Grad-CAM sur la photo")
    parser.add_argument("--output-heatmap", type=Path, default=Path("outputs/prediction_gradcam.png"), help="Chemin du fichier pour la carte thermique")
    parser.add_argument("--safe-mode", action="store_true", help="Active les garde-fous agronomiques (anti-flou, TTA, seuil de rejet)")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Seuil de confiance minimale en safe-mode")
    parser.add_argument("--min-sharpness", type=float, default=45.0, help="Score minimal de netteté (anti-flou)")
    args = parser.parse_args()

    _require_prediction_dependencies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint["architecture"],
        checkpoint["num_classes"],
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    transform = transforms.Compose(
        [
            transforms.Resize((checkpoint.get("image_size", 224), checkpoint.get("image_size", 224))),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["normalize_mean"], checkpoint["normalize_std"]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    if args.safe_mode:
        from reliability import safe_predict

        safe_res = safe_predict(
            model=model,
            image=image,
            labels=labels,
            transform=transform,
            device=device,
            min_confidence=args.min_confidence,
            min_sharpness=args.min_sharpness,
            use_tta=True,
        )
        print(f"Statut : {safe_res.badge_label}")
        print(f"Qualite image : Nettete={safe_res.quality['sharpness']:.1f} (seuil {args.min_sharpness}), Luminosite={safe_res.quality['brightness']:.1f}/255")
        print(f"Stabilite TTA : {safe_res.stability_score:.1%}")
        print(f"Explication : {safe_res.message}\n")

        print("Predictions :")
        for lbl, score in safe_res.top_predictions[: args.top_k]:
            print(f"  - {lbl}: {score:.3f}")
        top_class = next(
            index for index, label in labels.items() if label == safe_res.top_predictions[0][0]
        )
        top_class = int(top_class)
    else:
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        top_values, top_indices = torch.topk(probabilities, k=min(args.top_k, len(labels)))
        top_class = int(top_indices[0].item())

        print("Predictions :")
        for score, idx in zip(top_values.cpu().tolist(), top_indices.cpu().tolist()):
            print(f"  - {labels[str(idx)]}: {score:.3f}")


    if args.explain:
        from gradcam import GradCAM, overlay_heatmap_on_image

        cam = GradCAM(model, architecture=checkpoint["architecture"])
        heatmap = cam.generate_heatmap(tensor, target_class=top_class)
        cam.remove_hooks()

        overlay_img, _ = overlay_heatmap_on_image(image, heatmap, alpha=0.45)
        args.output_heatmap.parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(args.output_heatmap)
        print(f"Carte thermique Grad-CAM sauvegardee: {args.output_heatmap}")


if __name__ == "__main__":
    main()

