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

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    top_values, top_indices = torch.topk(probabilities, k=min(args.top_k, len(labels)))

    for score, idx in zip(top_values.cpu().tolist(), top_indices.cpu().tolist()):
        print(f"{labels[str(idx)]}: {score:.3f}")


if __name__ == "__main__":
    main()
