from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def _require_training_dependencies():
    global np, torch, nn, optim, DatasetDict, load_dataset
    global accuracy_score, classification_report, confusion_matrix
    global DataLoader, Image, models, transforms, tqdm

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from datasets import DatasetDict, load_dataset
        from PIL import Image
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        missing = exc.name or "une dependance"
        raise SystemExit(
            f"Dependance manquante: {missing}. Installez les dependances avec: "
            "pip install -r requirements.txt"
        ) from exc


def _standardize_label_column(dataset, source_label_column: str):
    if source_label_column == "label" and dataset.features["label"].__class__.__name__ == "ClassLabel":
        id_to_label = {idx: name for idx, name in enumerate(dataset.features["label"].names)}
        return dataset, id_to_label

    if source_label_column == "class_idx":
        pairs = {}
        for class_idx, class_label in zip(dataset["class_idx"], dataset["class_label"]):
            pairs[int(class_idx)] = str(class_label)
        id_to_label = dict(sorted(pairs.items()))
        return dataset.map(lambda item: {"label": int(item["class_idx"])}), id_to_label

    raw_labels = sorted(set(str(label) for label in dataset[source_label_column]))
    label_to_id = {label: idx for idx, label in enumerate(raw_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return dataset.map(lambda item: {"label": label_to_id[str(item[source_label_column])]}), id_to_label


def _load_image_dataset(seed: int):
    candidates = [
        ("mohanty/PlantVillage", "color", {}),
        ("mohanty/PlantVillage", "default", {}),
        ("geraldmc/plantvillage-full", None, {"revision": "v0.1.0"}),
        ("geraldmc/plantvillage-full", None, {}),
    ]
    last_error = None
    for name, config, kwargs in candidates:
        try:
            dataset = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
            base_split = "train" if "train" in dataset else list(dataset.keys())[0]
            columns = dataset[base_split].column_names
            if "image" not in columns:
                continue
            if "label" in columns:
                source_label_column = "label"
            elif "class_idx" in columns:
                source_label_column = "class_idx"
            elif "class_label" in columns:
                source_label_column = "class_label"
            else:
                continue
            print(f"Dataset utilise: {name} ({config or 'default'})")
            return dataset, base_split, source_label_column
        except Exception as exc:
            last_error = exc
            print(f"Dataset ignore: {name} ({config or 'default'}): {exc}")
    raise RuntimeError("Aucun dataset PlantVillage avec colonnes image/label n'a pu etre charge.") from last_error


def load_plantvillage_splits(seed: int = 42) -> Tuple[DatasetDict, Dict[int, str]]:
    """Charge PlantVillage et cree train/validation/test."""
    _require_training_dependencies()
    dataset_dict, base_split, source_label_column = _load_image_dataset(seed)
    full_dataset, id_to_label = _standardize_label_column(dataset_dict[base_split], source_label_column)

    if "split" in full_dataset.column_names:
        train_candidate = full_dataset.filter(lambda item: str(item["split"]).lower() == "train")
        test_candidate = full_dataset.filter(lambda item: str(item["split"]).lower() != "train")
        if len(train_candidate) > 0 and len(test_candidate) > 0:
            validation_split = train_candidate.train_test_split(test_size=0.10, seed=seed)
            return (
                DatasetDict(
                    train=validation_split["train"],
                    validation=validation_split["test"],
                    test=test_candidate,
                ),
                id_to_label,
            )

    split_1 = full_dataset.train_test_split(test_size=0.20, seed=seed)
    split_2 = split_1["test"].train_test_split(test_size=0.50, seed=seed)
    splits = DatasetDict(
        train=split_1["train"],
        validation=split_2["train"],
        test=split_2["test"],
    )
    return splits, id_to_label


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _split_samples(samples: List[Tuple[Path, int]], seed: int):
    if len(samples) < 3:
        raise ValueError("Le dataset local doit contenir au moins 3 images.")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * 0.10))
    validation_size = max(1, int(len(shuffled) * 0.10))
    test = shuffled[:test_size]
    validation = shuffled[test_size : test_size + validation_size]
    train = shuffled[test_size + validation_size :]
    if not train:
        raise ValueError("Le dataset local est trop petit pour creer un split train/validation/test.")
    return {"train": train, "validation": validation, "test": test}


def limit_samples_per_split(splits, max_samples_per_split: int | None, seed: int):
    if max_samples_per_split is None:
        return splits
    if max_samples_per_split <= 0:
        raise ValueError("--max-samples-per-split doit etre positif.")

    limited = {}
    for split_name, split in splits.items():
        sample_count = min(max_samples_per_split, len(split))
        if hasattr(split, "shuffle") and hasattr(split, "select"):
            limited[split_name] = split.shuffle(seed=seed).select(range(sample_count))
        else:
            shuffled = list(split)
            random.Random(f"{seed}-{split_name}").shuffle(shuffled)
            limited[split_name] = shuffled[:sample_count]
    return limited


def print_dataset_summary(splits, id_to_label: Dict[int, str]):
    print("Resume dataset:")
    print(f"- classes: {len(id_to_label)}")
    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            continue
        print(f"- {split_name}: {len(splits[split_name])} images")


def _scan_imagefolder_split(split_dir: Path, label_to_id: Dict[str, int]):
    samples = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label_id = label_to_id.setdefault(class_dir.name, len(label_to_id))
        for image_path in sorted(path for path in class_dir.rglob("*") if path.is_file() and _is_image_file(path)):
            samples.append((image_path, label_id))
    return samples


def load_local_imagefolder_splits(data_dir: Path, seed: int):
    """Charge un dataset local au format class folders ou train/validation/test/class."""
    label_to_id = {}
    split_names = {"train", "validation", "val", "test"}
    available_splits = {path.name.lower(): path for path in data_dir.iterdir() if path.is_dir() and path.name.lower() in split_names}

    if "train" in available_splits and "test" in available_splits:
        splits = {}
        for split_name in ("train", "validation", "test"):
            source_dir = available_splits.get(split_name)
            if split_name == "validation" and source_dir is None:
                source_dir = available_splits.get("val")
            if source_dir is not None:
                splits[split_name] = _scan_imagefolder_split(source_dir, label_to_id)
        if "validation" not in splits:
            train_split = _split_samples(splits["train"], seed)
            splits["train"] = train_split["train"]
            splits["validation"] = train_split["validation"]
        for split_name, samples in splits.items():
            if not samples:
                raise ValueError(f"Le split local '{split_name}' ne contient aucune image.")
    else:
        samples = _scan_imagefolder_split(data_dir, label_to_id)
        splits = _split_samples(samples, seed)

    id_to_label = {idx: label for label, idx in sorted(label_to_id.items(), key=lambda item: item[1])}
    return splits, id_to_label


def load_local_csv_splits(
    csv_path: Path,
    image_root: Path,
    image_column: str,
    label_column: str,
    split_column: str | None,
    seed: int,
):
    label_to_id = {}
    grouped = {"train": [], "validation": [], "test": []}
    all_samples = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = {image_column, label_column} - set(reader.fieldnames or [])
        if split_column:
            missing_columns -= {split_column}
            if split_column not in set(reader.fieldnames or []):
                missing_columns.add(split_column)
        if missing_columns:
            raise ValueError(f"Colonnes absentes du CSV: {', '.join(sorted(missing_columns))}")

        for row in reader:
            image_path = image_root / row[image_column]
            label = str(row[label_column]).strip()
            if not label:
                continue
            label_id = label_to_id.setdefault(label, len(label_to_id))
            sample = (image_path, label_id)
            if split_column:
                split_name = str(row[split_column]).strip().lower()
                if split_name == "val":
                    split_name = "validation"
                if split_name in grouped:
                    grouped[split_name].append(sample)
                else:
                    all_samples.append(sample)
            else:
                all_samples.append(sample)

    if split_column and grouped["train"] and grouped["test"]:
        splits = grouped
        if not splits["validation"]:
            train_split = _split_samples(splits["train"], seed)
            splits["train"] = train_split["train"]
            splits["validation"] = train_split["validation"]
    else:
        splits = _split_samples(all_samples, seed)

    for split_name, samples in splits.items():
        if not samples:
            raise ValueError(f"Le split CSV '{split_name}' ne contient aucune image.")
        missing_images = [str(path) for path, _ in samples if not path.exists()]
        if missing_images:
            preview = ", ".join(missing_images[:3])
            raise FileNotFoundError(f"Images introuvables dans '{split_name}': {preview}")

    id_to_label = {idx: label for label, idx in sorted(label_to_id.items(), key=lambda item: item[1])}
    return splits, id_to_label


class PlantVillageTorchDataset:
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        from PIL import Image
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"])
        return self.transform(image), label


class ImagePathTorchDataset:
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), int(label)



def load_training_data(args):
    if args.dataset == "plantvillage":
        splits, id_to_label = load_plantvillage_splits(seed=args.seed)
        dataset_class = PlantVillageTorchDataset
    elif args.dataset == "local-imagefolder":
        if args.data_dir is None:
            raise SystemExit("--data-dir est requis avec --dataset local-imagefolder")
        splits, id_to_label = load_local_imagefolder_splits(args.data_dir, args.seed)
        dataset_class = ImagePathTorchDataset
    elif args.dataset == "local-csv":
        if args.csv is None or args.data_dir is None:
            raise SystemExit("--csv et --data-dir sont requis avec --dataset local-csv")
        splits, id_to_label = load_local_csv_splits(
            args.csv,
            args.data_dir,
            args.image_column,
            args.label_column,
            args.split_column,
            args.seed,
        )
        dataset_class = ImagePathTorchDataset
    else:
        raise ValueError(f"Dataset inconnu: {args.dataset}")
    return splits, id_to_label, dataset_class


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    _require_training_dependencies()
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(
    architecture: str,
    num_classes: int,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> "nn.Module":
    _require_training_dependencies()

    class SimpleCNN(nn.Module):
        """CNN compact pour l'apprentissage pedagogique depuis zero."""

        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.35),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    if architecture == "simple_cnn":
        return SimpleCNN(num_classes)
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Architecture inconnue: {architecture}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    all_preds, all_labels = [], []
    running_loss = 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            outputs = model(images.to(device))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, class_names, output_path: Path):
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(10, len(class_names) * 0.35), max(8, len(class_names) * 0.30)))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediction")
    plt.ylabel("Classe reelle")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compute_class_weights(train_split, num_classes: int) -> "torch.Tensor":
    """Calcule les poids de classe inversement proportionnels aux frequences."""
    import torch
    counts = [0] * num_classes
    if hasattr(train_split, "column_names") and "label" in train_split.column_names:
        for lbl in train_split["label"]:
            counts[int(lbl)] += 1
    elif hasattr(train_split, "__iter__"):
        for item in train_split:
            if isinstance(item, dict) and "label" in item:
                counts[int(item["label"])] += 1
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                counts[int(item[1])] += 1

    total = sum(counts)
    if total == 0 or min(counts) == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = [total / (num_classes * max(c, 1)) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.mean()


def main():
    parser = argparse.ArgumentParser(description="Entrainement de diagnostic de maladies de plantes avec PyTorch.")
    parser.add_argument("--dataset", choices=["plantvillage", "local-imagefolder", "local-csv"], default="plantvillage")
    parser.add_argument("--data-dir", type=Path, help="Racine des images pour un dataset local.")
    parser.add_argument("--csv", type=Path, help="CSV local avec colonnes image/label, optionnellement split.")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--max-samples-per-split", type=int, help="Limite rapide pour tester le pipeline.")
    parser.add_argument("--architecture", choices=["simple_cnn", "resnet18", "mobilenet_v2"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-after-epoch", type=int, default=None, help="Degeler le backbone apres N epoques de prechauffage.")
    parser.add_argument("--balance-classes", action="store_true", help="Ponderer la fonction de perte pour corriger le desequilibre des classes.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Nombre d'epoques sans amelioration avant arret anticipe.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model"))
    args = parser.parse_args()

    _require_training_dependencies()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits, id_to_label, dataset_class = load_training_data(args)
    splits = limit_samples_per_split(splits, args.max_samples_per_split, args.seed)
    print_dataset_summary(splits, id_to_label)
    train_transform, eval_transform = build_transforms(args.image_size)
    loaders = {
        "train": DataLoader(
            dataset_class(splits["train"], train_transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        "validation": DataLoader(
            dataset_class(splits["validation"], eval_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        "test": DataLoader(
            dataset_class(splits["test"], eval_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, num_classes=len(id_to_label), freeze_backbone=args.freeze_backbone).to(device)

    if args.balance_classes:
        class_weights = compute_class_weights(splits["train"], len(id_to_label)).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Ponderation automatique des classes activee.")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and args.unfreeze_after_epoch and epoch == args.unfreeze_after_epoch + 1:
            print(f"[Epoch {epoch:03d}] Degel progressif du backbone pour affinage global (fine-tuning)...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, loaders["validation"], criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": args.architecture,
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(id_to_label),
                    "image_size": args.image_size,
                    "normalize_mean": NORMALIZE_MEAN,
                    "normalize_std": NORMALIZE_STD,
                    "freeze_backbone": args.freeze_backbone,
                },
                args.output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
                print(f"Arret anticipe (Early Stopping) declenche apres {epoch} epoques (patience: {args.early_stopping_patience}).")
                break

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model = build_model(
        checkpoint["architecture"],
        checkpoint["num_classes"],
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = evaluate(model, loaders["test"], device)
    class_names = [id_to_label[i] for i in range(len(id_to_label))]
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    test_acc = accuracy_score(y_true, y_pred)

    with open(args.output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, ensure_ascii=False, indent=2)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test_accuracy": test_acc, "classification_report": report}, f, indent=2)
    save_confusion_matrix(y_true, y_pred, class_names, args.output_dir / "confusion_matrix.png")

    print(f"Accuracy test: {test_acc:.4f}")
    print(f"Modele sauvegarde dans: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
