from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def count_imagefolder(data_dir: Path):
    split_names = {"train", "validation", "val", "test"}
    split_dirs = {
        path.name.lower(): path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.lower() in split_names
    }
    counts = defaultdict(Counter)

    if "train" in split_dirs or "test" in split_dirs or "validation" in split_dirs or "val" in split_dirs:
        for split_name, split_dir in split_dirs.items():
            normalized_split = "validation" if split_name == "val" else split_name
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                image_count = sum(1 for path in class_dir.rglob("*") if path.is_file() and is_image_file(path))
                counts[normalized_split][class_dir.name] += image_count
    else:
        for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            image_count = sum(1 for path in class_dir.rglob("*") if path.is_file() and is_image_file(path))
            counts["all"][class_dir.name] += image_count

    return counts


def count_csv(csv_path: Path, image_root: Path, image_column: str, label_column: str, split_column: str | None):
    counts = defaultdict(Counter)
    missing_images = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {image_column, label_column}
        if split_column:
            required.add(split_column)
        missing_columns = required - fieldnames
        if missing_columns:
            raise ValueError(f"Colonnes absentes du CSV: {', '.join(sorted(missing_columns))}")

        for row in reader:
            label = str(row[label_column]).strip()
            if not label:
                continue
            split_name = "all"
            if split_column:
                split_name = str(row[split_column]).strip().lower() or "all"
                if split_name == "val":
                    split_name = "validation"
            counts[split_name][label] += 1
            if not (image_root / row[image_column]).exists():
                missing_images += 1

    return counts, missing_images


def print_report(counts, missing_images=0):
    total = 0
    class_totals = Counter()
    for split_name in sorted(counts):
        split_total = sum(counts[split_name].values())
        total += split_total
        print(f"\n[{split_name}] {split_total} images")
        for label, count in counts[split_name].most_common():
            class_totals[label] += count
            print(f"  {label}: {count}")

    print(f"\nTotal: {total} images")
    print(f"Classes: {len(class_totals)}")
    if missing_images:
        print(f"Images introuvables: {missing_images}")

    if not class_totals:
        print("Statut: dataset vide ou structure non reconnue.")
        return

    smallest = min(class_totals.values())
    largest = max(class_totals.values())
    imbalance_ratio = largest / max(smallest, 1)
    print(f"Classe minimale: {smallest} images")
    print(f"Classe maximale: {largest} images")
    print(f"Ratio desequilibre: {imbalance_ratio:.1f}x")

    warnings = []
    if len(class_totals) < 2:
        warnings.append("au moins 2 classes sont necessaires")
    if smallest < 20:
        warnings.append("certaines classes ont moins de 20 images")
    if imbalance_ratio > 10:
        warnings.append("fort desequilibre entre classes")
    if "all" in counts:
        warnings.append("aucun split train/validation/test detecte; le script d'entrainement creera un split automatique")
    if missing_images:
        warnings.append("des chemins image du CSV sont introuvables")

    if warnings:
        print("Statut: a corriger avant entrainement serieux.")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Statut: pret pour un premier entrainement.")


def main():
    parser = argparse.ArgumentParser(description="Audite un dataset image local avant entrainement.")
    parser.add_argument("--format", choices=["imagefolder", "csv"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column")
    args = parser.parse_args()

    if args.format == "imagefolder":
        print_report(count_imagefolder(args.data_dir))
    else:
        if args.csv is None:
            raise SystemExit("--csv est requis avec --format csv")
        counts, missing_images = count_csv(
            args.csv,
            args.data_dir,
            args.image_column,
            args.label_column,
            args.split_column,
        )
        print_report(counts, missing_images)


if __name__ == "__main__":
    main()
