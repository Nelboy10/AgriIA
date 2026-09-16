from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _require_dependencies():
    global Image

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dependance manquante: Pillow. Installez les dependances avec: "
            "pip install -r requirements.txt"
        ) from exc


def sanitize_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return normalized.strip("_") or "unknown"


def find_column(fieldnames, candidates):
    lowered = {name.strip().lower(): name for name in fieldnames or []}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def build_image_index(image_dir: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {
        path.name: path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }


def clamp_box(xmin, ymin, xmax, ymax, width, height):
    left = max(0, min(width - 1, int(round(float(xmin)))))
    top = max(0, min(height - 1, int(round(float(ymin)))))
    right = max(left + 1, min(width, int(round(float(xmax)))))
    bottom = max(top + 1, min(height, int(round(float(ymax)))))
    return left, top, right, bottom


def export_crops(labelmap_csv: Path, image_dir: Path, output_dir: Path, min_size: int):
    _require_dependencies()
    image_index = build_image_index(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    with open(labelmap_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        image_col = find_column(reader.fieldnames, ["Image id", "image_id", "filename", "image", "file"])
        label_col = find_column(reader.fieldnames, ["Actual Label", "label", "class", "class_name"])
        xmin_col = find_column(reader.fieldnames, ["xmin", "x_min", "left"])
        ymin_col = find_column(reader.fieldnames, ["ymin", "y_min", "top"])
        xmax_col = find_column(reader.fieldnames, ["xmax", "x_max", "right"])
        ymax_col = find_column(reader.fieldnames, ["ymax", "y_max", "bottom"])
        required = [image_col, label_col, xmin_col, ymin_col, xmax_col, ymax_col]
        if any(column is None for column in required):
            raise ValueError(
                "CSV bounding-box incomplet. Colonnes attendues: image, label, "
                "xmin, ymin, xmax, ymax."
            )

        for index, row in enumerate(reader, start=1):
            image_name = Path(row[image_col]).name
            image_path = image_index.get(image_name)
            if image_path is None:
                skipped += 1
                continue

            label_dir = output_dir / sanitize_label(row[label_col])
            label_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                left, top, right, bottom = clamp_box(
                    row[xmin_col],
                    row[ymin_col],
                    row[xmax_col],
                    row[ymax_col],
                    image.width,
                    image.height,
                )
                if right - left < min_size or bottom - top < min_size:
                    skipped += 1
                    continue
                crop = image.crop((left, top, right, bottom))
                crop_name = f"{image_path.stem}_{index:06d}.jpg"
                crop.save(label_dir / crop_name, quality=92)
                exported += 1

    return exported, skipped


def main():
    parser = argparse.ArgumentParser(description="Recadre les bounding boxes KaraAgro en dataset de classification.")
    parser.add_argument("--labelmap-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-size", type=int, default=24)
    args = parser.parse_args()

    exported, skipped = export_crops(args.labelmap_csv, args.image_dir, args.output_dir, args.min_size)
    print(f"Crops exportes: {exported}")
    print(f"Elements ignores: {skipped}")
    print(f"Dataset pret: {args.output_dir}")


if __name__ == "__main__":
    main()
