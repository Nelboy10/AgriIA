# Diagnostic IA des maladies des plantes

Ce dossier contient uniquement le modele IA, son entrainement, un module de conseils climatiques et un module d'estimation de rendement. Il ne contient pas d'application, de bot ou d'interface.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrainement du modele image

Option rapide et recommandee, par transfert d'apprentissage avec ResNet18:

```bash
python train_plant_disease.py --architecture resnet18 --epochs 10 --batch-size 32 --output-dir outputs/resnet18
```

Option pedagogique, CNN simple entraine depuis zero:

```bash
python train_plant_disease.py --architecture simple_cnn --epochs 25 --batch-size 32 --output-dir outputs/simple_cnn
```

Par defaut, le script charge PlantVillage via Hugging Face avec `load_dataset("mohanty/PlantVillage", "color")`. Il peut aussi entrainer sur un dataset local.

Format dossier local:

```text
data/icassava/
  healthy/
  cassava_mosaic_disease/
  cassava_bacterial_blight/
```

Avant entrainement, verifier le dataset:

```bash
python audit_dataset.py --format imagefolder --data-dir data/icassava
```

```bash
python train_plant_disease.py --dataset local-imagefolder --data-dir data/icassava --architecture resnet18 --epochs 10 --output-dir outputs/icassava_resnet18
```

Test rapide avant entrainement complet:

```bash
python train_plant_disease.py --dataset local-imagefolder --data-dir data/icassava --architecture mobilenet_v2 --epochs 1 --batch-size 8 --max-samples-per-split 200 --output-dir outputs/icassava_quick
```

Format CSV local:

```csv
image,label,split
train/img_001.jpg,healthy,train
train/img_002.jpg,cassava_mosaic_disease,train
```

```bash
python audit_dataset.py --format csv --data-dir data/icassava --csv data/icassava/labels.csv --split-column split
python train_plant_disease.py --dataset local-csv --data-dir data/icassava --csv data/icassava/labels.csv --split-column split --architecture resnet18 --output-dir outputs/icassava_resnet18
```

Pour KaraAgro Maize/Cocoa avec bounding boxes, recadrer d'abord les zones annotees:

```bash
python prepare_karaagro_crops.py --labelmap-csv data/karaagro/train/labelmap.csv --image-dir data/karaagro/train --output-dir data/karaagro_crops
python audit_dataset.py --format imagefolder --data-dir data/karaagro_crops
python train_plant_disease.py --dataset local-imagefolder --data-dir data/karaagro_crops --architecture mobilenet_v2 --epochs 1 --batch-size 8 --max-samples-per-split 200 --output-dir outputs/karaagro_quick
python train_plant_disease.py --dataset local-imagefolder --data-dir data/karaagro_crops --architecture resnet18 --output-dir outputs/karaagro_resnet18
```

Le script cree les splits train/validation/test, applique les augmentations, entraine le modele, calcule les metriques et sauvegarde:

- `best_model.pt`
- `label_map.json`
- `metrics.json`
- `confusion_matrix.png`

## Prediction sur une nouvelle photo

```bash
python predict_photo.py --checkpoint outputs/resnet18/best_model.pt --labels outputs/resnet18/label_map.json --image chemin/vers/feuille.jpg
```

## Conseils climatiques

```bash
python climate_advice.py --crop maize --lat 6.37 --lon 2.43
```

Par defaut, le module utilise Open-Meteo, gratuit pour prototypage non commercial sans cle API.

## Estimation de rendement

```bash
python yield_estimation.py --crop maize --climate-factor 0.88 --health-factor 0.82
```

## Dossier d'etude

Les hypotheses, tableaux de calcul et limites sont dans [docs/feasibility_study.md](docs/feasibility_study.md).

## Structure du projet

```text
├── audit_dataset.py          # Audit et analyse statistique des jeux de donnees
├── climate_advice.py         # Moteur de regles agronomiques et meteo (Open-Meteo)
├── docs/
│   └── feasibility_study.md  # Etude detaillee de faisabilite et chiffrages
├── outputs/                  # Metriques, matrices de confusion et modeles
├── predict_photo.py          # Inference et prediction sur image
├── prepare_karaagro_crops.py # Recadrage bounding-boxes (KaraAgro) vers classification
├── requirements.txt          # Dependances Python
├── train_plant_disease.py    # Pipeline complet d'entrainement PyTorch
├── yield_estimation.py       # Estimation des rendements par facteurs de stress
└── LICENSE                   # Licence MIT
```

## Licence

Ce projet est distribue sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus d'informations.

