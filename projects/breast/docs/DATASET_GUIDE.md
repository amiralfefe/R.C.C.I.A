# BreakHis Dataset Guide

## Objectif

Ce guide explique comment preparer BreakHis pour R.C.C.I.A Breast.

Le projet cible une classification binaire :

- `benign`
- `malignant`

Rappel : ce projet est un demonstrateur educatif et portfolio. Il ne fournit pas de diagnostic medical.

## Dataset

Dataset cible : **BreakHis / Breast Cancer Histopathological Database**.

BreakHis contient des images histopathologiques de tumeurs mammaires avec plusieurs grossissements, typiquement :

- `40X`
- `100X`
- `200X`
- `400X`

Ces grossissements peuvent etre presents dans les chemins ou les noms de fichiers. Le script de preparation tente de les detecter et de les stocker dans `metadata.csv`.

## Structure Source Attendue

Le script accepte plusieurs structures tant que les chemins ou noms contiennent des indices de classe.

Exemples acceptables :

```text
C:\VSCODE\datasets\breakhis\
|-- benign\
|   `-- ...
`-- malignant\
    `-- ...
```

ou une structure BreakHis plus profonde :

```text
C:\VSCODE\datasets\breakhis\
`-- BreaKHis_v1\
    `-- histology_slides\
        `-- breast\
            |-- benign\
            `-- malignant\
```

Le script cherche aussi des indices dans les noms de fichiers BreakHis, par exemple `SOB_B_...` pour benign et `SOB_M_...` pour malignant.

## Preparation

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\prepare_breakhis_dataset.py --input C:\VSCODE\datasets\breakhis --output projects\breast\data\raw
```

Sortie attendue :

```text
projects/breast/data/raw/
|-- benign/
|-- malignant/
`-- metadata.csv
```

Le fichier `metadata.csv` contient au minimum :

- `class_name`
- `relative_path`
- `output_path`
- `source_path`
- `magnification`
- `patient_id`

## Split Classique

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\split_image_folder.py --input projects\breast\data\raw --output projects\breast\data\processed --val-ratio 0.15 --test-ratio 0.15
```

Sortie attendue :

```text
projects/breast/data/processed/
|-- train/
|   |-- benign/
|   `-- malignant/
|-- val/
|   |-- benign/
|   `-- malignant/
`-- test/
    |-- benign/
    `-- malignant/
```

## Split Patient-Aware

Si `metadata.csv` contient une colonne `patient_id` complete :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\split_image_folder.py --input projects\breast\data\raw --output projects\breast\data\processed --metadata projects\breast\data\raw\metadata.csv --patient-aware --val-ratio 0.15 --test-ratio 0.15
```

Si un `patient_id` manque, le script s'arrete au lieu de faire semblant de proteger contre la fuite de donnees.

## Donnees Non Versionnees

Ne pas commit :

- `projects/breast/data/raw/`
- `projects/breast/data/processed/`
- `projects/breast/outputs/`
- checkpoints `.pt`, `.pth`, `.ckpt`
- tokens Kaggle
- `.venv/`

Seuls les fichiers `.gitkeep` sont versionnes dans les dossiers de donnees et outputs.
