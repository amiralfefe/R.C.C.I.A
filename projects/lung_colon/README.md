# Lung + Colon Vision

Projet portfolio IA/data de computer vision pour classifier des images histopathologiques lung/colon en cinq classes.

> Important : Lung + Colon Vision est un demonstrateur educatif. Il ne fournit pas de diagnostic medical, ne remplace pas un avis professionnel et ne doit jamais orienter une decision de sante.

Ce dossier est un sous-projet du monorepo R.C.C.I.A. Les commandes ci-dessous supposent d'etre place dans `projects/lung_colon` et d'utiliser le venv cree a la racine du repo.

## Objectif V1

Initialiser une baseline multi-classe propre sur le dataset public LC25000 :

- preparation d'un dataset public Kaggle ;
- normalisation des noms de classes ;
- split `train` / `val` / `test` ;
- entrainement PyTorch / Torchvision ;
- evaluation accuracy, precision, recall, F1-score et matrice de confusion ;
- prediction CLI ;
- demo Streamlit avec Grad-CAM.

## Classes

| Classe normalisee | Classe LC25000 habituelle |
| --- | --- |
| `colon_adenocarcinoma` | `colon_aca` |
| `colon_benign` | `colon_n` |
| `lung_adenocarcinoma` | `lung_aca` |
| `lung_benign` | `lung_n` |
| `lung_squamous_cell_carcinoma` | `lung_scc` |

## Stack Technique

- Python
- PyTorch / Torchvision
- OpenCV
- Pandas / NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM
- Pytest

## Structure

```text
projects/lung_colon/
|-- app.py
|-- rccia_lung_colon/
|-- scripts/
|-- tests/
|-- docs/
|-- data/
|   |-- raw/
|   `-- processed/
`-- outputs/
```

Les dossiers `data/` et `outputs/` restent locaux et ne doivent pas etre committes, sauf les fichiers `.gitkeep`.

## Installation

Depuis la racine du repo :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cd projects\lung_colon
```

## Dataset

Dataset cible : **LC25000 / Lung and Colon Cancer Histopathological Images**.

Commande Kaggle indicative :

```powershell
..\..\.venv\Scripts\kaggle.exe datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images -p C:\VSCODE\datasets\lc25000 --unzip
```

Preparer `data/raw` :

```powershell
..\..\.venv\Scripts\python.exe scripts\prepare_lc25000_dataset.py --source C:\VSCODE\datasets\lc25000 --output data\raw --overwrite
```

Creer les splits :

```powershell
..\..\.venv\Scripts\python.exe scripts\split_image_folder.py --input data\raw --output data\processed --val-ratio 0.15 --test-ratio 0.15 --overwrite
```

## Pipeline V1

Entrainer une baseline ResNet18 :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_lung_colon.train --data-dir data\processed --epochs 5 --batch-size 16 --output-dir outputs
```

Evaluer :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_lung_colon.evaluate --data-dir data\processed --checkpoint outputs\best_model.pt --output-dir outputs\eval
```

Predire une image :

```powershell
..\..\.venv\Scripts\python.exe -m rccia_lung_colon.predict --checkpoint outputs\best_model.pt --image data\processed\test\lung_adenocarcinoma\example.jpeg --pretty
```

Lancer Streamlit :

```powershell
..\..\.venv\Scripts\streamlit.exe run app.py
```

## Modeles Supportes

- `resnet18` par defaut ;
- `mobilenet_v3_small` ;
- `efficientnet_b0`.

## Tests

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limites

- Demonstrateur educatif, pas outil medical.
- Dataset public, sans validation clinique independante.
- Baseline V1 non entrainee tant que le dataset reel n'est pas prepare.
- Scores futurs dependront du split, du preprocessing et du protocole d'entrainement.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
