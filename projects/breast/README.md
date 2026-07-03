# R.C.C.I.A Breast

Projet portfolio IA/data de computer vision pour classifier des images histopathologiques du cancer du sein en deux classes : `benign` et `malignant`.

> Important : R.C.C.I.A Breast est un demonstrateur educatif. Il ne fournit pas de diagnostic medical, ne remplace pas un avis professionnel et ne doit jamais orienter une decision de sante.

Ce dossier est un sous-projet du monorepo R.C.C.I.A. Les commandes ci-dessous supposent d'etre place a la racine du repo.

## Statut

V1 initialisation :

- structure du sous-projet creee ;
- package Python `rccia_breast` ;
- pipeline PyTorch prepare ;
- scripts BreakHis et split patient-aware ;
- app Streamlit V1 ;
- tests smoke CPU rapides ;
- aucune donnee reelle telechargee dans cette phase.

## Objectif V1

Preparer une baseline propre pour le dataset public BreakHis :

- classification binaire `benign` vs `malignant` ;
- preparation des images vers une structure `ImageFolder` ;
- extraction de grossissement `40X`, `100X`, `200X`, `400X` si possible ;
- extraction de `patient_id` si lisible dans les chemins ou noms de fichiers ;
- split classique ou patient-aware ;
- entrainement PyTorch / Torchvision ;
- evaluation accuracy, precision, recall, F1-score et matrice de confusion ;
- prediction CLI ;
- demo Streamlit avec Grad-CAM.

## Dataset Prevu

Dataset cible : **BreakHis / Breast Cancer Histopathological Database**.

Classes normalisees :

| Classe | Description |
| --- | --- |
| `benign` | images histopathologiques benignes |
| `malignant` | images histopathologiques malignes |

Point de vigilance : BreakHis contient plusieurs grossissements, typiquement `40X`, `100X`, `200X` et `400X`. Le projet conserve ces informations dans `metadata.csv` quand elles sont detectables.

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
projects/breast/
|-- README.md
|-- app.py
|-- rccia_breast/
|-- scripts/
|-- tests/
|-- docs/
|-- data/
|   |-- raw/
|   `-- processed/
`-- outputs/
```

Les dossiers `data/` et `outputs/` restent locaux et ne doivent pas etre committes, sauf les fichiers `.gitkeep`.

## Preparation Dataset

Commande indicative :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\prepare_breakhis_dataset.py --input C:\VSCODE\datasets\breakhis --output projects\breast\data\raw
```

Le script copie les images vers :

```text
projects/breast/data/raw/
|-- benign/
|-- malignant/
`-- metadata.csv
```

Il affiche :

- nombre d'images `benign` ;
- nombre d'images `malignant` ;
- grossissements detectes ;
- nombre de patients detectes si possible ;
- avertissement si `patient_id` n'est pas detecte.

## Split

Split classique :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\split_image_folder.py --input projects\breast\data\raw --output projects\breast\data\processed --val-ratio 0.15 --test-ratio 0.15
```

Split patient-aware si `metadata.csv` contient `patient_id` :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\split_image_folder.py --input projects\breast\data\raw --output projects\breast\data\processed --metadata projects\breast\data\raw\metadata.csv --patient-aware --val-ratio 0.15 --test-ratio 0.15
```

Si le split patient-aware est demande mais impossible, le script echoue avec un message clair.

## Entrainement

```powershell
.\.venv\Scripts\python.exe -m rccia_breast.train --data-dir projects\breast\data\processed --model resnet18 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\breast\outputs
```

Modeles supportes :

- `resnet18` par defaut ;
- `mobilenet_v3_small` ;
- `efficientnet_b0`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m rccia_breast.evaluate --data-dir projects\breast\data\processed --checkpoint projects\breast\outputs\best_model.pt --model resnet18 --output-dir projects\breast\outputs\eval
```

## Prediction CLI

```powershell
.\.venv\Scripts\python.exe -m rccia_breast.predict --checkpoint projects\breast\outputs\best_model.pt --image projects\breast\data\processed\test\benign\example.png --pretty
```

## Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run projects\breast\app.py
```

L'application ne plante pas si le checkpoint est absent : elle affiche les commandes a lancer pour entrainer un modele local.

## Patient-Aware Split

Le point fort attendu de Breast est de limiter le risque de fuite de donnees entre train, validation et test. Si plusieurs images du meme patient apparaissent dans plusieurs splits, les scores peuvent etre trop optimistes.

Le script de preparation tente donc d'extraire un `patient_id` depuis les noms de fichiers BreakHis ou depuis des patterns de type `patient`, `case` ou `pt` dans les chemins. Voir [docs/PATIENT_AWARE_SPLIT.md](docs/PATIENT_AWARE_SPLIT.md).

## Docs

- [Dataset guide](docs/DATASET_GUIDE.md)
- [Patient-aware split](docs/PATIENT_AWARE_SPLIT.md)

## Tests

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limites

- Demonstrateur educatif, pas outil medical.
- Dataset public, sans validation clinique independante.
- V1 initialisation uniquement : aucun resultat reel BreakHis n'est encore documente.
- Les futurs scores dependront du split, du niveau de grossissement, du preprocessing et du protocole d'entrainement.
- Le split patient-aware depend de la disponibilite et de la qualite des identifiants patients.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
