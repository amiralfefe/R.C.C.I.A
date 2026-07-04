# R.C.C.I.A Breast

Projet portfolio IA/data de computer vision pour classifier des images histopathologiques du cancer du sein en deux classes : `benign` et `malignant`.

> Important : R.C.C.I.A Breast est un demonstrateur educatif. Il ne fournit pas de diagnostic medical, ne remplace pas un avis professionnel et ne doit jamais orienter une decision de sante.

Ce dossier est un sous-projet du monorepo R.C.C.I.A. Les commandes ci-dessous supposent d'etre place a la racine du repo.

## Statut

V1 real dataset run :

- structure du sous-projet creee ;
- package Python `rccia_breast` ;
- pipeline PyTorch prepare ;
- scripts BreakHis et split patient-aware ;
- app Streamlit V1 ;
- tests smoke CPU rapides ;
- dataset BreakHis Kaggle prepare localement ;
- split patient-aware realise avec 81 patients detectes ;
- baseline ResNet18 pre-entrainee entrainee et evaluee.

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

## Dataset

Dataset cible : **BreakHis / Breast Cancer Histopathological Database**.

Classes normalisees :

| Classe | Description |
| --- | --- |
| `benign` | images histopathologiques benignes |
| `malignant` | images histopathologiques malignes |

Point de vigilance : BreakHis contient plusieurs grossissements, typiquement `40X`, `100X`, `200X` et `400X`. Le projet conserve ces informations dans `metadata.csv` quand elles sont detectables.

## Resultats V1 Reels

Dataset utilise : Kaggle `ambarish/breakhis`, extrait dans `C:\VSCODE\datasets\breakhis`.

Structure detectee :

```text
C:\VSCODE\datasets\breakhis\
|-- Folds.csv
`-- BreaKHis_v1/
    `-- BreaKHis_v1/
        `-- histology_slides/
            `-- breast/
                |-- benign/
                `-- malignant/
```

Images preparees dans `projects/breast/data/raw` :

| Classe | Images |
| --- | ---: |
| `benign` | 2 480 |
| `malignant` | 5 429 |
| **Total** | **7 909** |

Metadata detectee :

| Champ | Resultat |
| --- | ---: |
| Patients detectes | 81 |
| `40X` | 1 995 |
| `100X` | 2 081 |
| `200X` | 2 013 |
| `400X` | 1 820 |

Split patient-aware :

| Split | Benign | Malignant | Total | Patients |
| --- | ---: | ---: | ---: | ---: |
| Train | 1 633 | 3 520 | 5 153 | 55 |
| Val | 339 | 936 | 1 275 | 11 |
| Test | 508 | 973 | 1 481 | 15 |

Verification leakage : aucun `patient_id` partage entre `train`, `val` et `test`.

Baseline :

- modele : ResNet18 pre-entraine ;
- epochs : 3 ;
- batch size : 16 ;
- image size : 224 ;
- split : patient-aware.

Historique validation :

| Epoch | Train accuracy | Val accuracy |
| --- | ---: | ---: |
| 1 | 0.8830 | 0.7451 |
| 2 | 0.9280 | 0.8275 |
| 3 | 0.9488 | 0.8533 |

Evaluation test :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `benign` | 0.8860 | 0.7953 | 0.8382 | 508 |
| `malignant` | 0.8985 | 0.9466 | 0.9219 | 973 |
| **Macro avg** | **0.8923** | **0.8709** | **0.8800** | **1 481** |

Accuracy test : **0.8947**.

Confusions test :

| Vrai label | Prediction | Count |
| --- | --- | ---: |
| `benign` | `malignant` | 104 |
| `malignant` | `benign` | 52 |

Analyse par grossissement :

| Grossissement | Images test | Correctes | Accuracy |
| --- | ---: | ---: | ---: |
| `40X` | 378 | 332 | 0.8783 |
| `100X` | 397 | 354 | 0.8917 |
| `200X` | 375 | 348 | 0.9280 |
| `400X` | 331 | 291 | 0.8792 |

Prediction CLI :

- image `benign` testee : commande OK, prediction `benign`, confiance 0.9895 ;
- image `malignant` testee : commande OK, prediction `benign`, confiance 0.9988, donc exemple d'erreur a analyser dans une prochaine phase.

Streamlit : test local HTTP 200 OK avec `projects/breast/app.py`.

Lecture portfolio : le score est obtenu avec un split patient-aware, donc plus strict qu'un split random image-level. Les resultats restent experimentaux sur dataset public BreakHis et ne constituent pas une validation clinique.

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
.\.venv\Scripts\python.exe -m projects.breast.rccia_breast.train --data-dir projects\breast\data\processed --model resnet18 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\breast\outputs
```

Modeles supportes :

- `resnet18` par defaut ;
- `mobilenet_v3_small` ;
- `efficientnet_b0`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m projects.breast.rccia_breast.evaluate --data-dir projects\breast\data\processed --checkpoint projects\breast\outputs\best_model.pt --model resnet18 --output-dir projects\breast\outputs\eval
```

## Prediction CLI

```powershell
.\.venv\Scripts\python.exe -m projects.breast.rccia_breast.predict --checkpoint projects\breast\outputs\best_model.pt --image projects\breast\data\processed\test\benign\example.png --pretty
```

## Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run projects\breast\app.py
```

L'application ne plante pas si le checkpoint est absent : elle affiche les commandes a lancer pour entrainer un modele local.

## Apercu de l'application Streamlit

Accueil avec le modele local charge, les classes disponibles et le disclaimer medical :

![Streamlit home](docs/assets/streamlit-home.png)

Prediction `benign` sur une image du test set BreakHis :

![Prediction benign](docs/assets/prediction-benign.png)

Prediction `malignant` sur une image du test set BreakHis :

![Prediction malignant](docs/assets/prediction-malignant.png)

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
- Les resultats V1 viennent d'un split patient-aware local, pas d'une cohorte clinique externe.
- Les scores dependent du split, du niveau de grossissement, du preprocessing et du protocole d'entrainement.
- Le split patient-aware depend de la disponibilite et de la qualite des identifiants patients.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
