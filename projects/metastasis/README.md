# R.C.C.I.A Metastasis

Projet portfolio IA/data de computer vision pour classifier des patches histopathologiques
en deux classes : `non_metastatic` et `metastatic`.

> Important : R.C.C.I.A Metastasis est un demonstrateur educatif. Il ne fournit pas de
> diagnostic medical, ne remplace pas un avis professionnel et ne doit jamais orienter
> une decision de sante.

Ce dossier est un sous-projet du monorepo R.C.C.I.A. Les commandes ci-dessous supposent
d'etre place a la racine du repo.

## Statut

V1 real dataset run :

- structure du sous-projet creee ;
- package Python `rccia_metastasis` ;
- pipeline PyTorch prepare ;
- support ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- evaluation avec accuracy, precision, recall, F1, ROC-AUC et PR-AUC ;
- scripts de preparation PCam-like et split classique ;
- conversion HDF5 PCam vers ImageFolder preparee si `h5py` est installe ;
- app Streamlit V1 avec Grad-CAM si checkpoint disponible ;
- tests smoke CPU rapides.
- dataset PCam validation HDF5 telecharge et prepare localement ;
- baseline ResNet18 pre-entrainee entrainee sur un subset reel equilibre.

Le run reel V1 utilise un subset equilibre de la validation PCam pour garder un temps CPU
raisonnable. Les donnees, outputs et checkpoints restent locaux et ne sont pas versionnes.

## Objectif V1

Preparer une baseline propre pour un dataset de detection de metastases sur patches
histopathologiques, typiquement PCam / PatchCamelyon ou equivalent :

- classification binaire `non_metastatic` vs `metastatic` ;
- preparation des images vers une structure `ImageFolder` ;
- split classique `train` / `val` / `test` ;
- entrainement PyTorch / Torchvision ;
- evaluation avec metriques de classification et courbes ROC / precision-recall ;
- prediction CLI ;
- demo Streamlit avec probabilites, seuil `metastatic` et Grad-CAM.

## Dataset Prevu

Dataset cible probable : **PatchCamelyon / PCam** ou dataset equivalent de patches
histopathologiques annote en presence/absence de metastases.

Classes normalisees :

| Classe | Description |
| --- | --- |
| `non_metastatic` | patch sans metastase detectee dans le label du dataset |
| `metastatic` | patch annote positif / metastatic |

Le script V1 supporte :

- les datasets deja exportes en images classees par dossiers ou avec labels detectables
  dans les chemins/noms ;
- les fichiers PCam HDF5 non compresses via `--x-h5` et `--y-h5`.

Les archives officielles `.h5.gz` doivent etre decompressees avant conversion.

## Resultats V1 Reels

Dataset utilise : Kaggle `tyson04/pcam-validate`, base sur PCam / PatchCamelyon.

Telechargement :

| Fichier | Taille |
| --- | ---: |
| `pcam-validate.zip` | 766 MiB |
| `camelyonpatch_level_2_split_valid_x.h5` | 906 040 528 bytes |
| `camelyonpatch_level_2_split_valid_y.h5` | 34 816 bytes |

Structure HDF5 detectee :

| Fichier | Dataset | Shape | Type |
| --- | --- | --- | --- |
| `camelyonpatch_level_2_split_valid_x.h5` | `x` | `(32768, 96, 96, 3)` | `uint8` |
| `camelyonpatch_level_2_split_valid_y.h5` | `y` | `(32768, 1, 1, 1)` | `uint8` |

Labels dans le fichier complet :

| Label | Images |
| --- | ---: |
| `0` / `non_metastatic` | 16 399 |
| `1` / `metastatic` | 16 369 |

Subset converti pour la V1 CPU :

| Classe | Images |
| --- | ---: |
| `non_metastatic` | 2 500 |
| `metastatic` | 2 500 |
| **Total** | **5 000** |

Commande de preparation lancee :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\prepare_pcam_dataset.py --x-h5 C:\VSCODE\datasets\pcam\camelyonpatch_level_2_split_valid_x.h5 --y-h5 C:\VSCODE\datasets\pcam\camelyonpatch_level_2_split_valid_y.h5 --split-name valid --output projects\metastasis\data\raw --max-per-class 2500
```

Split :

| Split | `non_metastatic` | `metastatic` | Total |
| --- | ---: | ---: | ---: |
| Train | 1 750 | 1 750 | 3 500 |
| Val | 375 | 375 | 750 |
| Test | 375 | 375 | 750 |

Baseline :

- modele : ResNet18 pre-entraine ;
- epochs : 3 ;
- batch size : 32 ;
- image size : 96 ;
- dataset : subset reel PCam validation, equilibre.

Historique validation :

| Epoch | Train accuracy | Val accuracy |
| --- | ---: | ---: |
| 1 | 0.8111 | 0.8813 |
| 2 | 0.8820 | 0.9133 |
| 3 | 0.8957 | 0.9080 |

Best val accuracy : **0.9133**.

Evaluation test :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `metastatic` | 0.9040 | 0.9040 | 0.9040 | 375 |
| `non_metastatic` | 0.9040 | 0.9040 | 0.9040 | 375 |
| **Macro avg** | **0.9040** | **0.9040** | **0.9040** | **750** |

Metrices globales :

| Metrique | Valeur |
| --- | ---: |
| Accuracy | 0.9040 |
| ROC-AUC | 0.9598 |
| PR-AUC | 0.9616 |

Matrice de confusion, ordre des classes : `metastatic`, `non_metastatic`.

| Vrai \ Pred | `metastatic` | `non_metastatic` |
| --- | ---: | ---: |
| `metastatic` | 339 | 36 |
| `non_metastatic` | 36 | 339 |

Prediction CLI :

- image `non_metastatic` testee : OK, prediction `non_metastatic`, confiance 0.9112 ;
- image `metastatic` testee : OK, prediction `metastatic`, confiance 0.9998.

Streamlit :

- l'application fonctionne sans checkpoint ;
- le test headless avec checkpoint n'a pas pu etre relance dans cette session a cause
  d'une limite d'escalade d'outil.

Lecture portfolio : ce run valide le pipeline Metastasis sur un vrai subset PCam HDF5
converti localement. Les resultats restent experimentaux, sur subset validation et sans
validation clinique externe.

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

## Pipeline Prevu

1. Telecharger localement un dataset type PCam.
2. Convertir ou organiser les images en `non_metastatic` / `metastatic`.
3. Preparer `data/raw` avec `scripts/prepare_pcam_dataset.py`.
4. Creer `data/processed` avec `scripts/split_image_folder.py`.
5. Entrainer une baseline ResNet18.
6. Evaluer accuracy, precision, recall, F1, ROC-AUC, PR-AUC et confusion matrix.
7. Tester une prediction CLI.
8. Lancer la demo Streamlit avec Grad-CAM.
9. Ajouter plus tard une analyse de seuil de decision.

## Preparation Dataset

Commande indicative :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\prepare_pcam_dataset.py --input C:\VSCODE\datasets\pcam --output projects\metastasis\data\raw
```

Conversion HDF5 PCam officielle, apres decompression des fichiers `.h5.gz` :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\prepare_pcam_dataset.py --x-h5 C:\VSCODE\datasets\pcam\camelyonpatch_level_2_split_train_x.h5 --y-h5 C:\VSCODE\datasets\pcam\camelyonpatch_level_2_split_train_y.h5 --split-name train --output projects\metastasis\data\raw --max-per-class 5000
```

`--max-per-class` est optionnel mais recommande pour un premier run CPU raisonnable.

Structure raw attendue :

```text
projects/metastasis/data/raw/
|-- non_metastatic/
|-- metastatic/
`-- metadata.csv
```

## Split

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\split_image_folder.py --input projects\metastasis\data\raw --output projects\metastasis\data\processed --val-ratio 0.15 --test-ratio 0.15
```

## Entrainement

```powershell
.\.venv\Scripts\python.exe -m projects.metastasis.rccia_metastasis.train --data-dir projects\metastasis\data\processed --model resnet18 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\metastasis\outputs
```

Modeles supportes :

- `resnet18` par defaut ;
- `mobilenet_v3_small` ;
- `efficientnet_b0`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m projects.metastasis.rccia_metastasis.evaluate --data-dir projects\metastasis\data\processed --checkpoint projects\metastasis\outputs\best_model.pt --model resnet18 --output-dir projects\metastasis\outputs\eval
```

Rapports generes localement :

- `test_metrics.json`
- `test_classification_report.json`
- `test_classification_report.csv`
- `test_predictions.csv`
- `test_confusion_matrix.png`
- `test_roc_curve.png`
- `test_precision_recall_curve.png`

## Prediction CLI

```powershell
.\.venv\Scripts\python.exe -m projects.metastasis.rccia_metastasis.predict --checkpoint projects\metastasis\outputs\best_model.pt --image projects\metastasis\data\processed\test\metastatic\example.png --pretty
```

## Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run projects\metastasis\app.py
```

L'application ne plante pas si le checkpoint est absent : elle affiche les commandes a
lancer pour preparer et entrainer un modele local.

## Point Fort Metriques

Metastasis est pense comme un projet benchmark : l'accuracy ne suffit pas toujours, donc
la V1 prepare aussi ROC-AUC, PR-AUC, courbes ROC / precision-recall et analyse future
des seuils.

Voir [docs/METRICS_GUIDE.md](docs/METRICS_GUIDE.md).

## Docs

- [Dataset guide](docs/DATASET_GUIDE.md)
- [Metrics guide](docs/METRICS_GUIDE.md)

## Tests

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Limites

- Demonstrateur educatif, pas outil medical.
- Dataset reel non encore execute dans cette phase.
- Kaggle non authentifie avec le token local actuel pendant cette tentative.
- Les fichiers PCam officiels sont volumineux et distribues en `.h5.gz`.
- Pas de validation clinique, pas de certification, pas d'usage diagnostic.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
