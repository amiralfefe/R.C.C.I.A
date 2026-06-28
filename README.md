# Cancer Cell Vision

Projet portfolio de computer vision pour classifier des images microscopiques de cellules ou tissus en classes simples, par exemple `normal` et `cancer`.

> Important : ce projet est un demonstrateur educatif IA/data. Il ne doit pas etre utilise pour diagnostiquer, traiter ou orienter une decision medicale.

## Objectif V1

Construire une premiere pipeline complete :

- chargement d'un dataset public organise en dossiers par classe ;
- entrainement d'un modele de classification par transfer learning ;
- evaluation avec accuracy, precision, recall, F1-score et matrice de confusion ;
- interface Streamlit pour tester une image ;
- visualisation Grad-CAM pour montrer les zones qui influencent la prediction.

## Etat actuel

La V1 reelle est validee sur le dataset public Kaggle **Leukemia Classification** :

- dataset telecharge et prepare localement ;
- split `train` / `val` / `test` realise ;
- entrainement CPU 5 epochs termine ;
- evaluation test terminee avec accuracy, precision, recall, F1-score et matrice de confusion ;
- prediction CLI testee sur une image du test set ;
- checkpoint local cree dans `outputs/best_model.pt` ;
- demo Streamlit testee localement avec le checkpoint reel.

## Dataset conseille

Pour commencer, le meilleur cadrage est un dataset de leucemie aigue lymphoblastique ou un dataset binaire `normal` / `leukemia_blast`.

Dataset V1 recommande : **Leukemia Classification** sur Kaggle.

Lien manuel : https://www.kaggle.com/datasets/andrewmvd/leukemia-classification

Structure attendue apres telechargement :

```text
data/
  raw/
    normal/
    leukemia_blast/
```

Apres preparation, le projet attend :

```text
data/
  processed/
    train/
      normal/
      leukemia_blast/
    val/
      normal/
      leukemia_blast/
    test/
      normal/
      leukemia_blast/
```

Le projet fonctionne aussi avec plus de deux classes si les dossiers sont nommes par classe.
Voir aussi [docs/DATASET_GUIDE.md](docs/DATASET_GUIDE.md) et [docs/REAL_DATASET_V1.md](docs/REAL_DATASET_V1.md).

## Installation

Sur Windows, tu peux suivre [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) ou lancer :

```bat
scripts\setup_windows.bat
```

Installation manuelle :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour installer aussi les outils de test :

```powershell
python -m pip install -r requirements-dev.txt
```

## Preparation des donnees

Si ton dataset telecharge a une structure differente, commence par normaliser les classes :

```powershell
.\.venv\Scripts\python.exe scripts\prepare_leukemia_dataset.py --source path\to\downloaded_dataset --output data\raw --overwrite
```

Si ton dataset brut est deja organise par classe, par exemple `data/raw/normal` et `data/raw/leukemia_blast`, cree les splits avec :

```powershell
.\.venv\Scripts\python.exe scripts\split_image_folder.py --input data\raw --output data\processed --val-ratio 0.15 --test-ratio 0.15 --overwrite
```

## Entrainement

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --epochs 5 --batch-size 16 --output-dir outputs
```

Le meilleur modele est sauvegarde dans `outputs/best_model.pt`.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.evaluate --data-dir data\processed --checkpoint outputs\best_model.pt --output-dir outputs\eval
```

Les sorties d'evaluation sont sauvegardees dans `outputs/eval/`.

Pour presenter le projet, ne mets pas seulement l'accuracy en avant. Regarde aussi precision, recall, F1-score et matrice de confusion. Le recall sur `leukemia_blast` est important dans l'analyse experimentale, sans jamais presenter le modele comme fiable medicalement.

## Prediction CLI

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.predict --checkpoint outputs\best_model.pt --image data\processed\test\normal\example.jpg --pretty
```

## Interface Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

L'app permet de charger une image, d'obtenir une prediction et d'afficher une carte Grad-CAM.

## Resultats experimentaux V1

Dataset utilise : **andrewmvd/leukemia-classification**.

Images preparees :

| Classe | Images |
| --- | ---: |
| `normal` | 3389 |
| `leukemia_blast` | 7272 |

Split utilise :

| Classe | Train | Val | Test |
| --- | ---: | ---: | ---: |
| `normal` | 2372 | 508 | 509 |
| `leukemia_blast` | 5090 | 1090 | 1092 |

Baseline :

- modele : transfer learning ResNet18 ;
- entrainement : CPU, 5 epochs, batch size 16 ;
- meilleure validation accuracy : `0.9168` ;
- test accuracy : `0.9169`.

Rapport de classification sur le test set :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `leukemia_blast` | 0.9419 | 0.9359 | 0.9389 | 1092 |
| `normal` | 0.8643 | 0.8762 | 0.8702 | 509 |
| `macro avg` | 0.9031 | 0.9061 | 0.9046 | 1601 |
| `weighted avg` | 0.9173 | 0.9169 | 0.9171 | 1601 |

Matrice de confusion test :

| Vraie classe / prediction | `leukemia_blast` | `normal` |
| --- | ---: | ---: |
| `leukemia_blast` | 1022 | 70 |
| `normal` | 63 | 446 |

La matrice de confusion image est generee localement dans `outputs/eval/test_confusion_matrix.png`.
Le dossier `outputs/` est volontairement ignore par Git.

Exemple de prediction CLI validee :

- image : `data/processed/test/leukemia_blast/leukemia_blast_000004.bmp` ;
- prediction : `leukemia_blast` ;
- confiance : `0.7151`.

Ces resultats sont experimentaux et servent a presenter une pipeline IA/data reproductible. Ils ne constituent pas une preuve de performance clinique.

## Comment reproduire l'experience

Configurer l'acces Kaggle localement hors du depot, puis telecharger le dataset :

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
New-Item -ItemType Directory -Force C:\VSCODE\datasets\leukemia-classification
.\.venv\Scripts\kaggle.exe datasets download -d andrewmvd/leukemia-classification -p C:\VSCODE\datasets\leukemia-classification --unzip
```

Preparer les dossiers `data/raw/normal` et `data/raw/leukemia_blast` :

```powershell
.\.venv\Scripts\python.exe scripts\prepare_leukemia_dataset.py --source C:\VSCODE\datasets\leukemia-classification\C-NMC_Leukemia\training_data --output data\raw --overwrite
```

Creer les splits :

```powershell
.\.venv\Scripts\python.exe scripts\split_image_folder.py --input data\raw --output data\processed --val-ratio 0.15 --test-ratio 0.15 --overwrite
```

Entrainer la baseline :

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --epochs 5 --batch-size 16 --output-dir outputs
```

Evaluer le checkpoint :

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.evaluate --data-dir data\processed --checkpoint outputs\best_model.pt --output-dir outputs\eval
```

Tester une prediction CLI :

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.predict --checkpoint outputs\best_model.pt --image data\processed\test\leukemia_blast\leukemia_blast_000004.bmp --pretty
```

Lancer la demo Streamlit :

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Limites

Ce projet depend fortement du dataset, de sa qualite, de son equilibre et du protocole de validation.

- Le dataset est public et ne remplace pas une validation medicale.
- Les classes sont desequilibrees : `leukemia_blast` contient plus d'images que `normal`.
- La baseline a ete entrainee seulement 5 epochs sur CPU.
- Les scores dependent du split local et du preprocessing.
- Aucune validation clinique, multi-centrique ou par specialistes n'est realisee dans ce projet portfolio.
- Le modele ne doit pas etre utilise pour diagnostiquer, traiter ou orienter une decision medicale.

## Next steps

- Ajouter des captures d'ecran de la demo Streamlit au portfolio.
- Ameliorer la presentation Grad-CAM dans l'app.
- Ajouter de l'augmentation de donnees controlee.
- Comparer ResNet18, EfficientNet et un CNN plus leger.
- Tester une strategie de ponderation des classes ou de sampling.
- Ajouter une page portfolio avec captures, resultats et architecture.
