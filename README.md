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

Le projet est pret pour une premiere baseline reelle :

- pipeline end-to-end valide avec un mini dataset synthetique ;
- commandes CLI disponibles pour split, train, evaluate et predict ;
- interface Streamlit robuste quand le modele ou l'image manque ;
- workflow dataset reel documente dans [docs/REAL_DATASET_V1.md](docs/REAL_DATASET_V1.md).

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

## Resultats V1

Les resultats reels ne sont pas encore renseignes tant que le dataset public n'a pas ete telecharge et entraine localement.

Apres la premiere baseline, documenter :

- dataset utilise ;
- nombre d'images par classe et par split ;
- accuracy, precision, recall, F1-score ;
- matrice de confusion ;
- limites observees.

## Roadmap

- V1 : classification binaire sur cellules de leucemie ou tissu benin/malin.
- V2 : comparaison de plusieurs architectures, gestion du desequilibre de classes et augmentation avancee.
- V3 : segmentation de noyaux cellulaires avec annotations pixel-level.

## Limites

Ce projet depend fortement du dataset, de sa qualite, de son equilibre et du protocole de validation. Les scores ne representent pas une performance clinique et ne doivent pas etre interpretes comme une validation medicale.
