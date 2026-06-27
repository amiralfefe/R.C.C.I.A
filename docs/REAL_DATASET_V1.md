# Real Dataset V1

Cette V1 utilise une classification experimentale binaire :

- `normal`
- `leukemia_blast`

Important : ce projet reste un demonstrateur IA/data pour portfolio. Il ne fournit aucune aide au diagnostic et ne doit pas etre utilise dans un contexte medical reel.

## Dataset recommande

Dataset conseille : **Leukemia Classification** sur Kaggle, base sur le challenge C-NMC de cellules sanguines.

Lien manuel : https://www.kaggle.com/datasets/andrewmvd/leukemia-classification

Kaggle peut demander un compte et/ou une authentification. Ne commit pas l'archive telechargee, ni les images extraites.

## Structure cible

Avant split, le projet attend :

```text
data/
  raw/
    normal/
    leukemia_blast/
```

Apres split :

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

## Preparation depuis un dossier telecharge

Si le dataset telecharge contient des dossiers faciles a identifier comme `Normal`, `Leukemia`, `hem`, `all` ou `blast`, lance :

```powershell
.\.venv\Scripts\python.exe scripts\prepare_leukemia_dataset.py --source path\to\downloaded_dataset --output data\raw --overwrite
```

Si la detection automatique ne trouve pas les bons dossiers, indique les chemins explicitement. Les chemins peuvent etre absolus ou relatifs au dossier `--source`.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_leukemia_dataset.py --source path\to\downloaded_dataset --normal-dir path\to\normal_folder --blast-dir path\to\blast_folder --output data\raw --overwrite
```

Pour un premier essai rapide, cree un subset de developpement :

```powershell
.\.venv\Scripts\python.exe scripts\prepare_leukemia_dataset.py --source path\to\downloaded_dataset --output data\raw --max-per-class 300 --overwrite
```

## Split

```powershell
.\.venv\Scripts\python.exe scripts\split_image_folder.py --input data\raw --output data\processed --val-ratio 0.15 --test-ratio 0.15 --overwrite
```

## Baseline courte

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.train --data-dir data\processed --epochs 5 --batch-size 16 --output-dir outputs
```

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.evaluate --data-dir data\processed --checkpoint outputs\best_model.pt --output-dir outputs\eval
```

Les resultats a regarder en priorite :

- recall sur `leukemia_blast`
- precision sur `leukemia_blast`
- F1-score
- matrice de confusion

L'accuracy seule ne suffit pas a juger le comportement du modele.

## Prediction CLI

```powershell
.\.venv\Scripts\python.exe -m cancer_cell_vision.predict --checkpoint outputs\best_model.pt --image data\processed\test\normal\example.jpg --pretty
```

## Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Teste au moins une image `normal` et une image `leukemia_blast`. Grad-CAM peut aider a presenter le projet, mais il reste une visualisation exploratoire, pas une justification medicale.
