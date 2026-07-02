# Dataset Guide

Ce projet est un demonstrateur educatif / portfolio. Il ne doit pas etre utilise pour diagnostiquer, traiter ou orienter une decision medicale.

## Donnees brutes

Place les images dans `data/raw`, avec un dossier par classe :

```text
data/
  raw/
    normal/
      image_001.jpg
      image_002.jpg
    cancer/
      image_001.jpg
      image_002.jpg
```

Pour un dataset multi-classes, garde le meme principe :

```text
data/
  raw/
    class_1/
    class_2/
    class_3/
```

## Creation des splits

```powershell
python scripts/split_image_folder.py --input data/raw --output data/processed --val-ratio 0.15 --test-ratio 0.15
```

Structure produite :

```text
data/
  processed/
    train/
      normal/
      cancer/
    val/
      normal/
      cancer/
    test/
      normal/
      cancer/
```

## Donnees non committees

Les vrais fichiers images ne sont pas commit dans Git. Les dossiers `data/raw`, `data/processed` et `outputs` contiennent seulement un `.gitkeep` pour conserver la structure.

## Sources autorisees

Utilise uniquement des datasets publics, autorises et compatibles avec un usage portfolio / education. Ne mets pas de donnees medicales privees ou identifiantes dans ce depot.
