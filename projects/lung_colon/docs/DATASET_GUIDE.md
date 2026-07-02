# Dataset Guide - Lung + Colon

## Dataset Cible

Dataset : **LC25000 / Lung and Colon Cancer Histopathological Images**.

Commande Kaggle indicative :

```powershell
..\..\.venv\Scripts\kaggle.exe datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images -p C:\VSCODE\datasets\lc25000 --unzip
```

Kaggle peut demander une authentification locale. Ne jamais commit `kaggle.json`, `access_token`, archives zip ou images extraites.

## Classes Cibles

Le sous-projet normalise les dossiers vers ces cinq classes :

```text
data/raw/
  colon_adenocarcinoma/
  colon_benign/
  lung_adenocarcinoma/
  lung_benign/
  lung_squamous_cell_carcinoma/
```

Mapping habituel LC25000 :

| Dossier source | Classe normalisee |
| --- | --- |
| `colon_aca` | `colon_adenocarcinoma` |
| `colon_n` | `colon_benign` |
| `lung_aca` | `lung_adenocarcinoma` |
| `lung_n` | `lung_benign` |
| `lung_scc` | `lung_squamous_cell_carcinoma` |

## Preparation

Depuis `projects/lung_colon` :

```powershell
..\..\.venv\Scripts\python.exe scripts\prepare_lc25000_dataset.py --source C:\VSCODE\datasets\lc25000 --output data\raw --overwrite
```

Pour un subset rapide :

```powershell
..\..\.venv\Scripts\python.exe scripts\prepare_lc25000_dataset.py --source C:\VSCODE\datasets\lc25000 --output data\raw --max-per-class 300 --overwrite
```

## Split

```powershell
..\..\.venv\Scripts\python.exe scripts\split_image_folder.py --input data\raw --output data\processed --val-ratio 0.15 --test-ratio 0.15 --overwrite
```

Structure attendue apres split :

```text
data/processed/
  train/
  val/
  test/
```

Chaque split contient les cinq dossiers de classes.

## Fichiers Non Versionnes

Ne pas commit :

- `data/raw/`
- `data/processed/`
- `outputs/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- token Kaggle, `access_token`, `kaggle.json`
- `.venv/`

Le projet conserve seulement les `.gitkeep` pour garder la structure.
