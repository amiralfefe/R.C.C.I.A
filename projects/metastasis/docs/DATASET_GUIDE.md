# Metastasis Dataset Guide

## Dataset Cible

Le dataset cible probable est **PatchCamelyon / PCam** ou un dataset equivalent de
patches histopathologiques annote pour la presence de metastases.

Le projet reste educatif / portfolio uniquement. Il ne fournit pas de diagnostic medical.

## Objectif

Normaliser un dataset binaire vers deux classes :

- `non_metastatic`
- `metastatic`

## Structure Source Supportee En V1

Le script `prepare_pcam_dataset.py` supporte une source deja exportee sous forme
d'images, avec labels detectables dans les dossiers ou les noms de fichiers :

```text
C:\VSCODE\datasets\pcam\
|-- non_metastatic\
|   |-- image_001.png
|   `-- image_002.png
`-- metastatic\
    |-- image_101.png
    `-- image_102.png
```

Alias reconnus :

| Classe normalisee | Alias |
| --- | --- |
| `non_metastatic` | `non_metastatic`, `normal`, `negative`, `nonmetastatic`, `no_metastasis`, `no_tumor`, `0` |
| `metastatic` | `metastatic`, `metastasis`, `positive`, `tumor`, `tumour`, `cancer`, `1` |

## HDF5 / CSV

PCam est souvent distribue en formats HDF5 ou CSV selon les sources. La V1 du script ne
convertit pas encore directement les fichiers `.h5` / `.hdf5`. Si seuls des fichiers HDF5
sont detectes, le script echoue avec un message clair au lieu de faire semblant.

Une future version pourra ajouter un convertisseur dedie HDF5 -> images.

## Preparation

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\prepare_pcam_dataset.py --input C:\VSCODE\datasets\pcam --output projects\metastasis\data\raw
```

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

Structure processed attendue :

```text
projects/metastasis/data/processed/
|-- train/
|   |-- non_metastatic/
|   `-- metastatic/
|-- val/
|   |-- non_metastatic/
|   `-- metastatic/
`-- test/
    |-- non_metastatic/
    `-- metastatic/
```

## Donnees Non Versionnees

Les dossiers suivants restent locaux et ne doivent pas etre committes :

- `projects/metastasis/data/raw/`
- `projects/metastasis/data/processed/`
- `projects/metastasis/outputs/`
- checkpoints `.pt`, `.pth`, `.ckpt`
- tokens Kaggle et fichiers `kaggle.json` / `access_token`
- `.venv/`

## Disclaimer

Ce projet est un demonstrateur IA/data educatif. Il ne constitue pas un outil medical,
pas un diagnostic et pas une validation clinique.

