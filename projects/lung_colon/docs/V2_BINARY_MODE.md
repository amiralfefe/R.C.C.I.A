# V2.2 - Binary Benign vs Malignant Mode

## Objectif

La V2.2 ajoute un deuxieme mode de classification au sous-projet Lung + Colon Vision :

- mode 5 classes histopathologiques, conserve comme mode principal ;
- mode binaire `benign` vs `malignant`, utile pour analyser le meme dataset sous un angle plus simple.

Ce mode reste un demonstrateur educatif IA/data pour portfolio. Il ne fournit pas de diagnostic medical et ne doit pas etre interprete comme une validation clinique.

## Mapping Binaire

| Classe binaire | Classes sources LC25000 |
| --- | --- |
| `benign` | `colon_benign`, `lung_benign` |
| `malignant` | `colon_adenocarcinoma`, `lung_adenocarcinoma`, `lung_squamous_cell_carcinoma` |

Le mapping cree un dataset binaire desequilibre de facon attendue : deux classes sources benignes contre trois classes sources malignes.

## Preparation Du Dataset

Commande lancee depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\lung_colon\scripts\prepare_binary_dataset.py --input projects\lung_colon\data\processed --output projects\lung_colon\data\binary_processed
```

Le script copie les images, sans deplacer ni modifier `data/processed`.

Structure generee localement :

```text
projects/lung_colon/data/binary_processed/
|-- train/
|   |-- benign/
|   `-- malignant/
|-- val/
|   |-- benign/
|   `-- malignant/
`-- test/
    |-- benign/
    `-- malignant/
```

Counts generes :

| Split | Benign | Malignant | Total |
| --- | ---: | ---: | ---: |
| Train | 7 000 | 10 500 | 17 500 |
| Val | 1 500 | 2 250 | 3 750 |
| Test | 1 500 | 2 250 | 3 750 |

`projects/lung_colon/data/binary_processed/` est ignore par Git, sauf son `.gitkeep`.

## Entrainement

Commande lancee depuis la racine :

```powershell
.\.venv\Scripts\python.exe -m projects.lung_colon.rccia_lung_colon.train --data-dir projects\lung_colon\data\binary_processed --model resnet18 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\lung_colon\outputs\binary_resnet18
```

Configuration :

- modele : `resnet18`
- poids ImageNet pre-entraines
- `epochs=3`
- `batch_size=16`
- `image_size=224`
- `seed=42`
- CPU local

Historique :

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0250 | 0.9921 | 0.0037 | 0.9992 |
| 2 | 0.0056 | 0.9982 | 0.0024 | 0.9997 |
| 3 | 0.0052 | 0.9980 | 0.0003 | 1.0000 |

Best validation accuracy : **1.0000**.

EfficientNet-B0 n'a pas ete relance dans cette phase : les runs precedents montrent qu'il est beaucoup plus couteux sur CPU, et l'objectif V2.2 etait de valider le workflow binaire.

## Evaluation

Commande lancee :

```powershell
.\.venv\Scripts\python.exe -m projects.lung_colon.rccia_lung_colon.evaluate --data-dir projects\lung_colon\data\binary_processed --checkpoint projects\lung_colon\outputs\binary_resnet18\best_model.pt --model resnet18 --output-dir projects\lung_colon\outputs\binary_eval_resnet18
```

Resultats test sur 3 750 images :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `benign` | 1.0000 | 1.0000 | 1.0000 | 1 500 |
| `malignant` | 1.0000 | 1.0000 | 1.0000 | 2 250 |
| **Macro avg** | **1.0000** | **1.0000** | **1.0000** | **3 750** |

Accuracy test : **1.0000**.

Le recall `malignant` est **1.0000** sur ce split local. Ce chiffre doit rester interprete comme un resultat experimental sur LC25000, pas comme une preuve clinique.

## Prediction CLI

Image benign testee :

```powershell
.\.venv\Scripts\python.exe -m projects.lung_colon.rccia_lung_colon.predict --checkpoint projects\lung_colon\outputs\binary_resnet18\best_model.pt --image projects\lung_colon\data\binary_processed\test\benign\colon_benign_colon_benign_00002.jpeg --pretty
```

Resultat : `benign`, confiance `0.9999998808`.

Image malignant testee :

```powershell
.\.venv\Scripts\python.exe -m projects.lung_colon.rccia_lung_colon.predict --checkpoint projects\lung_colon\outputs\binary_resnet18\best_model.pt --image projects\lung_colon\data\binary_processed\test\malignant\colon_adenocarcinoma_colon_adenocarcinoma_00003.jpeg --pretty
```

Resultat : `malignant`, confiance `0.9999774694`.

## Streamlit

L'application propose maintenant deux modes dans la sidebar :

- `Mode 5 classes`
- `Mode binaire benign/malignant`

Le mode binaire charge par defaut :

```text
outputs/binary_resnet18/best_model.pt
```

Si le checkpoint binaire est absent, l'application affiche un message propre et ne plante pas. Le test local Streamlit a retourne HTTP 200.

## Fichiers Generes

Fichiers locaux generes :

```text
projects/lung_colon/data/binary_processed/
projects/lung_colon/outputs/binary_resnet18/
projects/lung_colon/outputs/binary_eval_resnet18/
```

Ces fichiers restent ignores et ne sont pas committes.

## Limites

- LC25000 est un dataset public tres propre et relativement facile.
- Le mode binaire simplifie fortement le probleme par rapport aux 5 classes.
- Le dataset binaire est desequilibre : 7 000 / 10 500 en train et 1 500 / 2 250 en test.
- Les scores parfaits sur ce split local ne valent pas preuve de robustesse clinique.
- Aucun resultat de ce projet ne doit etre utilise pour un diagnostic ou une decision medicale.
