# V2 - Model Comparison + Evaluation Reports

## Objectif

Cette V2 ajoute un workflow de comparaison de modeles pour le sous-projet Lung + Colon Vision. Le but est de comparer plusieurs architectures sur le meme dataset LC25000 prepare localement, avec les memes splits `train`, `val` et `test`.

Le projet reste un demonstrateur educatif IA/data pour portfolio. Il ne fournit pas de diagnostic medical et ne doit pas etre interprete comme une preuve de performance clinique.

## Dataset Et Classes

Dataset : **LC25000 / Lung and Colon Cancer Histopathological Images**.

Classes normalisees :

- `colon_adenocarcinoma`
- `colon_benign`
- `lung_adenocarcinoma`
- `lung_benign`
- `lung_squamous_cell_carcinoma`

Split utilise :

| Split | Images |
| --- | ---: |
| Train | 17 500 |
| Val | 3 750 |
| Test | 3 750 |

Chaque classe contient 3 500 images train, 750 images val et 750 images test.

## Modeles Compares

- `resnet18`
- `mobilenet_v3_small`
- `efficientnet_b0`

Les trois modeles utilisent des poids ImageNet pre-entraines pour ce benchmark reel V2.

## Commande

Commande lancee depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\lung_colon\scripts\run_model_comparison.py --data-dir projects\lung_colon\data\processed --models resnet18 mobilenet_v3_small efficientnet_b0 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\lung_colon\outputs\model_comparison
```

Configuration :

- `epochs=3`
- `batch_size=16`
- `image_size=224`
- `learning_rate=0.0001`
- `weight_decay=0.0001`
- `seed=42`
- `num_workers=0`
- pretrained active

## Resultats V2

| Modele | Accuracy | Macro precision | Macro recall | Macro F1 | Train time | Eval time | Eval ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.9965 | 0.9965 | 0.9965 | 0.9965 | 1 559.23 s | 43.35 s | 11.56 |
| `mobilenet_v3_small` | 0.9957 | 0.9958 | 0.9957 | 0.9957 | 874.02 s | 33.89 s | 9.04 |
| `efficientnet_b0` | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 3 156.67 s | 58.70 s | 15.65 |

`Eval ms/image` est une approximation basee sur le temps total d'evaluation divise par les 3 750 images test. Elle inclut le chargement/evaluation du pipeline local et ne doit pas etre lue comme une mesure d'inference pure bas niveau.

## F1 Par Classe

| Modele | colon_adenocarcinoma | colon_benign | lung_adenocarcinoma | lung_benign | lung_squamous_cell_carcinoma |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.9987 | 0.9993 | 0.9920 | 0.9993 | 0.9933 |
| `mobilenet_v3_small` | 1.0000 | 1.0000 | 0.9892 | 1.0000 | 0.9894 |
| `efficientnet_b0` | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9980 |

## Lecture Des Resultats

- `efficientnet_b0` obtient la meilleure accuracy et la meilleure macro F1, mais c'est aussi le modele le plus couteux a entrainer sur CPU local.
- `mobilenet_v3_small` est le plus rapide et conserve une accuracy tres proche des autres modeles.
- `resnet18` reste une baseline robuste et simple, avec un score V2 coherent avec la baseline V1.
- Les erreurs restantes concernent surtout la separation `lung_adenocarcinoma` / `lung_squamous_cell_carcinoma`.
- Les classes `colon_benign` et `lung_benign` sont quasiment parfaites dans ce protocole, ce qui confirme aussi que LC25000 est un benchmark public relativement facile.

## Fichiers Generes

Le script genere les fichiers suivants dans `projects/lung_colon/outputs/model_comparison/` :

```text
outputs/model_comparison/
|-- summary.csv
|-- summary.json
|-- resnet18/
|   |-- best_model.pt
|   |-- train/
|   |   |-- training_setup.json
|   |   |-- training_history.json
|   |   |-- training_history.csv
|   |   |-- training_loss.png
|   |   `-- training_accuracy.png
|   `-- eval/
|       |-- test_classification_report.json
|       |-- test_classification_report.csv
|       `-- test_confusion_matrix.png
|-- mobilenet_v3_small/
`-- efficientnet_b0/
```

Les fichiers `summary.csv` et `summary.json` sont sauvegardes apres chaque modele termine, pour conserver les resultats partiels si un benchmark long est interrompu.

## Interpreter Les Metriques

- Accuracy : proportion globale de predictions correctes.
- Precision : parmi les images predites dans une classe, proportion correcte.
- Recall : parmi les images reelles d'une classe, proportion retrouvee par le modele.
- F1-score : moyenne harmonique entre precision et recall.
- Macro F1 : moyenne non ponderee du F1 de chaque classe, utile ici car les classes sont equilibrees.
- Train time : temps total mesure par le script pour entrainer le modele.

## Limites

- LC25000 est un dataset public tres propre et relativement facile.
- Le benchmark est local, court et realise sur 3 epochs.
- Les scores ne valent pas validation clinique.
- Les images et checkpoints restent locaux et ne sont pas versionnes.
- Grad-CAM et les matrices de confusion aident a comprendre le comportement du modele, mais ne constituent pas une preuve medicale.

## Versionner Proprement

Ne pas commit :

- `projects/lung_colon/data/raw/`
- `projects/lung_colon/data/processed/`
- `projects/lung_colon/outputs/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- token Kaggle, `access_token`, `kaggle.json`
- `.venv/`
