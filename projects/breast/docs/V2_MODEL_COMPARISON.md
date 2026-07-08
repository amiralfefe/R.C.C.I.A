# V2 - Model Comparison + Patient-Aware Benchmark

## Objectif

Cette V2 compare plusieurs architectures sur BreakHis en conservant le split patient-aware existant. Le but est de mesurer autre chose que l'accuracy globale :

- macro F1 ;
- recall `malignant` ;
- F1 par classe ;
- performance par grossissement ;
- temps d'entrainement local.

Le projet reste un demonstrateur educatif IA/data pour portfolio. Il ne fournit pas de diagnostic medical et ne constitue pas une validation clinique.

## Dataset Et Split

Dataset : **BreakHis / Breast Cancer Histopathological Database**.

Classes :

- `benign`
- `malignant`

Split utilise :

| Split | Images | Patients |
| --- | ---: | ---: |
| Train | 5 153 | 55 |
| Val | 1 275 | 11 |
| Test | 1 481 | 15 |

Verification patient-aware :

- metadata disponible : oui ;
- patient overlap train/val/test : 0 ;
- metadata manquante dans le split : 0 image.

## Modeles Compares

- `resnet18`
- `mobilenet_v3_small`
- `efficientnet_b0`

Les trois modeles utilisent des poids ImageNet pre-entraines.

## Commande Lancee

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\run_model_comparison.py --data-dir projects\breast\data\processed --models resnet18 mobilenet_v3_small efficientnet_b0 --epochs 3 --batch-size 16 --image-size 224 --output-dir projects\breast\outputs\model_comparison
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
- test set : 1 481 images

## Resultats V2

| Modele | Accuracy | Macro precision | Macro recall | Macro F1 | Train time | Eval time | Eval ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8947 | 0.8923 | 0.8709 | 0.8800 | 510.19 s | 22.39 s | 15.12 |
| `mobilenet_v3_small` | 0.8575 | 0.9081 | 0.7933 | 0.8206 | 316.66 s | 13.25 s | 8.94 |
| `efficientnet_b0` | 0.9122 | 0.9114 | 0.8918 | 0.9004 | 762.23 s | 20.02 s | 13.51 |

`Eval ms/image` est une approximation basee sur le temps total d'evaluation divise par les 1 481 images test. Elle inclut le pipeline local et ne doit pas etre lue comme une mesure d'inference pure bas niveau.

## Metriques Par Classe

| Modele | Precision benign | Recall benign | F1 benign | Precision malignant | Recall malignant | F1 malignant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8860 | 0.7953 | 0.8382 | 0.8985 | 0.9466 | 0.9219 |
| `mobilenet_v3_small` | 0.9934 | 0.5886 | 0.7392 | 0.8229 | 0.9979 | 0.9020 |
| `efficientnet_b0` | 0.9091 | 0.8268 | 0.8660 | 0.9136 | 0.9568 | 0.9347 |

## Performance Par Grossissement

| Modele | 40X | 100X | 200X | 400X |
| --- | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8783 | 0.8917 | 0.9280 | 0.8792 |
| `mobilenet_v3_small` | 0.7778 | 0.8564 | 0.8987 | 0.9033 |
| `efficientnet_b0` | 0.8757 | 0.9194 | 0.9467 | 0.9063 |

## Lecture Des Resultats

- `efficientnet_b0` obtient la meilleure accuracy et la meilleure macro F1.
- `efficientnet_b0` obtient aussi les meilleurs F1 `benign` et `malignant`.
- `mobilenet_v3_small` est le plus rapide et obtient le meilleur recall `malignant` avec 0.9979.
- Le recall `malignant` tres eleve de MobileNetV3 vient avec un compromis : le recall `benign` chute a 0.5886, ce qui indique une tendance a predire plus souvent `malignant`.
- `resnet18` reste une baseline stable et reproductible, avec des resultats identiques a la V1.
- Le grossissement `200X` est le plus favorable pour les trois modeles dans ce benchmark.

## Fichiers Generes

Le script genere les fichiers suivants dans `projects/breast/outputs/model_comparison/` :

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
|       |-- test_confusion_matrix.png
|       `-- test_predictions.csv
|-- mobilenet_v3_small/
`-- efficientnet_b0/
```

Les fichiers `summary.csv` et `summary.json` sont sauvegardes apres chaque modele termine, afin de conserver les resultats partiels si un benchmark long est interrompu.

## Interpreter Les Metriques

- Accuracy : proportion globale de predictions correctes.
- Precision : parmi les images predites dans une classe, proportion correcte.
- Recall : parmi les images reelles d'une classe, proportion retrouvee par le modele.
- F1-score : moyenne harmonique entre precision et recall.
- Macro F1 : moyenne non ponderee du F1 des deux classes, utile ici car le dataset est desequilibre.
- Recall `malignant` : metrique importante a surveiller dans ce projet educatif, car elle mesure la proportion d'images `malignant` retrouvees par le modele.

## Limites

- BreakHis est un dataset public et les resultats dependent fortement du split, du preprocessing et du protocole.
- Le split patient-aware reduit le risque de fuite patient, mais ne remplace pas une validation clinique.
- Les identifiants patients sont extraits depuis les noms de fichiers BreakHis ; la qualite de cette extraction conditionne la rigueur du split.
- Le benchmark est court : 3 epochs sur CPU local.
- Les scores ne doivent pas etre presentes comme preuve de performance medicale.
- Les checkpoints, outputs et donnees restent locaux et ne sont pas versionnes.

## Versionner Proprement

Ne pas commit :

- `projects/breast/data/raw/`
- `projects/breast/data/processed/`
- `projects/breast/outputs/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- token Kaggle, `access_token`, `kaggle.json`
- `.venv/`
