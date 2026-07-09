# Metastasis V2 - Model Comparison

## Objectif

Metastasis V2 compare plusieurs architectures CNN sur le subset PCam deja prepare
localement. Le but est de rendre le benchmark plus credible qu'une seule baseline,
tout en gardant un protocole court et reproductible sur CPU.

Ce projet reste un demonstrateur educatif / portfolio. Il ne fournit pas de diagnostic
medical, ne remplace pas une validation clinique et ne doit jamais orienter une decision
de sante.

## Dataset et Protocole

Dataset utilise : subset PCam / PatchCamelyon prepare depuis Kaggle `tyson04/pcam-validate`.

Configuration :

| Parametre | Valeur |
| --- | --- |
| Classes | `non_metastatic`, `metastatic` |
| Images raw converties | 5 000 |
| Train | 3 500 images |
| Val | 750 images |
| Test | 750 images |
| Image size | 96 |
| Epochs | 3 |
| Batch size | 32 |
| Pretrained | oui |
| Learning rate | 0.0001 |

L'image size `96` est conservee car PCam fournit des patches natifs 96x96.

## Modeles Compares

- `resnet18`
- `mobilenet_v3_small`
- `efficientnet_b0`

## Metriques

Les resultats V2 utilisent :

- accuracy ;
- macro precision / recall / F1 ;
- ROC-AUC ;
- PR-AUC ;
- precision / recall / F1 par classe ;
- temps d'entrainement ;
- temps moyen d'inference par image pendant l'evaluation.

## Resultats Reels

| Modele | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Precision `non_metastatic` | Recall `non_metastatic` | F1 `non_metastatic` | Precision `metastatic` | Recall `metastatic` | F1 `metastatic` | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.9040 | 0.9039 | 0.9706 | 0.9672 | 0.9268 | 0.8773 | 0.9014 | 0.8835 | **0.9307** | 0.9065 | 112.86 s |
| MobileNetV3 small | 0.8573 | 0.8567 | 0.9374 | 0.9236 | 0.9110 | 0.7920 | 0.8474 | 0.8160 | 0.9227 | 0.8661 | **64.82 s** |
| EfficientNet-B0 | **0.9320** | **0.9320** | **0.9762** | **0.9780** | **0.9286** | **0.9360** | **0.9323** | **0.9355** | 0.9280 | **0.9317** | 93.17 s |

## Analyse Courte

EfficientNet-B0 est le meilleur modele global sur ce benchmark : il obtient la meilleure
accuracy, le meilleur macro F1, le meilleur ROC-AUC et le meilleur PR-AUC.

ResNet18 reste tres solide et obtient le meilleur recall `metastatic`, ce qui est une
metrique importante dans un probleme de detection positive. MobileNetV3 small est le
plus rapide a entrainer, mais il perd nettement en accuracy et macro F1 sur ce subset.

Ces resultats montrent un compromis classique : un modele plus compact peut etre plus
rapide, mais pas forcement meilleur sur les metriques critiques.

## Fichiers Generes Localement

Les fichiers suivants sont generes dans `projects/metastasis/outputs/model_comparison/`
et ne doivent pas etre commit :

- `summary.csv`
- `summary.json`
- `resnet18/best_model.pt`
- `resnet18/metrics.json`
- `resnet18/classification_report.json`
- `resnet18/confusion_matrix.png`
- `resnet18/training_curves.png`
- `mobilenet_v3_small/best_model.pt`
- `mobilenet_v3_small/metrics.json`
- `mobilenet_v3_small/classification_report.json`
- `mobilenet_v3_small/confusion_matrix.png`
- `mobilenet_v3_small/training_curves.png`
- `efficientnet_b0/best_model.pt`
- `efficientnet_b0/metrics.json`
- `efficientnet_b0/classification_report.json`
- `efficientnet_b0/confusion_matrix.png`
- `efficientnet_b0/training_curves.png`

## Reproduction

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\run_model_comparison.py --data-dir projects\metastasis\data\processed --models resnet18 mobilenet_v3_small efficientnet_b0 --epochs 3 --batch-size 32 --image-size 96 --output-dir projects\metastasis\outputs\model_comparison
```

Pour relancer un seul modele sans perdre les lignes existantes du resume :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\run_model_comparison.py --data-dir projects\metastasis\data\processed --models efficientnet_b0 --epochs 3 --batch-size 32 --image-size 96 --output-dir projects\metastasis\outputs\model_comparison
```

## Limites

- Benchmark educatif sur subset PCam public, pas validation clinique.
- Subset equilibre de 5 000 patches, pas corpus PCam complet.
- Trois epochs seulement pour garder un temps CPU raisonnable.
- Les metriques dependent du split local, du preprocessing et du seuil de decision.
- ROC-AUC et PR-AUC evaluent les scores probabilistes, mais ne suffisent pas a prouver
  une robustesse medicale.
