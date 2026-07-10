# Metastasis Vision - Project Summary

## Resume Court

Metastasis Vision est un sous-projet du monorepo R.C.C.I.A. Il explore la
classification de patches histopathologiques PCam / PatchCamelyon en deux classes :
`non_metastatic` et `metastatic`.

Le projet couvre un pipeline PyTorch complet : preparation HDF5 vers `ImageFolder`,
baseline ResNet18, benchmark multi-modeles, evaluation ROC-AUC / PR-AUC, analyse des
erreurs, analyse de seuils, Streamlit et Grad-CAM.

Ce projet est un demonstrateur IA/data educatif et portfolio. Il ne fournit pas de
diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter
une decision medicale.

## Objectif

Construire un projet reproductible pour :

- preparer un subset PCam depuis des fichiers HDF5 publics ;
- convertir les patches en structure `ImageFolder` ;
- entrainer une baseline de detection `non_metastatic` vs `metastatic` ;
- comparer ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- evaluer accuracy, precision, recall, F1, ROC-AUC et PR-AUC ;
- analyser les faux positifs, faux negatifs et seuils de decision ;
- presenter une demo Streamlit locale avec probabilites et Grad-CAM.

## Stack Technique

- Python
- PyTorch / Torchvision
- OpenCV
- Pandas / NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM
- Pytest

## Dataset

Dataset utilise : Kaggle `tyson04/pcam-validate`, base sur PCam / PatchCamelyon.

Subset V1/V2 prepare localement :

| Classe | Images |
| --- | ---: |
| `non_metastatic` | 2 500 |
| `metastatic` | 2 500 |
| **Total** | **5 000** |

Split local :

| Split | `non_metastatic` | `metastatic` | Total |
| --- | ---: | ---: | ---: |
| Train | 1 750 | 1 750 | 3 500 |
| Val | 375 | 375 | 750 |
| Test | 375 | 375 | 750 |

Les donnees, outputs et checkpoints restent locaux et ne sont pas versionnes dans Git.

## Pipeline ML

1. Telechargement local du dataset PCam / PatchCamelyon.
2. Inspection des fichiers HDF5 `x` et `y`.
3. Conversion vers `data/raw/non_metastatic` et `data/raw/metastatic`.
4. Split `train` / `val` / `test`.
5. Entrainement PyTorch avec transfer learning.
6. Evaluation avec metriques de classification, ROC-AUC et PR-AUC.
7. Prediction CLI sur exemples test.
8. Demo Streamlit avec probabilites et Grad-CAM.
9. Benchmark multi-modeles.
10. Analyse des erreurs et des seuils.

## Resultats V1 - Baseline ResNet18

Modele : ResNet18 pre-entraine, `epochs=3`, `batch_size=32`, `image_size=96`.

| Metrique | Valeur |
| --- | ---: |
| Accuracy test | 0.9040 |
| ROC-AUC | 0.9598 |
| PR-AUC | 0.9616 |
| F1 `non_metastatic` | 0.9040 |
| F1 `metastatic` | 0.9040 |

## Resultats V2 - Benchmark Modeles

Benchmark reel sur le meme split PCam subset.

| Modele | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Recall metastatic | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.9040 | 0.9039 | 0.9706 | 0.9672 | **0.9307** | 112.86 s |
| MobileNetV3 small | 0.8573 | 0.8567 | 0.9374 | 0.9236 | 0.9227 | **64.82 s** |
| EfficientNet-B0 | **0.9320** | **0.9320** | **0.9762** | **0.9780** | 0.9280 | 93.17 s |

Lecture portfolio :

- EfficientNet-B0 obtient le meilleur score global.
- ResNet18 obtient le meilleur recall `metastatic`.
- MobileNetV3 small est le plus rapide a entrainer, mais moins performant.

## Resultats V2.1 - Error And Threshold Analysis

Modele analyse : EfficientNet-B0.

| Metrique | Valeur |
| --- | ---: |
| Images test | 750 |
| Predictions correctes | 699 |
| Erreurs | 51 |
| Accuracy | 0.9320 |
| False positives `non_metastatic -> metastatic` | 24 |
| False negatives `metastatic -> non_metastatic` | 27 |
| Confiance moyenne correctes | 0.9071 |
| Confiance moyenne erreurs | 0.7107 |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.9780 |

Analyse des seuils :

| Threshold | Accuracy | Precision metastatic | Recall metastatic | F1 metastatic | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.8907 | 0.8399 | **0.9653** | 0.8983 | 69 | **13** |
| 0.40 | 0.9160 | 0.8861 | 0.9547 | 0.9191 | 46 | 17 |
| 0.50 | **0.9320** | 0.9355 | 0.9280 | **0.9317** | 24 | 27 |
| 0.60 | 0.9187 | 0.9460 | 0.8880 | 0.9161 | 19 | 42 |
| 0.70 | 0.9120 | **0.9668** | 0.8533 | 0.9065 | **11** | 55 |

Cette analyse montre que baisser le seuil augmente le recall `metastatic` et reduit les
faux negatifs, mais augmente les faux positifs. Monter le seuil reduit les faux positifs,
mais augmente les faux negatifs.

## Limites

- Dataset public PCam / PatchCamelyon, sans validation clinique externe.
- Run effectue sur un subset equilibre de 5 000 patches, pas sur tout le corpus.
- Split local, pas validation multi-centres.
- Trois epochs seulement pour garder un temps CPU raisonnable.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
- Aucun seuil ne doit etre interprete comme seuil de decision medicale.

## Statut Final

Statut : **Metastasis V2.2 portfolio publishing pack**.

Tag de reference : `metastasis-v2.1-threshold-analysis`.

Le projet est pret a etre presente comme projet portfolio IA/data, avec la formulation
suivante : demonstrateur educatif de computer vision sur dataset public, pas outil
medical, pas diagnostic et pas validation clinique.
