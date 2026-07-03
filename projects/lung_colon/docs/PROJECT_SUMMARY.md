# Lung + Colon Vision - Project Summary

## Resume Court

Lung + Colon Vision est un sous-projet du monorepo R.C.C.I.A. Il explore la classification d'images histopathologiques publiques du dataset LC25000 avec PyTorch, Torchvision, Streamlit et Grad-CAM.

Le projet est un demonstrateur IA/data educatif et portfolio. Il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter une decision medicale.

## Objectif

Construire un pipeline reproductible pour :

- preparer le dataset LC25000 ;
- entrainer des modeles de classification d'images ;
- evaluer les resultats avec accuracy, precision, recall, F1-score et matrice de confusion ;
- comparer plusieurs architectures ;
- analyser les erreurs ;
- presenter une demo locale Streamlit avec Grad-CAM ;
- ajouter un mode binaire `benign` vs `malignant`.

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

Dataset public : **LC25000 / Lung and Colon Cancer Histopathological Images**.

Le dataset local utilise contient 25 000 images, reparties en 5 classes equilibrees :

| Classe | Images |
| --- | ---: |
| `colon_adenocarcinoma` | 5 000 |
| `colon_benign` | 5 000 |
| `lung_adenocarcinoma` | 5 000 |
| `lung_benign` | 5 000 |
| `lung_squamous_cell_carcinoma` | 5 000 |

Split multi-classe :

| Split | Images |
| --- | ---: |
| Train | 17 500 |
| Val | 3 750 |
| Test | 3 750 |

Les donnees restent locales et ne sont pas versionnees dans Git.

## Pipeline ML

1. Telechargement local du dataset LC25000.
2. Normalisation vers une structure `ImageFolder`.
3. Split `train` / `val` / `test`.
4. Entrainement PyTorch avec transfer learning.
5. Evaluation sur test set.
6. Generation de rapports : classification report, matrice de confusion, courbes loss/accuracy.
7. Prediction CLI.
8. Demo Streamlit avec probabilites et Grad-CAM.
9. Benchmark multi-modeles.
10. Analyse des erreurs.
11. Mode binaire `benign` vs `malignant`.

## Resultats V1 - Classification 5 Classes

Modele : ResNet18 pre-entraine, `epochs=3`, `batch_size=16`, `image_size=224`.

Accuracy test : **0.9965**.

| Classe | F1-score |
| --- | ---: |
| `colon_adenocarcinoma` | 0.9987 |
| `colon_benign` | 0.9993 |
| `lung_adenocarcinoma` | 0.9920 |
| `lung_benign` | 0.9993 |
| `lung_squamous_cell_carcinoma` | 0.9933 |

## Resultats V2 - Benchmark Modeles

Benchmark reel sur le split test LC25000.

| Modele | Accuracy | Macro F1 | Train time |
| --- | ---: | ---: | ---: |
| EfficientNet-B0 | 0.9992 | 0.9992 | 3156.67 s |
| ResNet18 | 0.9965 | 0.9965 | 1559.23 s |
| MobileNetV3 small | 0.9957 | 0.9957 | 874.02 s |

Lecture portfolio :

- EfficientNet-B0 obtient le meilleur score sur ce benchmark.
- MobileNetV3 small est le plus rapide.
- ResNet18 reste une baseline robuste et simple a expliquer.

## Resultats V2.1 - Error Analysis

Modele analyse : EfficientNet-B0.

| Metrique | Valeur |
| --- | ---: |
| Images test | 3 750 |
| Predictions correctes | 3 747 |
| Erreurs | 3 |
| Accuracy | 0.9992 |
| Confiance moyenne correctes | 0.9986 |
| Confiance moyenne erreurs | 0.8624 |
| Erreurs benign/malignant | 0 |
| Erreurs lung/colon | 0 |

Les 3 erreurs observees sont des confusions entre sous-types malins pulmonaires :

- `lung_squamous_cell_carcinoma -> lung_adenocarcinoma` : 2
- `lung_adenocarcinoma -> lung_squamous_cell_carcinoma` : 1

## Resultats V2.2 - Mode Binaire

Mode : `benign` vs `malignant`.

Modele : ResNet18 pre-entraine, `epochs=3`, `batch_size=16`, `image_size=224`.

Split binaire :

| Split | Benign | Malignant | Total |
| --- | ---: | ---: | ---: |
| Train | 7 000 | 10 500 | 17 500 |
| Val | 1 500 | 2 250 | 3 750 |
| Test | 1 500 | 2 250 | 3 750 |

Resultats test :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `benign` | 1.0000 | 1.0000 | 1.0000 | 1 500 |
| `malignant` | 1.0000 | 1.0000 | 1.0000 | 2 250 |

Accuracy test : **1.0000**.

Ce score est a interpreter avec prudence : LC25000 est un benchmark public relativement facile et ces resultats ne constituent pas une validation clinique.

## Limites

- Dataset public, sans cohorte clinique externe.
- Split local, pas validation multi-centres.
- Resultats tres eleves probablement lies a la nature du dataset et au protocole experimental.
- Grad-CAM est une aide visuelle exploratoire, pas une preuve medicale.
- Le projet ne traite pas le deploiement clinique, la conformite reglementaire ou la validation par experts medicaux.

## Statut Final

Statut : **LungColon V2.2 complete + portfolio publishing pack**.

Tag de reference : `lung-colon-v2.2-binary-mode`.

Le projet est pret a etre presente comme projet portfolio IA/data, avec la formulation suivante : demonstrateur educatif de computer vision sur dataset public, pas outil medical et pas diagnostic.
