# Project Summary - R.C.C.I.A

## Resume Court

R.C.C.I.A, presente dans le README sous le nom Cancer Cell Vision, est un projet portfolio IA/data de computer vision. Il classe des images microscopiques de cellules sanguines en deux classes experimentales : `normal` et `leukemia_blast`.

Le projet montre une pipeline ML complete : preparation d'un dataset public, split train/validation/test, entrainement PyTorch, evaluation, prediction CLI, demo Streamlit, Grad-CAM, comparaison de modeles et analyse des erreurs.

Important : ce projet est un demonstrateur educatif. Il ne fournit pas de diagnostic medical, ne remplace pas un avis professionnel et ne doit jamais orienter une decision de sante.

## Objectif

L'objectif est de construire un projet ML presentable en portfolio, avec un cas d'usage visuel credible et un niveau de rigueur superieur a une simple notebook demo :

- pipeline reproductible ;
- metriques de classification ;
- comparaison de plusieurs architectures ;
- visualisation Grad-CAM ;
- analyse qualitative des erreurs ;
- documentation exploitable pour CV, LinkedIn et entretien.

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
- Git / GitHub

## Dataset Utilise

Dataset public : **Kaggle - andrewmvd/leukemia-classification**

Lien : https://www.kaggle.com/datasets/andrewmvd/leukemia-classification

Images preparees localement :

| Classe | Images |
| --- | ---: |
| `normal` | 3389 |
| `leukemia_blast` | 7272 |

Split utilise :

| Classe | Train | Val | Test |
| --- | ---: | ---: | ---: |
| `normal` | 2372 | 508 | 509 |
| `leukemia_blast` | 5090 | 1090 | 1092 |

Les donnees, checkpoints et outputs restent locaux et ne sont pas versionnes.

## Pipeline ML

```text
Dataset Kaggle
  -> preparation data/raw/normal + data/raw/leukemia_blast
  -> split train / val / test
  -> entrainement PyTorch
  -> evaluation test set
  -> prediction CLI
  -> demo Streamlit
  -> Grad-CAM
  -> comparaison de modeles
  -> analyse false positives / false negatives
```

## Resultats V1

Baseline V1 :

- modele : ResNet18 en transfer learning ;
- entrainement : 5 epochs sur CPU ;
- batch size : 16 ;
- test accuracy : `0.9169`.

Rapport test :

| Classe | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| `leukemia_blast` | 0.9419 | 0.9359 | 0.9389 | 1092 |
| `normal` | 0.8643 | 0.8762 | 0.8702 | 509 |
| `weighted avg` | 0.9173 | 0.9169 | 0.9171 | 1601 |

Matrice de confusion :

| Vraie classe / prediction | `leukemia_blast` | `normal` |
| --- | ---: | ---: |
| `leukemia_blast` | 1022 | 70 |
| `normal` | 63 | 446 |

## Resultats V2.1 Benchmark

La V2.1 compare trois architectures avec poids ImageNet pre-entraines, 3 epochs, batch size 16 et images 224x224.

| Modele | Accuracy | F1 normal | F1 leukemia_blast | Recall normal | Recall leukemia_blast | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8988 | 0.8251 | 0.9288 | 0.7505 | 0.9679 | 586.95 s |
| `mobilenet_v3_small` | 0.8976 | 0.8295 | 0.9268 | 0.7839 | 0.9505 | 277.76 s |
| `efficientnet_b0` | 0.8863 | 0.8080 | 0.9193 | 0.7525 | 0.9487 | 890.49 s |

Lecture rapide :

- `resnet18` obtient la meilleure accuracy et le meilleur recall `leukemia_blast` dans ce protocole court.
- `mobilenet_v3_small` est presque equivalent en accuracy, avec un entrainement nettement plus rapide sur CPU.
- `efficientnet_b0` est plus lent dans ce run et ne depasse pas les deux autres modeles.

## Resultats V2.2 Error Analysis

La V2.2 analyse les predictions du checkpoint V1 sur le test set.

| Indicateur | Valeur |
| --- | ---: |
| Images analysees | 1601 |
| Predictions correctes | 1468 |
| Erreurs | 133 |
| False positives | 63 |
| False negatives | 70 |
| Accuracy | 0.9169 |
| Confiance moyenne bonnes predictions | 0.9110 |
| Confiance moyenne erreurs | 0.7057 |

La V2.2 genere localement :

- `outputs/error_analysis/predictions.csv`
- `outputs/error_analysis/summary.json`
- exemples annotes ;
- Grad-CAM sur quelques erreurs.

Ces fichiers restent exclus de Git.

## Limites

- Dataset public Kaggle, sans validation clinique independante.
- Classes desequilibrees entre `normal` et `leukemia_blast`.
- Entrainements courts sur CPU.
- Resultats dependants du split local, du preprocessing et des hyperparametres.
- Grad-CAM aide a expliquer visuellement une prediction, mais ne prouve rien medicalement.
- Aucune validation par specialistes, aucune validation multi-centrique, aucune etude clinique.

## Statut Final

Version GitHub figee :

- tag : `v2.2-error-analysis`
- branche : `main`
- statut : projet portfolio complet, pret a etre presente.

La phase V2.3 ajoute ce pack de publication portfolio : resume projet, pitch entretien, brouillons LinkedIn et release notes.

## Disclaimer Medical

R.C.C.I.A est uniquement un demonstrateur educatif IA/data pour portfolio. Il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit pas etre utilise pour orienter une decision medicale.
