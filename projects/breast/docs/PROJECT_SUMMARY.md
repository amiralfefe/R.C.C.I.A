# Breast Vision - Project Summary

## Resume Court

Breast Vision est un sous-projet du monorepo R.C.C.I.A. Il explore la classification
`benign` vs `malignant` sur le dataset public BreakHis avec un split patient-aware,
PyTorch, Torchvision, Streamlit, Grad-CAM, benchmark multi-modeles et analyse d'erreurs.

Le projet est un demonstrateur IA/data educatif et portfolio. Il ne fournit pas de
diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter
une decision medicale.

## Objectif

Construire un pipeline reproductible pour :

- preparer le dataset BreakHis ;
- extraire les classes `benign` et `malignant` ;
- conserver les informations de grossissement `40X`, `100X`, `200X`, `400X` ;
- extraire un `patient_id` quand il est disponible ;
- realiser un split patient-aware pour limiter le risque de data leakage ;
- entrainer et evaluer des modeles Torchvision ;
- comparer ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- analyser les erreurs par classe, patient, grossissement et confiance ;
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

Dataset public : **BreakHis / Breast Cancer Histopathological Database**.

Images preparees localement :

| Classe | Images |
| --- | ---: |
| `benign` | 2 480 |
| `malignant` | 5 429 |
| **Total** | **7 909** |

Metadata detectee :

| Champ | Valeur |
| --- | ---: |
| Patients detectes | 81 |
| Magnifications | `40X`, `100X`, `200X`, `400X` |

Les donnees restent locales et ne sont pas versionnees dans Git.

## Split Patient-Aware

Le split est fait au niveau patient pour eviter que des images du meme patient se
retrouvent a la fois en train, validation et test.

| Split | Images | Patients |
| --- | ---: | ---: |
| Train | 5 153 | 55 |
| Val | 1 275 | 11 |
| Test | 1 481 | 15 |

Verification : `patient overlap = 0`.

Ce choix est un point central du projet : un split random image-level peut produire des
scores trop optimistes si plusieurs images d'un meme patient apparaissent dans plusieurs
splits.

## Pipeline ML

1. Telechargement local du dataset BreakHis.
2. Normalisation vers une structure `ImageFolder`.
3. Generation de `metadata.csv` avec `patient_id` et `magnification`.
4. Split patient-aware `train` / `val` / `test`.
5. Entrainement PyTorch avec transfer learning.
6. Evaluation avec accuracy, precision, recall, F1-score et matrice de confusion.
7. Prediction CLI.
8. Demo Streamlit avec probabilites et Grad-CAM.
9. Benchmark multi-modeles.
10. Analyse des erreurs par patient, grossissement et confiance.

## Resultats V1 - Baseline ResNet18

Modele : ResNet18 pre-entraine, `epochs=3`, `batch_size=16`, `image_size=224`.

Accuracy test : **0.8947**.

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| `benign` | 0.8860 | 0.7953 | 0.8382 |
| `malignant` | 0.8985 | 0.9466 | 0.9219 |
| **Macro avg** | **0.8923** | **0.8709** | **0.8800** |

## Resultats V2 - Benchmark Modeles

Benchmark reel sur le meme split patient-aware BreakHis.

| Modele | Accuracy | Macro F1 | Recall malignant | Train time |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0.8947 | 0.8800 | 0.9466 | 510.19 s |
| MobileNetV3 small | 0.8575 | 0.8206 | 0.9979 | 316.66 s |
| EfficientNet-B0 | 0.9122 | 0.9004 | 0.9568 | 762.23 s |

Lecture portfolio :

- EfficientNet-B0 obtient le meilleur compromis global.
- MobileNetV3 small maximise le recall `malignant`, mais avec un macro F1 plus faible.
- ResNet18 reste une baseline solide et simple a expliquer.

## Resultats V2.1 - Error Analysis

Modele analyse : EfficientNet-B0.

| Metrique | Valeur |
| --- | ---: |
| Images test | 1 481 |
| Predictions correctes | 1 351 |
| Erreurs | 130 |
| Accuracy | 0.9122 |
| False positives `benign -> malignant` | 88 |
| False negatives `malignant -> benign` | 42 |
| Confiance moyenne correctes | 0.9700 |
| Confiance moyenne erreurs | 0.8464 |

Erreurs par grossissement :

| Grossissement | Erreurs | Accuracy |
| --- | ---: | ---: |
| `40X` | 47 | 0.8757 |
| `100X` | 32 | 0.9194 |
| `200X` | 20 | 0.9467 |
| `400X` | 31 | 0.9063 |

Top patient avec erreurs :

| Patient | Erreurs | Lecture |
| --- | ---: | --- |
| `14-16184CD` | 76 | Variabilite patient importante |

L'analyse montre que les erreurs ne sont pas reparties uniformement : certains patients
concentrent une part importante des erreurs. C'est une raison forte de conserver une
lecture patient-aware du projet.

## Limites

- Dataset public BreakHis, sans validation clinique externe.
- Split patient-aware local, pas validation multi-centres.
- Les scores dependent du split, du preprocessing, du checkpoint et des hyperparametres.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
- Le projet ne traite pas le deploiement clinique, la conformite reglementaire ou la
  validation par experts medicaux.

## Statut Final

Statut : **Breast V2.2 portfolio publishing pack**.

Tag de reference : `breast-v2.1-error-analysis`.

Le projet est pret a etre presente comme projet portfolio IA/data, avec la formulation
suivante : demonstrateur educatif de computer vision sur dataset public, pas outil
medical, pas diagnostic et pas validation clinique.
