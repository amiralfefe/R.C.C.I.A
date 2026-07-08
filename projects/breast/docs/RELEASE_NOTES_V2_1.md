# Breast Vision - Release Notes V2.1

## Resume

La V2.1 finalise Breast Vision comme projet portfolio solide autour du dataset public
BreakHis.

Cette release couvre :

- classification `benign` vs `malignant` ;
- split patient-aware ;
- application Streamlit avec Grad-CAM ;
- benchmark multi-modeles ;
- analyse d'erreurs false positives / false negatives ;
- analyse par grossissement ;
- analyse par patient ;
- documentation portfolio.

Tag de reference : `breast-v2.1-error-analysis`.

## Fonctionnalites Principales

### V1 - Baseline Patient-Aware

- Preparation du dataset BreakHis.
- Extraction de `patient_id` et `magnification`.
- Split `train` / `val` / `test` patient-aware.
- Entrainement ResNet18 pre-entraine.
- Evaluation binaire.
- Prediction CLI.
- Demo Streamlit avec Grad-CAM.

### V1.1 - Streamlit Screenshots

- Captures de la page Streamlit.
- Prediction `benign`.
- Prediction `malignant`.
- Integration visuelle dans le README.

### V2 - Model Comparison

- Benchmark ResNet18, MobileNetV3 small et EfficientNet-B0.
- Generation de `summary.csv` et `summary.json`.
- Classification reports.
- Matrices de confusion.
- Courbes training loss / accuracy.
- Analyse par grossissement.

### V2.1 - Error Analysis

- Analyse de 1 481 images test.
- Separation des predictions correctes et erreurs.
- Comptage des faux positifs `benign -> malignant`.
- Comptage des faux negatifs `malignant -> benign`.
- Analyse par grossissement.
- Analyse par patient.
- Exemples locaux avec Grad-CAM.

### V2.2 - Portfolio Publishing Pack

- Resume projet.
- Pitch entretien.
- Brouillons LinkedIn.
- Release notes.
- README finalise pour presentation portfolio.

## Resultats Cles

### Dataset Et Split

| Element | Valeur |
| --- | ---: |
| Images BreakHis | 7 909 |
| Images `benign` | 2 480 |
| Images `malignant` | 5 429 |
| Patients detectes | 81 |
| Images train | 5 153 |
| Images val | 1 275 |
| Images test | 1 481 |
| Patient overlap | 0 |

### V1 - ResNet18

| Metrique | Valeur |
| --- | ---: |
| Accuracy test | 0.8947 |
| Macro F1 | 0.8800 |
| Recall `malignant` | 0.9466 |

### V2 - Model Comparison

| Modele | Accuracy | Macro F1 | Recall malignant | Train time |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0.8947 | 0.8800 | 0.9466 | 510.19 s |
| MobileNetV3 small | 0.8575 | 0.8206 | 0.9979 | 316.66 s |
| EfficientNet-B0 | 0.9122 | 0.9004 | 0.9568 | 762.23 s |

### V2.1 - Error Analysis EfficientNet-B0

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

Observation importante : le patient `14-16184CD` concentre 76 erreurs, ce qui montre que
les erreurs ne sont pas reparties uniformement et justifie une lecture patient-aware.

## Limites

- Dataset public BreakHis, sans validation clinique externe.
- Split patient-aware local, pas validation multi-centres.
- Resultats dependants du split, du preprocessing et des hyperparametres.
- Grad-CAM reste une visualisation exploratoire.
- Pas de certification, pas de validation medicale, pas d'usage clinique.

## Fichiers Volontairement Exclus

Les elements suivants restent locaux et ne sont pas versionnes :

- `projects/breast/data/`
- `projects/breast/outputs/`
- checkpoints `.pt`, `.pth`, `.ckpt`
- tokens Kaggle et fichiers `kaggle.json` / `access_token`
- `.venv/`

Cette release doit etre presentee comme un projet IA/data educatif et portfolio, jamais
comme un outil de diagnostic.
