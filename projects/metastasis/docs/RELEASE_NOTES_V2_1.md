# Metastasis Vision - Release Notes V2.1

## Resume

La V2.1 finalise Metastasis Vision comme projet portfolio autour de PCam /
PatchCamelyon.

Cette release couvre :

- classification `non_metastatic` vs `metastatic` ;
- conversion PCam HDF5 vers ImageFolder ;
- application Streamlit avec Grad-CAM ;
- benchmark multi-modeles ;
- evaluation ROC-AUC / PR-AUC ;
- analyse d'erreurs false positives / false negatives ;
- analyse de seuils ;
- documentation portfolio.

Tag de reference : `metastasis-v2.1-threshold-analysis`.

## Fonctionnalites Principales

### V1 - Baseline PCam

- Preparation d'un subset PCam equilibre.
- Conversion HDF5 vers `ImageFolder`.
- Split `train` / `val` / `test`.
- Entrainement ResNet18 pre-entraine.
- Evaluation binaire avec ROC-AUC et PR-AUC.
- Prediction CLI.
- Demo Streamlit avec Grad-CAM.

### V1.1 - Streamlit Screenshots

- Captures de la page Streamlit.
- Prediction `non_metastatic`.
- Prediction `metastatic`.
- Integration visuelle dans le README.

### V2 - Model Comparison

- Benchmark ResNet18, MobileNetV3 small et EfficientNet-B0.
- Generation de `summary.csv` et `summary.json`.
- Classification reports.
- Matrices de confusion.
- Courbes ROC et precision-recall.
- Courbes training loss / accuracy.

### V2.1 - Error And Threshold Analysis

- Analyse de 750 images test.
- Separation des predictions correctes et erreurs.
- Comptage des faux positifs `non_metastatic -> metastatic`.
- Comptage des faux negatifs `metastatic -> non_metastatic`.
- Analyse de seuils 0.30 / 0.40 / 0.50 / 0.60 / 0.70.
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
| Images PCam subset | 5 000 |
| Images `non_metastatic` | 2 500 |
| Images `metastatic` | 2 500 |
| Images train | 3 500 |
| Images val | 750 |
| Images test | 750 |

### V1 - ResNet18

| Metrique | Valeur |
| --- | ---: |
| Accuracy test | 0.9040 |
| ROC-AUC | 0.9598 |
| PR-AUC | 0.9616 |

### V2 - Model Comparison

| Modele | Accuracy | Macro F1 | ROC-AUC | PR-AUC | Recall metastatic | Train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.9040 | 0.9039 | 0.9706 | 0.9672 | **0.9307** | 112.86 s |
| MobileNetV3 small | 0.8573 | 0.8567 | 0.9374 | 0.9236 | 0.9227 | **64.82 s** |
| EfficientNet-B0 | **0.9320** | **0.9320** | **0.9762** | **0.9780** | 0.9280 | 93.17 s |

### V2.1 - Error And Threshold Analysis EfficientNet-B0

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

| Threshold | Accuracy | Recall metastatic | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| 0.30 | 0.8907 | **0.9653** | 69 | **13** |
| 0.40 | 0.9160 | 0.9547 | 46 | 17 |
| 0.50 | **0.9320** | 0.9280 | 24 | 27 |
| 0.60 | 0.9187 | 0.8880 | 19 | 42 |
| 0.70 | 0.9120 | 0.8533 | **11** | 55 |

Observation importante : baisser le seuil reduit les faux negatifs et augmente le recall
`metastatic`, mais augmente les faux positifs. Monter le seuil reduit les faux positifs,
mais augmente les faux negatifs.

## Limites

- Dataset public PCam / PatchCamelyon, sans validation clinique externe.
- Subset equilibre de 5 000 patches, pas corpus complet.
- Resultats dependants du split, du preprocessing, du checkpoint et des hyperparametres.
- Analyse limitee a cinq seuils fixes.
- Grad-CAM reste une visualisation exploratoire.
- Pas de certification, pas de validation medicale, pas d'usage clinique.

## Fichiers Volontairement Exclus

Les elements suivants restent locaux et ne sont pas versionnes :

- `projects/metastasis/data/`
- `projects/metastasis/outputs/`
- checkpoints `.pt`, `.pth`, `.ckpt`
- tokens Kaggle et fichiers `kaggle.json` / `access_token`
- `.venv/`

Cette release doit etre presentee comme un projet IA/data educatif et portfolio, jamais
comme un outil de diagnostic.
