# Lung + Colon Vision - Release Notes V2.2

## Resume

La V2.2 finalise Lung + Colon Vision comme projet portfolio complet autour du dataset public LC25000.

Cette release couvre :

- classification 5 classes lung/colon ;
- application Streamlit avec Grad-CAM ;
- benchmark multi-modeles ;
- analyse d'erreurs ;
- mode binaire `benign` vs `malignant` ;
- documentation portfolio.

Tag de reference : `lung-colon-v2.2-binary-mode`.

## Fonctionnalites Principales

### V1 - Baseline 5 Classes

- Preparation du dataset LC25000.
- Split `train` / `val` / `test`.
- Entrainement ResNet18 pre-entraine.
- Evaluation multi-classe.
- Prediction CLI.
- Demo Streamlit avec Grad-CAM.

### V2 - Model Comparison

- Benchmark ResNet18, MobileNetV3 small et EfficientNet-B0.
- Generation de `summary.csv` et `summary.json`.
- Classification reports.
- Matrices de confusion.
- Courbes training loss / accuracy.

### V2.1 - Error Analysis

- Analyse de 3 750 images test.
- Identification des predictions correctes et erreurs.
- Exemples Grad-CAM pour erreurs et predictions correctes a faible confiance.
- Synthese des confusions.

### V2.2 - Binary Mode

- Dataset binaire `benign` vs `malignant`.
- Entrainement ResNet18 pre-entraine.
- Evaluation binaire.
- Prediction CLI en mode binaire.
- Streamlit avec selection du mode 5 classes ou binaire.

## Resultats Cles

### Classification 5 Classes

| Modele | Accuracy | Macro F1 |
| --- | ---: | ---: |
| EfficientNet-B0 | 0.9992 | 0.9992 |
| ResNet18 | 0.9965 | 0.9965 |
| MobileNetV3 small | 0.9957 | 0.9957 |

### Error Analysis

| Metrique | Valeur |
| --- | ---: |
| Images test | 3 750 |
| Predictions correctes | 3 747 |
| Erreurs | 3 |
| Accuracy | 0.9992 |
| Erreurs benign/malignant | 0 |
| Erreurs lung/colon | 0 |

### Mode Binaire

| Classe | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| `benign` | 1.0000 | 1.0000 | 1.0000 |
| `malignant` | 1.0000 | 1.0000 | 1.0000 |

Accuracy test binaire : **1.0000**.

## Limites

- Dataset public LC25000, sans validation clinique externe.
- Scores tres eleves a interpreter comme benchmark educatif.
- Pas de certification, pas de validation medicale, pas d'usage clinique.
- Grad-CAM reste une visualisation exploratoire.

## Fichiers Volontairement Exclus

Les elements suivants restent locaux et ne sont pas versionnes :

- `projects/lung_colon/data/`
- `projects/lung_colon/outputs/`
- checkpoints `.pt`, `.pth`, `.ckpt`
- tokens Kaggle et fichiers `kaggle.json` / `access_token`
- `.venv/`

Cette release doit etre presentee comme un projet IA/data educatif et portfolio, jamais comme un outil de diagnostic.
