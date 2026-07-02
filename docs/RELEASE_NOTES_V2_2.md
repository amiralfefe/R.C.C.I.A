# Release Notes - v2.2-error-analysis

## Resume

La release `v2.2-error-analysis` fige une version portfolio complete de R.C.C.I.A, aussi presente sous le nom Cancer Cell Vision.

Elle inclut :

- V1 : pipeline ML reelle avec Streamlit et Grad-CAM ;
- V2 : workflow de comparaison de modeles ;
- V2.1 : benchmark reel ResNet18 / MobileNetV3 Small / EfficientNet-B0 ;
- V2.2 : analyse des erreurs et interpretabilite.

Cette release reste un demonstrateur educatif IA/data. Elle ne fournit pas de diagnostic medical.

## Fonctionnalites Principales

- Preparation du dataset Kaggle `andrewmvd/leukemia-classification`.
- Split train / validation / test.
- Entrainement PyTorch avec architectures Torchvision.
- Evaluation accuracy, precision, recall, F1-score.
- Matrice de confusion et rapports de classification.
- Prediction en ligne de commande.
- Interface Streamlit avec upload d'image.
- Probabilites par classe.
- Grad-CAM.
- Comparaison de modeles.
- Analyse false positives / false negatives.
- Exports locaux CSV / JSON pour l'analyse des erreurs.
- Tests automatises.
- README portfolio avec captures.

## Resultats Cles

### V1 Baseline

| Indicateur | Valeur |
| --- | ---: |
| Modele | ResNet18 |
| Epochs | 5 |
| Test accuracy | 0.9169 |
| F1 `normal` | 0.8702 |
| F1 `leukemia_blast` | 0.9389 |

### V2.1 Benchmark

| Modele | Accuracy | F1 normal | F1 leukemia_blast | Recall leukemia_blast |
| --- | ---: | ---: | ---: | ---: |
| `resnet18` | 0.8988 | 0.8251 | 0.9288 | 0.9679 |
| `mobilenet_v3_small` | 0.8976 | 0.8295 | 0.9268 | 0.9505 |
| `efficientnet_b0` | 0.8863 | 0.8080 | 0.9193 | 0.9487 |

### V2.2 Error Analysis

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

## Fichiers Exclus Volontairement

Les fichiers suivants restent locaux et ne sont pas committes :

- `data/raw/`
- `data/processed/`
- `outputs/`
- checkpoints `*.pt`, `*.pth`, `*.ckpt`
- token Kaggle
- `access_token`
- `kaggle.json`
- `.venv/`

## Limites

- Dataset public sans validation clinique independante.
- Classes desequilibrees.
- Entrainements courts sur CPU.
- Resultats dependants du split, du preprocessing et des hyperparametres.
- Grad-CAM reste une aide visuelle exploratoire.
- Aucun usage medical, aucun diagnostic, aucune decision de sante.

## Statut

- tag Git : `v2.2-error-analysis`
- branche : `main`
- statut : version portfolio publiable.
