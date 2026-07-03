# V2.1 - Error Analysis + Interpretability

## Objectif

Cette V2.1 ajoute une analyse des erreurs au sous-projet Lung + Colon Vision. Elle ne modifie pas le pipeline ML : elle exploite un checkpoint deja entraine pour analyser les predictions du test set.

Le projet reste un demonstrateur educatif IA/data pour portfolio. Il ne fournit pas de diagnostic medical, ne remplace pas un professionnel de sante et ne doit jamais orienter une decision medicale.

## Checkpoint Utilise

Checkpoint local :

```text
projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt
```

Modele :

```text
efficientnet_b0
```

Ce checkpoint vient du benchmark V2 multi-modeles. Il n'est pas versionne dans Git.

## Commande Lancee

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\lung_colon\scripts\analyze_errors.py --data-dir projects\lung_colon\data\processed --checkpoint projects\lung_colon\outputs\model_comparison\efficientnet_b0\best_model.pt --model efficientnet_b0 --output-dir projects\lung_colon\outputs\error_analysis --max-examples 15
```

## Fichiers Generes

```text
projects/lung_colon/outputs/error_analysis/
|-- predictions.csv
|-- summary.json
`-- examples/
    |-- most_confident_error_001.png
    |-- most_confident_error_001_gradcam.png
    |-- cancer_subtype_confusion_001.png
    |-- cancer_subtype_confusion_001_gradcam.png
    |-- correct_low_confidence_001.png
    `-- correct_low_confidence_001_gradcam.png
```

Ces fichiers restent locaux et ne doivent pas etre committes.

## Colonnes De `predictions.csv`

- `image_path`
- `true_label`
- `predicted_label`
- `confidence`
- `prob_colon_adenocarcinoma`
- `prob_colon_benign`
- `prob_lung_adenocarcinoma`
- `prob_lung_benign`
- `prob_lung_squamous_cell_carcinoma`
- `is_correct`
- `error_type`

## Types D'erreurs

- `correct` : prediction correcte.
- `same_organ_confusion` : erreur entre classes du meme organe.
- `lung_colon_confusion` : erreur entre poumon et colon.
- `benign_malignant_confusion` : confusion entre classe benigne et classe maligne.
- `cancer_subtype_confusion` : confusion entre sous-types malins du meme organe, ici surtout `lung_adenocarcinoma` et `lung_squamous_cell_carcinoma`.

## Resultats Reels

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

Erreurs par vraie classe :

| Vraie classe | Erreurs |
| --- | ---: |
| `lung_adenocarcinoma` | 1 |
| `lung_squamous_cell_carcinoma` | 2 |

Erreurs par classe predite :

| Classe predite | Erreurs |
| --- | ---: |
| `lung_adenocarcinoma` | 2 |
| `lung_squamous_cell_carcinoma` | 1 |

Confusions principales :

| Confusion | Count |
| --- | ---: |
| `lung_squamous_cell_carcinoma -> lung_adenocarcinoma` | 2 |
| `lung_adenocarcinoma -> lung_squamous_cell_carcinoma` | 1 |

Types d'erreurs :

| Type | Count |
| --- | ---: |
| `cancer_subtype_confusion` | 3 |
| `benign_malignant_confusion` | 0 |
| `lung_colon_confusion` | 0 |
| `same_organ_confusion` | 0 |

## Lecture

Le modele se trompe tres peu sur ce split LC25000 : 3 erreurs sur 3 750 images. Les erreurs restantes sont concentrees sur une distinction fine entre deux sous-types pulmonaires malins :

- `lung_squamous_cell_carcinoma -> lung_adenocarcinoma`
- `lung_adenocarcinoma -> lung_squamous_cell_carcinoma`

Il n'y a pas de confusion colon/poumon et pas de confusion benin/malin dans cette analyse locale.

La confiance moyenne des predictions correctes est tres elevee (`0.9986`). La confiance moyenne des erreurs reste elevee aussi (`0.8624`), ce qui rend utile l'analyse qualitative des exemples et des Grad-CAM.

## Streamlit

La demo Streamlit affiche automatiquement une section `Analyse des erreurs V2.1` si ce fichier existe localement :

```text
projects/lung_colon/outputs/error_analysis/summary.json
```

Si le fichier est absent, Streamlit affiche un message propre avec la commande a lancer pour generer l'analyse.

## Limites

- LC25000 est un dataset public propre et relativement facile.
- Les resultats sont experimentaux et dependent du split, du preprocessing et du checkpoint local.
- Les exemples Grad-CAM sont des visualisations exploratoires, pas des explications medicales.
- Aucune metrique de ce projet ne constitue une validation clinique.
- Les donnees, outputs, checkpoints et tokens restent exclus de Git.
