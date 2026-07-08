# Breast V2.1 - Error Analysis + Magnification/Patient Analysis

## Objectif

Breast V2.1 analyse les erreurs du meilleur modele global du benchmark V2 sur BreakHis :
`efficientnet_b0`.

Le but est de depasser l'accuracy globale et de comprendre :

- les erreurs `benign -> malignant` ;
- les erreurs `malignant -> benign` ;
- l'impact des grossissements `40X`, `100X`, `200X`, `400X` ;
- les patients qui concentrent le plus d'erreurs ;
- la difference de confiance entre predictions correctes et erreurs ;
- quelques exemples locaux avec Grad-CAM.

> Important : cette analyse reste un demonstrateur educatif / portfolio. Elle ne constitue
> pas une validation medicale et ne doit jamais etre interpretee comme un diagnostic.

## Checkpoint Analyse

Checkpoint utilise localement :

```text
projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt
```

Modele :

- architecture : `efficientnet_b0` ;
- dataset : BreakHis ;
- split : patient-aware ;
- classes : `benign`, `malignant` ;
- test set : 1 481 images ;
- metadata : `patient_id` et `magnification` disponibles pour toutes les images test.

## Commande

Depuis la racine du monorepo :

```powershell
.\.venv\Scripts\python.exe projects\breast\scripts\analyze_errors.py --data-dir projects\breast\data\processed --checkpoint projects\breast\outputs\model_comparison\efficientnet_b0\best_model.pt --model efficientnet_b0 --output-dir projects\breast\outputs\error_analysis --metadata projects\breast\data\raw\metadata.csv --max-examples 15
```

Fichiers generes localement :

```text
projects/breast/outputs/error_analysis/
|-- predictions.csv
|-- summary.json
`-- examples/
```

Ces fichiers ne sont pas versionnes.

## Resultats Reels

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

Lecture rapide : le modele EfficientNet-B0 fait davantage de faux positifs que de faux
negatifs. Il detecte fortement la classe `malignant`, mais classe parfois des images
`benign` comme `malignant`.

## Erreurs Par Grossissement

| Grossissement | Images test | Correctes | Erreurs | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `40X` | 378 | 331 | 47 | 0.8757 |
| `100X` | 397 | 365 | 32 | 0.9194 |
| `200X` | 375 | 355 | 20 | 0.9467 |
| `400X` | 331 | 300 | 31 | 0.9063 |

Le grossissement `40X` est le plus difficile dans cette analyse. Le `200X` reste le
plus favorable, comme dans le benchmark V2.

## Erreurs Par Patient

Top patients avec erreurs :

| Patient | Images | Correctes | Erreurs | Accuracy | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `14-16184CD` | 124 | 48 | 76 | 0.3871 | 76 | 0 |
| `14-10926` | 39 | 22 | 17 | 0.5641 | 0 | 17 |
| `14-18842D` | 64 | 51 | 13 | 0.7969 | 0 | 13 |
| `14-22549AB` | 121 | 110 | 11 | 0.9091 | 11 | 0 |
| `14-19440` | 142 | 137 | 5 | 0.9648 | 0 | 5 |

Lecture rapide : les erreurs ne sont pas reparties uniformement. Certains patients
concentrent une grande partie des erreurs, ce qui confirme l'interet du split
patient-aware et d'une analyse au niveau patient.

## Exemples Visuels Et Grad-CAM

Le script exporte localement jusqu'a 15 exemples par categorie :

- `false_positive_*.png` ;
- `false_negative_*.png` ;
- `most_confident_error_*.png` ;
- `correct_low_confidence_*.png` ;
- variantes `_gradcam.png` quand Grad-CAM est disponible.

Resultat local V2.1 :

- exemples exportes : 60 ;
- erreurs Grad-CAM : 0.

Ces images restent dans `projects/breast/outputs/error_analysis/examples/` et ne sont
pas commitees.

## Interpretation Portfolio

EfficientNet-B0 reste le meilleur compromis global sur ce split patient-aware :

- meilleure accuracy V2 : 0.9122 ;
- meilleure macro F1 V2 : 0.9004 ;
- recall `malignant` eleve : 0.9568 ;
- erreur dominante : `benign -> malignant`.

MobileNetV3 small maximise le recall `malignant` dans le benchmark V2, mais son F1
`benign` est plus faible. L'analyse V2.1 permet donc d'expliquer le compromis entre
recall `malignant` et faux positifs sur `benign`.

## Limites

- Dataset public BreakHis, sans validation clinique externe.
- Split patient-aware local, utile contre le leakage mais pas equivalent a une cohorte
  clinique independante.
- Les metriques dependent du split, du preprocessing, du checkpoint et des
  hyperparametres.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
- Le projet est educatif / portfolio uniquement, pas un outil medical.
