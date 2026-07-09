# Metastasis V2.1 - Error Analysis + ROC/Threshold Analysis

## Objectif

Metastasis V2.1 analyse les erreurs du meilleur modele global de V2, `efficientnet_b0`,
et mesure l'impact du seuil `metastatic` sur le compromis precision / recall / F1.

Ce projet reste un demonstrateur educatif / portfolio. Il ne fournit pas de diagnostic
medical, ne remplace pas une validation clinique et ne doit jamais orienter une decision
de sante.

## Modele et Checkpoint

| Element | Valeur |
| --- | --- |
| Modele | `efficientnet_b0` |
| Checkpoint | `projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt` |
| Dataset | subset PCam / PatchCamelyon local |
| Image size | 96 |
| Images test | 750 |
| Seuil par defaut | 0.50 |

## Resultats au Seuil 0.50

| Metrique | Valeur |
| --- | ---: |
| Accuracy | 0.9320 |
| Images correctes | 699 |
| Erreurs | 51 |
| False positives (`non_metastatic -> metastatic`) | 24 |
| False negatives (`metastatic -> non_metastatic`) | 27 |
| Confiance moyenne correctes | 0.9071 |
| Confiance moyenne erreurs | 0.7107 |
| ROC-AUC | 0.9762 |
| PR-AUC | 0.9780 |

Lecture : le modele fait legerement plus de faux negatifs que de faux positifs au seuil
0.50. Les erreurs ont une confiance moyenne plus basse que les predictions correctes,
mais certains cas faux restent tres confiants, ce qui justifie l'analyse qualitative et
Grad-CAM.

## Analyse des Seuils

| Threshold | Accuracy | Precision `metastatic` | Recall `metastatic` | F1 `metastatic` | FP | FN | Predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.8907 | 0.8399 | **0.9653** | 0.8983 | 69 | **13** | 431 |
| 0.40 | 0.9160 | 0.8861 | 0.9547 | 0.9191 | 46 | 17 | 404 |
| 0.50 | **0.9320** | 0.9355 | 0.9280 | **0.9317** | 24 | 27 | 372 |
| 0.60 | 0.9187 | 0.9460 | 0.8880 | 0.9161 | 19 | 42 | 352 |
| 0.70 | 0.9120 | **0.9668** | 0.8533 | 0.9065 | **11** | 55 | 331 |

Interpretation :

- seuil plus bas : plus de predictions `metastatic`, recall `metastatic` plus eleve,
  mais plus de faux positifs ;
- seuil plus haut : precision `metastatic` plus elevee et moins de faux positifs, mais
  davantage de faux negatifs ;
- le seuil 0.50 maximise le F1 `metastatic` dans cette grille, mais ce n'est pas un
  seuil medical recommande.

## Exemples et Grad-CAM

Le script a genere des exemples locaux dans :

```text
projects/metastasis/outputs/error_analysis/examples/
```

Categories exportees :

- `false_positive_*.png`
- `false_negative_*.png`
- `most_confident_error_*.png`
- `low_confidence_correct_*.png`
- variantes `*_gradcam.png`

Grad-CAM a ete genere avec succes pour les exemples exportes. Ces images restent locales
et ne sont pas versionnees.

## Fichiers Generes Localement

```text
projects/metastasis/outputs/error_analysis/
|-- error_cases.csv
|-- error_summary.json
|-- threshold_analysis.csv
|-- threshold_analysis.png
`-- examples/
```

Ces fichiers ne doivent pas etre commit.

## Reproduction

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\analyze_errors.py --data-dir projects\metastasis\data\processed --checkpoint projects\metastasis\outputs\model_comparison\efficientnet_b0\best_model.pt --model efficientnet_b0 --image-size 96 --output-dir projects\metastasis\outputs\error_analysis --thresholds 0.30 0.40 0.50 0.60 0.70 --max-examples 20
```

Pour ignorer Grad-CAM pendant un test rapide :

```powershell
.\.venv\Scripts\python.exe projects\metastasis\scripts\analyze_errors.py --data-dir projects\metastasis\data\processed --checkpoint projects\metastasis\outputs\model_comparison\efficientnet_b0\best_model.pt --model efficientnet_b0 --image-size 96 --output-dir projects\metastasis\outputs\error_analysis --thresholds 0.30 0.40 0.50 0.60 0.70 --max-examples 20 --skip-gradcam
```

## Limites

- Benchmark educatif sur subset PCam public, pas validation clinique.
- Subset equilibre de 5 000 patches, dont 750 images test.
- Analyse limitee a cinq seuils fixes.
- Grad-CAM est une visualisation exploratoire, pas une preuve medicale.
- Aucun seuil ne doit etre interprete comme seuil de decision medicale.
