# MultiCancer Project Matrix

Cette matrice decrit les quatre pipelines termines. Elle sert a comprendre leurs
differences, pas a etablir un classement medical ou une competition directe.

| Projet | Dataset | Modalite | Tache et classes | Resolution utilisee | Modele principal documente | Metriques principales | Split particulier | Analyse avancee | Limite principale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Leukemia | Kaggle `andrewmvd/leukemia-classification` | Cellules sanguines microscopiques | Binaire : `normal`, `leukemia_blast` | 224x224 | ResNet18 V1 | Accuracy 0.9169 ; F1 `normal` 0.8702 ; F1 `leukemia_blast` 0.9389 | Split image-level par classe ; classes desequilibrees | Benchmark multi-modeles, error analysis, Grad-CAM | Dataset public educatif, sans split patient-aware documente ni validation externe |
| LungColon | LC25000 | Histopathologie poumon et colon | 5 classes ; mode binaire `benign` / `malignant` additionnel | 224x224 | EfficientNet-B0 V2 pour le meilleur score 5 classes | Accuracy 0.9992 ; macro F1 0.9992 sur le benchmark V2 | Split image-level equilibre en 5 classes | Benchmark, analyse des erreurs, Grad-CAM, mode binaire | Scores tres eleves a interpreter avec prudence sur un benchmark public relativement facile |
| Breast | BreakHis | Histopathologie mammaire, grossissements 40X a 400X | Binaire : `benign`, `malignant` | 224x224 | EfficientNet-B0 V2 | Accuracy 0.9122 ; macro F1 0.9004 ; recall `malignant` 0.9568 | Split patient-aware, 0 patient commun entre train, val et test | Analyse par patient et grossissement, error analysis, Grad-CAM | Dataset public et split local ; variabilite patient importante, sans validation multi-centres |
| Metastasis | Kaggle `tyson04/pcam-validate`, subset PCam | Patches histopathologiques | Binaire : `non_metastatic`, `metastatic` | 96x96 | EfficientNet-B0 V2 | Accuracy 0.9320 ; ROC-AUC 0.9762 ; PR-AUC 0.9780 | Subset local equilibre de 5 000 patches | Error analysis, ROC/PR-AUC, threshold analysis, Grad-CAM | Subset du corpus et seuils experimentaux, sans validation clinique externe |

## Note D'Interpretation

Les accuracies, F1, ROC-AUC et PR-AUC ne sont pas directement comparables entre les
projets. Les datasets, classes, tailles d'image, splits, populations, niveaux de
difficulte et protocoles d'entrainement sont differents.

Le tableau doit donc etre lu comme une cartographie des choix methodologiques. Il ne
permet pas de conclure qu'un projet ou un modele est medicalement meilleur qu'un autre.
