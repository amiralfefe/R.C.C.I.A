# Roadmap R.C.C.I.A

R.C.C.I.A est organise comme une suite de sous-projets IA/data. Chaque sous-projet doit rester reproductible, documente, teste et presente comme demonstrateur educatif, jamais comme outil medical.

## Phase 1 - Leukemia

Statut : termine.

Objectif : classifier des images microscopiques de cellules sanguines en `normal` et `leukemia_blast`.

Livrables :

- pipeline PyTorch complet ;
- dataset Kaggle prepare localement ;
- split train / val / test ;
- Streamlit demo ;
- Grad-CAM ;
- benchmark multi-modeles ;
- analyse false positives / false negatives ;
- docs CV, LinkedIn, entretien.

Version de reference :

```text
v2.2-error-analysis
```

## Phase 2 - Lung + Colon

Statut : termine, V2.2 binary mode + portfolio publishing pack.

Objectif : construire un deuxieme projet histopathologique a partir du dataset public LC25000, avec cadrage portfolio et disclaimer medical.

V1 realisee :

- structure `projects/lung_colon` ;
- package `rccia_lung_colon` ;
- classification 5 classes ;
- scripts de preparation LC25000 et split ;
- entrainement, evaluation, prediction CLI ;
- demo Streamlit + Grad-CAM ;
- tests smoke sur dataset synthetique ;
- baseline ResNet18 pre-entrainee sur 25 000 images ;
- evaluation test multi-classe : accuracy 0.9965 ;
- comparaison ResNet18 / MobileNetV3 small / EfficientNet-B0 ;
- benchmark V2 : meilleur score observe avec EfficientNet-B0, accuracy 0.9992 ;
- analyse V2.1 : 3 erreurs sur 3 750 images test, toutes entre sous-types malins pulmonaires.
- mode V2.2 binaire `benign` vs `malignant` avec ResNet18, accuracy test 1.0000 et recall `malignant` 1.0000 sur le split local.
- pack final portfolio : resume projet, pitch entretien, brouillons LinkedIn et release notes.

Version de reference :

```text
lung-colon-v2.2-binary-mode
```

Prochaines etapes :

1. Stopper les ajouts majeurs sur Lung + Colon sauf correction documentaire.
2. Utiliser Lung + Colon pour CV, LinkedIn et portfolio.
3. Demarrer le prochain sous-projet : Breast.
4. Garder toute nouvelle metrique clairement separee d'une validation medicale.

## Phase 3 - Breast

Statut : V2 model comparison terminee.

Objectif : classification `benign` / `malignant` sur BreakHis / Breast Cancer Histopathological Database.

V1 realisee :

- structure `projects/breast` ;
- package `rccia_breast` ;
- classification binaire `benign` vs `malignant` ;
- scripts de preparation BreakHis ;
- dataset Kaggle `ambarish/breakhis` prepare localement ;
- 7 909 images : 2 480 `benign`, 5 429 `malignant` ;
- extraction de grossissement `40X`, `100X`, `200X`, `400X` ;
- 81 patients detectes ;
- split patient-aware sans overlap patient ;
- train / val / test : 5 153 / 1 275 / 1 481 images ;
- entrainement, evaluation, prediction CLI ;
- baseline ResNet18 pre-entrainee, epochs 3, image size 224 ;
- accuracy test 0.8947 ;
- recall `malignant` 0.9466 ;
- app Streamlit V1 testee HTTP 200 ;
- captures Streamlit ajoutees au README ;
- benchmark patient-aware ResNet18 / MobileNetV3 small / EfficientNet-B0 ;
- meilleur score V2 : EfficientNet-B0, accuracy 0.9122 et macro F1 0.9004 ;
- meilleur recall `malignant` V2 : MobileNetV3 small, recall 0.9979 ;
- tests smoke CPU rapides.

Prochaines etapes Breast :

1. Ajouter error analysis, notamment sur les faux `malignant -> benign`.
2. Comparer les erreurs par grossissement et par confiance.
3. Preparer un pack portfolio Breast.
4. Eventuellement tester une strategie de seuil pour arbitrer recall `malignant` vs faux positifs.
5. Garder le discours strictement educatif, sans promesse medicale.

## Phase 4 - Metastasis

Statut : prevu.

Objectif possible : detection ou classification de patches lies a des metastases sur dataset public.

Axes techniques :

- patch classification ;
- courbes ROC / AUC si pertinent ;
- analyse des seuils ;
- visualisation des erreurs.

## Phase 5 - MultiCancer

Statut : prevu.

Objectif : creer une vue transversale du portfolio R.C.C.I.A.

Livrables possibles :

- tableau comparatif des sous-projets ;
- harmonisation des docs ;
- page portfolio unique ;
- presentation courte recruteur ;
- synthese des limites communes.

## Regles Permanentes

- Ne pas commit `data/`, `outputs/`, checkpoints ou tokens.
- Garder les disclaimers medicaux visibles.
- Favoriser des pipelines simples, testables et reproductibles.
- Documenter les resultats reels uniquement.
- Distinguer clairement smoke tests, benchmarks et resultats finaux.
