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

Statut : termine, V2.2 portfolio publishing pack.

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
- analyse V2.1 : EfficientNet-B0, 130 erreurs sur 1 481 images test ;
- V2.1 : 88 false positives `benign -> malignant` et 42 false negatives `malignant -> benign` ;
- V2.1 : analyse par grossissement, par patient, confiance et exemples Grad-CAM locaux ;
- pack final portfolio : resume projet, pitch entretien, brouillons LinkedIn et release notes ;
- tests smoke CPU rapides.

Prochaines etapes Breast :

1. Stopper les ajouts majeurs sur Breast sauf correction documentaire.
2. Utiliser Breast pour CV, LinkedIn et portfolio.
3. Demarrer le prochain sous-projet : Metastasis.
4. Garder le discours strictement educatif, sans promesse medicale.

## Phase 4 - Metastasis

Statut : termine, V2.2 portfolio publishing pack.

Objectif : detection ou classification de patches lies a des metastases sur dataset public,
typiquement PCam / PatchCamelyon ou equivalent.

V1 initialisee :

- structure `projects/metastasis` ;
- package `rccia_metastasis` ;
- classification binaire `non_metastatic` vs `metastatic` ;
- scripts de preparation PCam-like et split classique ;
- conversion HDF5 PCam non compresse vers ImageFolder preparee ;
- support ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- entrainement, evaluation, prediction CLI ;
- evaluation avec accuracy, precision, recall, F1, ROC-AUC et PR-AUC ;
- matrices de confusion, courbes ROC et precision/recall ;
- demo Streamlit V1 avec Grad-CAM si checkpoint disponible ;
- dataset Kaggle `tyson04/pcam-validate` telecharge localement ;
- subset equilibre de 5 000 patches PCam converti en ImageFolder ;
- baseline ResNet18 pre-entrainee, accuracy test 0.9040 ;
- ROC-AUC 0.9598 et PR-AUC 0.9616 sur le test local ;
- captures Streamlit V1.1 ajoutees au README ;
- benchmark V2 ResNet18 / MobileNetV3 small / EfficientNet-B0 ;
- meilleur score V2 : EfficientNet-B0, accuracy 0.9320, macro F1 0.9320,
  ROC-AUC 0.9762 et PR-AUC 0.9780 ;
- meilleur recall `metastatic` V2 : ResNet18, recall 0.9307 ;
- modele le plus rapide V2 : MobileNetV3 small, 64.82 s d'entrainement ;
- analyse V2.1 sur EfficientNet-B0 : 699 correctes sur 750 images test ;
- V2.1 : 24 false positives et 27 false negatives au seuil 0.50 ;
- V2.1 : analyse de seuils 0.30 / 0.40 / 0.50 / 0.60 / 0.70 ;
- V2.1 : Grad-CAM genere localement pour les exemples exportes ;
- pack final portfolio : resume projet, pitch entretien, brouillons LinkedIn et release notes ;
- tests smoke CPU rapides.

Version de reference :

```text
metastasis-v2.1-threshold-analysis
```

Prochaines etapes Metastasis :

1. Stopper les ajouts majeurs sur Metastasis sauf correction documentaire.
2. Utiliser Metastasis pour CV, LinkedIn et portfolio.
3. Eventuellement elargir au train split complet si le temps CPU/GPU le permet.
4. Comparer les seuils sur un autre split PCam si disponible.
5. Demarrer la synthese transversale MultiCancer.
6. Garder le discours strictement educatif, sans promesse medicale.

## Phase 5 - MultiCancer

Statut : V1 complete / portfolio-ready.

Objectif : creer un hub transversal qui route explicitement vers les pipelines
specialises sans fusionner les datasets ni presenter un modele medical universel.

Livrables V0 :

- scope et decisions techniques documentes ;
- architecture cible par adaptateurs ;
- matrice comparative Leukemia / LungColon / Breast / Metastasis ;
- registre de metadonnees independant des checkpoints ;
- schemas communs `ProjectMetadata` et `PredictionResult` ;
- page Streamlit V0 sans modele, dataset ou prediction ;
- tests rapides du registre et des schemas.

Livrables V1.1 :

- contrat commun `BaseAdapter` ;
- schemas checkpoint, prediction et explication ;
- gestionnaire garantissant un seul modele actif ;
- adaptateur Leukemia reutilisant prediction et Grad-CAM existants ;
- chargement paresseux du checkpoint local ;
- prediction Streamlit normalisee et gestion des erreurs ;
- dechargement lors du changement de projet ;
- tests sans dependance au vrai checkpoint.

Livrables V1.2 :

- socle Torchvision reutilisable sans duplication du pipeline specialise ;
- adaptateur Breast avec checkpoint EfficientNet-B0 local ;
- prediction et Grad-CAM normalises ;
- contexte patient-aware et performances par grossissement visibles ;
- bascule Leukemia / Breast avec un seul modele actif ;
- tests Breast sans dependance au vrai checkpoint.

Livrables V1.3 :

- adaptateur Metastasis avec EfficientNet-B0 et preprocessing 96x96 ;
- probabilites et argmax originaux preserves ;
- couche `ThresholdDecision` separee de l'inference ;
- slider exploratoire sans rechargement ni nouvelle prediction ;
- tableau ROC/PR-AUC et FP/FN documente ;
- tests Metastasis et seuils sans dependance au vrai checkpoint.

Livrables V1.4 :

- adaptateur LungColon avec modes multiclass et binary explicites ;
- checkpoints EfficientNet-B0 et ResNet18 strictement separes ;
- cinq ou deux probabilites selon le mode actif ;
- unload et nettoyage de session lors d'une bascule ;
- Grad-CAM par mode et limites LC25000 visibles ;
- aucun chargement simultane des deux modeles.

Livrables V1.5 :

- QA reelle des cinq parcours avec checkpoints locaux ;
- verification Streamlit AppTest des bascules, probabilites, Grad-CAM et seuils ;
- validation des erreurs controlees et du lifecycle memoire ;
- captures finales du hub ;
- resume projet, pitch entretien, brouillons LinkedIn et release notes ;
- README final portfolio ;
- tag de reference `multicancer-v1`.

Statut final V1 :

1. Leukemia : integre.
2. Breast : integre.
3. Metastasis : integre.
4. LungColon : integre, avec modes 5 classes et binaire explicites.
5. Conserver le chargement paresseux d'un seul modele.
6. Afficher les limites methodologiques avec chaque prediction.

MultiCancer V1 est termine. Aucun chantier V2 n'est ajoute automatiquement a cette
roadmap.

Leukemia, LungColon, Breast et Metastasis restent termines. Leurs pipelines ne doivent
pas etre modifies par MultiCancer sauf correction d'interface ciblee et testee.

## Regles Permanentes

- Ne pas commit `data/`, `outputs/`, checkpoints ou tokens.
- Garder les disclaimers medicaux visibles.
- Favoriser des pipelines simples, testables et reproductibles.
- Documenter les resultats reels uniquement.
- Distinguer clairement smoke tests, benchmarks et resultats finaux.
