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

Statut : V1 reelle documentee.

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
- evaluation test multi-classe : accuracy 0.9965.

Prochaines etapes :

1. Ajouter une comparaison de modeles sur LC25000.
2. Ajouter une analyse d'erreurs multi-classe.
3. Ajouter des captures Streamlit dediees au sous-projet Lung + Colon.
4. Documenter toute nouvelle metrique sans la presenter comme validation medicale.

## Phase 3 - Breast

Statut : prevu.

Objectif possible : classification benin / malin sur un dataset public type histopathologie du cancer du sein.

Points de vigilance :

- clarifier le niveau image / patch ;
- surveiller le desequilibre de classes ;
- eviter toute promesse medicale ;
- comparer les resultats avec un protocole explicite.

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
