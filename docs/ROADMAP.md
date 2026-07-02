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

Statut : prochain.

Objectif propose : construire un deuxieme projet histopathologique a partir d'un dataset public lung/colon, avec cadrage portfolio et disclaimer medical.

Premieres etapes :

1. Choisir le dataset public.
2. Documenter les classes et les limites.
3. Creer un pipeline minimal separe dans `projects/lung_colon`.
4. Reutiliser les bonnes pratiques du projet leucemie sans copier aveuglement tout le code.
5. Ajouter une demo seulement apres une baseline propre.

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
