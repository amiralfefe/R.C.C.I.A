# MultiCancer Architecture

## Principe

MultiCancer adopte une architecture modulaire avec un adaptateur par projet specialise.
Le hub fournit l'interface commune ; chaque adaptateur reste responsable du contrat reel
de son pipeline.

La V0 ne contient aucun adaptateur fonctionnel. Elle definit uniquement les frontieres
qui seront implementees en V1.

## Composants Prevus

- **MultiCancer Streamlit Hub** : navigation, rendu commun et avertissements ;
- **Project Registry** : metadonnees stables des quatre projets ;
- **Specialized Adapters** : ponts vers les packages existants ;
- **Common Prediction Schema** : resultat normalise d'une inference ;
- **Common Metrics Schema** : presentation commune sans effacer le protocole source ;
- **Common Explainability Schema** : statut et artefact Grad-CAM optionnel ;
- **Disclaimer / Limitations Layer** : avertissement global et limites par projet.

## Architecture Cible

```text
projects/multicancer/
|-- README.md
|-- app.py
|-- multicancer/
|   |-- __init__.py
|   |-- registry.py
|   |-- schemas.py
|   |-- router.py
|   |-- adapters/
|   |   |-- __init__.py
|   |   |-- leukemia_adapter.py
|   |   |-- lung_colon_adapter.py
|   |   |-- breast_adapter.py
|   |   `-- metastasis_adapter.py
|   `-- ui/
|       |-- __init__.py
|       |-- comparison.py
|       `-- disclaimers.py
|-- docs/
|   `-- ROADMAP_V1.md
`-- tests/
    `-- test_multicancer_registry.py
```

`router.py`, `adapters/` et `ui/` sont des cibles V1, pas des livrables V0.

## Project Registry

Le registre V0 expose les metadonnees suivantes :

- `project_id` ;
- `display_name` ;
- `modality` ;
- `task` ;
- `classes` ;
- `image_size` ou description de resolution ;
- `status` ;
- `supports_gradcam` ;
- `primary_metrics` ;
- `dataset` ;
- `methodological_note`.

En V1, une indication relative et optionnelle de checkpoint pourra etre ajoutee. Son
existence ne sera jamais supposee au chargement du registre.

## Common Prediction Schema

Le schema conceptuel `PredictionResult` contient :

- `project_id` ;
- `predicted_class` ;
- `confidence` ;
- `class_probabilities` ;
- `model_name` ;
- `image_size` ;
- `explanation_path`, optionnel ;
- `warnings` ;
- `disclaimer`.

Une dataclass standard suffit en V0. Aucun framework de validation supplementaire n'est
necessaire.

## Common Metrics Schema

Le hub pourra normaliser le format d'affichage, mais chaque valeur devra conserver son
contexte : dataset, split, architecture, taille d'image, nombre d'epochs et nature du
protocole. Les metriques specifiques, comme ROC-AUC, PR-AUC ou analyse par
grossissement, restent attachees au projet source.

## Router

Le routage V1 sera explicite :

1. l'utilisateur choisit le projet ou la tache ;
2. le hub selectionne l'adaptateur correspondant ;
3. l'adaptateur valide l'image et la disponibilite du checkpoint ;
4. le pipeline specialise produit un `PredictionResult` ;
5. le hub affiche le resultat avec les limites du projet.

La V1 n'effectuera pas de detection automatique opaque de la modalite.

## Adapters

Chaque adaptateur encapsulera :

- chargement du modele et de son checkpoint ;
- preprocessing exact du projet ;
- prediction et probabilites ;
- Grad-CAM si disponible ;
- metadonnees et metriques documentees ;
- limites et avertissements.

Les quatre pipelines stockent deja dans leurs checkpoints `model_state`, `class_names`,
`image_size` et `model_name`. Ce socle commun facilitera les adaptateurs, mais ceux-ci
resteront independants afin de respecter les packages et comportements existants.

## Gestion Des Checkpoints

Les checkpoints restent locaux, facultatifs et exclus de Git. Le registre peut etre
charge sans checkpoint. En V1, le hub chargera au plus un modele a la fois et affichera
un message clair lorsque l'artefact attendu est absent.

## Gestion Des Erreurs

Les erreurs suivantes devront etre gerees sans crash global :

- checkpoint absent ou incompatible ;
- image invalide ;
- format non supporte ;
- Grad-CAM indisponible ;
- projet ou adaptateur indisponible.

## Securite Et Cadrage

- aucun stockage patient ;
- aucune donnee personnelle ;
- aucun diagnostic ou conseil medical ;
- avertissement global toujours visible ;
- limites propres au projet affichees avec chaque resultat ;
- aucune comparaison de score sans rappel du protocole.
