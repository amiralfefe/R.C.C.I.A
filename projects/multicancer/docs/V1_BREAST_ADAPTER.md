# MultiCancer V1.2 - Breast Adapter

## Objectif

La V1.2 integre le pipeline Breast au hub MultiCancer en reutilisant le contrat commun
stabilise avec Leukemia. Le pipeline specialise BreakHis reste inchange : preprocessing,
EfficientNet-B0, prediction et Grad-CAM sont appeles depuis `rccia_breast`.

> Demonstrateur educatif / portfolio uniquement. Aucun diagnostic, aucune recommandation
> medicale et aucune validation clinique.

## Pourquoi Breast En Deuxieme ?

Breast verifie que le contrat commun fonctionne avec un protocole plus exigeant : split
patient-aware, quatre grossissements et erreurs concentrees sur certains patients. Cette
integration teste la capacite du hub a afficher le contexte methodologique sans reduire
le projet a son accuracy.

## Contrat Reutilise

`BreastAdapter` implemente le meme contrat que Leukemia :

- `metadata()` ;
- `checkpoint_status()` ;
- `load()` ;
- `predict(image)` ;
- `explain(image, class_index)` ;
- `unload()` ;
- `is_loaded`.

Les deux adaptateurs heritent de `TorchvisionImageAdapter`, qui centralise le cycle de
vie, la normalisation des resultats et la gestion des erreurs. Chaque sous-classe ne
declare que son projet, son package specialise et ses checkpoints locaux plausibles.

## Checkpoint EfficientNet-B0

Checkpoint local cible :

```text
projects/breast/outputs/model_comparison/efficientnet_b0/best_model.pt
```

Le fichier reste exclu de Git. Le hub verifie uniquement son existence avant chargement.
La compatibilite du state dict est controlee par `rccia_breast.model.load_checkpoint` au
moment ou l'utilisateur demande le chargement ou une prediction.

## Preprocessing Et Prediction

- conversion RGB ;
- resize `224x224` ;
- normalisation ImageNet identique au pipeline Breast ;
- classes `benign` et `malignant` ;
- sortie `PredictionResult` commune avec confiance et probabilites.

Aucune image uploadee n'est ecrite sur disque.

## Grad-CAM

Le hub reutilise `rccia_breast.gradcam` et retourne une `ExplanationResult` en memoire.
Un echec de couche cible, de backward ou d'image devient une erreur controlee sans crash
Streamlit.

## Split Patient-Aware

Le dataset BreakHis local comprend 7 909 images et 81 patients detectes. Le split utilise
55 patients en train, 11 en validation et 15 en test, avec un overlap patient nul entre
les trois groupes.

Cette precaution reduit le risque de fuite d'images d'un meme patient entre apprentissage
et test. Elle ne remplace pas une validation externe ou multi-centres.

## Grossissements

Resultats V2.1 affiches dans le hub :

| Grossissement | Accuracy |
| --- | ---: |
| `40X` | 0.8757 |
| `100X` | 0.9194 |
| `200X` | 0.9467 |
| `400X` | 0.9063 |

Les performances varient donc avec le grossissement ; elles ne doivent pas etre
generalisees a d'autres protocoles d'acquisition.

## Analyse Par Patient

Le test set contient 1 481 images et 130 erreurs : 88 faux positifs `benign -> malignant`
et 42 faux negatifs `malignant -> benign`. Le patient `14-16184CD` concentre 76 erreurs.

Cette concentration illustre une variabilite patient importante. Elle ne permet aucune
conclusion clinique et ne doit pas etre extrapolee hors du dataset.

## Lazy Loading Et Unload

`ModelManager` enregistre Leukemia et Breast, mais ne conserve qu'un adaptateur actif :

```text
Leukemia charge -> selection Breast -> unload Leukemia -> Breast actif
Breast charge   -> selection Leukemia -> unload Breast -> Leukemia actif
```

Le cache CUDA est vide lors de l'unload lorsque CUDA est disponible.

## Gestion Des Erreurs

- checkpoint absent : hub consultable, prediction desactivee ;
- checkpoint incompatible : `CheckpointIncompatibleError` ;
- image invalide : `InvalidImageError` ;
- prediction avant chargement : `ModelNotLoadedError` ;
- Grad-CAM indisponible : `ExplanationUnavailableError`.

## Tests

Les tests utilisent des backends simules et ne dependent pas du checkpoint local. Ils
couvrent metadata, probabilites, erreurs, Grad-CAM, unload et bascules de projet.

Le parcours local peut aussi etre teste avec le vrai checkpoint, sans versionner les
images ou artefacts produits.

## Lancement

Depuis la racine :

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

Tests :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Prochaine Integration

Metastasis sera le troisieme adaptateur. Il ajoutera le contexte ROC-AUC, PR-AUC et
analyse de seuils au meme format commun.
