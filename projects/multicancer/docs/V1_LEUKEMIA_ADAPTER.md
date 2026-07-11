# MultiCancer V1.1 - Leukemia Adapter

## Objectif

La V1.1 stabilise le contrat commun des adaptateurs MultiCancer et integre Leukemia de
bout en bout. Elle reutilise le pipeline specialise existant sans modifier son
preprocessing, son modele, sa prediction ou son implementation Grad-CAM.

> MultiCancer et Leukemia sont des demonstrateurs educatifs pour portfolio. Ils ne
> fournissent aucun diagnostic, aucune recommandation medicale et aucune validation
> clinique.

## Pourquoi Leukemia En Premier ?

Leukemia fournit un parcours binaire deja termine : checkpoint documente, prediction
avec probabilites et Grad-CAM. Il permet de definir l'interface commune avec peu de cas
particuliers avant Breast, Metastasis et le double mode LungColon.

## Contrat Commun

`BaseAdapter` impose :

- `metadata()` : contexte stable, classes, modele, metriques et limites ;
- `checkpoint_status()` : `available`, `missing`, `incompatible` ou `unchecked` ;
- `load()` : chargement paresseux et idempotent ;
- `predict(image)` : sortie `PredictionResult` normalisee ;
- `explain(image, class_index)` : explication optionnelle ;
- `unload()` : liberation du modele et du cache CUDA ;
- `is_loaded` : etat observable sans charger le modele.

## Architecture

```text
Streamlit hub
  -> ModelManager
  -> LeukemiaAdapter
  -> projects/leukemia/rccia_leukemia
       |-- model.load_checkpoint / predict_image
       `-- gradcam.GradCAM
```

Le `ModelManager` decharge l'adaptateur actif avant tout changement de projet. Pour cette
phase, seul `leukemia` possede une factory de prediction.

## Lazy Loading

L'import du hub et la consultation du registre ne chargent pas PyTorch ni le checkpoint.
Le modele est charge seulement lorsque l'utilisateur clique sur le bouton de chargement
ou demande une analyse. Un second appel a `load()` reutilise le modele deja present.

## Gestion Du Checkpoint

Chemin local par defaut :

```text
projects/leukemia/outputs/best_model.pt
```

Le chemin est resolu relativement a la racine du monorepo. Il peut aussi etre fourni au
constructeur de l'adaptateur pour les tests ou une configuration locale. Le checkpoint
reste exclu de Git.

Si le fichier est absent ou incompatible, l'app affiche un message et reste consultable.

## PredictionResult

La prediction normalisee contient :

- projet, classe et indice predits ;
- confiance ;
- probabilites par classe ;
- modele et resolution ;
- avertissements et disclaimer ;
- metadonnees minimales du checkpoint si disponibles.

## Grad-CAM

L'adaptateur reutilise les fonctions `GradCAM`, `image_to_tensor`, `denormalize_image`
et `overlay_cam` du projet Leukemia. L'image explicative reste en memoire et n'est pas
ecrite sur disque. Un echec Grad-CAM devient `ExplanationUnavailableError` et ne fait pas
planter Streamlit.

## Gestion Des Erreurs

- contenu vide ou non-image : `InvalidImageError` ;
- checkpoint absent : `CheckpointMissingError` ;
- checkpoint illisible ou incompatible : `CheckpointIncompatibleError` ;
- prediction avant chargement : `ModelNotLoadedError` ;
- explication impossible : `ExplanationUnavailableError`.

L'upload accepte uniquement PNG et JPEG et n'est pas stocke durablement.

## Limites

- dataset public educatif et classes desequilibrees ;
- aucun split patient-aware documente ;
- aucune validation externe ou clinique ;
- Grad-CAM est exploratoire et ne constitue pas une preuve medicale ;
- seul Leukemia est integre dans cette phase.

## Lancement

Depuis la racine :

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

Tests :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Criteres De Validation

- Leukemia est selectionnable et les autres projets restent informatifs ;
- le checkpoint absent est gere sans crash ;
- le modele est charge uniquement a la demande ;
- la prediction, les probabilites et Grad-CAM utilisent le pipeline Leukemia existant ;
- un changement de projet decharge Leukemia ;
- aucun checkpoint, output, dataset ou upload n'est versionne.

## Prochaine Integration

Le prochain adaptateur sera Breast, avec preservation du contexte patient-aware et des
metriques par grossissement.
