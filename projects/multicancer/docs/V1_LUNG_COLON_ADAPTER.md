# MultiCancer V1.4 - LungColon Adapter Et Modes Explicites

## Objectif

La V1.4 integre le dernier pipeline specialise du hub. LungColon propose deux taches
distinctes et impose un choix explicite : aucune image n'est utilisee pour deviner le
mode et aucune probabilite d'un modele n'est transformee en sortie de l'autre mode.

> Demonstrateur educatif / portfolio uniquement. Aucun diagnostic, aucune validation
> clinique et aucune recommandation medicale.

## Pourquoi LungColon Est Integre En Dernier

LungColon valide que le contrat commun peut gerer plusieurs checkpoints au sein d'un
meme projet. Le hub doit conserver trois niveaux d'etat sans ambiguite : projet actif,
mode actif et checkpoint effectivement charge.

## Modes Disponibles

### Multiclass

- identifiant : `multiclass` ;
- modele : EfficientNet-B0 ;
- resolution : `224x224` ;
- checkpoint :
  `projects/lung_colon/outputs/model_comparison/efficientnet_b0/best_model.pt` ;
- accuracy locale : `0.9992` ;
- macro F1 local : `0.9992`.

Ordre exact des classes :

1. `colon_adenocarcinoma`
2. `colon_benign`
3. `lung_adenocarcinoma`
4. `lung_benign`
5. `lung_squamous_cell_carcinoma`

### Binary

- identifiant : `binary` ;
- modele : ResNet18 ;
- resolution : `224x224` ;
- checkpoint : `projects/lung_colon/outputs/binary_resnet18/best_model.pt` ;
- accuracy locale : `1.0000` ;
- recall `malignant` local : `1.0000`.

Ordre exact des classes :

1. `benign`
2. `malignant`

Le resultat binaire vient de ce checkpoint specialise. Il n'est jamais derive du modele
cinq classes.

## Preprocessing Confirme

Les deux checkpoints utilisent le preprocessing d'inference du projet specialise :

- conversion RGB ;
- resize `224x224` ;
- conversion tensor ;
- normalisation ImageNet.

Ce preprocessing est partage parce que le code LungColon l'utilise reellement pour les
deux modes, pas par convention du hub.

## AdapterModeMetadata

Chaque mode declare independamment sa tache, ses classes, son architecture, sa
resolution, son checkpoint, ses metriques, ses limites et le support Grad-CAM. Le
registre de projet indique seulement que LungColon supporte les modes `multiclass` et
`binary`.

## PredictionResult Commun

`predict(image)` reste conforme au contrat commun : argmax, indice, confiance et
probabilites softmax brutes. `raw_metadata` ajoute uniquement `mode_id` et
`mode_display_name`.

- multiclass retourne exactement cinq probabilites ;
- binary retourne exactement deux probabilites ;
- l'ordre suit strictement les classes du checkpoint et du mode actif.

## Lazy Loading Et Bascule

`loaded_mode` identifie le checkpoint reellement charge. Un modele n'est valide que si
`loaded_mode == current_mode`.

Lors d'une bascule :

1. l'ancien modele est decharge ;
2. `loaded_mode` revient a `None` ;
3. la prediction, l'image uploadee et l'etat Grad-CAM sont effaces ;
4. le nouveau checkpoint est seulement charge a la prochaine action explicite.

Choisir de nouveau le meme mode ne recharge pas le modele.

## Grad-CAM

Le hub reutilise la couche cible du pipeline LungColon : bloc `features` final pour
EfficientNet-B0 et `layer4[-1].conv2` pour ResNet18. L'overlay reste en memoire et tout
indice hors limites est refuse proprement.

## Etat Streamlit

La sidebar affiche `Mode de classification` avec deux options explicites. La fiche,
les classes, le modele, les metriques, le statut du checkpoint et les limites suivent
le mode selectionne. Aucune prediction n'est relancee automatiquement apres une
bascule.

## Limites LC25000

- LC25000 est un benchmark public relativement facile ;
- les scores tres eleves peuvent dependre des caracteristiques du dataset ;
- les trois erreurs multiclass concernent des sous-types malins pulmonaires ;
- le score binaire parfait concerne uniquement le split local controle ;
- aucune validation externe ou clinique n'a ete realisee ;
- les scores ne sont pas directement comparables a Breast, Metastasis ou Leukemia.

## Tests

Les tests utilisent des checkpoints et backends simules. Ils couvrent les configurations,
les statuts independants, l'incompatibilite, les predictions cinq/deux classes,
Grad-CAM, l'unload, les transitions de projet et le nettoyage de session Streamlit.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
```

## Statut Final MultiCancer V1

Leukemia, Breast, Metastasis et LungColon sont tous integres comme pipelines
specialises. MultiCancer reste un routeur educatif transparent, pas un modele universel
et pas un outil medical.
