# MultiCancer V1.3 - Metastasis Adapter And Threshold Decision Layer

## Objectif

La V1.3 integre le pipeline Metastasis au hub et ajoute une couche de decision par seuil
strictement separee de l'inference. `predict(image)` conserve les probabilites brutes,
l'argmax et sa confiance ; le slider Streamlit applique ensuite une regle exploratoire
sans recalculer ni modifier le resultat du modele.

> Demonstrateur educatif / portfolio uniquement. Aucun diagnostic, aucune recommandation
> medicale, aucune validation clinique et aucun seuil medical recommande.

## Pourquoi Metastasis En Troisieme ?

Metastasis reutilise le contrat Torchvision valide avec Leukemia et Breast, mais ajoute
des metriques ROC-AUC / PR-AUC et une analyse du compromis faux positifs / faux negatifs.
Il permet donc de montrer qu'inference probabiliste et regle de decision sont deux
concepts distincts.

## Contrat Commun Reutilise

`MetastasisAdapter` herite de `TorchvisionImageAdapter` et declare seulement :

- projet `metastasis` ;
- package `rccia_metastasis` ;
- classes `non_metastatic`, `metastatic` ;
- checkpoint EfficientNet-B0 local.

Le chargement, la prediction, Grad-CAM, les erreurs et l'unload utilisent le meme contrat
que Leukemia et Breast.

## Checkpoint Et Preprocessing

Checkpoint local cible :

```text
projects/metastasis/outputs/model_comparison/efficientnet_b0/best_model.pt
```

Le preprocessing reste celui du projet specialise : conversion RGB, resize `96x96` et
normalisation ImageNet. Le checkpoint et les outputs restent exclus de Git.

## Prediction Normalisee

`PredictionResult` contient uniquement :

- argmax original et indice ;
- confiance de l'argmax ;
- probabilites `non_metastatic` et `metastatic` ;
- modele, resolution, avertissements et metadata optionnelle.

Il ne contient aucun threshold ni `thresholded_class`.

## ThresholdDecision

`apply_binary_threshold()` lit une prediction existante et retourne une nouvelle
`ThresholdDecision` :

- classe positive et classe negative ;
- probabilite positive ;
- seuil applique et seuil par defaut ;
- classe derivee ;
- indicateur d'override ;
- avertissement educatif.

La fonction est pure : elle ne modifie pas le `PredictionResult`. Une probabilite egale
au seuil est classee positive.

## Slider Exploratoire

Le slider couvre `0.30` a `0.70`, avec `0.50` comme valeur par defaut. Il recalcule
uniquement `ThresholdDecision` depuis les probabilites conservees en session.

Le hub affiche simultanement :

- argmax original ;
- probabilite `metastatic` ;
- seuil applique ;
- decision derivee ;
- statut default/override.

Argmax et decision au seuil peuvent differer sans nouvelle inference.

## Resultats De Seuils V2.1

Resultats agreges sur les 750 images test locales :

| Threshold | Accuracy | Precision metastatic | Recall metastatic | F1 metastatic | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.8907 | 0.8399 | 0.9653 | 0.8983 | 69 | 13 |
| 0.40 | 0.9160 | 0.8861 | 0.9547 | 0.9191 | 46 | 17 |
| 0.50 | 0.9320 | 0.9355 | 0.9280 | 0.9317 | 24 | 27 |
| 0.60 | 0.9187 | 0.9460 | 0.8880 | 0.9161 | 19 | 42 |
| 0.70 | 0.9120 | 0.9668 | 0.8533 | 0.9065 | 11 | 55 |

Un seuil plus bas augmente generalement le recall `metastatic` et les faux positifs. Un
seuil plus haut reduit les faux positifs et augmente les faux negatifs. Cette observation
de benchmark ne definit aucun seuil medical.

## Grad-CAM

L'adaptateur reutilise `rccia_metastasis.gradcam`. L'overlay reste en memoire et suit
l'argmax ou la classe explicitement demandee ; il n'est pas influence par le slider sauf
si l'utilisateur demande explicitement une autre classe.

## Lazy Loading Et Unload

`ModelManager` connait maintenant Leukemia, Breast et Metastasis. Toute activation
decharge l'adaptateur precedent avant de construire le suivant. Changer le slider ne
touche pas au manager et ne recharge pas le modele.

## Gestion Des Erreurs

- checkpoint absent ou incompatible ;
- image invalide ;
- modele non charge ;
- Grad-CAM indisponible ;
- seuil hors `[0, 1]` ;
- classe positive absente ou prediction non binaire.

Chaque erreur est controlee sans transformer l'interface en outil medical.

## Tests Et Lancement

Les tests utilisent des backends simules et ne dependent pas du checkpoint local. Ils
couvrent l'adaptateur, la fonction de seuil, l'immutabilite de la prediction, le manager,
le registre et les etats Streamlit.

```powershell
.\.venv\Scripts\streamlit.exe run projects\multicancer\app.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Prochaine Integration

LungColon sera le quatrieme adaptateur. Il devra exposer une selection explicite entre
le mode 5 classes et le mode binaire, avec des checkpoints potentiellement differents.
