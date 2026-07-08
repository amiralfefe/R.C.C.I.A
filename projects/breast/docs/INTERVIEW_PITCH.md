# Breast Vision - Interview Pitch

## Pitch 30 Secondes

J'ai construit un sous-projet de computer vision dans mon monorepo R.C.C.I.A pour
classifier des images histopathologiques BreakHis en `benign` et `malignant`. Le point
fort est le split patient-aware : les images d'un meme patient ne sont pas reparties
entre train et test, ce qui limite le data leakage. Le projet couvre PyTorch,
benchmark multi-modeles, Streamlit, Grad-CAM et analyse d'erreurs par patient et
grossissement. C'est un demonstrateur educatif et portfolio, pas un outil medical.

## Pitch 2 Minutes

Breast Vision est le troisieme sous-projet du monorepo R.C.C.I.A. Il utilise BreakHis,
un dataset public d'images histopathologiques du sein, pour une classification binaire
`benign` vs `malignant`.

J'ai commence par preparer les donnees en structure `ImageFolder`, en extrayant aussi
les informations de grossissement `40X`, `100X`, `200X`, `400X` et les identifiants
patients quand ils etaient presents dans les noms de fichiers. Ensuite, j'ai realise un
split patient-aware : 55 patients en train, 11 en validation, 15 en test, avec 0 overlap
patient entre les splits.

La V1 utilise ResNet18 pre-entraine en baseline et atteint 0.8947 d'accuracy test, avec
un recall `malignant` de 0.9466. En V2, j'ai compare ResNet18, MobileNetV3 small et
EfficientNet-B0. EfficientNet-B0 obtient le meilleur compromis global avec 0.9122
d'accuracy et 0.9004 de macro F1. MobileNetV3 small maximise le recall `malignant`
avec 0.9979, mais au prix d'un F1 `benign` plus faible.

La V2.1 ajoute une analyse d'erreurs : EfficientNet-B0 fait 130 erreurs sur 1 481 images
test, dont 88 faux positifs `benign -> malignant` et 42 faux negatifs
`malignant -> benign`. L'analyse montre aussi que le grossissement `40X` est le plus
difficile et qu'un patient concentre 76 erreurs, ce qui renforce l'interet du
patient-aware split.

Je presente le projet comme un portfolio ML educatif : il montre un pipeline serieux,
mais il ne constitue pas un outil medical ni une validation clinique.

## Points Techniques A Expliquer

- Structure monorepo avec un sous-projet autonome `projects/breast`.
- Dataset BreakHis prepare en `ImageFolder`.
- Extraction de `patient_id` et `magnification`.
- Split patient-aware pour limiter le data leakage.
- Transfer learning avec Torchvision.
- Support des architectures `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`.
- Evaluation avec classification report, matrice de confusion et metriques par classe.
- Benchmark multi-modeles avec temps d'entrainement.
- Analyse d'erreurs par patient, grossissement, classe et confiance.
- Grad-CAM pour visualiser les zones influencant la prediction.
- Tests automatises et exclusion volontaire des donnees, outputs, checkpoints et tokens.

## Questions Possibles Et Reponses Courtes

### Pourquoi BreakHis ?

BreakHis est un dataset public connu pour la classification histopathologique du sein. Il
est interessant car il contient plusieurs grossissements et permet d'aborder le risque de
data leakage entre images d'un meme patient.

### Pourquoi patient-aware split ?

Parce que plusieurs images peuvent venir du meme patient. Si un patient apparait en train
et en test, le score peut devenir trop optimiste. Le split patient-aware rend le protocole
plus strict.

### Pourquoi eviter un split random image-level ?

Un split random image-level peut melanger des images tres proches ou issues du meme
patient entre les splits. Le modele peut alors apprendre des caracteristiques patient ou
acquisition plutot que generaliser.

### Pourquoi PyTorch ?

PyTorch est lisible, standard pour la computer vision et bien integre avec Torchvision
pour le transfer learning et les architectures pre-entrainees.

### Pourquoi Grad-CAM ?

Grad-CAM donne une visualisation exploratoire des zones qui influencent la prediction.
C'est utile pour analyser le comportement du modele, mais ce n'est pas une preuve
medicale.

### Pourquoi comparer ResNet18, MobileNetV3 et EfficientNet ?

Pour montrer le compromis entre performance, vitesse et complexite. ResNet18 sert de
baseline, MobileNetV3 est leger et rapide, EfficientNet cherche un bon compromis
accuracy/complexite.

### Pourquoi EfficientNet est le meilleur compromis global ?

Sur ce benchmark patient-aware, EfficientNet-B0 obtient la meilleure accuracy
(`0.9122`) et la meilleure macro F1 (`0.9004`) tout en gardant un recall `malignant`
eleve (`0.9568`).

### Pourquoi MobileNetV3 a un tres fort recall malignant mais un macro F1 plus faible ?

Il detecte presque toutes les images `malignant`, mais il classe davantage d'images
`benign` comme `malignant`. Le recall `malignant` monte, mais le F1 `benign` et le macro
F1 baissent.

### Comment interpreter precision, recall et F1 ?

La precision mesure la fiabilite des predictions d'une classe, le recall mesure la
capacite a retrouver les exemples de cette classe, et le F1 combine les deux. Dans ce
projet, le recall `malignant` est important a analyser, mais il ne transforme pas le
modele en outil medical.

### Pourquoi analyser les erreurs par patient ?

Parce que les erreurs peuvent etre concentrees sur certains patients. Dans V2.1, un
patient concentre 76 erreurs, ce qui montre une variabilite patient importante et
renforce l'interet du protocole patient-aware.

### Pourquoi 40X est plus difficile que 200X ?

Dans l'analyse V2.1, `40X` a la plus faible accuracy (`0.8757`) tandis que `200X` a la
meilleure (`0.9467`). Cela suggere que certains niveaux de grossissement donnent au
modele des informations plus discriminantes dans ce split local.

### Pourquoi ce n'est pas un outil medical ?

Parce qu'il n'a pas ete valide cliniquement, n'a pas ete teste sur des cohortes externes
controlees, n'est pas certifie, et n'a pas ete concu pour orienter une decision de sante.
Le projet est uniquement educatif et portfolio.
