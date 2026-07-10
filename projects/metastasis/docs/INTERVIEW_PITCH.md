# Metastasis Vision - Interview Pitch

## Pitch 30 Secondes

J'ai construit Metastasis Vision, un sous-projet de mon monorepo R.C.C.I.A pour
classifier des patches histopathologiques PCam en `non_metastatic` et `metastatic`.
Le projet couvre la conversion HDF5 vers ImageFolder, PyTorch, Streamlit, Grad-CAM,
benchmark multi-modeles et surtout une analyse ROC/PR-AUC avec seuils de decision. Au
seuil 0.50, EfficientNet-B0 atteint 0.9320 d'accuracy, 0.9762 de ROC-AUC et 0.9780 de
PR-AUC sur le test local. C'est un demonstrateur educatif et portfolio, pas un outil
medical.

## Pitch 2 Minutes

Metastasis Vision est un sous-projet R.C.C.I.A centre sur PCam / PatchCamelyon, un
dataset public de patches histopathologiques. Le probleme est une classification binaire
`non_metastatic` vs `metastatic`.

J'ai d'abord prepare un subset equilibre de 5 000 patches a partir de fichiers HDF5 :
2 500 images par classe, puis un split train / validation / test de 3 500 / 750 / 750.
L'image size reste a 96 car PCam fournit nativement des patches 96x96.

La V1 utilise ResNet18 pre-entraine comme baseline et obtient 0.9040 d'accuracy test,
0.9598 de ROC-AUC et 0.9616 de PR-AUC. En V2, j'ai compare ResNet18, MobileNetV3 small
et EfficientNet-B0. EfficientNet-B0 obtient le meilleur score global avec 0.9320
d'accuracy, 0.9320 de macro F1, 0.9762 de ROC-AUC et 0.9780 de PR-AUC.

La V2.1 ajoute une analyse d'erreurs et de seuils. Au seuil 0.50, le modele fait 24 faux
positifs et 27 faux negatifs sur 750 images test. En abaissant le seuil a 0.30, le recall
`metastatic` monte a 0.9653 et les faux negatifs baissent a 13, mais les faux positifs
montent a 69. En montant le seuil a 0.70, les faux positifs baissent a 11, mais les faux
negatifs montent a 55.

Le projet montre donc que l'accuracy seule ne suffit pas : il faut analyser ROC-AUC,
PR-AUC, recall, faux positifs, faux negatifs et seuils. Je ne choisis pas un seuil
medical ; je documente le compromis dans un cadre educatif.

## Points Techniques A Expliquer

- Structure monorepo avec un sous-projet autonome `projects/metastasis`.
- Dataset PCam / PatchCamelyon prepare depuis HDF5.
- Conversion vers une structure `ImageFolder`.
- Classification binaire `non_metastatic` vs `metastatic`.
- Transfer learning avec Torchvision.
- Support des architectures `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`.
- Evaluation avec accuracy, precision, recall, F1, ROC-AUC et PR-AUC.
- Benchmark multi-modeles avec temps d'entrainement.
- Analyse des faux positifs, faux negatifs, confiance et seuils.
- Grad-CAM pour visualiser les zones influencant la prediction.
- Tests automatises et exclusion volontaire des donnees, outputs, checkpoints et tokens.

## Questions Possibles Et Reponses Courtes

### Pourquoi PCam / PatchCamelyon ?

PCam est un dataset public de patches histopathologiques adapte a une classification
binaire `non_metastatic` vs `metastatic`. Il permet de travailler des metriques utiles
comme ROC-AUC, PR-AUC et analyse de seuils.

### Pourquoi garder `image-size 96` ?

PCam fournit des patches natifs en 96x96. Garder cette taille evite un resizing inutile
et rend le benchmark coherent avec la structure originale du dataset.

### Pourquoi PyTorch ?

PyTorch est lisible, standard pour la computer vision et bien integre avec Torchvision
pour le transfer learning et les architectures pre-entrainees.

### Pourquoi comparer ResNet18, MobileNetV3 et EfficientNet ?

Pour montrer le compromis entre performance, vitesse et complexite. ResNet18 sert de
baseline, MobileNetV3 est leger et rapide, EfficientNet cherche un meilleur compromis
global.

### Quel modele est le meilleur ?

EfficientNet-B0 est le meilleur global sur ce benchmark : 0.9320 d'accuracy, 0.9320 de
macro F1, 0.9762 de ROC-AUC et 0.9780 de PR-AUC.

### Pourquoi ResNet18 reste interessant ?

ResNet18 obtient le meilleur recall `metastatic` dans le benchmark V2, avec 0.9307. Il
reste aussi simple a expliquer comme baseline robuste.

### Pourquoi MobileNetV3 est utile ?

MobileNetV3 small est le plus rapide a entrainer dans V2, avec 64.82 secondes, mais il
perd en accuracy et macro F1. Il illustre le compromis vitesse / performance.

### Pourquoi ROC-AUC et PR-AUC ?

ROC-AUC mesure la capacite du modele a separer les classes sur tous les seuils. PR-AUC
est utile pour analyser le compromis precision / recall, surtout quand on s'interesse a
la classe positive `metastatic`.

### Pourquoi l'accuracy seule ne suffit pas ?

Deux modeles peuvent avoir une accuracy proche mais des profils d'erreurs tres
differents. Dans ce projet, les faux positifs, faux negatifs et seuils changent
fortement le comportement du modele.

### Que montre l'analyse des seuils ?

A 0.30, le recall `metastatic` augmente et les faux negatifs diminuent, mais les faux
positifs augmentent. A 0.70, les faux positifs diminuent, mais les faux negatifs
augmentent. Le projet documente ce compromis sans choisir de seuil medical.

### Pourquoi Grad-CAM ?

Grad-CAM donne une visualisation exploratoire des zones qui influencent la prediction.
C'est utile pour analyser le comportement du modele, mais ce n'est pas une preuve
medicale.

### Pourquoi ce n'est pas un outil medical ?

Parce qu'il n'a pas ete valide cliniquement, n'a pas ete teste sur des cohortes externes
controlees, n'est pas certifie, et n'a pas ete concu pour orienter une decision de sante.
Le projet est uniquement educatif et portfolio.
