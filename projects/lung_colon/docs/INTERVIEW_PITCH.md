# Lung + Colon Vision - Interview Pitch

## Pitch 30 Secondes

J'ai construit un sous-projet de computer vision dans mon monorepo R.C.C.I.A pour classifier des images histopathologiques lung/colon du dataset public LC25000. Le pipeline couvre la preparation des donnees, le split train/val/test, l'entrainement PyTorch, l'evaluation, la comparaison de modeles, l'analyse des erreurs et une demo Streamlit avec Grad-CAM. Le projet est strictement educatif et portfolio, pas un outil de diagnostic.

## Pitch 2 Minutes

Lung + Colon Vision est un projet IA/data autour du dataset public LC25000, qui contient 25 000 images histopathologiques reparties en 5 classes : deux classes colon et trois classes poumon.

J'ai construit un pipeline complet en PyTorch : normalisation du dataset, split train/validation/test, entrainement avec transfer learning, evaluation avec accuracy, precision, recall, F1-score, matrice de confusion et classification report. La V1 utilise ResNet18 en baseline et atteint 0.9965 d'accuracy test sur le split local.

Ensuite, j'ai ajoute une V2 de comparaison de modeles avec ResNet18, MobileNetV3 small et EfficientNet-B0. EfficientNet-B0 obtient le meilleur score sur ce benchmark, MobileNetV3 small est le plus rapide, et ResNet18 reste la baseline la plus simple a expliquer.

La V2.1 ajoute une analyse des erreurs : sur 3 750 images test, EfficientNet-B0 fait 3 erreurs, toutes entre deux sous-types malins pulmonaires. Il n'y a pas de confusion benign/malignant ni lung/colon dans cette analyse locale.

La V2.2 ajoute un mode binaire `benign` vs `malignant`, avec Streamlit capable de basculer entre le mode 5 classes et le mode binaire. Je presente ces scores avec prudence, car LC25000 est un dataset public relativement facile et ce projet n'a aucune validation clinique.

## Points Techniques A Expliquer

- Structure monorepo avec un sous-projet autonome `projects/lung_colon`.
- Dataset converti en structure compatible `ImageFolder`.
- Transfer learning avec Torchvision.
- Support des architectures `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`.
- Evaluation multi-classe avec classification report et matrice de confusion.
- Benchmark comparatif avec temps d'entrainement et metriques par classe.
- Grad-CAM pour visualiser les zones qui influencent la prediction.
- Analyse d'erreurs pour comprendre les confusions restantes.
- Mode binaire construit a partir du split 5 classes.
- Tests automatises et exclusion volontaire des donnees, outputs, checkpoints et tokens.

## Questions Possibles Et Reponses Courtes

### Pourquoi PyTorch ?

PyTorch est standard pour les projets de computer vision, lisible pour le prototypage et bien integre avec Torchvision pour le transfer learning.

### Pourquoi Grad-CAM ?

Grad-CAM donne une visualisation approximative des zones qui influencent la prediction. C'est utile pour l'interpretabilite portfolio, mais ce n'est pas une preuve medicale.

### Pourquoi comparer plusieurs modeles ?

Pour montrer le compromis entre performance, vitesse et complexite. EfficientNet-B0 score le mieux ici, MobileNetV3 small est le plus rapide, et ResNet18 sert de baseline robuste.

### Comment interpreter precision, recall et F1 ?

La precision mesure la fiabilite des predictions positives d'une classe, le recall mesure la capacite a retrouver les exemples de cette classe, et le F1 combine les deux. Dans un contexte medical educatif, le recall d'une classe sensible peut etre particulierement important, mais ce projet ne fait pas de diagnostic.

### Pourquoi ajouter un mode binaire ?

Le mode binaire `benign` vs `malignant` teste une autre formulation du probleme, plus simple que les 5 classes. Il montre aussi que l'application Streamlit peut supporter plusieurs modes.

### Quelles sont les limites ?

Le dataset est public et relativement facile, le split est local, il n'y a pas de validation clinique externe, et les performances ne prouvent pas une robustesse medicale.

### Pourquoi ce n'est pas un outil medical ?

Parce qu'il n'a pas ete valide cliniquement, n'a pas ete teste sur des cohortes externes controlees, n'est pas certifie, et n'a pas ete concu pour orienter une decision de sante.

### Comment eviter de survendre le score de 100 pour cent en binaire ?

Je le presente comme un resultat experimental sur LC25000, pas comme une preuve clinique. Le point fort du projet est le pipeline complet : benchmark, interpretabilite, analyse d'erreurs, tests et documentation.
