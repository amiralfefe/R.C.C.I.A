# Interview Pitch - R.C.C.I.A

## Pitch 30 Secondes

J'ai developpe R.C.C.I.A, un projet portfolio de computer vision applique a des images microscopiques de cellules sanguines. Le projet classe des images en `normal` ou `leukemia_blast` a partir d'un dataset public Kaggle. J'ai construit une pipeline complete en PyTorch : preparation des donnees, split train/validation/test, entrainement, evaluation, Streamlit, Grad-CAM, comparaison de modeles et analyse des erreurs. La V1 atteint 91.69 % d'accuracy test, et la V2 ajoute un benchmark multi-modeles ainsi qu'une analyse false positives / false negatives. Le projet reste strictement educatif et ne constitue pas un outil medical.

## Pitch 2 Minutes

R.C.C.I.A est un projet ML portfolio concu pour montrer une demarche complete de computer vision, pas seulement un notebook d'entrainement. Je suis parti d'un dataset public Kaggle de cellules sanguines, que j'ai prepare en deux classes experimentales : `normal` et `leukemia_blast`.

J'ai construit une pipeline PyTorch reproductible : preparation des images, split train/validation/test, entrainement d'une baseline ResNet18, evaluation avec accuracy, precision, recall, F1-score et matrice de confusion, puis prediction en ligne de commande.

Pour rendre le projet presentable et interpretable, j'ai ajoute une application Streamlit avec upload d'image, probabilites par classe et visualisation Grad-CAM. Ensuite, j'ai cree une V2 de comparaison de modeles pour evaluer ResNet18, MobileNetV3 Small et EfficientNet-B0 avec un protocole comparable. Enfin, la V2.2 ajoute une analyse des erreurs : predictions CSV, resume JSON, false positives, false negatives, exemples annotes et Grad-CAM sur certaines erreurs.

La baseline V1 obtient 91.69 % d'accuracy sur le test set. Le benchmark V2.1 montre que ResNet18 garde le meilleur recall `leukemia_blast` dans ce protocole, tandis que MobileNetV3 est presque aussi performant et beaucoup plus rapide sur CPU.

Je presente toujours ce projet comme un demonstrateur educatif IA/data : il n'a pas de validation clinique, ne remplace pas un avis medical et ne doit pas etre utilise pour prendre une decision de sante.

## Points Techniques A Expliquer

- Preparation d'un dataset public et separation propre des donnees locales non versionnees.
- Importance du split train/validation/test.
- Choix d'une baseline ResNet18 pour demarrer rapidement.
- Transfer learning avec Torchvision.
- Difference entre accuracy, precision, recall et F1-score.
- Pourquoi analyser le recall `leukemia_blast` sans transformer le projet en outil medical.
- Comparaison ResNet18, MobileNetV3 Small et EfficientNet-B0.
- Role de Grad-CAM pour expliquer visuellement une prediction.
- Analyse qualitative des erreurs : false positives, false negatives, confiance moyenne.
- Limites : dataset public, desequilibre, protocole court CPU, absence de validation clinique.

## Questions Possibles Et Reponses Courtes

### Pourquoi PyTorch ?

PyTorch est flexible, lisible et tres utilise en recherche comme en prototypage ML. Pour ce projet, il permet de construire une pipeline claire, de reutiliser Torchvision pour le transfer learning et de garder le code facile a expliquer en entretien.

### Pourquoi Grad-CAM ?

Grad-CAM donne une visualisation des zones de l'image qui influencent la prediction du modele. Dans un projet portfolio, cela aide a rendre le comportement du modele plus concret. Ce n'est pas une preuve medicale, seulement un outil d'interpretabilite.

### Pourquoi comparer plusieurs modeles ?

Comparer plusieurs architectures evite de se limiter a une seule baseline. Cela permet d'analyser le compromis performance / temps d'entrainement : par exemple, ResNet18 a obtenu le meilleur recall `leukemia_blast`, tandis que MobileNetV3 etait beaucoup plus rapide.

### Comment interpreter precision, recall et F1 ?

La precision mesure la proportion de predictions correctes parmi les images predites dans une classe. Le recall mesure la proportion d'images reelles d'une classe que le modele retrouve. Le F1-score combine precision et recall. L'accuracy seule peut masquer des erreurs importantes, surtout si les classes sont desequilibrees.

### Quelles sont les limites ?

Le dataset est public, les classes sont desequilibrees, les entrainements sont courts et realises sur CPU, et les resultats dependent du split et du preprocessing. Le projet n'a aucune validation clinique ni validation par specialistes.

### Pourquoi ce n'est pas un outil medical ?

Parce qu'un outil medical exige une validation clinique stricte, des protocoles controles, des donnees representatives, des tests multi-centriques, une evaluation par experts et un cadre reglementaire. Ici, le projet sert uniquement a demontrer des competences IA/data sur un dataset public.

### Pourquoi analyser les erreurs ?

L'analyse des erreurs montre que je ne me contente pas d'une accuracy globale. Elle permet de regarder les false positives, les false negatives, la confiance du modele et des exemples concrets, ce qui rend l'evaluation plus serieuse et plus explicable.

### Que ferais-tu ensuite ?

Je travaillerais sur l'augmentation de donnees, la calibration des probabilites, l'analyse des seuils, un benchmark plus long, et une evaluation qualitative plus poussee avec davantage d'exemples Grad-CAM. Pour un contexte medical reel, il faudrait surtout changer de cadre : validation clinique, experts et donnees controlees.
