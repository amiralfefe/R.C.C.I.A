# Metastasis Metrics Guide

## Pourquoi L'Accuracy Ne Suffit Pas

Dans un probleme binaire `non_metastatic` vs `metastatic`, l'accuracy peut etre
trompeuse si les classes sont desequilibrees ou si le cout des erreurs n'est pas
symetrique.

Ce projet prepare donc plusieurs metriques :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- ROC-AUC ;
- PR-AUC ;
- confusion matrix ;
- courbe ROC ;
- courbe precision/recall.

Le projet reste educatif : aucune metrique ne doit etre interpretee comme validation
clinique.

## Precision

La precision repond a la question :

> Parmi les patches predits `metastatic`, combien sont vraiment `metastatic` ?

Une precision faible signifie plus de faux positifs.

## Recall

Le recall repond a la question :

> Parmi les patches vraiment `metastatic`, combien le modele retrouve-t-il ?

Dans un contexte educatif autour de metastases, le recall `metastatic` est important a
suivre, car il mesure les cas positifs retrouves par le modele. Cela ne transforme pas le
modele en outil medical.

## F1-Score

Le F1-score combine precision et recall. Il est utile quand on veut une mesure unique qui
penalise a la fois les faux positifs et les faux negatifs.

## ROC-AUC

La ROC-AUC mesure la capacite du modele a separer les classes sur differents seuils de
decision. Elle utilise la probabilite `metastatic`.

## PR-AUC

La PR-AUC est souvent informative quand la classe positive est rare ou quand les faux
positifs comptent beaucoup. Elle resume la courbe precision/recall sur differents seuils.

## Confusion Matrix

La matrice de confusion permet de lire directement :

- vrais `non_metastatic` ;
- faux positifs `non_metastatic -> metastatic` ;
- faux negatifs `metastatic -> non_metastatic` ;
- vrais `metastatic`.

## Seuil De Decision

Le seuil par defaut est souvent `0.5` sur la probabilite `metastatic`. Une future version
pourra comparer plusieurs seuils pour analyser le compromis precision/recall.

Changer le seuil peut augmenter le recall `metastatic`, mais aussi augmenter les faux
positifs. Ce compromis doit etre documente avec prudence.

## Disclaimer

Ces metriques servent a analyser un benchmark educatif sur dataset public. Elles ne
permettent aucune decision medicale, aucun diagnostic et aucune validation clinique.

