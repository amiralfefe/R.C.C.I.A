# LinkedIn Draft - R.C.C.I.A

## Version Courte

J'ai finalise une V2.2 de R.C.C.I.A, un projet portfolio IA/data de computer vision autour de la classification d'images microscopiques de cellules sanguines.

Le projet inclut :

- pipeline PyTorch complete ;
- dataset public Kaggle ;
- evaluation accuracy / precision / recall / F1 ;
- application Streamlit ;
- Grad-CAM ;
- benchmark ResNet18 / MobileNetV3 / EfficientNet ;
- analyse false positives / false negatives ;
- tests automatises.

Resultat V1 : 91.69 % d'accuracy test sur une classification `normal` vs `leukemia_blast`.

Important : c'est un demonstrateur educatif pour portfolio, pas un outil medical ni un diagnostic.

#AI #DataScience #MachineLearning #ComputerVision #PyTorch #Streamlit

## Version Detaillee

J'ai termine une nouvelle version de R.C.C.I.A, un projet portfolio IA/data de computer vision applique a des images microscopiques de cellules sanguines.

L'objectif n'etait pas de creer un outil medical, mais de construire un projet ML propre, reproductible et presentable : un vrai pipeline de bout en bout plutot qu'un simple notebook.

Ce que le projet couvre :

- preparation d'un dataset public Kaggle ;
- split train / validation / test ;
- entrainement PyTorch avec transfer learning ;
- evaluation avec accuracy, precision, recall, F1-score et matrice de confusion ;
- prediction CLI ;
- interface Streamlit ;
- visualisation Grad-CAM ;
- comparaison ResNet18, MobileNetV3 Small et EfficientNet-B0 ;
- analyse des erreurs avec false positives, false negatives et confiance moyenne ;
- tests automatises.

Quelques resultats :

- V1 ResNet18 : 91.69 % d'accuracy test ;
- `leukemia_blast` : precision 0.9419, recall 0.9359, F1 0.9389 ;
- V2.1 benchmark : ResNet18 garde le meilleur recall `leukemia_blast`, MobileNetV3 est presque aussi performant et beaucoup plus rapide sur CPU ;
- V2.2 error analysis : 1601 images test analysees, 1468 predictions correctes, 133 erreurs, 63 false positives et 70 false negatives.

Ce que j'ai surtout voulu montrer : aller au-dela de "mon modele a 91 % d'accuracy" en documentant les limites, les erreurs, les comparaisons de modeles et les visualisations d'interpretabilite.

Disclaimer important : ce projet est uniquement un demonstrateur educatif IA/data pour portfolio. Il ne fournit aucun diagnostic medical et ne doit pas etre utilise dans un contexte de sante.

#AI #DataScience #MachineLearning #ComputerVision #DeepLearning #PyTorch #Streamlit #Portfolio

## Version Sobre / Professionnelle

Projet portfolio finalise : R.C.C.I.A, une pipeline de computer vision en PyTorch pour classifier des images microscopiques de cellules sanguines a partir d'un dataset public.

Le projet inclut la preparation des donnees, l'entrainement, l'evaluation, une interface Streamlit, Grad-CAM, un benchmark multi-modeles et une analyse des erreurs.

Resultats principaux :

- accuracy test V1 : 91.69 % ;
- benchmark ResNet18 / MobileNetV3 Small / EfficientNet-B0 ;
- analyse false positives / false negatives ;
- documentation complete et tests automatises.

Le projet est presente comme un demonstrateur educatif IA/data uniquement, sans usage medical ou diagnostique.

#MachineLearning #ComputerVision #PyTorch #DataScience

## Hashtags Moderes

#AI #DataScience #MachineLearning #ComputerVision #PyTorch #Streamlit

## Notes Avant Publication

- Ajouter le lien GitHub.
- Ajouter 1 a 3 captures Streamlit.
- Eviter toute formulation qui laisserait penser que le modele aide au diagnostic.
- Preferer "demonstrateur educatif", "projet portfolio" et "dataset public".
- Ne pas publier de token, checkpoint, images sensibles ou outputs complets.
