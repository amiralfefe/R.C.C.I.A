# Breast Vision - LinkedIn Drafts

## Version Courte

J'ai finalise **Breast Vision**, une nouvelle brique de mon monorepo R.C.C.I.A.

Objectif : construire un demonstrateur educatif de computer vision sur BreakHis pour
classifier des images histopathologiques en `benign` vs `malignant`, avec un point
important : un split patient-aware pour limiter le risque de data leakage.

Stack : Python, PyTorch, Torchvision, Streamlit, Scikit-learn, Grad-CAM, Pytest.

Resultats experimentaux :

- BreakHis : 7 909 images, 81 patients detectes
- Split patient-aware : 0 overlap patient
- EfficientNet-B0 : 91.22 % accuracy test, 90.04 % macro F1
- Error analysis : 130 erreurs sur 1 481 images test
- Analyse par patient : un patient concentre 76 erreurs
- Analyse par grossissement : `200X` meilleur, `40X` plus difficile

Projet educatif / portfolio uniquement : pas outil medical, pas diagnostic, pas
validation clinique.

#MachineLearning #ComputerVision #PyTorch #DataScience #Portfolio

## Version Detaillee

J'ai termine **Breast Vision**, un sous-projet du monorepo R.C.C.I.A consacre a des
projets IA/data de computer vision sur datasets publics lies au cancer.

Le but n'etait pas de creer un outil medical, mais un projet portfolio plus rigoureux
sur le protocole experimental.

Ce que le projet couvre :

- preparation du dataset public BreakHis ;
- classification `benign` vs `malignant` ;
- extraction de `patient_id` et du grossissement `40X`, `100X`, `200X`, `400X` ;
- split patient-aware avec 0 overlap patient entre train/val/test ;
- entrainement PyTorch avec transfer learning ;
- evaluation avec accuracy, precision, recall, F1-score et matrice de confusion ;
- benchmark ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- demo Streamlit avec prediction, probabilites et Grad-CAM ;
- analyse d'erreurs par classe, patient, grossissement et confiance ;
- tests automatises et documentation portfolio.

Quelques resultats experimentaux :

- ResNet18 baseline : 89.47 % accuracy test ;
- EfficientNet-B0 benchmark V2 : 91.22 % accuracy test et 90.04 % macro F1 ;
- MobileNetV3 small : meilleur recall `malignant` avec 99.79 %, mais macro F1 plus faible ;
- Error analysis V2.1 : 88 faux positifs `benign -> malignant` et 42 faux negatifs
  `malignant -> benign` ;
- le grossissement `40X` est le plus difficile, `200X` le plus favorable ;
- un patient concentre 76 erreurs, ce qui montre l'interet d'une analyse patient-aware.

Je garde une lecture prudente : BreakHis est un dataset public et ces resultats ne sont
pas une validation clinique. La valeur du projet est surtout dans le pipeline ML complet,
le split patient-aware, l'analyse d'erreurs, l'interpretabilite et la presentation
reproductible.

#MachineLearning #ComputerVision #PyTorch #Streamlit #DataScience #AI

## Version Sobre Professionnelle

Nouveau projet portfolio finalise : **Breast Vision**, sous-projet du monorepo R.C.C.I.A.

Le projet met en place un pipeline de computer vision sur le dataset public BreakHis :

- classification `benign` vs `malignant` ;
- split patient-aware pour limiter le data leakage ;
- entrainement PyTorch / Torchvision ;
- benchmark ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- Streamlit avec Grad-CAM ;
- analyse d'erreurs par patient et grossissement.

Resultat principal : EfficientNet-B0 atteint 91.22 % d'accuracy test sur le split
patient-aware local. L'analyse V2.1 montre 130 erreurs sur 1 481 images test, avec une
concentration notable sur certains patients.

Projet strictement educatif et portfolio. Il ne constitue pas un outil medical, pas un
diagnostic et pas une validation clinique.

#ComputerVision #PyTorch #MachineLearning #Portfolio
