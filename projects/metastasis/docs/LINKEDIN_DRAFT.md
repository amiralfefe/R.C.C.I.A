# Metastasis Vision - LinkedIn Drafts

## Version Courte

J'ai finalise **Metastasis Vision**, une nouvelle brique de mon monorepo R.C.C.I.A.

Objectif : construire un demonstrateur educatif de computer vision sur PCam /
PatchCamelyon pour classifier des patches histopathologiques en `non_metastatic` vs
`metastatic`.

Stack : Python, PyTorch, Torchvision, Streamlit, Scikit-learn, Grad-CAM, Pytest.

Resultats experimentaux :

- subset PCam : 5 000 patches equilibres
- split local : 3 500 train / 750 val / 750 test
- EfficientNet-B0 : 93.20 % accuracy test
- ROC-AUC : 97.62 %
- PR-AUC : 97.80 %
- analyse seuils : compromis clair entre recall `metastatic`, faux positifs et faux negatifs
- Grad-CAM et app Streamlit locale

Projet educatif / portfolio uniquement : pas outil medical, pas diagnostic, pas
validation clinique.

#MachineLearning #ComputerVision #PyTorch #DataScience #Portfolio

## Version Detaillee

J'ai termine **Metastasis Vision**, un sous-projet du monorepo R.C.C.I.A consacre a des
projets IA/data de computer vision sur datasets publics lies au cancer.

Le but n'etait pas de creer un outil medical, mais de construire un projet portfolio
rigoureux autour d'un probleme de classification binaire sur patches histopathologiques :
`non_metastatic` vs `metastatic`.

Ce que le projet couvre :

- preparation d'un subset PCam / PatchCamelyon depuis fichiers HDF5 ;
- conversion vers une structure `ImageFolder` ;
- entrainement PyTorch avec transfer learning ;
- evaluation avec accuracy, precision, recall, F1, ROC-AUC et PR-AUC ;
- benchmark ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- application Streamlit avec prediction, probabilites et Grad-CAM ;
- analyse d'erreurs false positives / false negatives ;
- analyse de seuils 0.30 / 0.40 / 0.50 / 0.60 / 0.70 ;
- tests automatises et documentation portfolio.

Quelques resultats experimentaux :

- ResNet18 baseline : 90.40 % accuracy test ;
- EfficientNet-B0 benchmark V2 : 93.20 % accuracy test et 93.20 % macro F1 ;
- ROC-AUC EfficientNet-B0 : 97.62 % ;
- PR-AUC EfficientNet-B0 : 97.80 % ;
- au seuil 0.50 : 24 faux positifs et 27 faux negatifs sur 750 images test ;
- au seuil 0.30 : recall `metastatic` 96.53 %, mais plus de faux positifs ;
- au seuil 0.70 : moins de faux positifs, mais plus de faux negatifs.

La partie la plus interessante du projet est l'analyse de seuils : elle montre pourquoi
l'accuracy seule ne suffit pas. Modifier le seuil change directement le compromis entre
faux positifs et faux negatifs.

Je garde une lecture prudente : PCam est un dataset public et ces resultats ne sont pas
une validation clinique. La valeur du projet est dans le pipeline ML complet, le
benchmark, les metriques ROC/PR, l'analyse d'erreurs, l'interpretabilite et la
presentation reproductible.

#MachineLearning #ComputerVision #PyTorch #Streamlit #DataScience #AI

## Version Sobre Professionnelle

Nouveau projet portfolio finalise : **Metastasis Vision**, sous-projet du monorepo
R.C.C.I.A.

Le projet met en place un pipeline de computer vision sur PCam / PatchCamelyon :

- classification `non_metastatic` vs `metastatic` ;
- conversion HDF5 vers ImageFolder ;
- entrainement PyTorch / Torchvision ;
- benchmark ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- Streamlit avec Grad-CAM ;
- evaluation ROC-AUC / PR-AUC ;
- analyse d'erreurs et de seuils.

Resultat principal : EfficientNet-B0 atteint 93.20 % d'accuracy test, 97.62 % de
ROC-AUC et 97.80 % de PR-AUC sur le split local. L'analyse V2.1 montre le compromis
entre seuil bas, meilleur recall `metastatic`, et hausse des faux positifs.

Projet strictement educatif et portfolio. Il ne constitue pas un outil medical, pas un
diagnostic et pas une validation clinique.

#ComputerVision #PyTorch #MachineLearning #Portfolio
