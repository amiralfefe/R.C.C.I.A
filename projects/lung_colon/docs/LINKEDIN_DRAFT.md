# Lung + Colon Vision - LinkedIn Drafts

## Version Courte

J'ai finalise une nouvelle brique de mon monorepo R.C.C.I.A : **Lung + Colon Vision**.

Objectif : construire un demonstrateur educatif de computer vision sur le dataset public LC25000, avec classification histopathologique lung/colon en 5 classes, comparaison de modeles, Grad-CAM, analyse d'erreurs et mode binaire `benign` vs `malignant`.

Stack : Python, PyTorch, Torchvision, Streamlit, Scikit-learn, Grad-CAM, Pytest.

Resultats experimentaux sur LC25000 :

- EfficientNet-B0 : 99.92 % accuracy test en mode 5 classes
- 3 erreurs sur 3 750 images test dans l'analyse V2.1
- Mode binaire : 100 % accuracy test sur le split local

Ces resultats restent a interpreter comme un benchmark educatif sur dataset public, pas comme une validation clinique ni un outil de diagnostic.

#MachineLearning #ComputerVision #PyTorch #DataScience #Portfolio

## Version Detaillee

J'ai termine **Lung + Colon Vision**, un sous-projet de mon monorepo R.C.C.I.A consacre a des projets IA/data de computer vision sur datasets publics lies au cancer.

Le but n'etait pas de creer un outil medical, mais un projet portfolio complet, reproductible et bien documente.

Ce que le projet couvre :

- preparation du dataset public LC25000 ;
- classification 5 classes lung/colon ;
- entrainement PyTorch avec transfer learning ;
- evaluation avec accuracy, precision, recall, F1-score et matrice de confusion ;
- comparaison ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- demo Streamlit avec predictions, probabilites et Grad-CAM ;
- analyse des erreurs ;
- mode binaire `benign` vs `malignant` ;
- tests automatises et documentation portfolio.

Quelques resultats experimentaux :

- ResNet18 baseline 5 classes : 99.65 % accuracy test ;
- EfficientNet-B0 benchmark V2 : 99.92 % accuracy test ;
- Error analysis V2.1 : 3 erreurs sur 3 750 images test ;
- Mode binaire V2.2 : 100 % accuracy test sur le split local.

Je garde une lecture prudente : LC25000 est un dataset public relativement facile, donc ces scores ne sont pas une preuve de performance clinique. La valeur du projet est surtout dans le pipeline ML complet, l'analyse des erreurs, l'interpretabilite et la presentation reproductible.

#MachineLearning #ComputerVision #PyTorch #Streamlit #DataScience #AI

## Version Sobre Professionnelle

Nouveau projet portfolio finalise : **Lung + Colon Vision**, sous-projet du monorepo R.C.C.I.A.

Le projet met en place un pipeline de computer vision sur le dataset public LC25000 :

- classification histopathologique en 5 classes ;
- entrainement PyTorch / Torchvision ;
- evaluation detaillee ;
- benchmark ResNet18, MobileNetV3 small et EfficientNet-B0 ;
- Streamlit avec Grad-CAM ;
- analyse d'erreurs ;
- mode binaire `benign` vs `malignant`.

Resultat principal : EfficientNet-B0 atteint 99.92 % d'accuracy test sur le split local LC25000, avec 3 erreurs sur 3 750 images test dans l'analyse V2.1.

Projet strictement educatif et portfolio. Il ne constitue pas un outil medical, pas un diagnostic et pas une validation clinique.

#ComputerVision #PyTorch #MachineLearning #Portfolio
