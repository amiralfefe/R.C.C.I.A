# MultiCancer - LinkedIn Drafts

## Version Courte

J'ai finalise **MultiCancer V1**, le hub de synthese de mon monorepo R.C.C.I.A.

Le hub regroupe quatre pipelines de computer vision specialises : Leukemia, Breast,
Metastasis et LungColon, avec cinq parcours reels lorsque les modes multiclass et binary
de LungColon sont comptes separement.

Points techniques : architecture Adapter, lazy loading, un seul modele actif,
probabilites normalisees, Grad-CAM, seuil exploratoire Metastasis et modes LungColon
explicites.

Le projet conserve les differences de datasets, classes, resolutions et protocoles. Il
ne cherche pas a creer une IA universelle.

Demonstrateur educatif / portfolio uniquement : pas diagnostic, pas dispositif medical,
pas validation clinique.

#MachineLearning #ComputerVision #PyTorch #Streamlit #Portfolio

## Version Detaillee

Je viens de finaliser **MultiCancer V1**, le projet de synthese de R.C.C.I.A, mon
monorepo de computer vision sur datasets publics lies au cancer.

MultiCancer reunit quatre projets specialises :

- Leukemia : `normal` vs `leukemia_blast` ;
- Breast : BreakHis `benign` vs `malignant`, split patient-aware ;
- Metastasis : PCam `non_metastatic` vs `metastatic`, ROC/PR et analyse de seuils ;
- LungColon : LC25000 en cinq classes ou en mode binaire explicite.

Le principal choix d'architecture a ete de ne pas fusionner artificiellement ces taches.
Le hub route explicitement vers le bon pipeline et normalise seulement la presentation.

Ce que V1 apporte :

- contrat commun d'adaptateurs ;
- `ModelManager` garantissant un seul modele en memoire ;
- lazy loading des checkpoints locaux ;
- sortie `PredictionResult` commune ;
- probabilites et Grad-CAM ;
- `ThresholdDecision` Metastasis separee de l'argmax ;
- deux modes LungColon avec unload et nettoyage d'etat ;
- gestion controlee des checkpoints absents et images invalides ;
- QA reelle des cinq parcours et tests automatises.

Un resultat interessant de la QA est une prediction Breast incorrecte avec une forte
confiance. Elle rappelle qu'une probabilite elevee ne garantit pas une decision correcte
et que l'analyse par patient, les limites du dataset et la validation externe comptent
autant que l'accuracy.

MultiCancer n'est pas une IA qui detecte tous les cancers. C'est un hub educatif qui met
en valeur plusieurs pipelines specialises, leurs compromis et leurs limites.

#MachineLearning #ComputerVision #PyTorch #Streamlit #DataScience #MLOps

## Version Sobre Professionnelle

Projet portfolio finalise : **R.C.C.I.A MultiCancer V1**.

Le projet propose une interface Streamlit commune pour quatre pipelines specialises de
computer vision, sans melanger leurs datasets ou leurs modeles.

Fonctionnalites principales :

- routage explicite par projet ;
- architecture Adapter et resultat normalise ;
- chargement paresseux d'un seul checkpoint ;
- probabilites et Grad-CAM ;
- analyse de seuils exploratoire pour Metastasis ;
- modes multiclass/binary distincts pour LungColon ;
- tests du lifecycle, des erreurs et des transitions d'etat.

Le projet est strictement educatif et portfolio. Il n'a aucune validation clinique et ne
constitue ni un diagnostic ni un dispositif medical.

#ComputerVision #MachineLearning #Python #Portfolio
