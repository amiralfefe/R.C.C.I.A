# MultiCancer V1 Roadmap

## Phase 1 - Registry Et Schemas

Statut : termine en V1.1.

- metadonnees communes finalisees ;
- contrat `BaseAdapter` defini ;
- schemas checkpoint, prediction et explication ajoutes ;
- erreurs communes et limites par projet validees.

## Phase 2 - Specialized Adapters

- Leukemia : integre en V1.1 ;
- Breast : prochaine integration ;
- Metastasis : troisieme integration ;
- LungColon : quatrieme integration ;
- reutiliser les fonctions existantes sans modifier les pipelines source.

LungColon est volontairement dernier car son adaptateur devra distinguer la
classification 5 classes et le mode binaire `benign` / `malignant`.

## Phase 3 - Streamlit Hub

Statut : parcours Leukemia termine, autres adaptateurs en attente.

- selection explicite des quatre fiches ;
- metadonnees affichees avant inference ;
- chargement paresseux d'un seul modele ;
- page utilisable si le checkpoint est absent ;
- dechargement lors du changement de projet.

## Phase 4 - Predictions Normalisees

Statut : contrat et parcours Leukemia termines.

- classe predite ;
- confiance ;
- probabilites par classe ;
- nom du modele et taille d'entree ;
- avertissements et disclaimer.

## Phase 5 - Explainability

Statut : Grad-CAM Leukemia integre, autres projets en attente.

- integrer Grad-CAM lorsque l'adaptateur le supporte ;
- afficher un message propre lorsqu'il est indisponible ;
- conserver le caractere exploratoire de l'explication.

## Phase 6 - Comparaison Methodologique

- afficher datasets, classes, resolutions et splits ;
- presenter les metriques avec leur protocole ;
- interdire tout classement global simpliste.

## Phase 7 - Tests Et Gestion Des Erreurs

- checkpoint absent ou incompatible ;
- image invalide ou format non supporte ;
- adaptateur indisponible ;
- contrat commun de prediction ;
- absence de chargement multiple de modeles.

## Criteres De Fin V1

- les quatre projets sont selectionnables ;
- un seul modele est charge a la fois ;
- une prediction specialisee fonctionne pour chaque checkpoint local disponible ;
- le resultat est affiche au format commun ;
- le disclaimer reste visible ;
- un checkpoint absent est gere proprement ;
- aucun dataset ou modele n'est melange avec un autre projet ;
- aucun artefact lourd ou sensible n'est versionne.
