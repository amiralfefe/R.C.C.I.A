# MultiCancer V1 Roadmap

## Phase 1 - Registry Et Schemas

- finaliser les metadonnees communes ;
- definir le contrat des adaptateurs ;
- valider les avertissements et limites par projet.

## Phase 2 - Specialized Adapters

- creer un adaptateur Leukemia ;
- creer un adaptateur LungColon ;
- creer un adaptateur Breast ;
- creer un adaptateur Metastasis ;
- reutiliser les fonctions existantes sans modifier les pipelines source.

## Phase 3 - Streamlit Hub

- selection explicite du projet ;
- affichage des metadonnees avant inference ;
- chargement paresseux d'un seul modele ;
- page utilisable meme si certains checkpoints sont absents.

## Phase 4 - Predictions Normalisees

- classe predite ;
- confiance ;
- probabilites par classe ;
- nom du modele et taille d'entree ;
- avertissements et disclaimer.

## Phase 5 - Explainability

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
