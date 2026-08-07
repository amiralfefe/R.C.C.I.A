# MultiCancer V1 Roadmap

Statut global : **MultiCancer V1 complete / portfolio-ready**.

## Phase 1 - Registry Et Schemas

Statut : termine en V1.1.

- metadonnees communes finalisees ;
- contrat `BaseAdapter` defini ;
- schemas checkpoint, prediction et explication ajoutes ;
- erreurs communes et limites par projet validees.

## Phase 2 - Specialized Adapters

- Leukemia : integre en V1.1 ;
- Breast : integre en V1.2 ;
- Metastasis : integre en V1.3 ;
- LungColon : integre en V1.4 avec modes multiclass et binary explicites ;
- reutiliser les fonctions existantes sans modifier les pipelines source.

LungColon a ete integre en dernier afin de valider deux checkpoints specialises dans un
meme adaptateur sans chargement simultane.

## Phase 3 - Streamlit Hub

Statut : parcours des quatre projets termines.

- selection explicite des quatre fiches ;
- metadonnees affichees avant inference ;
- chargement paresseux d'un seul modele ;
- page utilisable si le checkpoint est absent ;
- dechargement lors du changement de projet.

## Phase 4 - Predictions Normalisees

Statut : contrat et parcours des quatre projets termines.

- classe predite ;
- confiance ;
- probabilites par classe ;
- nom du modele et taille d'entree ;
- avertissements et disclaimer.

## Phase 5 - Explainability

Statut : Grad-CAM integre pour les quatre projets et les deux modes LungColon.

La V1.3 ajoute une couche `ThresholdDecision` pure et distincte de l'argmax du modele.
La V1.4 ajoute `AdapterModeMetadata` et un unload strict lors des bascules LungColon.

- integrer Grad-CAM lorsque l'adaptateur le supporte ;
- afficher un message propre lorsqu'il est indisponible ;
- conserver le caractere exploratoire de l'explication.

## Phase 6 - Comparaison Methodologique

Statut : termine.

- afficher datasets, classes, resolutions et splits ;
- presenter les metriques avec leur protocole ;
- interdire tout classement global simpliste.

## Phase 7 - Tests Et Gestion Des Erreurs

Statut : termine.

- checkpoint absent ou incompatible ;
- image invalide ou format non supporte ;
- adaptateur indisponible ;
- contrat commun de prediction ;
- absence de chargement multiple de modeles.

## Phase 8 - Final QA Et Portfolio Pack

Statut : termine en V1.5.

- QA reelle des cinq parcours avec checkpoints locaux ;
- probabilites et dimensions Grad-CAM verifiees ;
- AppTest final des transitions Leukemia, Breast, Metastasis et LungColon ;
- seuil Metastasis verifie sans nouvelle inference ;
- changement de mode LungColon sans sortie obsolette ;
- resume projet, pitch entretien, brouillons LinkedIn et release notes ;
- captures Streamlit et README final ;
- scan des fichiers sensibles, 120 tests passes et tag `multicancer-v1`.

## Criteres De Fin V1

- les quatre projets sont selectionnables ;
- un seul modele est charge a la fois ;
- une prediction specialisee fonctionne pour chaque checkpoint local disponible ;
- le resultat est affiche au format commun ;
- le disclaimer reste visible ;
- un checkpoint absent est gere proprement ;
- aucun dataset ou modele n'est melange avec un autre projet ;
- aucun artefact lourd ou sensible n'est versionne.

Tous les criteres de fin V1 sont valides. Aucune V2 n'est planifiee automatiquement.
