# MultiCancer Decisions

## Decisions Techniques V0

### Hub Specialise Plutot Que Modele Universel

Les quatre projets traitent des modalites, tissus, classes et resolutions differents.
MultiCancer les orchestre sans fusionner leurs datasets ni leurs modeles.

### Routage Explicite Par Projet

L'utilisateur choisira la tache. Ce choix limite l'ambiguite et rend visible le contexte
de la prediction.

### Adaptateurs Independants

Chaque adaptateur reutilisera le package existant du projet et conservera son
preprocessing, son chargement de checkpoint et son Grad-CAM.

### Schema Commun Minimal

Deux dataclasses standard suffisent en V0 : `ProjectMetadata` et `PredictionResult`.
Cette approche evite une nouvelle dependance et laisse les details specifiques dans les
adaptateurs.

### Reutilisation Des Checkpoints Existants

La V1 reutilisera les checkpoints locaux existants lorsqu'ils sont disponibles. Aucun
checkpoint n'est requis pour consulter le registre ou la page V0.

### Aucun Reentrainement Et Aucune Fusion

La V0 n'entraine aucun modele, ne telecharge aucune donnee et ne fusionne aucun dataset.

### Limites Affichees Avec Chaque Prediction

Le disclaimer global ne suffit pas : la V1 devra aussi afficher la note methodologique
et les limites du projet selectionne.

## Decisions Reportees

- detection automatique de modalite ;
- API FastAPI commune ;
- deploiement public ;
- cache de modeles ;
- chargement paresseux avance ;
- authentification ;
- suivi d'usage ;
- schema persistant de metriques ;
- choix ou calibration de seuil a usage medical.

## Risques Identifies

| Risque | Reponse prevue |
| --- | --- |
| Interfaces existantes incompatibles | Isoler chaque package derriere un adaptateur teste |
| Dependances ou imports differents | Charger uniquement l'adaptateur selectionne |
| Poids memoire de plusieurs modeles | Un seul modele charge a la fois en V1 |
| Metriques non comparables | Afficher le protocole et une note methodologique avec les scores |
| Checkpoint absent sur une autre machine | Registre independant des checkpoints et message d'erreur clair |
| Duplication de preprocessing | Reutiliser le preprocessing du package specialise |
| Confusion entre demonstration et diagnostic | Disclaimer global et limites par projet toujours visibles |
| Donnees sensibles chargees par erreur | Aucun stockage, aucune persistance et consignes explicites |

## Critere De Reexamen

Une decision reportee ne sera reouverte que si elle apporte une valeur portfolio claire,
reste compatible avec le cadre educatif et peut etre testee sans fragiliser les quatre
projets existants.
